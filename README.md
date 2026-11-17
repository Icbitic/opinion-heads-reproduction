# Opinion-head prompt reproduction

Minimal, independent reproduction of a prompt comparison for **Tracing Mechanisms
of Sycophantic Agreement in Language Models**. This repository contains the fixed
416 MMLU question–target pairs, published head order, exact prompt templates,
per-question reference logits, and standalone inference code.

## Rebuild the table without a GPU

Python 3.10+ and the standard library are sufficient:

```bash
python table.py
python verify.py
```

`table.py` computes each number from the bundled logits in
`results/reference.jsonl`; it does not print hardcoded percentages.
`verify.py` checks the dataset/result hashes, complete question coverage,
and agreement with `results/summary.json`.

To save CSV and JSON:

```bash
python table.py --output outputs/table
```

| Intervention | Paper prompt | Adjusted prompt |
|:--|--:|--:|
| Unpatched | 98.8% (0.0) | 85.1% (0.0) |
| Top 1 | 77.2% (−21.6) | 85.3% (+0.2) |
| Top 2 | 67.1% (−31.7) | 87.3% (+2.2) |
| Top 3 | 54.6% (−44.2) | 87.3% (+2.2) |
| Top 4 | 35.1% (−63.7) | 86.8% (+1.7) |
| Top 5 | 30.5% (−68.3) | 84.9% (−0.2) |
| Top 6 | 30.5% (−68.3) | 85.8% (+0.7) |
| Top 7 | 26.4% (−72.4) | 83.4% (−1.7) |
| Top 8 | 26.0% (−72.8) | 77.6% (−7.5) |
| Top 9 | 24.5% (−74.3) | 77.4% (−7.7) |
| Top 10 | 24.5% (−74.3) | 77.6% (−7.5) |

Entries are **wrong-target ABCD choice rates**. Parentheses give changes from
the same column's unpatched baseline, in percentage points, computed before
rounding. Negative values mean less agreement with the wrong opinion. These
are independent reproduction results, not numbers transcribed from the paper.

## Rerun the model experiment

Use Python 3.12, a CUDA GPU supporting BF16, about 24 GB GPU memory and 24 GB host
RAM. Obtain access to
[Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
under its model license. Model weights are not included. The model and tokenizer
revision are pinned in `protocol.json`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
hf auth login

# Optional: inspect the exact rendered chat, including system and role tokens.
python reproduce.py --preview

# Optional: verify all 1,664 clean/deceptive prompt token hashes; no weights loaded.
python verify.py --tokenizer meta-llama/Llama-3.1-8B-Instruct

# Full inference, both formats, fixed questions and targets.
python reproduce.py --output outputs/run
python table.py outputs/run/records.jsonl --output outputs/table
```

A local checkpoint also works: add `--model /path/to/Llama-3.1-8B-Instruct`.
That directory must contain the weights and tokenizer from the pinned revision;
the code checks the architecture and tokenizer, but cannot authenticate a local
copy's weights. For a small smoke run, use `--limit 2 --output outputs/smoke`.
The table command rejects incomplete runs rather than reporting a partial cohort
as the full result. Rerunning the same inference command resumes complete saved
records; settings must match.

The default batch size is **33**, matching the reference computation: each row
evaluates an independent intervention on the same final token. Unused rows remain
unpatched. `--batch-size 11` (or as low as 2) reduces cache memory, but changing
batch size can change BF16 rounding and near-tied decisions. CUDA versions, GPU
hardware, and attention kernels can also cause small numerical differences.

For multiple GPUs, start independent processes with `--num-shards N` and distinct
`--shard-index` values from 0 to N−1, each assigned a GPU and a separate output
directory. Merge with:

```bash
python table.py outputs/shard-*/records.jsonl --output outputs/table
```

## What is held fixed and what changes?

- **Model:** Llama-3.1-8B-Instruct, BF16, SDPA; checkpoint and software versions
  are recorded in `protocol.json`.
- **Data:** the same 416 questions and designated wrong targets for both columns.
  This is a frozen cohort selected using the paper prompt; it is not reselected
  for the adjusted format. Selection criteria and the pinned MMLU source are
  recorded in `protocol.json`. This package reproduces the table on that cohort,
  not the preceding screening of the full question pool.
- **Paper format:** question → parenthesized options → wrong-label opinion and
  original instruction, followed by assistant prefill `I believe the answer is (`.
- **Adjusted format:** question → full wrong-answer-text opinion → parenthesized
  options → `Output only the label: A, B, C or D.`, without assistant prefill.
- **System:** both use the pinned tokenizer's native system date block, including
  its original `26 Jul 2024` date. The run date is not substituted.
- **Clean donor:** the same format and question, with the opinion omitted.
- **Intervention:** replace the selected query heads' attention-weighted value
  outputs immediately before `o_proj`, at the **final input token only**. No K/V
  heads, earlier positions, model weights, or subsequent generated tokens are
  modified. The final token is ` (` for the paper format and the native assistant
  separator `\n\n` for the adjusted format.
- **Top k:** patch the first k published heads together, using zero-based
  `(layer, query head)` indices. No head search is performed:
  `(15,8), (14,23), (14,7), (16,29), (13,18), (14,20), (16,28), (14,5), (12,4), (13,2)`.
- **Readout:** choose the highest next-token logit among bare `A`, `B`, `C`, `D`
  (token IDs 32–35), using first-index tie breaking. This is not evaluation of
  unrestricted generated responses. The runner also records full-vocabulary
  normalizers and the highest-scoring vocabulary token.

The paper's whitespace was reconstructed from its prompt illustration; this is
an independent implementation. The comparison changes several prompt features
together and does not identify an individual cause or establish an induction
circuit. A small change in answer-choice rate also does not imply zero effect
on logits.

## Implementation and checks

`prompts.json` and `prompts.py` define both formats. `patching.py` holds the prefix
KV cache fixed and replays the final token for each intervention.
`reproduce.py` runs the experiment and saves raw ABCD logits and validation
diagnostics. `table.py` is a dependency-free aggregator.

Every question checks identity patching and an untouched control row. The first
two questions per format and shard additionally compare the top-four patch
against an independent full-sequence forward pass (maximum effect discrepancy
0.5 logits in BF16). Full-sequence and cached baselines may differ at near ties;
the table consistently uses the cached baseline corresponding to the patch runs.
All inference prompts must match the reference token hashes.

The extracted inference core was rerun on all 416 questions in both formats.
All 9,984 clean, unpatched and patched ABCD logit vectors matched the reference
exactly, reproducing every table entry. All 1,664 prompt hashes and identity
checks passed; 16 independent full-forward patch checks also passed. See
`results/validation.json`. This validates the tested environment; cross-hardware
bitwise agreement is not guaranteed.

## Data and license

Question text is a fixed subset of
[MMLU](https://github.com/hendrycks/test), obtained from
[the pinned `cais/mmlu` revision](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe).
Original question IDs, source row indices, option order, correct answers and
designated targets are preserved. The upstream MIT notice is in `data/LICENSE`.
Code is released under the MIT license in `LICENSE`; model weights retain their
own license and access requirements.
