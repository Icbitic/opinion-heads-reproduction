#!/usr/bin/env python3
"""Run the fixed-cohort experiment on one CUDA GPU; no scheduler required."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from prompts import FORMATS, PROTOCOL, ROOT, digest, encode, read_rows, render


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score(logits):
    selected = logits[PROTOCOL["answer_token_ids"]].float()
    return dict(logits=selected.cpu().tolist(), logsumexp=float(logits.float().logsumexp(-1)),
                vocabulary_top_token_id=int(logits.argmax()))


def evaluate(model, tokenizer, device, output, *, forms=FORMATS, batch_size=33,
             shard_index=0, num_shards=1, limit=None, progress=None):
    """Reusable experiment core; writes plain JSON files under output."""
    import torch
    from patching import HeadPatcher, replay_cache

    if not 0 <= shard_index < num_shards or batch_size < 2 or (limit is not None and limit < 1):
        raise ValueError("Invalid shard, batch size or limit")
    if not forms or len(set(forms)) != len(forms) or any(f not in FORMATS for f in forms):
        raise ValueError("Invalid prompt formats")
    config = model.config
    if (config.model_type, config.num_hidden_layers, config.num_attention_heads,
            config.num_key_value_heads, config.hidden_size) != ("llama", 32, 32, 8, 4096):
        raise ValueError("Expected Llama-3.1-8B-Instruct architecture")
    if model.dtype != torch.bfloat16 or config._attn_implementation != "sdpa":
        raise ValueError("Reference uses BF16 with SDPA attention")
    for form in forms:
        from prompts import TEMPLATES
        prefix = TEMPLATES[form]["assistant_prefill"]
        before = tokenizer.encode(prefix, add_special_tokens=False)
        for label, token in zip("ABCD", PROTOCOL["answer_token_ids"], strict=True):
            if tokenizer.encode(prefix + label, add_special_tokens=False) != before + [token]:
                raise ValueError("Answer continuation tokenization differs")
    model.eval()
    torch.set_num_threads(1)
    torch.cuda.reset_peak_memory_stats(device)
    for relative, expected in PROTOCOL["files_sha256"].items():
        if file_sha(ROOT / relative) != expected:
            raise ValueError(f"Bundled input changed: {relative}")
    all_rows = read_rows(ROOT / "data/cohort.jsonl")
    rows = [r for i,r in enumerate(all_rows) if i % num_shards == shard_index]
    if limit is not None:
        rows = rows[:limit]
    reference = {(r["id"],r["format"]): r for r in read_rows(ROOT / "results/reference.jsonl")}
    settings = dict(model=PROTOCOL["model"], revision=PROTOCOL["revision"],
                    formats=list(forms), batch_size=batch_size, shard_index=shard_index,
                    num_shards=num_shards, limit=limit,
                    source_sha256={p.name:file_sha(p) for p in ROOT.iterdir()
                                   if p.is_file() and p.suffix in (".py", ".json", ".txt")},
                    cohort_sha256=file_sha(ROOT / "data/cohort.jsonl"),
                    versions={name:version(name) for name in ("torch","transformers","numpy","tokenizers")})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    meta_path, records_path = output / "run.json", output / "records.jsonl"
    if meta_path.exists():
        if json.loads(meta_path.read_text())["settings"] != settings:
            raise ValueError("Existing output has different settings; choose a new directory")
    elif records_path.exists():
        raise ValueError("Existing records have no run metadata")
    meta_path.write_text(json.dumps(dict(settings=settings, complete=False),indent=2)+"\n")
    previous = read_rows(records_path) if records_path.exists() else []
    done = {(r["id"],r["format"]) for r in previous}
    expected_keys = {(r["id"],f) for r in rows for f in forms}
    if len(done) != len(previous) or not done <= expected_keys:
        raise ValueError("Duplicate or unexpected records in output")
    heads, answer_ids = PROTOCOL["heads"], PROTOCOL["answer_token_ids"]
    patcher = HeadPatcher(model, device, heads)
    try:
        with records_path.open("a") as file, torch.inference_mode():
            for index,row in enumerate(rows):
                for form in forms:
                    if (row["id"],form) in done:
                        continue
                    ids = {kind:encode(tokenizer,row,form,kind) for kind in ("clean","deceptive")}
                    hashes = {kind:digest(value) for kind,value in ids.items()}
                    if hashes != reference[row["id"],form]["prompt_sha256"]:
                        raise ValueError("Prompt differs from the reference experiment")
                    prefix = patcher.prefill(ids["clean"])
                    clean_logits, donor = patcher.last(ids["clean"],replay_cache(prefix,batch_size),batch_size,capture=True)
                    clean_score = score(clean_logits[0])
                    del prefix, clean_logits
                    prefix = patcher.prefill(ids["deceptive"])
                    cache = replay_cache(prefix,batch_size)
                    base, recipient = patcher.last(ids["deceptive"],cache,batch_size,capture=True)
                    full = patcher.full(ids["deceptive"])
                    validation = dict(full_vs_replay_max_error=float((base[0,answer_ids]-full[0,answer_ids]).abs().max()),
                                      full_vs_replay_prediction_agrees=int(base[0,answer_ids].argmax())==int(full[0,answer_ids].argmax()))
                    with patcher.intervene([heads[:4]],recipient):
                        identity, _ = patcher.last(ids["deceptive"],cache,batch_size)
                    if not torch.equal(identity,base):
                        raise ValueError("Identity patch changed logits")
                    validation["identity_exact"] = True
                    patched_scores = []
                    for start in range(0,10,batch_size-1):
                        ks = list(range(start+1,min(start+batch_size,11)))
                        specs = [heads[:k] for k in ks]
                        with patcher.intervene(specs,donor):
                            patched, _ = patcher.last(ids["deceptive"],cache,batch_size)
                        if not torch.equal(patched[-1],base[-1]):
                            raise ValueError("Unpatched control row changed")
                        patched_scores.extend(score(patched[j]) for j in range(len(specs)))
                        if index < 2 and 4 in ks:
                            j = ks.index(4)
                            with patcher.intervene([heads[:4]],donor):
                                full_patch = patcher.full(ids["deceptive"])
                            error = float(((patched[j,answer_ids]-base[j,answer_ids])-
                                           (full_patch[0,answer_ids]-full[0,answer_ids])).abs().max())
                            validation["top4_full_forward_effect_error"] = error
                            if error > .5:
                                raise ValueError("Full-forward patch effect differs by more than 0.5 logits")
                    record = dict(id=row["id"],format=form,correct_index=row["correct_index"],
                                  target_index=row["target_index"],clean=clean_score,unpatched=score(base[0]),
                                  patched=patched_scores,prompt_sha256=hashes,validation=validation)
                    file.write(json.dumps(record,separators=(",",":"))+"\n")
                    file.flush()
                    done.add((row["id"],form))
                    del prefix,cache,recipient,donor,base,full,identity,patched
                print(f"Questions {index+1}/{len(rows)}",flush=True)
                if progress:
                    progress(index+1,len(rows))
    finally:
        patcher.close()
    metadata = dict(settings=settings,complete=done==expected_keys,records=len(done),
                    records_sha256=file_sha(records_path),peak_reserved_gib=torch.cuda.max_memory_reserved(device)/2**30)
    meta_path.write_text(json.dumps(metadata,indent=2)+"\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model",default=PROTOCOL["model"],help="Official model ID or local checkpoint directory")
    parser.add_argument("--output",type=Path,default=ROOT/"outputs/run")
    parser.add_argument("--format",choices=FORMATS,action="append",dest="forms")
    parser.add_argument("--batch-size",type=int,default=33,help="33 matches the original numerical setup; minimum 2")
    parser.add_argument("--shard-index",type=int,default=0)
    parser.add_argument("--num-shards",type=int,default=1)
    parser.add_argument("--limit",type=int,help="Smoke-run only; cannot generate a complete table")
    parser.add_argument("--preview",action="store_true",help="Print both live-rendered prompts without loading weights")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model,revision=PROTOCOL["revision"])
    if args.preview:
        row = next(r for r in read_rows(ROOT/"data/cohort.jsonl") if r["id"]=="mmlu-test-college_biology-0073")
        for form in args.forms or FORMATS:
            print(f"\n--- {form} ---\n"+render(tokenizer,row,form,"deceptive"))
        return
    import torch
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("A CUDA GPU with BF16 support is required; table.py works on CPU without dependencies")
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(args.model,revision=PROTOCOL["revision"],
                                               dtype=torch.bfloat16,attn_implementation="sdpa").to(device)
    evaluate(model,tokenizer,device,args.output,forms=args.forms or FORMATS,batch_size=args.batch_size,
             shard_index=args.shard_index,num_shards=args.num_shards,limit=args.limit)


if __name__ == "__main__":
    main()
