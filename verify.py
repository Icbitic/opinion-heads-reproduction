#!/usr/bin/env python3
"""Verify bundled data/results; optionally check every prompt with the tokenizer."""
import argparse
import hashlib
import json
from pathlib import Path

from prompts import PROTOCOL, ROOT, digest, encode, read_rows
from table import aggregate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer",help="Model ID or local tokenizer directory; enables 1,664 prompt checks")
    args=parser.parse_args()
    for name,expected in PROTOCOL["files_sha256"].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f"Checksum mismatch: {name}")
    reference=ROOT/"results/reference.jsonl"
    summary=aggregate([reference])
    expected=json.loads((ROOT/"results/summary.json").read_text())
    if summary!=expected:
        raise ValueError("Reference table does not match per-question logits")
    checked=0
    if args.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(args.tokenizer,revision=PROTOCOL["revision"])
        data={r["id"]:r for r in read_rows(ROOT/"data/cohort.jsonl")}
        for r in read_rows(reference):
            for kind in ("clean","deceptive"):
                if digest(encode(tokenizer,data[r["id"]],r["format"],kind))!=r["prompt_sha256"][kind]:
                    raise ValueError(f"Prompt mismatch: {r['id']} {r['format']} {kind}")
                checked+=1
    print(json.dumps(dict(questions=summary["n"],records=2*summary["n"],verified_prompt_hashes=checked,table_matches=True)))


if __name__=="__main__":
    main()
