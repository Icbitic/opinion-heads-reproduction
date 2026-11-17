#!/usr/bin/env python3
"""Build the three-column table from per-question logits (standard library only)."""
import argparse
import csv
import json
import math
from pathlib import Path

from prompts import FORMATS, ROOT, read_rows


def prediction(score):
    logits = score["logits"]
    if len(logits) != 4 or not all(math.isfinite(x) for x in logits):
        raise ValueError("Expected four finite ABCD logits")
    return max(range(4),key=logits.__getitem__)


def aggregate(paths):
    cohort = read_rows(ROOT/"data/cohort.jsonl")
    rows = {}
    for path in paths:
        for row in read_rows(path):
            key = row["id"],row["format"]
            if key in rows:
                raise ValueError(f"Duplicate result: {key}")
            if len(row["patched"]) != 10:
                raise ValueError("Expected all ten cumulative interventions")
            rows[key] = row
    expected = {(r["id"],f) for r in cohort for f in FORMATS}
    if set(rows) != expected:
        raise ValueError(f"Incomplete or unexpected coverage: missing {len(expected-set(rows))}, extra {len(set(rows)-expected)}")
    counts = {f:[0]*11 for f in FORMATS}
    for fixed in cohort:
        for form in FORMATS:
            row = rows[fixed["id"],form]
            if any(row[k] != fixed[k] for k in ("correct_index","target_index")):
                raise ValueError("Correct answer or target changed")
            for k,score in enumerate([row["unpatched"],*row["patched"]]):
                counts[form][k] += prediction(score)==row["target_index"]
    n = len(cohort)
    return dict(n=n,counts=counts,rates={f:[100*x/n for x in counts[f]] for f in FORMATS},
                changes_pp={f:[100*(x-counts[f][0])/n for x in counts[f]] for f in FORMATS})


def markdown(report):
    lines = ["| Intervention | Paper prompt | Adjusted prompt |", "|:--|--:|--:|"]
    for k in range(11):
        cells = []
        for form in FORMATS:
            rate,change = report["rates"][form][k],report["changes_pp"][form][k]
            diff = "0.0" if change==0 else f"{change:+.1f}".replace("-","−")
            cells.append(f"{rate:.1f}% ({diff})")
        lines.append("| "+" | ".join(["Unpatched" if k==0 else f"Top {k}",*cells])+" |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("records",nargs="*",type=Path,help="One or more complete run/shard JSONL files")
    parser.add_argument("--output",type=Path,help="Write table.csv and summary.json to this directory")
    args = parser.parse_args()
    report = aggregate(args.records or [ROOT/"results/reference.jsonl"])
    print(markdown(report))
    print(f"\nN={report['n']}; wrong-target ABCD choice rate. Parentheses: change from unpatched, in percentage points.")
    if args.output:
        args.output.mkdir(parents=True,exist_ok=True)
        (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
        with (args.output/"table.csv").open("w",newline="") as file:
            writer=csv.writer(file,lineterminator="\n")
            writer.writerow(["Intervention","Paper prompt","Adjusted prompt"])
            for k in range(11):
                writer.writerow(["Unpatched" if k==0 else f"Top {k}",
                                 *[f"{report['rates'][f][k]:.1f}% ({report['changes_pp'][f][k]:+.1f} pp)" for f in FORMATS]])


if __name__ == "__main__":
    main()
