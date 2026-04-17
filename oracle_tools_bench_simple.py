#!/usr/bin/env python3
"""Simpler sequential oracle+tools benchmark with reliable file writing."""

import csv
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from oracle_solve_tools import oracle_solve_tools

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "officeqa_full.csv"
OUTPUT = HERE / "oracle_tools_eval.jsonl"

benchmark = {}
with open(BENCHMARK) as f:
    for row in csv.DictReader(f):
        benchmark[row["uid"]] = row

all_uids = sorted(benchmark.keys())

# Parse args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=246)
parser.add_argument("--offset", type=int, default=0)
args = parser.parse_args()

uids = all_uids[args.offset : args.offset + args.n]

print(f"Oracle+Tools: {len(uids)} questions → {OUTPUT}\n", flush=True)

correct = 0
with open(OUTPUT, "a") as out:
    for i, uid in enumerate(uids, 1):
        row = benchmark[uid]
        result = oracle_solve_tools(uid, row["question"], row.get("source_files", ""))

        if "answer" in result:
            from reward import score_answer

            gold = row.get("answer", "")
            score = score_answer(gold, result["answer"])
            result["score"] = score
            result["gold"] = gold
            if score == 1.0:
                correct += 1

        out.write(json.dumps(result) + "\n")
        out.flush()

        if i % 10 == 0:
            print(f"  {i}/{len(uids)}  ({correct} correct so far)", flush=True)

print(f"\nDone. {correct}/{len(uids)} correct")
