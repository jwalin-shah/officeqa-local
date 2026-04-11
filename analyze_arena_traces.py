#!/usr/bin/env python3
"""Classify failures in an archived arena trace directory.

Joins officeqa-uid*.json trajectory files against officeqa_full.csv for
gold answers, then runs archive_from_arena/analyst/classify_failures.py
on each failing trace. Writes a bucket summary and a JSONL of misses.

Usage:
    python3 analyze_arena_traces.py [trace_dir] [--csv FILE] [--out FILE]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "archive_from_arena" / "analyst"))
import contextlib

from classify_failures import classify_task  # type: ignore

DEFAULT_TRACE_DIR = Path("/Users/jwalinshah/archive/officeqa-arena/traces_comprehensive/v20_183.8")

SUBMIT_NAMES = ("submit_answer", "final_answer", "submit", "answer", "report")


def task_id_to_uid(task_id: str) -> str:
    m = re.search(r"uid(\d+)", task_id, flags=re.I)
    return f"UID{m.group(1).zfill(4)}" if m else task_id


def load_gold(csv_path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out[row["uid"]] = {
                "question": row["question"],
                "gold": row["answer"],
                "source_files": row.get("source_files", ""),
                "difficulty": row.get("difficulty", ""),
            }
    return out


def extract_predicted(steps: list[dict]) -> str:
    for step in reversed(steps):
        for tc in step.get("tool_calls") or []:
            name = (tc.get("function_name") or tc.get("name") or "").lower()
            if any(s in name for s in SUBMIT_NAMES):
                args = tc.get("arguments") or tc.get("args") or {}
                if isinstance(args, str):
                    with contextlib.suppress(Exception):
                        args = json.loads(args)
                if isinstance(args, dict):
                    for k in ("answer", "final_answer", "value", "result"):
                        if k in args:
                            return str(args[k])
                return str(args)[:200]
    for step in reversed(steps):
        if step.get("source") != "agent":
            continue
        msg = step.get("message") or ""
        if msg:
            nums = re.findall(r"-?\$?[\d,]+\.?\d*%?", msg)
            if nums:
                return nums[-1]
            return msg.strip().splitlines()[-1][:200]
    return ""


def last_agent_snippet(steps: list[dict]) -> str:
    for s in reversed(steps):
        if s.get("source") == "agent" and s.get("message"):
            return s["message"][:300]
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace_dir", type=Path, nargs="?", default=DEFAULT_TRACE_DIR)
    ap.add_argument("--csv", type=Path, default=HERE / "officeqa_full.csv")
    ap.add_argument("--out", type=Path, default=HERE / "arena_misses.jsonl")
    args = ap.parse_args()

    if not args.trace_dir.is_dir():
        sys.exit(f"No trace dir: {args.trace_dir}")
    gold = load_gold(args.csv)
    files = sorted(args.trace_dir.glob("officeqa-uid*.json"))
    print(f"Loaded {len(files)} traces from {args.trace_dir.name}")
    print(f"Loaded {len(gold)} gold rows from {args.csv.name}")

    misses: list[dict] = []
    pass_count = 0
    unmapped = 0
    buckets: Counter = Counter()

    for p in files:
        try:
            with open(p) as f:
                trace = json.load(f)
        except Exception as e:
            print(f"  [warn] {p.name}: {e}", file=sys.stderr)
            continue
        task_id = trace.get("task_id", p.stem)
        uid = task_id_to_uid(task_id)
        g = gold.get(uid)
        if g is None:
            unmapped += 1
            continue

        reward = trace.get("reward")
        if reward is not None and float(reward) >= 1.0:
            pass_count += 1
            continue

        steps = (trace.get("trajectory") or {}).get("steps") or []
        for s in steps:
            if s.get("tool_calls") is None:
                s["tool_calls"] = []
            if s.get("observation") is None:
                s["observation"] = {}
            obs = s["observation"]
            if isinstance(obs, dict):
                results = obs.get("results")
                if results is None:
                    obs["results"] = []
                else:
                    for r in results:
                        if isinstance(r, dict) and r.get("content") is None:
                            r["content"] = ""
        predicted = extract_predicted(steps)
        row = {
            "uid": uid,
            "question": g["question"],
            "gold": g["gold"],
            "predicted": predicted,
            "elapsed_sec": None,
            "tool_calls": sum(len(s.get("tool_calls") or []) for s in steps),
        }
        category = classify_task(row, steps)
        buckets[category] += 1
        misses.append(
            {
                "uid": uid,
                "question": g["question"],
                "gold": g["gold"],
                "predicted": predicted,
                "category": category,
                "difficulty": g.get("difficulty", ""),
                "source_files": g.get("source_files", ""),
                "last_agent_msg_snippet": last_agent_snippet(steps),
                "tool_call_count": row["tool_calls"],
                "trace_path": str(p),
            }
        )

    total = len(files) - unmapped
    n_fail = len(misses)
    print()
    print(f"Results for {args.trace_dir.name}")
    print(f"  total matched: {total}")
    print(f"  pass:          {pass_count}  ({pass_count / max(total, 1) * 100:.1f}%)")
    print(f"  fail:          {n_fail}  ({n_fail / max(total, 1) * 100:.1f}%)")
    if unmapped:
        print(f"  unmapped:      {unmapped}")
    print()
    print("Failure buckets:")
    for cat, n in buckets.most_common():
        pct = n / max(n_fail, 1) * 100
        print(f"  {cat:<22} {n:>4}  ({pct:.1f}%)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for m in misses:
            f.write(json.dumps(m) + "\n")
    print()
    print(f"Wrote {n_fail} misses to {args.out}")


if __name__ == "__main__":
    main()
