#!/usr/bin/env python3
"""Run multiple questions in parallel and report accuracy."""

import argparse
import concurrent.futures
import csv
import json
import sys
import time

from solve import solve

sys.stdout.reconfigure(line_buffering=True)


def run_one(row):
    uid = row["uid"]
    q = row["question"]
    expected = row["answer"].strip()
    try:
        got = solve(q).strip()
        from reward import fuzzy_match_answer

        correct, rationale = fuzzy_match_answer(expected, got, tolerance=0.01)
        return uid, correct, expected, got, rationale
    except Exception as e:
        return uid, False, expected, f"ERROR: {e}", str(e)


def _parse_uids(raw: str) -> set[str]:
    return {u.strip() for u in raw.split(",") if u.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument(
        "--parallel", type=int, default=10, help="Number of parallel workers (default 10)"
    )
    ap.add_argument("--uids", type=str, default="")
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--summary-out", type=str, default="")
    args = ap.parse_args()

    questions = []
    with open("officeqa_full.csv") as f:
        for row in csv.DictReader(f):
            questions.append(row)

    selected = _parse_uids(args.uids)
    if selected:
        sample = [row for row in questions if row["uid"] in selected]
    else:
        sample = questions[: args.n]

    print(f"Testing {len(sample)} questions ({args.parallel} workers)...\n")
    t0 = time.time()
    results = []
    detailed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        futures = {pool.submit(run_one, row): row for row in sample}
        for future in concurrent.futures.as_completed(futures):
            uid, correct, expected, got, rationale = future.result()
            mark = "✅" if correct else "❌"
            if correct:
                print(f"{mark} {uid}: {got!r}")
            else:
                print(f"{mark} {uid}: expected={expected!r}  got={got!r}  ({rationale})")
            results.append(correct)
            detailed.append(
                {
                    "uid": uid,
                    "correct": correct,
                    "expected": expected,
                    "got": got,
                    "rationale": rationale,
                }
            )

    accuracy = sum(results) / len(results) * 100
    print(f"\nAccuracy: {sum(results)}/{len(results)} = {accuracy:.0f}%")
    if args.out:
        with open(args.out, "w") as f:
            for row in sorted(detailed, key=lambda x: x["uid"]):
                f.write(json.dumps(row) + "\n")
        print(f"Wrote detailed results to {args.out}")
    if args.summary_out:
        summary = {
            "total_questions": len(results),
            "correct": sum(results),
            "accuracy": round(accuracy, 1),
            "elapsed_s": round(time.time() - t0, 2),
        }
        with open(args.summary_out, "w") as f:
            f.write(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote summary to {args.summary_out}")


if __name__ == "__main__":
    main()
