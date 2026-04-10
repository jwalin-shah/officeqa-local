#!/usr/bin/env python3
"""Run multiple questions in parallel and report accuracy."""

import concurrent.futures
import contextlib
import csv
import sys

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


def main():
    questions = []
    with open("officeqa_full.csv") as f:
        for row in csv.DictReader(f):
            questions.append(row)

    test_n = 10
    for i, a in enumerate(sys.argv):
        if a == "--n" and i + 1 < len(sys.argv):
            with contextlib.suppress(ValueError):
                test_n = int(sys.argv[i + 1])
    sample = questions[:test_n]

    print(f"Testing {test_n} questions...\n")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(run_one, row): row for row in sample}
        for future in concurrent.futures.as_completed(futures):
            uid, correct, expected, got, rationale = future.result()
            mark = "✅" if correct else "❌"
            if correct:
                print(f"{mark} {uid}: {got!r}")
            else:
                print(f"{mark} {uid}: expected={expected!r}  got={got!r}  ({rationale})")
            results.append(correct)

    accuracy = sum(results) / len(results) * 100
    print(f"\nAccuracy: {sum(results)}/{len(results)} = {accuracy:.0f}%")


if __name__ == "__main__":
    main()
