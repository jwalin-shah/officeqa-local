#!/usr/bin/env python3
"""Run decompose() on a sample of benchmark questions concurrently.

Usage:
  uv run python eval_decompose.py [--n N] [--workers W] [--out FILE] [--validate]

Dumps one JSON object per line with {uid, question, gold_files, spec, error, elapsed_s}
so we can inspect whether the LLM is correctly identifying what each question asks for.

With ``--validate``, runs :func:`validate_decompose.check_spec` on each result (no
extra LLM calls) and prints aggregate failure-tag counts.
"""

import concurrent.futures
import csv
import json
import sys
import time

from solve import decompose
from validate_decompose import check_spec

sys.stdout.reconfigure(line_buffering=True)


def run_one(row: dict) -> dict:
    t0 = time.time()
    try:
        spec = decompose(row["question"])
        err = None if spec is not None else "DECOMPOSE_RETURNED_NONE"
    except Exception as e:
        spec, err = None, f"{type(e).__name__}: {e}"
    return {
        "uid": row["uid"],
        "question": row["question"],
        "gold_answer": row["answer"],
        "gold_files": [f.strip() for f in row["source_files"].split("\n") if f.strip()],
        "spec": spec,
        "error": err,
        "elapsed_s": round(time.time() - t0, 2),
    }


def main():
    n = 20
    workers = 20
    out_path = "decompose_eval.jsonl"
    do_validate = False
    for i, a in enumerate(sys.argv):
        if a == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
        elif a == "--workers" and i + 1 < len(sys.argv):
            workers = int(sys.argv[i + 1])
        elif a == "--out" and i + 1 < len(sys.argv):
            out_path = sys.argv[i + 1]
        elif a == "--validate":
            do_validate = True

    with open("officeqa_full.csv") as f:
        rows = list(csv.DictReader(f))[:n]

    print(f"Decomposing {len(rows)} questions with {workers} workers → {out_path}")
    t_start = time.time()
    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, row): row for row in rows}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            mark = "FAIL" if res["error"] else "OK  "
            print(f"  [{done:>3}/{len(rows)}] {mark} {res['uid']}  ({res['elapsed_s']}s)")

    results.sort(key=lambda r: r["uid"])
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    n_fail = sum(1 for r in results if r["error"])
    print(
        f"\nDone in {time.time() - t_start:.1f}s — "
        f"{len(results) - n_fail}/{len(results)} succeeded, {n_fail} failed"
    )

    if do_validate:
        fail_tags: dict[str, int] = {}
        n_flagged = 0
        for r in results:
            row = {
                "uid": r["uid"],
                "question": r["question"],
                "spec": r.get("spec"),
            }
            fails = check_spec(row)
            if fails:
                n_flagged += 1
                for f in fails:
                    key = f.split(":", 1)[-1].split("(", 1)[0]
                    fail_tags[key] = fail_tags.get(key, 0) + 1
        print("\n── validate_decompose (live run) ──")
        print(f"  flagged rows: {n_flagged}/{len(results)}")
        for k, v in sorted(fail_tags.items(), key=lambda x: -x[1]):
            print(f"    {k:32s} {v:4d}")


if __name__ == "__main__":
    main()
