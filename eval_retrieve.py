#!/usr/bin/env python3
"""Measure retrieve_v2.retrieve() recall@K against gold source files.

Loads cached decompose specs from decompose_eval.full.jsonl so we never
hit the LLM. Runs retrieve() with ``top_k=TOP_K`` (50) and records whether
any gold source file appears within ranks 5, 10, 20, 30, and 50 of the
retrieved list.
"""

import concurrent.futures
import json
import sys
import time
from pathlib import Path

from retrieve_v2 import retrieve

sys.stdout.reconfigure(line_buffering=True)

CACHE = Path("decompose_eval.full.jsonl")
TOP_K = 50
WORKERS = 12


def stem(name: str) -> str:
    return Path(name).stem


def run_one(row: dict) -> dict:
    spec = row.get("spec")
    gold = {stem(f) for f in row.get("gold_files") or []}
    if not spec or not gold:
        return {"uid": row["uid"], "skipped": True, "gold": list(gold)}

    t0 = time.time()
    try:
        results = retrieve(spec, row["question"], top_k=TOP_K, load_html=False)
    except Exception as e:
        return {"uid": row["uid"], "error": f"{type(e).__name__}: {e}"}

    retrieved_stems = [stem(r["file"]) for r in results]
    ranks = [i + 1 for i, s in enumerate(retrieved_stems) if s in gold]
    first_hit = ranks[0] if ranks else None
    return {
        "uid": row["uid"],
        "gold": list(gold),
        # Output field name is legacy; value is up to TOP_K (50) stems, not 20.
        "retrieved_top20": retrieved_stems,
        "first_hit_rank": first_hit,
        "hit_at_5": first_hit is not None and first_hit <= 5,
        "hit_at_10": first_hit is not None and first_hit <= 10,
        "hit_at_20": first_hit is not None and first_hit <= 20,
        "hit_at_30": first_hit is not None and first_hit <= 30,
        "hit_at_50": first_hit is not None and first_hit <= 50,
        "elapsed_s": round(time.time() - t0, 2),
    }


def main():
    rows = [json.loads(l) for l in CACHE.open()]  # noqa: E741
    usable = [r for r in rows if r.get("spec") and r.get("gold_files")]
    print(f"Loaded {len(rows)} cached specs; {len(usable)} usable (non-null spec + gold)")

    results: list[dict] = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, r): r for r in usable}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1  # noqa: SIM113
            if res.get("error"):
                mark = "ERR "
            elif res.get("skipped"):
                mark = "SKIP"
            else:
                rank = res["first_hit_rank"]
                mark = f"@{rank:>2}" if rank else "MISS"
            print(f"  [{done:>3}/{len(usable)}] {mark} {res['uid']}")

    n = sum(1 for r in results if not r.get("error") and not r.get("skipped"))
    h5 = sum(1 for r in results if r.get("hit_at_5"))
    h10 = sum(1 for r in results if r.get("hit_at_10"))
    h20 = sum(1 for r in results if r.get("hit_at_20"))
    h30 = sum(1 for r in results if r.get("hit_at_30"))
    h50 = sum(1 for r in results if r.get("hit_at_50"))

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"  n      = {n}")
    print(f"  recall@5  = {h5}/{n} ({h5 / n * 100:.1f}%)")
    print(f"  recall@10 = {h10}/{n} ({h10 / n * 100:.1f}%)")
    print(f"  recall@20 = {h20}/{n} ({h20 / n * 100:.1f}%)")
    print(f"  recall@30 = {h30}/{n} ({h30 / n * 100:.1f}%)")
    print(f"  recall@50 = {h50}/{n} ({h50 / n * 100:.1f}%)")

    Path("retrieve_eval.jsonl").write_text(
        "\n".join(json.dumps(r) for r in sorted(results, key=lambda x: x["uid"])) + "\n"
    )


if __name__ == "__main__":
    main()
