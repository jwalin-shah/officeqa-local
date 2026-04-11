"""Evaluate search_agent fallback against retrieve_v2 misses.

Loads:
  - retrieve_eval.jsonl  (baseline hits / misses at top_k=20)
  - decompose_eval.full.jsonl  (cached decompose specs per uid)

For each hit@5 miss, runs retrieve_with_agent_fallback with the cached
plan and checks whether any returned entry's file matches a gold file.
Reports: hit@5 / hit@10 / hit@20 lift over baseline, plus per-question
status streamed in real time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from retrieve_v2 import _ledger_conn
from search_agent import (
    _hydrate_entries,
    retrieve_with_agent_fallback,
    run_search_agent,
)

BASELINE = Path("retrieve_eval.jsonl")
DECOMPOSE_CACHE = Path("decompose_eval.full.jsonl")


def _load_baseline() -> list[dict]:
    with BASELINE.open() as f:
        return [json.loads(line) for line in f]


def _load_decompose_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if not DECOMPOSE_CACHE.exists():
        return cache
    with DECOMPOSE_CACHE.open() as f:
        for line in f:
            rec = json.loads(line)
            cache[rec["uid"]] = rec
    return cache


def _gold_files_set(rec: dict) -> set[str]:
    golds = rec.get("gold") or rec.get("gold_files") or []
    if isinstance(golds, str):
        golds = [golds]
    return {g.replace(".txt", "").replace(".json", "") for g in golds}


def _entry_file_stem(entry: dict) -> str:
    f = entry.get("file", "")
    return f.replace(".txt", "").replace(".json", "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0, help="limit to first N misses (0=all)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default="agent_fallback_eval.jsonl")
    ap.add_argument(
        "--force",
        action="store_true",
        help="Bypass fallback trigger and always run the agent (measures ceiling).",
    )
    args = ap.parse_args()

    baseline = _load_baseline()
    cache = _load_decompose_cache()

    # Also load questions from full decompose cache (it has them) for misses
    # that aren't in decompose_cache we need the question text somewhere.
    misses = [r for r in baseline if not r["hit_at_5"]]
    if args.n:
        misses = misses[: args.n]

    print(
        f"baseline: {len(baseline)} total, "
        f"{sum(1 for r in baseline if r['hit_at_5'])} hit@5, "
        f"{len(misses)} hit@5 misses — evaluating on {len(misses)}",
        flush=True,
    )

    out_fh = open(args.out, "w")
    agent_hit5 = 0
    agent_hit10 = 0
    agent_triggered = 0
    errors = 0
    t0 = time.time()

    for i, rec in enumerate(misses, 1):
        uid = rec["uid"]
        dc = cache.get(uid, {})
        question = dc.get("question")
        if not question:
            print(f"  [{i}/{len(misses)}] {uid} SKIP no question in cache", flush=True)
            continue
        plan = dc.get("spec") or None
        gold = _gold_files_set(rec)

        t_start = time.time()
        try:
            if args.force:
                agent_result = run_search_agent(question, plan=plan, verbose=args.verbose)
                table_ids = agent_result.get("table_ids") or []
                entries = _hydrate_entries(_ledger_conn(), table_ids, load_html=False)
            else:
                entries = retrieve_with_agent_fallback(
                    plan,
                    question,
                    top_k=args.top_k,
                    verbose=args.verbose,
                    load_html=False,
                )
            err = None
        except Exception as exc:
            entries = []
            err = f"{type(exc).__name__}: {exc}"
            errors += 1

        triggered = args.force or any((e.get("retrieval_channel") == "agent") for e in entries)
        if triggered:
            agent_triggered += 1

        files = [_entry_file_stem(e) for e in entries]
        hit5 = any(f in gold for f in files[:5])
        hit10 = any(f in gold for f in files[:10])
        if hit5:
            agent_hit5 += 1
        if hit10:
            agent_hit10 += 1

        dt = time.time() - t_start
        status = "HIT5" if hit5 else ("HIT10" if hit10 else "MISS")
        trig = "AG" if triggered else "--"
        err_note = f" ERR={err}" if err else ""
        print(
            f"  [{i}/{len(misses)}] {uid} {status} {trig} "
            f"gold={sorted(gold)[:1]} top1={files[:1]} "
            f"n={len(entries)} {dt:.1f}s{err_note}",
            flush=True,
        )
        out_fh.write(
            json.dumps(
                {
                    "uid": uid,
                    "question": question,
                    "gold": sorted(gold),
                    "agent_triggered": triggered,
                    "hit_at_5": hit5,
                    "hit_at_10": hit10,
                    "top_files": files,
                    "error": err,
                    "elapsed_s": dt,
                }
            )
            + "\n"
        )
        out_fh.flush()

    out_fh.close()
    total_elapsed = time.time() - t0
    n = len(misses)
    print(
        f"\n=== agent fallback on {n} baseline hit@5 misses ===\n"
        f"  agent triggered   : {agent_triggered}/{n}\n"
        f"  recovered hit@5   : {agent_hit5}/{n} ({agent_hit5 / max(n, 1) * 100:.1f}%)\n"
        f"  recovered hit@10  : {agent_hit10}/{n} ({agent_hit10 / max(n, 1) * 100:.1f}%)\n"
        f"  errors            : {errors}\n"
        f"  elapsed           : {total_elapsed:.1f}s "
        f"(avg {total_elapsed / max(n, 1):.1f}s/question)",
        flush=True,
    )


if __name__ == "__main__":
    main()
