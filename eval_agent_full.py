"""Agent-only retrieval eval — compare against retrieve_v2 baseline.

Runs run_search_agent on every question in the sample (no deterministic
pipeline) and reports hit@5 / hit@10 vs baseline.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from retrieve_v2 import _ledger_conn
from search_agent import _hydrate_entries, run_search_agent

BASELINE = Path("retrieve_eval.jsonl")
DECOMPOSE_CACHE = Path("decompose_eval.full.jsonl")


def _gold_stems(rec: dict) -> set[str]:
    golds = rec.get("gold") or rec.get("gold_files") or []
    if isinstance(golds, str):
        golds = [golds]
    return {g.replace(".txt", "").replace(".json", "") for g in golds}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--uids",
        type=str,
        default="/tmp/eval_sample_uids.json",
        help="JSON file with list of UIDs to evaluate",
    )
    ap.add_argument("--n", type=int, default=0, help="Limit to first N (0=all)")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--out", type=str, default="agent_full_eval.jsonl")
    args = ap.parse_args()

    # Load UIDs to evaluate
    if args.uids and Path(args.uids).exists():
        with open(args.uids) as f:
            uid_list: list[str] = json.load(f)
    else:
        uid_list = []

    # Load baseline
    baseline_by_uid: dict[str, dict] = {}
    with BASELINE.open() as f:
        for line in f:
            r = json.loads(line)
            baseline_by_uid[r["uid"]] = r

    # Load decompose cache
    cache: dict[str, dict] = {}
    with DECOMPOSE_CACHE.open() as f:
        for line in f:
            r = json.loads(line)
            cache[r["uid"]] = r

    # Filter to requested UIDs
    if uid_list:
        rows = [baseline_by_uid[u] for u in uid_list if u in baseline_by_uid]
    else:
        rows = list(baseline_by_uid.values())
    if args.n:
        rows = rows[: args.n]

    baseline_hit5 = sum(1 for r in rows if r["hit_at_5"])
    print(
        f"evaluating {len(rows)} questions — "
        f"baseline hit@5={baseline_hit5}/{len(rows)} ({baseline_hit5 / len(rows) * 100:.0f}%)",
        flush=True,
    )

    conn = _ledger_conn()
    out_fh = open(args.out, "w")

    agent_hit5 = 0
    agent_hit10 = 0
    both_hit5 = 0  # agent AND baseline both hit
    agent_only_hit5 = 0  # agent hit, baseline miss
    baseline_only_hit5 = 0  # baseline hit, agent miss
    errors = 0
    t0 = time.time()

    for i, rec in enumerate(rows, 1):
        uid = rec["uid"]
        dc = cache.get(uid, {})
        question = dc.get("question", "")
        if not question:
            print(f"  [{i}/{len(rows)}] {uid} SKIP no question", flush=True)
            continue
        plan = dc.get("spec") or None
        gold = _gold_stems(rec)
        b_hit5 = rec["hit_at_5"]

        t_start = time.time()
        try:
            agent_result = run_search_agent(question, plan=plan, verbose=args.verbose)
            table_ids = agent_result.get("table_ids") or []
            entries = _hydrate_entries(conn, table_ids, load_html=False)
            err = None
        except Exception as exc:
            entries = []
            err = f"{type(exc).__name__}: {exc}"
            errors += 1

        files = [e["file"].replace(".json", "").replace(".txt", "") for e in entries]
        a_hit5 = any(f in gold for f in files[:5])
        a_hit10 = any(f in gold for f in files[:10])
        if a_hit5:
            agent_hit5 += 1
        if a_hit10:
            agent_hit10 += 1
        if a_hit5 and b_hit5:
            both_hit5 += 1
        elif a_hit5 and not b_hit5:
            agent_only_hit5 += 1
        elif b_hit5 and not a_hit5:
            baseline_only_hit5 += 1

        dt = time.time() - t_start
        a_status = "HIT5" if a_hit5 else ("HIT10" if a_hit10 else "MISS")
        b_status = "B-HIT" if b_hit5 else "B-MISS"
        err_note = f" ERR={err}" if err else ""
        print(
            f"  [{i}/{len(rows)}] {uid} agent={a_status} {b_status} "
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
                    "baseline_hit5": b_hit5,
                    "agent_hit5": a_hit5,
                    "agent_hit10": a_hit10,
                    "agent_top_files": files,
                    "n_entries": len(entries),
                    "error": err,
                    "elapsed_s": dt,
                }
            )
            + "\n"
        )
        out_fh.flush()

    out_fh.close()
    total = len(rows)
    elapsed = time.time() - t0

    print(
        f"\n=== agent-only vs baseline on {total} questions ===\n"
        f"  baseline hit@5   : {baseline_hit5}/{total} ({baseline_hit5 / total * 100:.0f}%)\n"
        f"  agent hit@5      : {agent_hit5}/{total} ({agent_hit5 / total * 100:.0f}%)\n"
        f"  agent hit@10     : {agent_hit10}/{total} ({agent_hit10 / total * 100:.0f}%)\n"
        f"  --- breakdown ---\n"
        f"  both hit@5       : {both_hit5}\n"
        f"  agent-only hit@5 : {agent_only_hit5}  (recovered by agent)\n"
        f"  baseline-only@5  : {baseline_only_hit5}  (lost by switching to agent)\n"
        f"  errors           : {errors}\n"
        f"  elapsed          : {elapsed:.0f}s (avg {elapsed / max(total, 1):.1f}s/q)",
        flush=True,
    )


if __name__ == "__main__":
    main()
