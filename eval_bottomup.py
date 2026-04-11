"""Eval bottom-up retrieval only — no LLM, no extraction, no HTML fetching.

Uses cached decompose specs. For each DR calls search_cells_bottomup() and
checks whether the gold file appears in top-k cell hits. Fast because we
never fetch raw_html.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

sys.path.insert(0, ".")
from find import search_cells_bottomup

DECOMPOSE_FILE = "decompose_eval.full.jsonl"
TOP_KS = [1, 5, 10, 30]


def gold_stem(f: str) -> str:
    return f.replace(".txt", "").replace(".json", "")


def main() -> None:
    records = []
    with open(DECOMPOSE_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("error") or not d.get("spec"):
                continue
            records.append(d)
    print(f"Loaded {len(records)} questions\n", flush=True)

    hit_at_q = defaultdict(int)
    hit_at_dr = defaultdict(int)
    total_drs = 0
    no_hint = 0
    no_hits = 0

    for i, rec in enumerate(records, 1):
        gold_stems = {gold_stem(f) for f in (rec.get("gold_files") or [])}
        drs = rec["spec"].get("data_requests") or []

        q_hits: dict[int, bool] = {k: False for k in TOP_KS}

        for dr in drs:
            total_drs += 1
            row_hint = dr.get("row_hint", "")
            if not row_hint:
                no_hint += 1
                continue

            years = dr.get("years") or []
            col_year = years[0] if years else None
            topic = " ".join(filter(None, [dr.get("label", ""), row_hint]))

            hits = search_cells_bottomup(
                row_hint=row_hint, col_year=col_year, topic=topic, limit=30
            )
            if not hits:
                no_hits += 1
                continue

            hit_files = [gold_stem(h["file"]) for h in hits]
            for k in TOP_KS:
                if gold_stems & set(hit_files[:k]):
                    hit_at_dr[k] += 1
                    q_hits[k] = True

        for k in TOP_KS:
            if q_hits[k]:
                hit_at_q[k] += 1

        tag = "HIT" if q_hits[1] else "miss"
        print(
            f"[{i:3d}/246] {rec['uid']:10s} {tag:4s}  DRs={len(drs)}  "
            f"gold={str(rec['gold_answer'])[:12]:12s}  "
            f"{(drs[0].get('row_hint', '') if drs else '')[:30]}",
            flush=True,
        )

    n = len(records)
    attempted_drs = total_drs - no_hint
    print(f"\n{'=' * 60}")
    print(f"Questions: {n}  |  DRs: {total_drs}  |  no row_hint: {no_hint}  |  no hits: {no_hits}")
    print(f"Attempted DRs (had row_hint): {attempted_drs}\n")
    print("File hit@k — question level (any DR found gold file):")
    for k in TOP_KS:
        print(f"  top-{k:2d}: {hit_at_q[k]:3d}/{n}  ({100 * hit_at_q[k] / n:.1f}%)")
    print(f"\nFile hit@k — DR level (over {attempted_drs} DRs with row_hint):")
    for k in TOP_KS:
        pct = 100 * hit_at_dr[k] / attempted_drs if attempted_drs else 0
        print(f"  top-{k:2d}: {hit_at_dr[k]:3d}/{attempted_drs}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
