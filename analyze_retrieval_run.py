#!/usr/bin/env python3
"""Summarize retrieval run artifacts into rank buckets and tuning headroom."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def _bucket(rank: int | None) -> str:
    if rank is None:
        return "miss"
    if rank <= 5:
        return "top_5"
    if rank <= 10:
        return "rank_6_10"
    if rank <= 20:
        return "rank_11_20"
    if rank <= 30:
        return "rank_21_30"
    if rank <= 50:
        return "rank_31_50"
    return "beyond_50"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("details", type=Path, help="retrieval details jsonl from eval_retrieve.py")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    rows = _load_rows(args.details)
    usable = [r for r in rows if not r.get("error") and not r.get("skipped")]
    counts: Counter[str] = Counter()
    table_counts: Counter[str] = Counter()
    rank_sum = 0.0
    rank_count = 0
    examples: dict[str, list[str]] = {
        k: [] for k in ("rank_11_20", "rank_21_30", "rank_31_50", "miss")
    }
    table_examples: dict[str, list[str]] = {
        k: [] for k in ("rank_11_20", "rank_21_30", "rank_31_50", "miss")
    }

    for row in usable:
        rank = row.get("first_hit_rank")
        bucket = _bucket(rank)
        counts[bucket] += 1
        table_rank = row.get("first_table_hit_rank")
        table_bucket = _bucket(table_rank)
        table_counts[table_bucket] += 1
        if rank is not None:
            rank_sum += 1.0 / rank
            rank_count += 1
        if bucket in examples and len(examples[bucket]) < 10:
            examples[bucket].append(row["uid"])
        if table_bucket in table_examples and len(table_examples[table_bucket]) < 10:
            table_examples[table_bucket].append(row["uid"])

    n = len(usable)
    summary = {
        "total_questions": n,
        "file_recall_at_5": round(counts["top_5"] / n * 100, 1) if n else 0.0,
        "file_recall_at_10": round((counts["top_5"] + counts["rank_6_10"]) / n * 100, 1)
        if n
        else 0.0,
        "file_recall_at_20": round(
            (counts["top_5"] + counts["rank_6_10"] + counts["rank_11_20"]) / n * 100, 1
        )
        if n
        else 0.0,
        "file_recall_at_30": round(
            (counts["top_5"] + counts["rank_6_10"] + counts["rank_11_20"] + counts["rank_21_30"])
            / n
            * 100,
            1,
        )
        if n
        else 0.0,
        "file_recall_at_50": round(
            (
                counts["top_5"]
                + counts["rank_6_10"]
                + counts["rank_11_20"]
                + counts["rank_21_30"]
                + counts["rank_31_50"]
            )
            / n
            * 100,
            1,
        )
        if n
        else 0.0,
        "table_recall_at_5": round(table_counts["top_5"] / n * 100, 1) if n else 0.0,
        "table_recall_at_10": round(
            (table_counts["top_5"] + table_counts["rank_6_10"]) / n * 100, 1
        )
        if n
        else 0.0,
        "table_recall_at_20": round(
            (table_counts["top_5"] + table_counts["rank_6_10"] + table_counts["rank_11_20"])
            / n
            * 100,
            1,
        )
        if n
        else 0.0,
        "table_recall_at_30": round(
            (
                table_counts["top_5"]
                + table_counts["rank_6_10"]
                + table_counts["rank_11_20"]
                + table_counts["rank_21_30"]
            )
            / n
            * 100,
            1,
        )
        if n
        else 0.0,
        "table_recall_at_50": round(
            (
                table_counts["top_5"]
                + table_counts["rank_6_10"]
                + table_counts["rank_11_20"]
                + table_counts["rank_21_30"]
                + table_counts["rank_31_50"]
            )
            / n
            * 100,
            1,
        )
        if n
        else 0.0,
        "mrr": round(rank_sum / n, 4) if n else 0.0,
        "rank_buckets": dict(counts),
        "table_rank_buckets": dict(table_counts),
        "headroom": {
            "move_11_20_into_top10": counts["rank_11_20"],
            "move_21_30_into_top10": counts["rank_21_30"],
            "move_31_50_into_top10": counts["rank_31_50"],
            "misses": counts["miss"],
        },
        "table_headroom": {
            "move_11_20_into_top10": table_counts["rank_11_20"],
            "move_21_30_into_top10": table_counts["rank_21_30"],
            "move_31_50_into_top10": table_counts["rank_31_50"],
            "misses": table_counts["miss"],
        },
        "examples": examples,
        "table_examples": table_examples,
    }

    print(json.dumps(summary, indent=2))
    if args.out:
        args.out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote summary to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
