#!/usr/bin/env python3
"""Compare two eval_suite run directories, focusing on retrieval deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> dict[str, dict]:
    return {row["uid"]: row for row in (json.loads(line) for line in path.open() if line.strip())}


def _retrieval_delta(base_dir: Path, head_dir: Path) -> dict:
    base_summary = _read_json(base_dir / "retrieval" / "summary.json")
    head_summary = _read_json(head_dir / "retrieval" / "summary.json")
    base_rows = _read_jsonl(base_dir / "retrieval" / "details.jsonl")
    head_rows = _read_jsonl(head_dir / "retrieval" / "details.jsonl")

    improvements: list[dict] = []
    regressions: list[dict] = []
    unchanged = 0
    for uid in sorted(base_rows.keys() & head_rows.keys()):
        b = base_rows[uid].get("first_hit_rank")
        h = head_rows[uid].get("first_hit_rank")
        if b == h:
            unchanged += 1
            continue
        row = {"uid": uid, "base_rank": b, "head_rank": h}
        if b is None and h is not None:
            improvements.append(row)
        elif b is not None and h is None:
            regressions.append(row)
        elif b is not None and h is not None and h < b:
            improvements.append(row)
        else:
            regressions.append(row)

    metric_keys = (
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_at_30",
        "recall_at_50",
    )
    return {
        "summary_delta": {
            key: round(head_summary.get(key, 0.0) - base_summary.get(key, 0.0), 1)
            for key in metric_keys
        },
        "improved_uids": improvements[:50],
        "regressed_uids": regressions[:50],
        "counts": {
            "improved": len(improvements),
            "regressed": len(regressions),
            "unchanged": unchanged,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("base", type=Path)
    ap.add_argument("head", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    result = {
        "base": str(args.base),
        "head": str(args.head),
        "retrieval": _retrieval_delta(args.base, args.head),
    }
    print(json.dumps(result, indent=2))
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"Wrote comparison to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
