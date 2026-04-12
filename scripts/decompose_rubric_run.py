#!/usr/bin/env python3
"""Run decomposition checks: structural validation, fallback scan, rubric sample export.

uv run python scripts/decompose_rubric_run.py
uv run python scripts/decompose_rubric_run.py --random 40 --out decompose_rubric_sample_40.json
uv run python scripts/decompose_rubric_run.py --jsonl path.jsonl --out rubric_sample.json
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_check_spec():
    path = ROOT / "validate_decompose.py"
    spec = importlib.util.spec_from_file_location("validate_decompose", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(mod)
    return mod.check_spec


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(
        description="Decompose structural validation + rubric sample export"
    )
    p.add_argument(
        "--jsonl",
        default=str(ROOT / "decompose_eval.full.jsonl"),
        help="Decompose cache (one JSON object per line)",
    )
    p.add_argument(
        "--out",
        default=str(ROOT / "decompose_rubric_sample.json"),
        help="Write rubric sample JSON here",
    )
    p.add_argument(
        "--random",
        type=int,
        metavar="N",
        default=0,
        help="If N>0, sample N random UIDs (uniform) instead of stratified mix",
    )
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling")
    args = p.parse_args()

    check_spec = _load_check_spec()
    jsonl_path = Path(args.jsonl)
    rows = [json.loads(line) for line in jsonl_path.open()]

    # Structural counts (same buckets as validate_decompose.py)
    fail_counts: Counter[str] = Counter()
    n_clean = 0
    for row in rows:
        fails = check_spec(row)
        if not fails:
            n_clean += 1
        else:
            for f in fails:
                key = f.split(":", 1)[-1].split("(", 1)[0]
                fail_counts[key] += 1

    print(f"=== Structural validation ({jsonl_path.name}, n={len(rows)}) ===")
    for k, v in fail_counts.most_common():
        print(f"  {k:30s} {v:4d}")
    print(f"  clean:   {n_clean}/{len(rows)} ({100 * n_clean / len(rows):.1f}%)")
    print(
        f"  flagged: {len(rows) - n_clean}/{len(rows)} ({100 * (len(rows) - n_clean) / len(rows):.1f}%)"
    )

    # Fallback / broken semantic signals in cache
    fallback_uids: list[str] = []
    for row in rows:
        spec = row.get("spec") or {}
        notes = (spec.get("resolution_notes") or "").lower()
        if "decompose fallback" in notes or "llm failed" in notes:
            fallback_uids.append(row["uid"])
    print(f"\n=== Fallback specs (resolution_notes) ===\n  count: {len(fallback_uids)}")
    if fallback_uids:
        print("  uids:", ", ".join(sorted(fallback_uids)))

    # Gold join + stratified sample (same strategy as prior chat)
    csv_path = ROOT / "officeqa_full.csv"
    with csv_path.open() as f:
        gold = {r["uid"]: r for r in csv.DictReader(f)}
    by_uid = {r["uid"]: r for r in rows}

    rnd = random.Random(args.seed)  # nosec
    all_uids = [r["uid"] for r in rows]
    if args.random and args.random > 0:
        n = min(args.random, len(all_uids))
        uids = rnd.sample(all_uids, n)
        uids.sort(key=lambda u: u)  # stable order for diffing; randomness is which set
    else:
        flagged = [r["uid"] for r in rows if check_spec(r)]
        clean = [r["uid"] for r in rows if not check_spec(r)]
        multi = [
            r["uid"] for r in rows if len((r.get("spec") or {}).get("data_requests") or []) >= 4
        ]
        sample_flagged = rnd.sample(flagged, min(6, len(flagged)))
        sample_clean = rnd.sample(clean, min(6, len(clean)))
        sample_multi = rnd.sample(multi, min(4, len(multi)))
        uids = list(dict.fromkeys(sample_flagged + sample_clean + sample_multi))

    bundle: list[dict] = []
    for uid in uids:
        r = by_uid[uid]
        g = gold[uid]
        spec = r["spec"]
        drs = spec.get("data_requests") or []
        dr_brief = []
        for dr in drs:
            dr_brief.append(
                {
                    "id": dr.get("id"),
                    "label": (dr.get("label") or "")[:160],
                    "row_hint": (dr.get("row_hint") or "")[:140],
                    "column_hint": (dr.get("column_hint") or "")[:100],
                    "years": dr.get("years"),
                    "granularity": dr.get("granularity"),
                    "cohort": dr.get("cohort"),
                }
            )
        bundle.append(
            {
                "uid": uid,
                "struct_fails": check_spec(r),
                "question": r["question"],
                "gold_answer": g["answer"],
                "difficulty": g.get("difficulty", ""),
                "resolution_notes": (spec.get("resolution_notes") or "")[:800],
                "computation": spec.get("computation"),
                "python_template": (spec.get("computation_spec") or {}).get("python_template"),
                "output_format": spec.get("output_format"),
                "vintage": spec.get("vintage"),
                "data_requests_brief": dr_brief,
                "rubric": {
                    "Q": None,
                    "R": None,
                    "C": None,
                    "G": None,
                    "verdict": None,
                    "notes": None,
                },
            }
        )

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.write_text(json.dumps(bundle, indent=2))
    mode = f"random n={args.random}" if args.random > 0 else "stratified"
    print(f"\n=== Rubric sample ({len(bundle)} uids, {mode}, seed={args.seed}) ===")
    print("  uids:", " ".join(uids))
    try:
        rel = out_path.relative_to(ROOT)
    except ValueError:
        rel = out_path
    print(f"  wrote: {rel}")


if __name__ == "__main__":
    main()
