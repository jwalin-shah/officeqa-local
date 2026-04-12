#!/usr/bin/env python3
"""Run retrieve (+ optional extract/compute) for UIDs listed in a rubric JSON sample.

The rubric file (e.g. decompose_rubric_sample_40.json) only holds *brief* data_requests
for human review. This script joins each UID to the **full** cached spec in
decompose_eval.full.jsonl — that is what retrieve / fast-path / extract need.

Examples:
  # No LLM: show which DRs the deterministic fast-path can resolve after retrieve
  uv run python scripts/run_rubric_uids_pipeline.py --report-only

  # Full extract+compute (uses API when fast-path does not cover all DRs)
  uv run python scripts/run_rubric_uids_pipeline.py

  uv run python scripts/run_rubric_uids_pipeline.py --rubric path/to/other.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_jsonl_by_uid(path: Path) -> dict[str, dict]:
    by_uid: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_uid[row["uid"]] = row
    return by_uid


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    p = argparse.ArgumentParser(description="Rubric UIDs → cached spec → retrieve / extract probe")
    p.add_argument(
        "--rubric",
        default=str(ROOT / "decompose_rubric_sample_40.json"),
        help="JSON array with objects containing 'uid'",
    )
    p.add_argument(
        "--jsonl",
        default=str(ROOT / "decompose_eval.full.jsonl"),
        help="Cached decompose specs (full QuestionSpec per UID)",
    )
    p.add_argument(
        "--report-only",
        action="store_true",
        help="Retrieve + fast-path probe only (no extract LLM / no compute)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        metavar="N",
        help="Process only the first N rubric entries (0 = all)",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    from solve import _run_extract_and_compute, _try_deterministic_fast_path, retrieve_for_spec

    rubric_path = Path(args.rubric)
    if not rubric_path.is_absolute():
        rubric_path = ROOT / rubric_path
    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_absolute():
        jsonl_path = ROOT / jsonl_path

    bundle = json.loads(rubric_path.read_text())
    if args.limit and args.limit > 0:
        bundle = bundle[: args.limit]
    by_uid = _load_jsonl_by_uid(jsonl_path)

    for item in bundle:
        uid = item["uid"]
        cached = by_uid.get(uid)
        if not cached:
            print(f"{uid}  MISSING in {jsonl_path.name}", flush=True)
            continue
        question = cached["question"]
        spec = cached.get("spec")
        if not spec:
            print(f"{uid}  no spec in cache", flush=True)
            continue

        per_dr = retrieve_for_spec(spec, question, verbose=args.verbose)
        resolved, unresolved = _try_deterministic_fast_path(spec, per_dr, verbose=False)
        n_dr = len(spec.get("data_requests") or [])
        n_res = len(resolved)
        n_un = len(unresolved)

        if args.report_only:
            print(
                f"{uid}  drs={n_dr}  fastpath_resolved={n_res}  need_llm={n_un}  "
                f"unresolved={unresolved}",
                flush=True,
            )
            continue

        ans, extraction = _run_extract_and_compute(spec, per_dr, question, verbose=args.verbose)
        mode = (
            "deterministic" if (extraction or {}).get("notes") == "deterministic" else "mixed/llm"
        )
        print(f"{uid}  {mode}  answer={ans!r}", flush=True)


if __name__ == "__main__":
    main()
