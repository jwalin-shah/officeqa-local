#!/usr/bin/env python3
"""Cross-reference pulled Arena traces with STABILITY_REPORT.md task buckets.

Usage:
    python3 scripts/triage_traces_vs_stability.py \\
        --traces-dir nomcp/results/traces/v3.0.0_166

Parses UID lists from ## 1. Task Categories in STABILITY_REPORT.md and reports
pass/fail counts per bucket for the trace JSON files present.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "STABILITY_REPORT.md"


def parse_stability_buckets() -> dict[str, set[str]]:
    """Return bucket name -> set of task ids e.g. officeqa-uid0002."""
    text = REPORT.read_text(encoding="utf-8")
    buckets: dict[str, set[str]] = {}
    current: str | None = None
    for line in text.splitlines():
        m = re.match(r"^### (ALWAYS_PASS|USUALLY_PASS|FLIP_FLOP|USUALLY_FAIL|ALWAYS_FAIL)", line)
        if m:
            current = m.group(1)
            buckets[current] = set()
            continue
        if current and "**UIDs:**" in line:
            uids = re.findall(r"UID(\d{4})", line)
            for u in uids:
                buckets[current].add(f"officeqa-uid{u}")
    return buckets


def task_id_from_filename(name: str) -> str:
    if name.endswith(".json"):
        return name[:-5]
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description="Triage traces vs stability buckets")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        default=None,
        help="Directory of *.json trajectory files (default: latest under nomcp/results/traces/)",
    )
    args = parser.parse_args()

    trace_dir = args.traces_dir
    if trace_dir is None:
        base = ROOT / "nomcp" / "results" / "traces"
        if not base.is_dir():
            raise SystemExit(f"No traces base: {base}")
        candidates = sorted(
            [p for p in base.iterdir() if p.is_dir() and p.name != "latest"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise SystemExit(f"No versioned trace dirs under {base}")
        trace_dir = candidates[0]
        print(f"Using latest trace dir: {trace_dir}\n")

    buckets = parse_stability_buckets()
    if not buckets:
        raise SystemExit(f"Could not parse buckets from {REPORT}")

    # Load per-task outcome from traces
    outcomes: dict[str, str] = {}
    for path in sorted(trace_dir.glob("officeqa-uid*.json")):
        tid = task_id_from_filename(path.name)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            outcomes[tid] = f"error:{e}"
            continue
        reward = data.get("reward", 0)
        outcomes[tid] = "pass" if reward and float(reward) > 0 else "fail"

    print("=" * 72)
    print("TRACE RUN vs STABILITY BUCKETS (historical pass-rate groups)")
    print("=" * 72)
    print(f"Trace directory: {trace_dir}")
    print(f"Tasks with JSON: {len(outcomes)}\n")

    for bucket in (
        "ALWAYS_PASS",
        "USUALLY_PASS",
        "FLIP_FLOP",
        "USUALLY_FAIL",
        "ALWAYS_FAIL",
    ):
        uids = buckets.get(bucket, set())
        present = [u for u in uids if u in outcomes]
        passed = sum(1 for u in present if outcomes[u] == "pass")
        failed = sum(1 for u in present if outcomes[u] == "fail")
        missing = len(uids) - len(present)
        total = len(present)
        rate = (passed / total * 100.0) if total else 0.0
        print(f"{bucket} (historical cohort: {len(uids)} tasks)")
        print(f"  In this trace dir: {total} present, {missing} missing JSON")
        print(f"  This run: {passed} pass, {failed} fail  ({rate:.1f}% pass of present)")
        print()

    # Priority hint: high historical pass bucket but fail now
    regressions = []
    for uid in buckets.get("ALWAYS_PASS", set()):
        if uid in outcomes and outcomes[uid] == "fail":
            regressions.append(uid)
    if regressions:
        print("Potential regressions (ALWAYS_PASS historically but FAIL this run):")
        for uid in sorted(regressions)[:40]:
            print(f"  {uid}")
        if len(regressions) > 40:
            print(f"  ... and {len(regressions) - 40} more")
        print()

    # High-priority fixes: ALWAYS_FAIL cohort still failing
    still_fail = [
        uid
        for uid in buckets.get("ALWAYS_FAIL", set())
        if uid in outcomes and outcomes[uid] == "fail"
    ]
    print(
        f"ALWAYS_FAIL cohort: {len(still_fail)} still failing in this run (focus for tool/prompt work)."
    )


if __name__ == "__main__":
    main()
