#!/usr/bin/env python3
"""Run the standard retrieval tuning cycle: tune, holdout, audit, optional full."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(cmd: list[str], *, label: str) -> None:
    print(f"\n== {label} ==")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)


def _summary(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="current-main")
    ap.add_argument("--include-full", action="store_true")
    ap.add_argument("--miss-k", type=int, default=10)
    args = ap.parse_args()

    runs_root = ROOT / "runs"
    stages = [
        ("retrieval_tune", runs_root / f"{args.prefix}-retrieval-tune"),
        ("retrieval_holdout", runs_root / f"{args.prefix}-retrieval-holdout"),
        ("retrieval_audit", runs_root / f"{args.prefix}-retrieval-audit"),
    ]
    if args.include_full:
        stages.append(("retrieval_dev", runs_root / f"{args.prefix}-retrieval-full"))

    for subset, out_dir in stages:
        _run(
            [
                sys.executable,
                "eval_suite.py",
                "--subset",
                subset,
                "--stages",
                "retrieval",
                "--out-dir",
                str(out_dir),
            ],
            label=f"eval {subset}",
        )
        _run(
            [
                sys.executable,
                "analyze_retrieval_run.py",
                str(out_dir / "retrieval" / "details.jsonl"),
                "--out",
                str(out_dir / "retrieval" / "analysis.json"),
            ],
            label=f"analyze {subset}",
        )

    for subset, out_dir in stages:
        details_dir = out_dir / "retrieval"
        manifest = {
            "retrieval_tune": ROOT / "eval_subsets" / "retrieval_tune_uids.json",
            "retrieval_holdout": ROOT / "eval_subsets" / "retrieval_holdout_uids.json",
            "retrieval_audit": ROOT / "eval_subsets" / "retrieval_audit_uids.json",
            "retrieval_dev": None,
        }[subset]
        cmd = [
            sys.executable,
            "analyze_retrieve_misses.py",
            "--miss-k",
            str(args.miss_k),
            "--out",
            str(details_dir / "miss_buckets.jsonl"),
            "--summary-out",
            str(details_dir / "miss_buckets_summary.json"),
        ]
        if manifest is not None:
            uids = json.loads(manifest.read_text())
            cmd.extend(["--uids", ",".join(uids)])
        _run(cmd, label=f"misses {subset}")

    report = {}
    for subset, out_dir in stages:
        report[subset] = {
            "summary": _summary(out_dir / "retrieval" / "summary.json"),
            "analysis": _summary(out_dir / "retrieval" / "analysis.json"),
            "miss_buckets": _summary(out_dir / "retrieval" / "miss_buckets_summary.json"),
        }

    report_path = runs_root / f"{args.prefix}-retrieval-cycle-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nWrote cycle report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
