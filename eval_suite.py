#!/usr/bin/env python3
"""Canonical eval wrapper for current-main baselines and commit-to-commit runs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_STAGES = ("ledger", "decompose", "retrieval")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _git_rev_parse(arg: str) -> str:
    return subprocess.check_output(["git", "rev-parse", arg], cwd=ROOT, text=True).strip()


def _load_manifest(path: Path) -> list[str]:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError(f"{path} must be a JSON list of uid strings")
    return data


def _all_uids() -> list[str]:
    with (ROOT / "officeqa_full.csv").open() as f:
        return [row["uid"] for row in csv.DictReader(f)]


def _resolve_subset(name: str, manifest_path: Path | None) -> tuple[str, list[str], str]:
    if manifest_path is not None:
        uids = _load_manifest(manifest_path)
        return manifest_path.stem, uids, _sha256(manifest_path)
    builtins = {
        "smoke": ROOT / "eval_subsets" / "smoke_uids.json",
        "extraction_dev": ROOT / "eval_subsets" / "extraction_dev_uids.json",
        "retrieval_tune": ROOT / "eval_subsets" / "retrieval_tune_uids.json",
        "retrieval_holdout": ROOT / "eval_subsets" / "retrieval_holdout_uids.json",
        "retrieval_audit": ROOT / "eval_subsets" / "retrieval_audit_uids.json",
        "retrieval_dev": None,
        "full_eval": None,
    }
    if name not in builtins:
        raise ValueError(f"Unknown subset {name!r}")
    if builtins[name] is None:
        uids = _all_uids()
        subset_hash = hashlib.sha256(json.dumps(uids).encode()).hexdigest()
        return name, uids, subset_hash
    path = builtins[name]
    assert path is not None
    return name, _load_manifest(path), _sha256(path)


def _run_stage(stage: str, uids: list[str], out_dir: Path) -> dict:
    uids_arg = ",".join(uids)
    stage_dir = out_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    cmd: list[str]
    if stage == "ledger":
        cmd = [
            sys.executable,
            "eval_ledger.py",
            "--uids",
            uids_arg,
            "--out",
            str(stage_dir / "details.jsonl"),
            "--summary-out",
            str(stage_dir / "summary.json"),
        ]
    elif stage == "decompose":
        cmd = [
            sys.executable,
            "eval_decompose_oracle.py",
            "--uids",
            uids_arg,
            "--out",
            str(stage_dir / "details.jsonl"),
            "--summary-out",
            str(stage_dir / "summary.json"),
        ]
    elif stage == "retrieval":
        cmd = [
            sys.executable,
            "eval_retrieve.py",
            "--uids",
            uids_arg,
            "--out",
            str(stage_dir / "details.jsonl"),
            "--summary-out",
            str(stage_dir / "summary.json"),
        ]
    elif stage == "extraction":
        cmd = [
            sys.executable,
            "eval_extract_oracle.py",
            "--uids",
            uids_arg,
            "--out",
            str(stage_dir / "details.jsonl"),
        ]
    elif stage == "end_to_end":
        cmd = [
            sys.executable,
            "batch_test.py",
            "--uids",
            uids_arg,
            "--out",
            str(stage_dir / "details.jsonl"),
            "--summary-out",
            str(stage_dir / "summary.json"),
        ]
    else:
        raise ValueError(f"Unsupported stage {stage!r}")

    print(f"\n== {stage} ==")
    print(" ".join(cmd))
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    (stage_dir / "stdout.log").write_text(proc.stdout)
    (stage_dir / "stderr.log").write_text(proc.stderr)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        return {"status": "failed", "returncode": proc.returncode}
    return {"status": "ok", "returncode": 0}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="smoke")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--stages", default=",".join(DEFAULT_STAGES))
    ap.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "current-main")
    args = ap.parse_args()

    subset_name, uids, subset_hash = _resolve_subset(args.subset, args.manifest)
    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_rev_parse("HEAD"),
        "git_branch": _git_rev_parse("--abbrev-ref HEAD"),
        "python": sys.version,
        "cwd": str(ROOT),
        "subset": subset_name,
        "subset_size": len(uids),
        "subset_hash": subset_hash,
        "stages": stages,
        "inputs": {
            "officeqa_full_csv": {
                "path": "officeqa_full.csv",
                "sha256": _sha256(ROOT / "officeqa_full.csv"),
            },
            "decompose_eval_full_jsonl": {
                "path": "decompose_eval.full.jsonl",
                "sha256": _sha256(ROOT / "decompose_eval.full.jsonl"),
            },
            "ledger_sqlite": {
                "path": "ledger.sqlite",
                "sha256": _sha256(ROOT / "ledger.sqlite"),
            },
        },
        "env": {
            "uv_cache_dir": os.environ.get("UV_CACHE_DIR", ""),
        },
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (args.out_dir / "uids.json").write_text(json.dumps(uids, indent=2) + "\n")

    stage_status: dict[str, dict] = {}
    failed = False
    for stage in stages:
        status = _run_stage(stage, uids, args.out_dir)
        stage_status[stage] = status
        if status["status"] != "ok":
            failed = True
            break

    (args.out_dir / "stage_status.json").write_text(json.dumps(stage_status, indent=2) + "\n")
    print(f"\nWrote run manifest to {args.out_dir / 'manifest.json'}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
