#!/usr/bin/env python3
"""Run Layer 1 bottom-up tests on Daytona.

Creates a lightweight sandbox (no corpus or DB needed — these tests use
hand-curated evidence or question-only), uploads the test scripts, runs them.

Usage:
    python3 nomcp/bottomup/run_daytona.py          # run both layer1a and layer1b
    python3 nomcp/bottomup/run_daytona.py --layer 1a  # just extraction
    python3 nomcp/bottomup/run_daytona.py --layer 1b  # just planning
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Load .env
_env_file = Path(__file__).resolve().parent.parent.parent / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env_file)
    except ImportError:
        for line in _env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

try:
    from daytona import CreateSandboxFromSnapshotParams, Daytona, DaytonaConfig
except ImportError:
    print("ERROR: daytona SDK not installed. Run: pip install daytona")
    sys.exit(1)

BOTTOMUP_DIR = Path(__file__).resolve().parent
SNAPSHOT_NAME = "officeqa-nomcp"


def _get_client() -> Daytona:
    api_key = os.environ.get("DAYTONA_API_KEY", "")
    if not api_key:
        print("ERROR: Set DAYTONA_API_KEY env var")
        sys.exit(1)
    return Daytona(
        DaytonaConfig(
            api_key=api_key,
            target=os.environ.get("DAYTONA_TARGET", "us"),
        )
    )


def _get_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    # Try arena.yaml
    for yaml_path in [
        BOTTOMUP_DIR.parent / "arena.yaml",
        BOTTOMUP_DIR.parent.parent / "arena.yaml",
    ]:
        if yaml_path.exists():
            try:
                import yaml

                cfg = yaml.safe_load(yaml_path.read_text())
                key = cfg.get("agent", {}).get("env", {}).get("OPENROUTER_API_KEY", "")
                if key:
                    return key
            except Exception:
                pass
    return ""


def run(layer: str = "both", uids: str = ""):
    client = _get_client()
    api_key = _get_api_key()
    if not api_key:
        print("ERROR: No OPENROUTER_API_KEY found")
        sys.exit(1)

    print("Creating sandbox...")
    sandbox = client.create(
        CreateSandboxFromSnapshotParams(
            snapshot=SNAPSHOT_NAME,
            language="python",
            env_vars={
                "OPENROUTER_API_KEY": api_key,
                "LLM_API_KEY": api_key,
                "PYTHONUNBUFFERED": "1",
                "CORPUS_DIR": "/app/corpus",
            },
        ),
        timeout=120,
    )
    print(f"Sandbox ready: {sandbox.id}")

    try:
        # Upload test files
        sandbox.process.exec("mkdir -p /tmp/bottomup", timeout=5)
        for fname in os.listdir(BOTTOMUP_DIR):
            fpath = BOTTOMUP_DIR / fname
            if fpath.is_file() and fname.endswith(".py"):
                sandbox.fs.upload_file(fpath.read_bytes(), f"/tmp/bottomup/{fname}")
                print(f"  Uploaded {fname}")

        if layer == "pipeline":
            # Full E2E pipeline test (needs corpus)
            print("\n" + "=" * 60)
            print("PIPELINE: Full decompose solve (E2E)")
            print("=" * 60)
            cmd = "cd /tmp/bottomup && python3 test_pipeline.py"
            if uids:
                cmd += f" --uids {uids}"
            result = sandbox.process.exec(cmd, timeout=900)
            print(result.result)
            if result.exit_code != 0:
                print(f"[Exit code: {result.exit_code}]")
        else:
            # Layer tests (no corpus needed)
            if layer in ("both", "1a"):
                print("\n" + "=" * 60)
                print("LAYER 1A: Extraction (MiniMax as Intern)")
                print("=" * 60)
                result = sandbox.process.exec(
                    "cd /tmp/bottomup && python3 layer1_extraction.py",
                    timeout=300,
                )
                print(result.result)
                if result.exit_code != 0:
                    print(f"[Exit code: {result.exit_code}]")

            if layer in ("both", "1b"):
                print("\n" + "=" * 60)
                print("LAYER 1B: Planning (MiniMax as Mentor)")
                print("=" * 60)
                result = sandbox.process.exec(
                    "cd /tmp/bottomup && python3 layer1b_planning.py",
                    timeout=300,
                )
                print(result.result)
                if result.exit_code != 0:
                    print(f"[Exit code: {result.exit_code}]")

        # Download results
        print("\n--- Downloading results ---")
        for fname in ["layer1_results.json", "layer1b_results.json", "test_pipeline_results.json"]:
            try:
                content = sandbox.fs.download_file(f"/tmp/bottomup/{fname}")
                out_path = BOTTOMUP_DIR / fname
                out_path.write_bytes(content)
                print(f"  Saved {out_path}")
            except Exception:
                pass

    finally:
        print(f"\nSandbox {sandbox.id} will auto-stop. Done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=["1a", "1b", "both", "pipeline"], default="both")
    parser.add_argument(
        "--uids", type=str, default="", help="Comma-separated UIDs for pipeline test"
    )
    args = parser.parse_args()
    run(args.layer, args.uids)


if __name__ == "__main__":
    main()
