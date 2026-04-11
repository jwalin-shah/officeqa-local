#!/usr/bin/env python3
"""Scan pulled Arena trajectory JSON for harness mode and corpus contract signals.

Looks for:
  - trajectory.agent.name (goose vs openhands)
  - harbor-task, manifest.json, /app/resources in step messages
  - officeqa_* MCP tool names in serialized messages

Usage:
    python3 scripts/audit_traces.py nomcp/results/traces/v3.0.0_166
    python3 scripts/audit_traces.py --summary-only nomcp/results/traces/latest
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

MCP_TOOL = re.compile(r"officeqa_[a-zA-Z0-9_]+")
HARBOR = re.compile(r"harbor-task|harbor task", re.I)
MANIFEST = re.compile(r"manifest\.json|/app/resources/", re.I)


def load_traces(trace_dir: Path) -> list[Path]:
    return sorted(trace_dir.glob("officeqa-uid*.json"))


def step_messages(data: dict) -> str:
    traj = data.get("trajectory") or {}
    steps = traj.get("steps") or []
    parts: list[str] = []
    for s in steps:
        if isinstance(s, dict):
            parts.append(str(s.get("message") or ""))
    return "\n".join(parts)


def audit_file(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    blob = step_messages(data)
    traj = data.get("trajectory") or {}
    agent = (traj.get("agent") or {}) if isinstance(traj.get("agent"), dict) else {}
    name = (agent.get("name") or "").lower()
    mcp_hits = MCP_TOOL.findall(blob)
    return {
        "task_id": data.get("task_id", path.stem),
        "reward": data.get("reward"),
        "agent_name": name or None,
        "harbor": bool(HARBOR.search(blob)),
        "manifest_path_signals": bool(MANIFEST.search(blob)),
        "mcp_tool_mentions": len(mcp_hits),
        "mcp_tool_kinds": len(set(mcp_hits)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Arena trace JSON files")
    parser.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        default=None,
        help="Directory containing officeqa-uid*.json (default: latest under nomcp/results/traces/)",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print aggregate stats only",
    )
    args = parser.parse_args()

    trace_dir = args.trace_dir
    if trace_dir is None:
        base = ROOT / "nomcp" / "results" / "traces"
        candidates = sorted(
            [p for p in base.iterdir() if p.is_dir() and p.name != "latest"],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        trace_dir = candidates[0] if candidates else None
        if trace_dir is None:
            raise SystemExit(f"No trace dirs under {base}")
        print(f"Using: {trace_dir}\n")

    if not trace_dir.is_dir():
        raise SystemExit(f"Not a directory: {trace_dir}")

    files = load_traces(trace_dir)
    if not files:
        raise SystemExit(f"No officeqa-uid*.json in {trace_dir}")

    rows = [audit_file(p) for p in files]
    agents = Counter(r["agent_name"] or "unknown" for r in rows)
    harbor_n = sum(1 for r in rows if r["harbor"])
    manifest_n = sum(1 for r in rows if r["manifest_path_signals"])
    mcp_any = sum(1 for r in rows if r["mcp_tool_mentions"] > 0)
    mcp_total = sum(r["mcp_tool_mentions"] for r in rows)

    print("=" * 72)
    print("TRACE AUDIT")
    print("=" * 72)
    print(f"Directory: {trace_dir}")
    print(f"Files: {len(files)}")
    print()
    print("Agent (trajectory.agent.name):")
    for k, v in agents.most_common():
        print(f"  {k or 'unknown'}: {v}")
    print()
    print("Signals in step message text:")
    print(f"  harbor-task / harbor recipe: {harbor_n}/{len(rows)} traces")
    print(f"  manifest.json or /app/resources/: {manifest_n}/{len(rows)} traces")
    print(f"  traces with officeqa_* substring: {mcp_any}/{len(rows)}")
    print(f"  total officeqa_* substring hits: {mcp_total}")
    print()

    if not args.summary_only:
        print("Per-task (first 30 with mcp_tool_mentions > 0 or agent mismatch):")
        shown = 0
        for r in rows:
            if r["mcp_tool_mentions"] > 0:
                print(
                    f"  {r['task_id']}: agent={r['agent_name']} reward={r['reward']} "
                    f"mcp_substr_hits={r['mcp_tool_mentions']}"
                )
                shown += 1
                if shown >= 30:
                    break
        if mcp_any == 0:
            print(
                "  (no officeqa_* substrings in messages — expect OpenHands+MCP traces for tool names in JSON-RPC)"
            )
        print()

    print("Interpretation:")
    print(
        "  - goose + harbor + manifest signals => file/manifest task contract (subset under /app/resources)."
    )
    print("  - openhands + officeqa_* hits => MCP tool loop with server/mcp_stdio-style tools.")
    print(
        "  - officeqa_* count 0 with goose + shell-heavy runs => prompt/harness mismatch vs root MCP prompt."
    )


if __name__ == "__main__":
    main()
