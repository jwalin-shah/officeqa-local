#!/usr/bin/env python3
"""Pull full agent trajectories from Arena API for a submission.

Requires: arena-cli installed at ~/.arena/ (uses its venv + auth)

Usage:
    # Pull traces for the latest submission
    ~/.arena/venv/bin/python scripts/pull_arena_traces.py

    # Pull for a specific submission ID
    ~/.arena/venv/bin/python scripts/pull_arena_traces.py --submission-id <ID>

    # Pull and analyze tool usage
    ~/.arena/venv/bin/python scripts/pull_arena_traces.py --analyze

Cross-reference with STABILITY_REPORT buckets:
    python3 scripts/triage_traces_vs_stability.py --traces-dir nomcp/results/traces/<label>
"""

import argparse
import asyncio
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


async def get_latest_submission_id(client, slug: str) -> tuple[str, str]:
    """Get the most recent completed submission ID and a label."""
    from arena_cli.client.submissions import list_submissions

    offset = 0
    page = 50
    while True:
        resp = await list_submissions(client, slug=slug, limit=page, offset=offset)
        for sub in resp.items:
            if sub.status != "completed":
                continue
            version = sub.agent.version if sub.agent else "unknown"
            score = 0.0
            if sub.stages and sub.stages.execute:
                score = sub.stages.execute.score
            label = f"v{version}_{int(score)}"
            return str(sub.id), label
        if not resp.items or offset + len(resp.items) >= resp.total:
            break
        offset += len(resp.items)
    raise RuntimeError("No completed submissions found")


async def pull_traces(submission_id: str, slug: str, out_dir: Path, analyze: bool):
    from arena_cli.client.base import ArenaClient
    from arena_cli.client.trajectories import get_trajectory, list_trajectories

    async with ArenaClient() as client:
        # If no submission ID provided, get the latest
        if not submission_id:
            submission_id, label = await get_latest_submission_id(client, slug)
            out_dir = ROOT / "nomcp" / "results" / "traces" / label
            print(f"Latest submission: {submission_id} ({label})")

        out_dir.mkdir(parents=True, exist_ok=True)

        # Paginate to get all trajectories
        all_items = []
        offset = 0
        while True:
            resp = await list_trajectories(
                client, submission_id=submission_id, slug=slug, limit=100, offset=offset
            )
            all_items.extend(resp.items)
            if len(all_items) >= resp.total or len(resp.items) == 0:
                break
            offset += 100

        passed = sum(1 for i in all_items if i.reward > 0)
        failed = len(all_items) - passed
        print(f"Total: {len(all_items)}, Passed: {passed}, Failed: {failed}")
        print(f"Score: ~{sum(i.reward for i in all_items):.1f}")
        print(f"Output: {out_dir}/")

        # Download full traces
        for idx, item in enumerate(all_items):
            task_id = item.task_id
            out_path = out_dir / f"{task_id}.json"
            if out_path.exists():
                continue
            try:
                detail = await get_trajectory(client, item.trajectory_id, slug=slug)
                traj_dict = detail.model_dump(mode="json")
                with open(out_path, "w") as f:
                    json.dump(traj_dict, f)
                status = "PASS" if item.reward > 0 else "FAIL"
                print(f"  [{idx + 1}/{len(all_items)}] {task_id}: {status}")
            except Exception as e:
                print(f"  [{idx + 1}/{len(all_items)}] {task_id}: ERROR - {e}")

        print(f"\nDone. {len(list(out_dir.glob('*.json')))} traces in {out_dir}/")

        if analyze:
            analyze_traces(out_dir)


def analyze_traces(trace_dir: Path):
    """Analyze tool usage patterns across all traces."""
    print(f"\n{'=' * 60}")
    print("TRACE ANALYSIS")
    print(f"{'=' * 60}\n")

    tool_counts = {}
    mcp_tool_counts = {}
    pass_count = 0
    fail_count = 0
    mcp_used = 0
    shell_used = 0

    for f in sorted(trace_dir.glob("*.json")):
        with open(f) as fh:
            data = json.load(fh)

        reward = data.get("reward", 0)
        if reward > 0:
            pass_count += 1
        else:
            fail_count += 1

        traj = data.get("trajectory", {})
        steps = traj.get("steps", [])
        # Concatenate all step messages (multi-turn trajectories use more than step 0)
        msg = "\n".join((s.get("message") or "") if isinstance(s, dict) else "" for s in steps)

        # Count tool calls
        task_has_mcp = False
        task_has_shell = False

        # MCP tool calls (prefixed with officeqa_)
        mcp_tools = re.findall(r'"name":\s*"officeqa_(\w+)"', msg)
        for t in mcp_tools:
            mcp_tool_counts[t] = mcp_tool_counts.get(t, 0) + 1
            task_has_mcp = True

        # All tool calls (goose built-in)
        all_tools = re.findall(r'"toolCall":\s*\{[^}]*"name":\s*"(\w+)"', msg)
        for t in all_tools:
            tool_counts[t] = tool_counts.get(t, 0) + 1
            if t == "shell":
                task_has_shell = True

        if task_has_mcp:
            mcp_used += 1
        if task_has_shell:
            shell_used += 1

    total = pass_count + fail_count
    print(f"Results: {pass_count}/{total} passed ({pass_count / total * 100:.1f}%)")
    print(f"Tasks using MCP tools: {mcp_used}/{total}")
    print(f"Tasks using shell: {shell_used}/{total}")

    print("\nGoose tool calls:")
    for t, c in sorted(tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:30s} {c:5d}")

    print("\nMCP tool calls (officeqa_*):")
    for t, c in sorted(mcp_tool_counts.items(), key=lambda x: -x[1]):
        print(f"  {t:30s} {c:5d}")


def main():
    parser = argparse.ArgumentParser(description="Pull Arena traces")
    parser.add_argument("--submission-id", default="", help="Submission ID (default: latest)")
    parser.add_argument("--slug", default="grounded-reasoning", help="Competition slug")
    parser.add_argument("--output", default="", help="Output directory")
    parser.add_argument("--analyze", action="store_true", help="Analyze tool usage after download")
    args = parser.parse_args()

    out_dir = Path(args.output) if args.output else ROOT / "nomcp" / "results" / "traces" / "latest"
    asyncio.run(pull_traces(args.submission_id, args.slug, out_dir, args.analyze))


if __name__ == "__main__":
    main()
