"""Invoke a local coding CLI with a task file. Used by agent_iterate.sh.

Environment:
  OFFICEQA_AGENT — shell-style command prefix (parsed with shlex.split), e.g.
    'codex exec --sandbox workspace-write --full-auto'

Args: <task_spec.txt> <worktree_directory>
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: _invoke_agent.py <task_spec.txt> <worktree_dir>", file=sys.stderr)
        return 2
    raw = os.environ.get("OFFICEQA_AGENT", "").strip()
    if not raw:
        print("OFFICEQA_AGENT is not set", file=sys.stderr)
        return 2
    spec_path = Path(sys.argv[1])
    worktree = Path(sys.argv[2])
    prompt = spec_path.read_text()
    try:
        prefix = shlex.split(raw)
    except ValueError as e:
        print(f"OFFICEQA_AGENT shlex error: {e}", file=sys.stderr)
        return 2
    cmd = prefix + [prompt]
    return subprocess.call(cmd, cwd=worktree)


if __name__ == "__main__":
    raise SystemExit(main())
