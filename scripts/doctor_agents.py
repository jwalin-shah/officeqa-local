from __future__ import annotations

import os
import shutil
from pathlib import Path


def _check_file(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "missing"
    if path.is_dir():
        return True, "dir"
    if os.access(path, os.X_OK):
        return True, "executable"
    return True, "file"


def main() -> int:
    home = Path.home()
    checks: list[tuple[str, bool, str]] = []

    for binary in ("rtk", "llm-tldr", "claude", "codex", "gemini", "cursor-agent", "uv", "rg"):
        resolved = shutil.which(binary)
        checks.append((f"bin:{binary}", resolved is not None, resolved or "missing"))

    for rel in (
        ".agent-rules/COMMON_TRAVERSAL.md",
        ".agent-hooks/rtk-rewrite.sh",
        ".agent-hooks/source-read-warn.sh",
        ".agent-hooks/context-budget-warn.sh",
        ".agent-hooks/agent-doctor.sh",
        ".claude/CLAUDE.md",
        ".claude/hooks/tldr-warn.sh",
        ".cursor/hooks.json",
        ".cursor/hooks/rtk-rewrite.sh",
        ".cursor/cli-config.json",
        ".codex/config.toml",
    ):
        ok, detail = _check_file(home / rel)
        checks.append((f"home:{rel}", ok, detail))

    repo_checks = [
        Path("AGENTS.md"),
        Path("scripts/agent_task_specs/retrieval.txt"),
        Path("docs/AGENT_ORCHESTRATION.md"),
    ]
    for path in repo_checks:
        ok, detail = _check_file(path)
        checks.append((f"repo:{path}", ok, detail))

    failures = [label for label, ok, _ in checks if not ok]
    width = max(len(label) for label, _, _ in checks)
    for label, ok, detail in checks:
        mark = "OK" if ok else "MISS"
        print(f"{mark:4} {label:<{width}}  {detail}")

    if failures:
        print("\nMissing pieces:")
        for label in failures:
            print(f"- {label}")
        return 1

    print("\nAgent environment looks ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
