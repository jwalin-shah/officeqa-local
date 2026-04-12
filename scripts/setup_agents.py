from __future__ import annotations

from pathlib import Path


def main() -> int:
    home = Path.home()
    print("Agent setup entrypoint")
    print()
    print("Shared home policy:")
    print(f"- {home / '.agent-rules' / 'COMMON_TRAVERSAL.md'}")
    print()
    print("Recommended launchers:")
    for name in (
        "agent-doctor",
        "agent-claude",
        "agent-codex",
        "agent-gemini",
        "agent-cursor",
    ):
        print(f"- {name}")
    print()
    print("Repo doctor:")
    print("- UV_CACHE_DIR=/tmp/uv-officeqa uv run python -m scripts.doctor_agents")
    print("- UV_CACHE_DIR=/tmp/uv-officeqa uv run python -m scripts.setup_agents")
    print()
    print("Repo policy:")
    print(f"- {Path.cwd() / 'AGENTS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
