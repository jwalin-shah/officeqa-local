#!/usr/bin/env bash
# Run Cursor Agent CLI once in a git worktree with a task spec (non-interactive).
#
# Usage:
#   ./scripts/cursor_agent_once.sh retrieval
#   ./scripts/cursor_agent_once.sh /abs/path/to/worktree /abs/path/to/spec.txt
#
# Environment:
#   OFFICEQA_WT_ROOT   — parent of per-stream dirs (default: ~/projects/officeqa-wt)
#   CURSOR_AGENT_BIN    — override path to cursor-agent
#   CURSOR_AGENT_MODE   — empty = edit-capable headless (--force); "plan" = --mode plan (read-only)
#   CURSOR_AGENT_MODEL  — e.g. sonnet-4 (optional)
#   OFFICEQA_CURSOR_SANDBOX — pass through to --sandbox (e.g. "disabled" for fewer prompts)
#
# Examples:
#   CURSOR_AGENT_MODE=plan ./scripts/cursor_agent_once.sh retrieval
#   OFFICEQA_WT_ROOT="$PWD/.agent-worktrees" ./scripts/cursor_agent_once.sh eval
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT_ROOT="${OFFICEQA_WT_ROOT:-$HOME/projects/officeqa-wt}"
SPECS_DIR="$ROOT/scripts/agent_task_specs"

resolve_paths() {
  if [[ "$#" -eq 1 ]]; then
    local stream="$1"
    case "$stream" in
      retrieval|decompose|fastpath|extract|eval|ingest) ;;
      *)
        echo "usage: $0 <stream>|<<worktree-dir> <spec.txt>>" >&2
        echo "  stream one of: retrieval decompose fastpath extract eval ingest" >&2
        exit 2
        ;;
    esac
    WORKTREE="$WT_ROOT/$stream"
    SPEC="$SPECS_DIR/${stream}.txt"
  elif [[ "$#" -eq 2 ]]; then
    WORKTREE="$(cd "$1" && pwd)"
    SPEC="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
  else
    echo "usage: $0 <stream>" >&2
    echo "   or: $0 <worktree-dir> <spec.txt>" >&2
    exit 2
  fi
}

resolve_paths "$@"

if [[ ! -d "$WORKTREE" ]]; then
  echo "error: worktree not found: $WORKTREE" >&2
  exit 1
fi
if [[ ! -f "$SPEC" ]]; then
  echo "error: spec not found: $SPEC" >&2
  exit 1
fi

BIN="${CURSOR_AGENT_BIN:-}"
if [[ -z "$BIN" ]]; then
  BIN="$(command -v cursor-agent 2>/dev/null || true)"
fi
if [[ -z "$BIN" ]]; then
  BIN="$HOME/.local/bin/cursor-agent"
fi
if [[ ! -x "$BIN" ]]; then
  echo "error: cursor-agent not executable at: $BIN" >&2
  exit 127
fi

PROMPT="$(cat "$SPEC")"

ARGS=(--workspace "$WORKTREE" -p --trust)
if [[ "${CURSOR_AGENT_MODE:-}" == "plan" ]]; then
  ARGS+=(--mode plan)
else
  ARGS+=(--force)
fi
if [[ -n "${CURSOR_AGENT_MODEL:-}" ]]; then
  ARGS+=(--model "$CURSOR_AGENT_MODEL")
fi
if [[ -n "${OFFICEQA_CURSOR_SANDBOX:-}" ]]; then
  ARGS+=(--sandbox "$OFFICEQA_CURSOR_SANDBOX")
fi

echo "Running: $BIN ${ARGS[*]} -- <prompt from $(basename "$SPEC")>" >&2
exec "$BIN" "${ARGS[@]}" -- "$PROMPT"
