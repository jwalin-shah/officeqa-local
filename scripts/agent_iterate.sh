#!/usr/bin/env bash
# Run workstream checks in each git worktree in a loop; optionally invoke a
# local coding agent when checks fail (you must set OFFICEQA_AGENT yourself).
#
# This script does NOT ship with any cloud API keys. It only shells out to
# whatever command you put in OFFICEQA_AGENT (e.g. codex exec, claude -p).
#
# Usage:
#   ./scripts/setup_worktrees.sh
#   ./scripts/link_worktree_artifacts.sh
#   MAX_ROUNDS=3 ./scripts/agent_iterate.sh
#
# With a local agent (example — adjust flags to your policy):
#   export OFFICEQA_AGENT='codex exec --sandbox workspace-write'
#   MAX_ROUNDS=2 ./scripts/agent_iterate.sh
#
# With Cursor Agent CLI (uses scripts/cursor_agent_once.sh):
#   OFFICEQA_AGENT_CURSOR=1 MAX_ROUNDS=2 ./scripts/agent_iterate.sh
#
# Dry run (checks only, never invoke agent):
#   DRY_RUN=1 ./scripts/agent_iterate.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT_ROOT="${OFFICEQA_WT_ROOT:-$HOME/projects/officeqa-wt}"
MAX_ROUNDS="${MAX_ROUNDS:-1}"
SPECS_DIR="$ROOT/scripts/agent_task_specs"
DRY_RUN="${DRY_RUN:-0}"

# Space-separated override, e.g. OFFICEQA_STREAMS="retrieval eval"
if [[ -n "${OFFICEQA_STREAMS:-}" ]]; then
  # shellcheck disable=SC2206
  STREAMS=($OFFICEQA_STREAMS)
else
  STREAMS=(retrieval decompose fastpath extract eval ingest)
fi

run_checks() {
  local dir="$1"
  local stream="$2"
  if [[ ! -d "$dir/.venv" ]]; then
    (cd "$dir" && uv sync --all-groups --quiet) || return 1
  fi
  (cd "$dir" && uv run ruff check . && uv run pytest -q --no-cov)
}

run_agent() {
  local stream="$1"
  local dir="$2"
  local spec="$SPECS_DIR/${stream}.txt"
  if [[ ! -f "$spec" ]]; then
    echo "missing spec: $spec"
    return 1
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN=1 — would invoke agent for $stream in $dir (see $spec)"
    return 0
  fi
  if [[ "${OFFICEQA_AGENT_CURSOR:-}" == "1" ]]; then
    echo ">>> cursor-agent for $stream (OFFICEQA_AGENT_CURSOR=1)"
    OFFICEQA_WT_ROOT="$WT_ROOT" bash "$ROOT/scripts/cursor_agent_once.sh" "$stream"
    return $?
  fi
  if [[ -z "${OFFICEQA_AGENT:-}" ]]; then
    echo "OFFICEQA_AGENT unset — cannot spawn agent for $stream"
    echo "Hint: OFFICEQA_AGENT_CURSOR=1 to use scripts/cursor_agent_once.sh"
    return 1
  fi
  echo ">>> Invoking OFFICEQA_AGENT for $stream in $dir"
  python3 "$ROOT/scripts/_invoke_agent.py" "$spec" "$dir"
}

main() {
  local round failed any_fail
  for round in $(seq 1 "$MAX_ROUNDS"); do
    echo "========== Round $round / $MAX_ROUNDS =========="
    failed=()
    for stream in "${STREAMS[@]}"; do
      dir="$WT_ROOT/$stream"
      if [[ ! -d "$dir" ]]; then
        echo "skip (no worktree): $dir — run ./scripts/setup_worktrees.sh"
        continue
      fi
      echo "-- check $stream ($dir)"
      if run_checks "$dir" "$stream"; then
        echo "ok: $stream"
      else
        echo "FAIL: $stream"
        failed+=("$stream")
      fi
    done

    if [[ ${#failed[@]} -eq 0 ]]; then
      echo "All streams green."
      exit 0
    fi

    if [[ "$DRY_RUN" == "1" ]]; then
      echo "DRY_RUN=1 — checks failed on: ${failed[*]} (no agent invoked)."
      exit 2
    fi
    if [[ -z "${OFFICEQA_AGENT:-}" && "${OFFICEQA_AGENT_CURSOR:-}" != "1" ]]; then
      echo "Streams still failing: ${failed[*]}"
      echo "Set OFFICEQA_AGENT (generic CLI) or OFFICEQA_AGENT_CURSOR=1 for cursor-agent."
      exit 2
    fi

    for stream in "${failed[@]}"; do
      dir="$WT_ROOT/$stream"
      run_agent "$stream" "$dir" || true
    done
  done

  echo "Max rounds reached; some streams may still fail."
  exit 3
}

main "$@"
