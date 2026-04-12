#!/usr/bin/env bash
# Create one git worktree per parallel workstream (see docs/AGENT_ORCHESTRATION.md).
# Usage:
#   ./scripts/setup_worktrees.sh              # create all default worktrees
#   ./scripts/setup_worktrees.sh --list       # print branch/path pairs only
#   OFFICEQA_WT_ROOT=~/wt ./scripts/setup_worktrees.sh   # custom parent directory
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WT_ROOT="${OFFICEQA_WT_ROOT:-$HOME/projects/officeqa-wt}"
BASE_BRANCH="${OFFICEQA_WT_BASE:-main}"

STREAMS=(retrieval decompose fastpath extract eval ingest)

if [[ "${1:-}" == "--list" ]]; then
  for s in "${STREAMS[@]}"; do
    printf '%s\t%s\n' "agent/$s" "$WT_ROOT/$s"
  done
  exit 0
fi

mkdir -p "$WT_ROOT"

for stream in "${STREAMS[@]}"; do
  branch="agent/$stream"
  path="$WT_ROOT/$stream"
  if [[ -d "$path" ]]; then
    echo "skip (exists): $path"
    continue
  fi
  if git show-ref --verify --quiet "refs/heads/$branch"; then
    git worktree add "$path" "$branch"
  else
    git worktree add -b "$branch" "$path" "$BASE_BRANCH"
  fi
  echo "ok: $path -> $branch"
done

echo
echo "Optional: symlink shared artifacts into each worktree:"
echo "  for d in $WT_ROOT/*/; do ln -sf \"$ROOT/ledger.sqlite\" \"\$d/ledger.sqlite\"; done"
echo "  for d in $WT_ROOT/*/; do ln -sf \"$ROOT/corpus_json\" \"\$d/corpus_json\"; done"
echo "See docs/AGENT_ORCHESTRATION.md for headless agent commands and merge protocol."
