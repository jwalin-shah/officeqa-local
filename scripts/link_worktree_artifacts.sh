#!/usr/bin/env bash
# Symlink large gitignored artifacts from the main repo into each worktree.
# Run from repo root after ./scripts/setup_worktrees.sh
#
# Usage:
#   ./scripts/link_worktree_artifacts.sh
#   OFFICEQA_WT_ROOT=... ./scripts/link_worktree_artifacts.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT_ROOT="${OFFICEQA_WT_ROOT:-$HOME/projects/officeqa-wt}"

for name in retrieval decompose fastpath extract eval ingest; do
  d="$WT_ROOT/$name"
  [[ -d "$d" ]] || continue
  if [[ -e "$ROOT/ledger.sqlite" ]]; then
    rm -f "$d/ledger.sqlite"
    ln -sf "$ROOT/ledger.sqlite" "$d/ledger.sqlite"
    echo "linked ledger.sqlite -> $d"
  else
    echo "skip ledger (missing at $ROOT/ledger.sqlite)"
  fi
  if [[ -d "$ROOT/corpus_json" ]]; then
    rm -f "$d/corpus_json"
    ln -sf "$ROOT/corpus_json" "$d/corpus_json"
    echo "linked corpus_json -> $d"
  fi
done
