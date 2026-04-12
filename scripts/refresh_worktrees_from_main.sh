#!/usr/bin/env bash
# Merge latest main into each agent/* worktree branch (run from main after you commit).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WT_ROOT="${OFFICEQA_WT_ROOT:-$HOME/projects/officeqa-wt}"

git -C "$ROOT" fetch origin main 2>/dev/null || git -C "$ROOT" fetch origin master 2>/dev/null || true

for name in retrieval decompose fastpath extract eval ingest; do
  d="$WT_ROOT/$name"
  [[ -d "$d" ]] || continue
  echo "==> $d"
  git -C "$d" merge --no-edit main 2>/dev/null || git -C "$d" merge --no-edit master || true
done
