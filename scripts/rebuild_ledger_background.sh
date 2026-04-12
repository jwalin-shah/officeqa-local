#!/usr/bin/env bash
# Build a fresh ledger in the background without removing the current DB until
# the build finishes (so other agents can keep using ledger.sqlite, or point
# OFFICEQA_LEDGER_PATH at the timestamped backup).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
NEXT="ledger.build.${STAMP}.sqlite"
LOG="build_ledger.${STAMP}.log"
PIDFILE="build_ledger.${STAMP}.pid"

if [[ -f ledger.sqlite ]]; then
  cp -p ledger.sqlite "ledger.sqlite.prev.${STAMP}"
  echo "Backed up current DB to ledger.sqlite.prev.${STAMP}" | tee "$LOG"
else
  echo "No existing ledger.sqlite — fresh build" | tee "$LOG"
fi

(
  set -euo pipefail
  cd "$ROOT"
  uv run python build_ledger.py --rebuild --output "$NEXT" >>"$LOG" 2>&1
  mv -f "$NEXT" ledger.sqlite
  echo "$(date -u +"%Y-%m-%dT%H:%M:%SZ") finished — ledger.sqlite updated" >>"$LOG"
) &
echo $! >"$PIDFILE"

echo "Background rebuild PID $(cat "$PIDFILE")"
echo "Log: $ROOT/$LOG"
echo "During the run, ledger.sqlite stays the old file until the final mv."
echo "To pin another shell/agent to the pre-build copy:"
echo "  export OFFICEQA_LEDGER_PATH=\"$ROOT/ledger.sqlite.prev.${STAMP}\""
