"""Resolve path to ledger.sqlite for all runtime tools.

Default: ``<repo_root>/ledger.sqlite``.

Override for parallel agents or A/B testing::

    export OFFICEQA_LEDGER_PATH=/path/to/ledger.sqlite.prev.20260411_1323
    # or repo-relative:
    export OFFICEQA_LEDGER_PATH=ledger.sqlite.prev.20260411_1323
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent


def get_ledger_sqlite_path() -> Path:
    raw = (os.environ.get("OFFICEQA_LEDGER_PATH") or "").strip()
    if not raw:
        return _REPO_ROOT / "ledger.sqlite"
    p = Path(raw).expanduser()
    if p.is_absolute():
        return p.resolve()
    return (_REPO_ROOT / p).resolve()
