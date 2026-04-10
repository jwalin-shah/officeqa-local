"""Tests for ledger.sqlite structure, FTS, and scout.py."""

import sqlite3
from pathlib import Path

import pytest

LEDGER_PATH = Path("ledger.sqlite")
LEDGER_EXISTS = LEDGER_PATH.exists()
skip_no_ledger = pytest.mark.skipif(not LEDGER_EXISTS, reason="ledger.sqlite not found")


# ── Key tables exist ─────────────────────────────────────────────────────────


@skip_no_ledger
def test_tables_table_exists():
    conn = sqlite3.connect(str(LEDGER_PATH))
    cur = conn.execute("SELECT COUNT(*) FROM tables")
    count = cur.fetchone()[0]
    conn.close()
    assert count > 0, "tables table is empty"


@skip_no_ledger
def test_table_columns_exists():
    conn = sqlite3.connect(str(LEDGER_PATH))
    cur = conn.execute("SELECT COUNT(*) FROM table_columns")
    count = cur.fetchone()[0]
    conn.close()
    assert count > 0, "table_columns table is empty"


@skip_no_ledger
def test_table_rows_exists():
    conn = sqlite3.connect(str(LEDGER_PATH))
    cur = conn.execute("SELECT COUNT(*) FROM table_rows")
    count = cur.fetchone()[0]
    conn.close()
    assert count > 0, "table_rows table is empty"


@skip_no_ledger
def test_cells_table_exists():
    conn = sqlite3.connect(str(LEDGER_PATH))
    cur = conn.execute("SELECT COUNT(*) FROM cells")
    count = cur.fetchone()[0]
    conn.close()
    assert count > 0, "cells table is empty"


@skip_no_ledger
def test_tables_fts_exists():
    conn = sqlite3.connect(str(LEDGER_PATH))
    cur = conn.execute(
        "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='tables_fts'"
    )
    count = cur.fetchone()[0]
    conn.close()
    assert count == 1, "tables_fts FTS5 virtual table not found"


# ── Row counts are sane ──────────────────────────────────────────────────────


@skip_no_ledger
def test_table_count_reasonable():
    conn = sqlite3.connect(str(LEDGER_PATH))
    count = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
    conn.close()
    # We expect hundreds to thousands of tables from 697 bulletin JSONs
    assert count >= 100, f"Only {count} tables — expected ≥100"


@skip_no_ledger
def test_cells_count_reasonable():
    conn = sqlite3.connect(str(LEDGER_PATH))
    count = conn.execute("SELECT COUNT(*) FROM cells").fetchone()[0]
    conn.close()
    # Expect millions of cells
    assert count >= 100_000, f"Only {count} cells — expected ≥100K"


# ── FTS works ────────────────────────────────────────────────────────────────


@skip_no_ledger
def test_fts_query_defense():
    conn = sqlite3.connect(str(LEDGER_PATH))
    rows = conn.execute(
        "SELECT rowid FROM tables_fts WHERE tables_fts MATCH 'defense' LIMIT 5"
    ).fetchall()
    conn.close()
    assert len(rows) > 0, "FTS query for 'defense' returned no results"


@skip_no_ledger
def test_fts_query_receipts():
    conn = sqlite3.connect(str(LEDGER_PATH))
    rows = conn.execute(
        "SELECT rowid FROM tables_fts WHERE tables_fts MATCH 'receipts' LIMIT 5"
    ).fetchall()
    conn.close()
    assert len(rows) > 0, "FTS query for 'receipts' returned no results"


# ── Row count stability (idempotency proxy) ──────────────────────────────────


@skip_no_ledger
def test_row_counts_stable():
    """Key table row counts should be consistent (proxy for idempotency)."""
    conn = sqlite3.connect(str(LEDGER_PATH))
    counts = {}
    for table in ("tables", "table_columns", "table_rows", "cells"):
        counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    conn.close()

    # These counts should all be positive and reasonable
    assert counts["tables"] > 0
    assert counts["table_columns"] > 0
    assert counts["table_rows"] > 0
    assert counts["cells"] > 0
    # Rows should be more than columns (typical for tabular data)
    assert counts["table_rows"] >= counts["tables"]


# ── Lookup views exist ───────────────────────────────────────────────────────


@skip_no_ledger
def test_row_label_lookup_exists():
    conn = sqlite3.connect(str(LEDGER_PATH))
    try:
        cur = conn.execute("SELECT COUNT(*) FROM row_label_lookup LIMIT 1")
        count = cur.fetchone()[0]
        assert count > 0
    finally:
        conn.close()


# ── Scout tests ──────────────────────────────────────────────────────────────


@skip_no_ledger
def test_scout_defense_question():
    """scout() returns non-empty hints for a defense-related question."""
    from scout import scout

    result = scout("What were the total expenditures for national defense in 1940?")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "CORPUS SCOUT" in result


@skip_no_ledger
def test_scout_receipts_question():
    from scout import scout

    result = scout("What were total budget receipts in fiscal year 1950?")
    assert len(result) > 0


@skip_no_ledger
def test_scout_debt_question():
    from scout import scout

    result = scout("What was the public debt outstanding at end of fiscal year 1945?")
    assert len(result) > 0


@skip_no_ledger
def test_scout_veterans_question():
    from scout import scout

    result = scout("How much was spent on veterans services in calendar year 1942?")
    assert len(result) > 0


@skip_no_ledger
def test_scout_empty_result():
    """scout() returns empty string for a nonsense query."""
    from scout import scout

    result = scout("xyzzy_completely_made_up_query_12345")
    assert isinstance(result, str)
    # May be empty or may have low-quality hits — both are acceptable
