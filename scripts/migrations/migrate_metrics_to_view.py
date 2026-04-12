#!/usr/bin/env python3
"""Migrate metrics from materialized table to VIEW.

Steps:
1. Add metric_slug column to table_rows and populate it.
2. Create metrics_synthesized table with the ~10K synthesized rows.
3. Drop the metrics TABLE and its 4 indices.
4. Create a metrics VIEW = join query UNION ALL metrics_synthesized.
5. Verify row_label_lookup and col_label_lookup still work.
6. Delete is_missing=1 cells (~4.3M rows).
7. VACUUM the database.

Target: ledger.sqlite < 3GB (down from ~7.8GB).

This script is idempotent — safe to run multiple times.
"""

import re
import sqlite3
import sys
import time
from pathlib import Path

LEDGER_PATH = Path("ledger.sqlite")

# normalize_metric_slug must match build_ledger.py's implementation exactly
_METRIC_SLUG_FOOTNOTE_RE = re.compile(r"\s*\d+/")
_METRIC_SLUG_PUNCT_RE = re.compile(r"[^\w\s\-]")
_METRIC_SLUG_WS_RE = re.compile(r"\s+")


def normalize_metric_slug(metric: str | None) -> str:
    if not metric:
        return ""
    s = metric.lower().strip()
    s = _METRIC_SLUG_FOOTNOTE_RE.sub("", s)
    s = _METRIC_SLUG_PUNCT_RE.sub("", s)
    s = _METRIC_SLUG_WS_RE.sub(" ", s).strip()
    return s


METRICS_VIEW_DDL = """
CREATE VIEW metrics AS
SELECT
    NULL AS id,
    t.id AS table_id,
    r.metric_slug,
    r.row_path,
    col.col_path,
    CASE
      WHEN COALESCE(col.year_extracted, r.year_extracted) IS NOT NULL
       AND COALESCE(col.month_extracted, r.month_extracted) IS NOT NULL
        THEN printf('%04d-%02d',
                    COALESCE(col.year_extracted, r.year_extracted),
                    COALESCE(col.month_extracted, r.month_extracted))
      WHEN COALESCE(col.year_extracted, r.year_extracted, t.file_year) IS NOT NULL
        THEN printf('%04d',
                    COALESCE(col.year_extracted, r.year_extracted, t.file_year))
      ELSE NULL
    END AS time_key,
    COALESCE(col.year_extracted, r.year_extracted, t.file_year) AS year,
    COALESCE(col.month_extracted, r.month_extracted) AS month,
    t.period AS period_basis,
    c.numeric_value AS value,
    c.raw_value AS value_raw,
    t.unit,
    t.file,
    t.page_id,
    c.has_footnote,
    c.is_revised,
    c.is_preliminary,
    c.is_estimated,
    0 AS is_synthesized
FROM cells c
JOIN tables t          ON c.table_id = t.id
JOIN table_rows r      ON c.row_id   = r.id
JOIN table_columns col ON c.col_id   = col.id
WHERE t.parse_ok = 1
  AND t.table_kind = 'data'
  AND r.is_section_header = 0
  AND r.row_path IS NOT NULL
  AND LENGTH(TRIM(r.row_path)) > 0
  AND c.parse_status IN ('ok', 'missing')

UNION ALL

SELECT
    id, table_id, metric_slug, row_path, col_path, time_key,
    year, month, period_basis, value, value_raw, unit,
    file, page_id,
    has_footnote, is_revised, is_preliminary, is_estimated,
    is_synthesized
FROM metrics_synthesized
"""


def main() -> None:
    if not LEDGER_PATH.exists():
        print(f"ERROR: {LEDGER_PATH} not found", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA cache_size=-2000000")  # 2GB cache for big ops
    conn.create_function("norm_slug", 1, normalize_metric_slug, deterministic=True)

    # --- Check current state ---
    metrics_entry = conn.execute("SELECT type FROM sqlite_master WHERE name='metrics'").fetchone()
    if metrics_entry and metrics_entry[0] == "view":
        print("metrics is already a VIEW — checking for missing cell cleanup...", flush=True)
        (missing_cells,) = conn.execute(
            "SELECT COUNT(*) FROM cells WHERE is_missing = 1"
        ).fetchone()
        if missing_cells == 0:
            print("  No missing cells to clean. Migration already complete.", flush=True)
            conn.close()
            return
        else:
            print(f"  Found {missing_cells:,} missing cells to clean up.", flush=True)
            # Skip to step 6
            _delete_missing_cells(conn, missing_cells)
            _vacuum(conn)
            conn.close()
            _report_size()
            return

    # --- Pre-migration checks ---
    print("=== Pre-migration checks ===", flush=True)

    (total_metrics,) = conn.execute("SELECT COUNT(*) FROM metrics").fetchone()
    (synth_count,) = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE is_synthesized = 1"
    ).fetchone()
    (non_synth_count,) = conn.execute(
        "SELECT COUNT(*) FROM metrics WHERE is_synthesized = 0"
    ).fetchone()
    (rll_count,) = conn.execute("SELECT COUNT(*) FROM row_label_lookup").fetchone()
    (cll_count,) = conn.execute("SELECT COUNT(*) FROM col_label_lookup").fetchone()
    (missing_cells,) = conn.execute("SELECT COUNT(*) FROM cells WHERE is_missing = 1").fetchone()

    print(f"  metrics total:          {total_metrics:,}", flush=True)
    print(f"  metrics synthesized:    {synth_count:,}", flush=True)
    print(f"  metrics non-synthesized:{non_synth_count:,}", flush=True)
    print(f"  row_label_lookup:       {rll_count:,}", flush=True)
    print(f"  col_label_lookup:       {cll_count:,}", flush=True)
    print(f"  cells is_missing=1:     {missing_cells:,}", flush=True)

    # --- Step 1: Add metric_slug to table_rows ---
    print("\n=== Step 1: Add metric_slug to table_rows ===", flush=True)
    t0 = time.time()
    try:
        conn.execute("ALTER TABLE table_rows ADD COLUMN metric_slug TEXT")
        conn.commit()
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            raise
        print("  Column already exists", flush=True)

    conn.execute("UPDATE table_rows SET metric_slug = norm_slug(row_path)")
    conn.commit()
    print(f"  Populated metric_slug in {time.time() - t0:.1f}s", flush=True)

    # --- Step 2: Create metrics_synthesized table ---
    print("\n=== Step 2: Create metrics_synthesized table ===", flush=True)
    t0 = time.time()

    conn.execute("DROP TABLE IF EXISTS metrics_synthesized")
    conn.execute("""
        CREATE TABLE metrics_synthesized (
            id              INTEGER PRIMARY KEY,
            table_id        INTEGER NOT NULL,
            metric_slug     TEXT NOT NULL,
            row_path        TEXT,
            col_path        TEXT,
            time_key        TEXT,
            year            INTEGER,
            month           INTEGER,
            period_basis    TEXT,
            value           REAL,
            value_raw       TEXT,
            unit            TEXT,
            file            TEXT NOT NULL,
            page_id         INTEGER,
            has_footnote    INTEGER DEFAULT 0,
            is_revised      INTEGER DEFAULT 0,
            is_preliminary  INTEGER DEFAULT 0,
            is_estimated    INTEGER DEFAULT 0,
            is_synthesized  INTEGER DEFAULT 1
        )
    """)

    conn.execute("""
        INSERT INTO metrics_synthesized
            (table_id, metric_slug, row_path, col_path, time_key,
             year, month, period_basis, value, value_raw, unit,
             file, page_id,
             has_footnote, is_revised, is_preliminary, is_estimated,
             is_synthesized)
        SELECT
            table_id, metric_slug, row_path, col_path, time_key,
            year, month, period_basis, value, value_raw, unit,
            file, page_id,
            has_footnote, is_revised, is_preliminary, is_estimated,
            is_synthesized
        FROM metrics
        WHERE is_synthesized = 1
    """)
    conn.commit()

    (synth_migrated,) = conn.execute("SELECT COUNT(*) FROM metrics_synthesized").fetchone()
    print(
        f"  Migrated {synth_migrated:,} synthesized rows in {time.time() - t0:.1f}s",
        flush=True,
    )
    assert synth_migrated == synth_count, (
        f"Synthesized count mismatch: {synth_migrated} vs {synth_count}"
    )

    # --- Step 3: Drop metrics TABLE and its indices ---
    print("\n=== Step 3: Drop metrics table and indices ===", flush=True)
    t0 = time.time()

    conn.execute("DROP INDEX IF EXISTS idx_metrics_slug")
    conn.execute("DROP INDEX IF EXISTS idx_metrics_year")
    conn.execute("DROP INDEX IF EXISTS idx_metrics_table")
    conn.execute("DROP INDEX IF EXISTS idx_metrics_file")
    conn.execute("DROP TABLE IF EXISTS metrics")
    conn.commit()
    print(f"  Dropped in {time.time() - t0:.1f}s", flush=True)

    # --- Step 4: Create metrics VIEW ---
    print("\n=== Step 4: Create metrics VIEW ===", flush=True)
    t0 = time.time()
    conn.execute(METRICS_VIEW_DDL)
    conn.commit()
    print(f"  Created VIEW in {time.time() - t0:.1f}s", flush=True)

    # --- Step 5: Verify ---
    print("\n=== Step 5: Verify ===", flush=True)
    (new_type,) = conn.execute("SELECT type FROM sqlite_master WHERE name='metrics'").fetchone()
    print(f"  metrics type: {new_type}", flush=True)
    assert new_type == "view", f"Expected 'view', got '{new_type}'"

    (new_rll,) = conn.execute("SELECT COUNT(*) FROM row_label_lookup").fetchone()
    (new_cll,) = conn.execute("SELECT COUNT(*) FROM col_label_lookup").fetchone()
    print(f"  row_label_lookup: {new_rll:,} (was {rll_count:,})", flush=True)
    print(f"  col_label_lookup: {new_cll:,} (was {cll_count:,})", flush=True)
    assert new_rll == rll_count, f"RLL count changed: {new_rll} vs {rll_count}"
    assert new_cll == cll_count, f"CLL count changed: {new_cll} vs {cll_count}"

    # --- Step 6: Delete is_missing=1 cells ---
    _delete_missing_cells(conn, missing_cells)

    # --- Step 7: VACUUM ---
    _vacuum(conn)

    conn.close()
    _report_size()
    print("\nMigration complete!", flush=True)


def _delete_missing_cells(conn: sqlite3.Connection, count: int) -> None:
    print(f"\n=== Deleting {count:,} is_missing=1 cells ===", flush=True)
    t0 = time.time()
    conn.execute("DELETE FROM cells WHERE is_missing = 1")
    conn.commit()
    (remaining,) = conn.execute("SELECT COUNT(*) FROM cells").fetchone()
    print(f"  Deleted in {time.time() - t0:.1f}s, {remaining:,} cells remaining", flush=True)


def _vacuum(conn: sqlite3.Connection) -> None:
    print("\n=== VACUUM ===", flush=True)
    t0 = time.time()
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("VACUUM")
    print(f"  Completed in {time.time() - t0:.1f}s", flush=True)


def _report_size() -> None:
    size_bytes = LEDGER_PATH.stat().st_size
    size_gb = size_bytes / (1024**3)
    print(f"\n=== Final DB size: {size_gb:.2f} GB ===", flush=True)
    if size_gb < 3.0:
        print("  ✓ Target met: < 3GB", flush=True)
    else:
        print(f"  ✗ Target NOT met: {size_gb:.2f} GB >= 3GB", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
