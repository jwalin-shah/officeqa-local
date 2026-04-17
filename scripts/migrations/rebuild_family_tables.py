"""Rebuild table_family* tables from the tables/table_columns/table_rows schema.

family_id = LOWER(TRIM(title)) — same grouping used by the old ledger.
tables.family_id column must exist before running this (add it if missing).

Run after build_ledger.py:
    uv run python scripts/migrations/rebuild_family_tables.py
"""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ledger_paths import get_ledger_sqlite_path


def rebuild(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")

    print(f"Rebuilding family tables in {db_path}")

    # ── Ensure tables.family_id exists ────────────────────────────────────────
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tables)").fetchall()]
    if "family_id" not in cols:
        print("  Adding tables.family_id column...")
        conn.execute("ALTER TABLE tables ADD COLUMN family_id TEXT")
        conn.execute("UPDATE tables SET family_id = LOWER(TRIM(COALESCE(title, '')))")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tables_family ON tables(family_id)")
        conn.commit()
        print("    Done")

    # ── Drop + recreate family tables ─────────────────────────────────────────
    conn.executescript("""
        DROP TABLE IF EXISTS table_families;
        DROP TABLE IF EXISTS table_family_columns;
        DROP TABLE IF EXISTS table_family_rows;
        DROP TABLE IF EXISTS table_family_years;
        DROP TABLE IF EXISTS families_fts;

        CREATE TABLE table_families(
            family_id      TEXT,
            primary_title  TEXT,
            member_count   INTEGER,
            latest_ver     INTEGER,
            latest_table_id INTEGER
        );
        CREATE INDEX idx_families_fid    ON table_families(family_id);
        CREATE INDEX idx_families_latest ON table_families(latest_table_id);

        CREATE TABLE table_family_columns(
            family_id      TEXT,
            col_leaf_lower TEXT
        );
        CREATE INDEX idx_family_columns_fid   ON table_family_columns(family_id);
        CREATE INDEX idx_family_columns_lower ON table_family_columns(col_leaf_lower);

        CREATE TABLE table_family_rows(
            family_id  TEXT,
            metric_slug TEXT
        );
        CREATE INDEX idx_family_rows_fid  ON table_family_rows(family_id);
        CREATE INDEX idx_family_rows_slug ON table_family_rows(metric_slug);

        CREATE TABLE table_family_years(
            family_id     TEXT,
            year          INT,
            best_table_id INT
        );
        CREATE INDEX idx_family_years_fid      ON table_family_years(family_id);
        CREATE INDEX idx_family_years_year     ON table_family_years(year);
        CREATE INDEX idx_family_years_fid_year ON table_family_years(family_id, year);
        CREATE INDEX idx_family_years_best     ON table_family_years(best_table_id);
    """)

    # ── table_families ─────────────────────────────────────────────────────────
    print("  Building table_families...")
    conn.execute("""
        INSERT INTO table_families(family_id, primary_title, member_count, latest_ver, latest_table_id)
        SELECT
            family_id,
            (SELECT title FROM tables t2
             WHERE t2.family_id = t.family_id
             ORDER BY t2.file_year DESC, t2.file_month DESC, t2.id DESC
             LIMIT 1) AS primary_title,
            COUNT(*)  AS member_count,
            MAX(file_year * 100 + COALESCE(file_month, 0)) AS latest_ver,
            (SELECT id FROM tables t3
             WHERE t3.family_id = t.family_id
             ORDER BY t3.file_year DESC, t3.file_month DESC, t3.id DESC
             LIMIT 1) AS latest_table_id
        FROM tables t
        WHERE family_id IS NOT NULL AND family_id != ''
        GROUP BY family_id
    """)
    fam_count = conn.execute("SELECT COUNT(*) FROM table_families").fetchone()[0]
    print(f"    {fam_count:,} families")

    # ── table_family_columns ───────────────────────────────────────────────────
    print("  Building table_family_columns...")
    conn.execute("""
        INSERT INTO table_family_columns(family_id, col_leaf_lower)
        SELECT DISTINCT
            t.family_id,
            LOWER(TRIM(tc.col_leaf)) AS col_leaf_lower
        FROM table_columns tc
        JOIN tables t ON t.id = tc.table_id
        WHERE t.family_id IS NOT NULL AND t.family_id != ''
          AND tc.col_leaf IS NOT NULL AND TRIM(tc.col_leaf) != ''
    """)
    col_count = conn.execute("SELECT COUNT(*) FROM table_family_columns").fetchone()[0]
    print(f"    {col_count:,} family-column pairs")

    # ── table_family_rows ──────────────────────────────────────────────────────
    print("  Building table_family_rows...")
    conn.execute("""
        INSERT INTO table_family_rows(family_id, metric_slug)
        SELECT DISTINCT
            t.family_id,
            tr.metric_slug
        FROM table_rows tr
        JOIN tables t ON t.id = tr.table_id
        WHERE t.family_id IS NOT NULL AND t.family_id != ''
          AND tr.metric_slug IS NOT NULL AND TRIM(tr.metric_slug) != ''
    """)
    row_count = conn.execute("SELECT COUNT(*) FROM table_family_rows").fetchone()[0]
    print(f"    {row_count:,} family-row pairs")

    # ── table_family_years ─────────────────────────────────────────────────────
    # For each (family, data_year) pair, pick the most recently published
    # table_id. data_year = COALESCE(col.year_extracted, row.year_extracted,
    # file_year) — same resolution as the facts view.  This mirrors how the
    # original migration was built and ensures year coverage is broad.
    print("  Building table_family_years (from facts view)...")
    conn.execute("""
        INSERT INTO table_family_years(family_id, year, best_table_id)
        SELECT
            tf.family_id,
            f.data_year AS year,
            (SELECT t2.id FROM tables t2
             JOIN facts f2 ON f2.table_id = t2.id
             WHERE t2.family_id = tf.family_id
               AND f2.data_year = f.data_year
             ORDER BY t2.file_year DESC, t2.file_month DESC, t2.id DESC
             LIMIT 1) AS best_table_id
        FROM facts f
        JOIN tables tf ON tf.id = f.table_id
        WHERE tf.family_id IS NOT NULL AND tf.family_id != ''
          AND f.data_year IS NOT NULL
        GROUP BY tf.family_id, f.data_year
    """)
    yr_count = conn.execute("SELECT COUNT(*) FROM table_family_years").fetchone()[0]
    print(f"    {yr_count:,} family-year pairs")

    # ── families_fts ───────────────────────────────────────────────────────────
    print("  Building families_fts...")
    conn.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS families_fts USING fts5(
            family_id UNINDEXED,
            primary_title,
            tokenize = 'unicode61'
        );
        INSERT INTO families_fts(family_id, primary_title)
        SELECT family_id, COALESCE(primary_title, '') FROM table_families;
    """)
    fts_count = conn.execute("SELECT COUNT(*) FROM families_fts").fetchone()[0]
    print(f"    {fts_count:,} families_fts rows")

    conn.commit()
    conn.close()
    print("Done.")


if __name__ == "__main__":
    db_path = sys.argv[1] if len(sys.argv) > 1 else get_ledger_sqlite_path()
    rebuild(db_path)
