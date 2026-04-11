"""One-time migration: fill year_extracted on month-only table_columns.

For Type-B rolling-series tables (title = "March 1979 through February 1980"),
columns are just month abbreviations with month_extracted set but year_extracted=NULL.
This migration reads each such table's title, parses the date range, and assigns
year_extracted to its month-only columns by walking the sequence.

Run once: uv run python migrate_fill_column_years.py
"""

import sqlite3
import sys

from build_ledger import _fill_column_years, _title_date_range

DB = "ledger.sqlite"


def migrate(db_path: str = DB, dry_run: bool = False) -> None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Find tables that have at least one month-only column (year=NULL, month!=NULL)
    # and whose title parses to a date range.
    tables = conn.execute(
        """
        SELECT DISTINCT t.id, t.title
        FROM tables t
        JOIN table_columns c ON c.table_id = t.id
        WHERE c.month_extracted IS NOT NULL
          AND c.year_extracted IS NULL
        ORDER BY t.id
        """
    ).fetchall()

    print(f"Found {len(tables)} tables with month-only columns", flush=True)

    updated_cols = 0
    updated_tables = 0

    for trow in tables:
        tid = trow["id"]
        title = trow["title"] or ""

        date_range = _title_date_range(title)
        if not date_range:
            continue  # title doesn't encode a date range — skip

        # Load existing columns for this table
        col_rows = conn.execute(
            """
            SELECT id, col_index, col_path, col_leaf, year_extracted, month_extracted
            FROM table_columns WHERE table_id = ? ORDER BY col_index
            """,
            (tid,),
        ).fetchall()

        cols = [dict(r) for r in col_rows]
        filled = _fill_column_years(cols, title)

        # Write back any cols whose year_extracted changed
        n_updated = 0
        for orig, new in zip(col_rows, filled, strict=False):
            if orig["year_extracted"] is None and new["year_extracted"] is not None:
                if not dry_run:
                    conn.execute(
                        "UPDATE table_columns SET year_extracted = ? WHERE id = ?",
                        (new["year_extracted"], orig["id"]),
                    )
                n_updated += 1

        if n_updated:
            updated_cols += n_updated
            updated_tables += 1

    if not dry_run:
        conn.commit()

    print(
        f"{'[dry-run] ' if dry_run else ''}Updated {updated_cols} columns "
        f"across {updated_tables} tables",
        flush=True,
    )
    conn.close()


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    migrate(dry_run=dry_run)
