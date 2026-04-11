#!/usr/bin/env python3
"""M2.1: Derived index — extract data years covered by each table.

Principle: NO dedup, NO data removal. This is a pure lookup accelerator built
on top of files.db's `tables` table. If it's ever wrong we throw it out and
rebuild — the source of truth (json_blob, raw_html) is untouched.

What counts as a "data year" on a table:
  - Any 4-digit year (19xx, 20xx) appearing in a column header
  - Also includes fiscal year references like "FY 1940", "Fiscal Year 1940"
  - Both expanded ranges ("1938-1940" → 1938, 1939, 1940) and bare years

Adds two things to files.db:
  - tables.data_years TEXT — JSON array of ints, NULL if none detected
  - table_years(file, table_id, data_year) — normalized for fast filtering
  - idx_table_years_year — for O(log N) lookup by year

Usage:
    uv run python build_data_years.py
    uv run python build_data_years.py --verify   # show samples
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = Path("files.db")

BARE_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
# "1938-1940" or "1938 - 40" or "1938-40"
RANGE_RE = re.compile(r"\b(19\d{2}|20\d{2})\s*[-–]\s*(\d{2,4})\b")
# "FY 1940", "Fiscal Year 1940", "CY 1940", "Calendar Year 1940"
FY_RE = re.compile(
    r"\b(?:FY|CY|Fiscal\s+Year|Calendar\s+Year)\s*(19\d{2}|20\d{2})\b",
    re.IGNORECASE,
)


def extract_years_from_text(text: str) -> set[int]:
    """Extract all data years from a block of column header text."""
    if not text:
        return set()
    years: set[int] = set()

    # Bare years
    for m in BARE_YEAR_RE.findall(text):
        years.add(int(m))

    # FY/CY prefixed years
    for m in FY_RE.findall(text):
        years.add(int(m))

    # Ranges: "1938-1940" or "1938-40" — expand inclusive
    for m in RANGE_RE.finditer(text):
        start = int(m.group(1))
        end_raw = m.group(2)
        if len(end_raw) == 2:
            # "1938-40" → 1940. Use century of start as baseline.
            end = (start // 100) * 100 + int(end_raw)
            if end < start:
                end += 100
        else:
            end = int(end_raw)
        if 1900 <= end <= 2100 and end - start <= 40:
            for y in range(start, end + 1):
                years.add(y)

    return years


def extract_years_from_table(col_labels: str, json_blob: str) -> set[int]:
    """Walk both col_labels text and column structure for year references."""
    years = extract_years_from_text(col_labels or "")

    # Also walk structured columns in case col_labels dropped something
    if json_blob:
        try:
            data = json.loads(json_blob)
            for c in data.get("columns", []):
                years |= extract_years_from_text(c.get("label", ""))
                if c.get("parent"):
                    years |= extract_years_from_text(c["parent"])
        except Exception:
            pass

    return years


def build(db_path: Path) -> None:
    if not db_path.exists():
        print(f"{db_path} does not exist. Run build_files_db.py first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")

    # Add data_years column if it doesn't exist (idempotent)
    cols = {r[1] for r in conn.execute("PRAGMA table_info(tables)")}
    if "data_years" not in cols:
        conn.execute("ALTER TABLE tables ADD COLUMN data_years TEXT")
        print("  added column: tables.data_years")

    # Rebuild side table fresh every time — it's derived
    conn.execute("DROP TABLE IF EXISTS table_years")
    conn.execute("DROP INDEX IF EXISTS idx_table_years_year")
    conn.execute(
        """
        CREATE TABLE table_years (
            file TEXT,
            table_id TEXT,
            data_year INTEGER,
            PRIMARY KEY (file, table_id, data_year)
        )
        """
    )

    total = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
    print(f"Scanning {total} tables for data years...")

    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.perf_counter()
    processed = 0
    with_years = 0
    total_year_rows = 0
    yr_histogram: dict[int, int] = {}

    cur = conn.execute("SELECT file, table_id, col_labels, json_blob FROM tables")
    updates: list[tuple[str, str, str]] = []
    year_inserts: list[tuple[str, str, int]] = []

    BATCH = 1000
    for file, table_id, col_labels, json_blob in cur:
        processed += 1
        years = extract_years_from_table(col_labels or "", json_blob or "")

        # Sanity bounds — nothing before 1900 or after 2050
        years = {y for y in years if 1900 <= y <= 2050}

        if years:
            with_years += 1
            total_year_rows += len(years)
            years_json = json.dumps(sorted(years))
            updates.append((years_json, file, table_id))
            for y in years:
                year_inserts.append((file, table_id, y))
                yr_histogram[y] = yr_histogram.get(y, 0) + 1

        if len(updates) >= BATCH:
            conn.executemany(
                "UPDATE tables SET data_years = ? WHERE file = ? AND table_id = ?",
                updates,
            )
            conn.executemany(
                "INSERT OR IGNORE INTO table_years (file, table_id, data_year) VALUES (?, ?, ?)",
                year_inserts,
            )
            conn.commit()
            updates.clear()
            year_inserts.clear()

        if processed % 10000 == 0:
            print(
                f"  [{processed:>6}/{total}] with_years={with_years} "
                f"total_year_rows={total_year_rows} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    if updates:
        conn.executemany(
            "UPDATE tables SET data_years = ? WHERE file = ? AND table_id = ?",
            updates,
        )
        conn.executemany(
            "INSERT OR IGNORE INTO table_years (file, table_id, data_year) VALUES (?, ?, ?)",
            year_inserts,
        )
        conn.commit()

    print("Building index...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_table_years_year ON table_years(data_year)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_table_years_file ON table_years(file)")
    conn.commit()

    elapsed = time.perf_counter() - t0
    print(f"\n✓ data_years built in {elapsed:.1f}s")
    print(f"  tables scanned:      {processed}")
    print(f"  tables with years:   {with_years} ({with_years / processed * 100:.1f}%)")
    print(f"  total (table, year): {total_year_rows}")
    print(f"  distinct years:      {len(yr_histogram)}")
    if yr_histogram:
        ys = sorted(yr_histogram)
        print(f"  year range:          {ys[0]}..{ys[-1]}")
        # Show a few popular years
        popular = sorted(yr_histogram.items(), key=lambda x: -x[1])[:5]
        print(f"  most-covered:        {popular}")


def verify(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    print("\n=== Spot checks ===")

    # How many files cover year 1940?
    for yr in (1940, 1950, 1980, 2000, 2020):
        n = conn.execute(
            "SELECT COUNT(DISTINCT file) FROM table_years WHERE data_year = ?",
            (yr,),
        ).fetchone()[0]
        print(f"  year {yr}: {n} distinct files")

    print("\n=== Sample tables with data_years ===")
    for row in conn.execute(
        "SELECT file, table_id, title, data_years FROM tables "
        "WHERE data_years IS NOT NULL AND title IS NOT NULL "
        "ORDER BY RANDOM() LIMIT 5"
    ):
        f, tid, title, years = row
        print(f"  {f[-25:]} {tid}")
        print(f"    title: {(title or '')[:70]}")
        print(f"    years: {years}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if not args.verify:
        build(args.db)
    verify(args.db)


if __name__ == "__main__":
    main()
