#!/usr/bin/env python3
"""Propagate year context onto bare-month rows in table_rows.

Treasury bulletins store monthly data in long format with the year anchored
on the first row of a month block:

    row 0: "1940-January"   year=1940 month=1
    row 1: "February"       year=None month=2   ← should inherit 1940
    row 2: "March"          year=None month=3   ← should inherit 1940
    ...
    row 11: "December"      year=None month=12  ← should inherit 1940

Ingest extracts `month_extracted` correctly for the bare rows but doesn't
carry year context forward, so structural queries like "tables with 12
distinct months of 1940" come up empty. This script walks each table in
row order and fills in the missing year.

Rules:
  - Only update rows where year_extracted IS NULL AND month_extracted IS NOT NULL.
  - Track `current_year` per-table. Set it whenever we see an explicit
    year_extracted value. Reset it on section-header rows (context boundary).
  - Skip tables with no explicit year anchors — nothing to propagate.

Usage:
  uv run python migrate_propagate_year.py           # dry-run (default)
  uv run python migrate_propagate_year.py --apply   # commit updates
"""

import sqlite3
import sys
import time
from collections import Counter

DB_PATH = "ledger.sqlite"


def walk_table(rows: list[sqlite3.Row]) -> list[tuple[int, int]]:
    """Return (year, row_id) updates for one table. Empty if nothing to fix."""
    updates: list[tuple[int, int]] = []
    current_year: int | None = None
    for r in rows:
        if r["is_section_header"]:
            current_year = None
            continue
        if r["year_extracted"] is not None:
            current_year = r["year_extracted"]
            continue
        # year is None — candidate for propagation
        if r["month_extracted"] is not None and current_year is not None:
            updates.append((current_year, r["id"]))
    return updates


def main():
    apply = "--apply" in sys.argv

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # All data tables
    table_ids = [r[0] for r in conn.execute("SELECT id FROM tables WHERE table_kind='data'")]
    print(f"Scanning {len(table_ids):,} data tables (dry-run={not apply})")

    all_updates: list[tuple[int, int]] = []
    tables_touched = 0
    year_counts: Counter = Counter()
    examples: list[dict] = []

    t0 = time.time()
    for i, tid in enumerate(table_ids):
        rows = conn.execute(
            """SELECT id, row_index, row_leaf, year_extracted, month_extracted, is_section_header
                 FROM table_rows
                WHERE table_id = ?
             ORDER BY row_index""",
            (tid,),
        ).fetchall()
        updates = walk_table(rows)
        if updates:
            tables_touched += 1
            all_updates.extend(updates)
            for year, _row_id in updates:
                year_counts[year] += 1
            # Stash a few examples for sanity-checking
            if len(examples) < 5:
                # Find the row that got propagated and its anchor
                row_map = {r["id"]: r for r in rows}
                for year, rid in updates[:3]:
                    examples.append(
                        {
                            "table_id": tid,
                            "row": row_map[rid]["row_leaf"],
                            "month": row_map[rid]["month_extracted"],
                            "propagated_year": year,
                        }
                    )
        if (i + 1) % 10000 == 0:
            print(
                f"  [{i + 1:,}/{len(table_ids):,}] tables scanned, "
                f"{len(all_updates):,} propagations queued  ({time.time() - t0:.1f}s)"
            )

    elapsed = time.time() - t0
    print(f"\n── Scan complete in {elapsed:.1f}s ──")
    print(f"  tables touched:     {tables_touched:,} / {len(table_ids):,}")
    print(f"  rows to propagate:  {len(all_updates):,}")
    print("\n  Top 10 years by propagation count:")
    for year, n in year_counts.most_common(10):
        print(f"    {year}: {n:,}")

    print("\n  First 5 example propagations:")
    for ex in examples:
        print(
            f"    table={ex['table_id']}  row={ex['row']!r:30s}  "
            f"month={ex['month']}  → year={ex['propagated_year']}"
        )

    if not apply:
        print("\nDry-run complete. Re-run with --apply to commit.")
        return

    # Apply in a single transaction
    print(f"\nApplying {len(all_updates):,} updates...")
    conn.executemany(
        "UPDATE table_rows SET year_extracted = ? WHERE id = ?",
        all_updates,
    )
    conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
