#!/usr/bin/env python3
"""M2.1b: Derived index — normalized row labels per table.

Principle (still): no dedup, no data removal. Pure lookup accelerator built
from tables.json_blob. Rebuild at any time from source of truth.

Output: row_slugs(file, table_id, slug, label_raw)
  - slug       — normalized form for matching (lowercase, alphanum + underscores,
                 stopwords dropped, whitespace collapsed)
  - label_raw  — original row label for display / debug

An FTS index over `slug` lets us quickly find "which tables have a row labeled
like X". Much more discriminating than BM25 over full text because row labels
are short, specific strings like "National defense" or "Individual income tax".

Usage:
    uv run python build_row_slugs.py
    uv run python build_row_slugs.py --verify
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

# Very small stopword set — we want to keep MOST words in row labels
# because row labels are already short and specific
ROW_STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "by",
    "with",
    "total",
    "subtotal",
}

SLUG_CLEAN_RE = re.compile(r"[^\w\s]")
WHITESPACE_RE = re.compile(r"\s+")


def slugify(text: str) -> str:
    """Normalize a row label into a searchable slug."""
    if not text:
        return ""
    s = text.lower().strip()
    s = SLUG_CLEAN_RE.sub(" ", s)
    s = WHITESPACE_RE.sub(" ", s).strip()
    tokens = [t for t in s.split() if t and t not in ROW_STOPWORDS and len(t) > 1]
    return "_".join(tokens)


def extract_row_labels(json_blob: str) -> list[str]:
    """Walk the JSON blob and return every row label (including subheader rows)."""
    if not json_blob:
        return []
    try:
        data = json.loads(json_blob)
    except Exception:
        return []
    out: list[str] = []
    for row in data.get("rows", []):
        label = (row.get("label") or "").strip()
        if label:
            out.append(label)
    return out


def build(db_path: Path) -> None:
    if not db_path.exists():
        print(f"{db_path} does not exist.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode = WAL")

    # Fresh rebuild every time — it's derived
    conn.execute("DROP TABLE IF EXISTS row_slugs")
    conn.execute("DROP TABLE IF EXISTS row_slugs_fts")
    conn.execute(
        """
        CREATE TABLE row_slugs (
            file TEXT,
            table_id TEXT,
            row_ord INTEGER,
            slug TEXT,
            label_raw TEXT,
            PRIMARY KEY (file, table_id, row_ord)
        )
        """
    )

    total = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
    print(f"Scanning {total} tables for row labels...")

    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.perf_counter()
    processed = 0
    total_rows = 0
    slug_hist: dict[str, int] = {}

    BATCH = 2000
    buf: list[tuple] = []
    for file, table_id, blob in conn.execute("SELECT file, table_id, json_blob FROM tables"):
        processed += 1
        labels = extract_row_labels(blob or "")
        for ord_i, lbl in enumerate(labels):
            slug = slugify(lbl)
            if not slug:
                continue
            buf.append((file, table_id, ord_i, slug, lbl[:500]))
            total_rows += 1
            slug_hist[slug] = slug_hist.get(slug, 0) + 1

        if len(buf) >= BATCH:
            conn.executemany(
                "INSERT OR IGNORE INTO row_slugs (file, table_id, row_ord, slug, label_raw) VALUES (?, ?, ?, ?, ?)",
                buf,
            )
            conn.commit()
            buf.clear()

        if processed % 10000 == 0:
            print(
                f"  [{processed:>6}/{total}] total_rows={total_rows} "
                f"distinct_slugs={len(slug_hist)} "
                f"({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    if buf:
        conn.executemany(
            "INSERT OR IGNORE INTO row_slugs (file, table_id, row_ord, slug, label_raw) VALUES (?, ?, ?, ?, ?)",
            buf,
        )
        conn.commit()

    print("Building index + FTS over row slugs...")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_row_slugs_slug ON row_slugs(slug)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_row_slugs_file ON row_slugs(file)")

    # FTS over slug_tokens (space-separated form for phrase matching)
    conn.execute(
        """
        CREATE VIRTUAL TABLE row_slugs_fts USING fts5(
            file, table_id, slug_tokens, label_raw,
            tokenize = 'porter unicode61'
        )
        """
    )
    # Insert with underscores converted to spaces so FTS can tokenize properly
    conn.execute(
        """
        INSERT INTO row_slugs_fts (file, table_id, slug_tokens, label_raw)
        SELECT file, table_id, REPLACE(slug, '_', ' '), label_raw FROM row_slugs
        """
    )
    conn.commit()

    elapsed = time.perf_counter() - t0
    print(f"\n✓ row_slugs built in {elapsed:.1f}s")
    print(f"  tables scanned:    {processed}")
    print(f"  row labels stored: {total_rows}")
    print(f"  distinct slugs:    {len(slug_hist)}")
    print("  most common:")
    for slug, n in sorted(slug_hist.items(), key=lambda x: -x[1])[:10]:
        print(f"    {slug:40s} {n}")


def verify(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    print("\n=== Sample lookups ===")
    for phrase in ["national defense", "individual income tax", "customs duties", "public debt"]:
        slug = slugify(phrase)
        n = conn.execute(
            "SELECT COUNT(DISTINCT file) FROM row_slugs WHERE slug = ?", (slug,)
        ).fetchone()[0]
        print(f"  slug={slug!r:35s} → {n} distinct files (exact)")

        # FTS variant
        fts_q = slug.replace("_", " ")
        n_fts = conn.execute(
            "SELECT COUNT(DISTINCT file) FROM row_slugs_fts WHERE row_slugs_fts MATCH ?",
            (fts_q,),
        ).fetchone()[0]
        print(f"  {'FTS':>38s} → {n_fts} distinct files")


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
