#!/usr/bin/env python3
"""Build files.db — lossless structured index over corpus_json/.

Design principles (learned the hard way from ledger.sqlite):
  - DO NOT normalize table content into cells/rows/cols tables
  - DO store whole tables as structured JSON blobs, keyed by (file, table_id)
  - DO preserve hierarchy (multi-level headers, row indentation, footnotes)
  - FTS5 indexes for retrieval — no custom BM25, no sentence-transformers

Schema:
  files(file PK, year, month, title)
  sections(file, section_id, header, ord)
  tables(file, table_id, section_id, title, units, n_rows, n_cols, json_blob)
  prose(file, section_id, text)
  tables_fts(title, row_labels, col_labels, content=tables) — fts5
  prose_fts(text, content=prose) — fts5

Usage:
    uv run python build_files_db.py
    uv run python build_files_db.py --out /tmp/files.db --limit 20
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import warnings
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

CORPUS_DIR = Path("corpus_json")
DEFAULT_OUT = Path("files.db")

# treasury_bulletin_YYYY_MM.json
FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")
UNIT_RE = re.compile(
    r"\(\s*([Ii]n\s+)?(millions?|thousands?|billions?|percent|dollars?)"
    r"[^)]*\)",
    re.IGNORECASE,
)


def parse_filename(name: str) -> tuple[int | None, int | None]:
    m = FILENAME_RE.search(name)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def extract_units(text: str) -> str | None:
    """Pull a units declaration out of nearby text (title, caption, etc)."""
    if not text:
        return None
    m = UNIT_RE.search(text)
    return m.group(0) if m else None


def df_to_structured(df: pd.DataFrame) -> dict[str, Any]:
    """Convert a pandas DataFrame (possibly with MultiIndex cols) to our schema.

    Output:
      {
        "columns": [{"level": int, "parent": str|None, "label": str}, ...],
        "rows":    [{"label": str, "cells": [val, ...]}, ...]
      }
    """
    cols: list[dict[str, Any]] = []
    if isinstance(df.columns, pd.MultiIndex):
        # Flatten MultiIndex: one column entry per leaf, with parent chain
        for tup in df.columns:
            parts = [str(p) for p in tup if p and str(p) != "nan" and "Unnamed" not in str(p)]
            if not parts:
                cols.append({"level": 0, "parent": None, "label": ""})
                continue
            leaf = parts[-1]
            parent = " / ".join(parts[:-1]) if len(parts) > 1 else None
            cols.append({"level": len(parts) - 1, "parent": parent, "label": leaf})
    else:
        for c in df.columns:
            label = "" if ("Unnamed" in str(c) or str(c) == "nan") else str(c)
            cols.append({"level": 0, "parent": None, "label": label})

    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        values = row.tolist()
        # First cell is the row label by convention — but only if it's text-ish
        label = ""
        cells: list[Any] = []
        if values:
            first = values[0]
            if isinstance(first, str) or (
                first is not None and not isinstance(first, (int, float))
            ):
                label = "" if (pd.isna(first) or "Unnamed" in str(first)) else str(first)
                cells = values[1:]
            else:
                cells = values
        # Clean NaN → None for JSON
        cleaned = [None if (isinstance(v, float) and pd.isna(v)) else v for v in cells]
        rows.append({"label": label, "cells": cleaned})

    return {"columns": cols, "rows": rows}


def row_labels_text(structured: dict) -> str:
    return " | ".join(r["label"] for r in structured["rows"] if r["label"])


def col_labels_text(structured: dict) -> str:
    parts = []
    for c in structured["columns"]:
        if c["parent"]:
            parts.append(f"{c['parent']} / {c['label']}")
        elif c["label"]:
            parts.append(c["label"])
    return " | ".join(parts)


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS files (
            file TEXT PRIMARY KEY,
            year INTEGER,
            month INTEGER,
            title TEXT
        );
        CREATE TABLE IF NOT EXISTS sections (
            file TEXT,
            section_id INTEGER,
            header TEXT,
            ord INTEGER,
            PRIMARY KEY (file, section_id)
        );
        CREATE TABLE IF NOT EXISTS tables (
            file TEXT,
            table_id TEXT,
            section_id INTEGER,
            page_id INTEGER,
            title TEXT,
            units TEXT,
            n_rows INTEGER,
            n_cols INTEGER,
            json_blob TEXT,
            raw_html TEXT,
            row_labels TEXT,
            col_labels TEXT,
            PRIMARY KEY (file, table_id)
        );
        CREATE TABLE IF NOT EXISTS prose (
            file TEXT,
            section_id INTEGER,
            ord INTEGER,
            text TEXT
        );
        CREATE TABLE IF NOT EXISTS footnotes (
            file TEXT,
            ord INTEGER,
            text TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tables_file ON tables(file);
        CREATE INDEX IF NOT EXISTS idx_sections_file ON sections(file);
        CREATE INDEX IF NOT EXISTS idx_prose_file ON prose(file);
        """
    )


def create_fts(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS tables_fts;
        CREATE VIRTUAL TABLE tables_fts USING fts5(
            file, table_id, title, row_labels, col_labels,
            tokenize = 'porter unicode61'
        );
        INSERT INTO tables_fts (file, table_id, title, row_labels, col_labels)
            SELECT file, table_id, title, row_labels, col_labels FROM tables;

        DROP TABLE IF EXISTS prose_fts;
        CREATE VIRTUAL TABLE prose_fts USING fts5(
            file, section_id, text,
            tokenize = 'porter unicode61'
        );
        INSERT INTO prose_fts (file, section_id, text)
            SELECT file, section_id, text FROM prose;

        DROP TABLE IF EXISTS files_fts;
        CREATE VIRTUAL TABLE files_fts USING fts5(
            file, year, title, all_sections, all_table_titles, all_row_labels,
            tokenize = 'porter unicode61'
        );
        """
    )
    # Aggregate file-level FTS: one row per file with concatenated text
    conn.execute(
        """
        INSERT INTO files_fts (file, year, title, all_sections, all_table_titles, all_row_labels)
        SELECT
            f.file,
            CAST(f.year AS TEXT),
            COALESCE(f.title, ''),
            COALESCE((SELECT GROUP_CONCAT(header, ' | ') FROM sections WHERE file = f.file), ''),
            COALESCE((SELECT GROUP_CONCAT(title, ' | ') FROM tables WHERE file = f.file), ''),
            COALESCE((SELECT GROUP_CONCAT(row_labels, ' | ') FROM tables WHERE file = f.file), '')
        FROM files f
        """
    )


def process_file(conn: sqlite3.Connection, fp: Path) -> dict[str, int]:
    stats = {"tables": 0, "sections": 0, "prose": 0, "footnotes": 0, "parse_fail": 0}
    try:
        data = json.loads(fp.read_text())
    except Exception:
        return stats

    doc = data.get("document") or {}
    elements = doc.get("elements") or []

    year, month = parse_filename(fp.name)
    file_title = None

    current_section_id = 0
    current_section_header = ""
    sections_written: set[int] = set()

    table_counter = 0
    prose_counter = 0
    footnote_counter = 0
    last_prose_near_table: str | None = None

    for e in elements:
        etype = e.get("type")
        content = e.get("content") or ""
        bbox_list = e.get("bbox") or []
        page_id = bbox_list[0].get("page_id") if bbox_list else None

        if etype == "title" and file_title is None:
            file_title = content.strip()[:500]

        elif etype == "section_header":
            current_section_id += 1
            current_section_header = content.strip()
            if current_section_id not in sections_written:
                conn.execute(
                    "INSERT OR REPLACE INTO sections (file, section_id, header, ord) VALUES (?, ?, ?, ?)",
                    (fp.name, current_section_id, current_section_header, current_section_id),
                )
                sections_written.add(current_section_id)

        elif etype == "text":
            txt = content.strip()
            if txt:
                prose_counter += 1
                conn.execute(
                    "INSERT INTO prose (file, section_id, ord, text) VALUES (?, ?, ?, ?)",
                    (fp.name, current_section_id, prose_counter, txt),
                )
                last_prose_near_table = txt
                stats["prose"] += 1

        elif etype == "footnote":
            txt = content.strip()
            if txt:
                footnote_counter += 1
                conn.execute(
                    "INSERT INTO footnotes (file, ord, text) VALUES (?, ?, ?)",
                    (fp.name, footnote_counter, txt),
                )
                stats["footnotes"] += 1

        elif etype == "table":
            if not content.strip():
                continue
            try:
                dfs = pd.read_html(StringIO(content), flavor="lxml")
            except Exception:
                try:
                    dfs = pd.read_html(StringIO(content))
                except Exception:
                    stats["parse_fail"] += 1
                    # Fallback: treat as prose
                    prose_counter += 1
                    conn.execute(
                        "INSERT INTO prose (file, section_id, ord, text) VALUES (?, ?, ?, ?)",
                        (fp.name, current_section_id, prose_counter, content.strip()[:4000]),
                    )
                    stats["prose"] += 1
                    continue

            if not dfs:
                continue

            # A single element can hold multiple logical tables
            for sub_idx, df in enumerate(dfs):
                if df.empty:
                    continue
                table_counter += 1
                table_id = f"t{table_counter:04d}" + (f".{sub_idx}" if len(dfs) > 1 else "")
                structured = df_to_structured(df)

                # Title/units heuristic:
                # - Title = section header (most reliable anchor)
                # - If nearest prose is short AND isn't just a units declaration, append it
                # - Units = first unit-looking string from prose / section header / content
                title_candidate = current_section_header or ""
                nearby = last_prose_near_table
                nearby_is_units_only = bool(
                    nearby and UNIT_RE.search(nearby) and len(nearby.strip()) < 60
                )
                if nearby and len(nearby) < 200 and not nearby_is_units_only:
                    # Enrich title with nearby prose context (often the actual table caption)
                    if title_candidate and nearby not in title_candidate:
                        title_candidate = f"{title_candidate} — {nearby}"
                    elif not title_candidate:
                        title_candidate = nearby
                units = (
                    extract_units(nearby or "")
                    or extract_units(title_candidate)
                    or extract_units(content[:400])
                )

                rl = row_labels_text(structured)
                cl = col_labels_text(structured)

                conn.execute(
                    """
                    INSERT INTO tables
                      (file, table_id, section_id, page_id, title, units,
                       n_rows, n_cols, json_blob, raw_html, row_labels, col_labels)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        fp.name,
                        table_id,
                        current_section_id,
                        page_id,
                        title_candidate[:500],
                        units,
                        len(structured["rows"]),
                        len(structured["columns"]),
                        json.dumps(structured, ensure_ascii=False),
                        content,
                        rl[:4000],
                        cl[:2000],
                    ),
                )
                stats["tables"] += 1

    conn.execute(
        "INSERT OR REPLACE INTO files (file, year, month, title) VALUES (?, ?, ?, ?)",
        (fp.name, year, month, file_title),
    )
    stats["sections"] = len(sections_written)
    return stats


def build(out_path: Path, limit: int | None = None) -> None:
    files = sorted(CORPUS_DIR.glob("*.json"))
    if limit:
        files = files[:limit]

    if out_path.exists():
        out_path.unlink()

    conn = sqlite3.connect(str(out_path))
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    create_schema(conn)

    totals = {"tables": 0, "sections": 0, "prose": 0, "footnotes": 0, "parse_fail": 0}
    sys.stdout.reconfigure(line_buffering=True)
    for i, fp in enumerate(files, 1):
        stats = process_file(conn, fp)
        for k, v in stats.items():
            totals[k] += v
        if i % 25 == 0 or i == len(files):
            print(
                f"  [{i:4d}/{len(files)}] {fp.name}  "
                f"tables={totals['tables']:>6}  "
                f"prose={totals['prose']:>6}  "
                f"fail={totals['parse_fail']}",
                flush=True,
            )
        conn.commit()

    print("Building FTS indexes...")
    create_fts(conn)
    conn.commit()

    # Size report
    out_mb = out_path.stat().st_size / 1_048_576
    print(f"\n✓ {out_path} built: {out_mb:.1f} MB")
    print(f"  files:     {len(files)}")
    print(f"  tables:    {totals['tables']}")
    print(f"  sections:  {totals['sections']}")
    print(f"  prose:     {totals['prose']}")
    print(f"  footnotes: {totals['footnotes']}")
    print(f"  parse_fail:{totals['parse_fail']}")
    conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--limit", type=int, default=None, help="Only process first N files")
    args = ap.parse_args()
    build(args.out, args.limit)


if __name__ == "__main__":
    main()
