#!/usr/bin/env python3
"""Build the master ledger from corpus_json/ into ledger.sqlite.

Pipeline:
  walk corpus_json/*.json
   ↓
  for each element of type=='table':
    lxml primary parse + pd.read_html cross-check
   ↓
  normalize cells (raw / missing / zero / unparseable / footnote / revised / preliminary)
   ↓
  insert into SQLite: tables, table_columns, table_rows, cells
   ↓
  create views: facts, canonical_facts, supersessions
   ↓
  per-file logging + final sanity checks

Layers:
  tables           — one row per source table, with signature hash for dedup
  table_columns    — columns per table, with parsed year/month
  table_rows       — rows per table, with row_path + indent level
  cells            — raw parsed cells, with missing/zero/footnote flags
  facts (view)     — cells joined to context, with resolved data_year
  canonical_facts  — facts deduped by (signature, row_path, col_path, data_year),
                     keeping the latest-published cell
  supersessions    — everything canonical_facts dropped, for audit

Usage:
  uv run python build_ledger.py --limit 10        # sanity check on 10 files
  uv run python build_ledger.py                   # full build
  uv run python build_ledger.py --rebuild         # drop and rebuild
  uv run python build_ledger.py --rebuild --output ledger.next.sqlite
      # build to a side file; swap into place when done (keeps old DB for agents)
"""

import argparse
import hashlib
import io
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd
from lxml import html as lhtml

sys.stdout.reconfigure(line_buffering=True)

CORPUS_JSON = Path("corpus_json")
LEDGER_PATH = Path("ledger.sqlite")

FILENAME_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")
YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b")
MONTH_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december|"
    r"jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)\b",
    re.IGNORECASE,
)
MONTH_TO_INT = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

MISSING_TOKENS = {
    "",
    "-",
    "—",
    "–",
    "...",
    "....",
    ".....",
    "nan",
    "na",
    "n.a.",
    "n/a",
    "(x)",
    "x",
}

UNIT_PATTERNS = [
    # Scale + currency (the dominant case in Treasury bulletins)
    # Handle common OCR typos: "thousande", "thousand of" (missing s), "million" (singular)
    (re.compile(r"\b(in\s+)?millions?\s+of\s+dollars?\b", re.I), "millions_usd"),
    (re.compile(r"\bin\s+million\b", re.I), "millions_usd"),
    (re.compile(r"\b(in\s+)?billions?\s+of\s+dollars?\b", re.I), "billions_usd"),
    (re.compile(r"\bin\s+billion\b", re.I), "billions_usd"),
    (re.compile(r"\b(in\s+)?thousande?s?\s+of\s+dollars?\b", re.I), "thousands_usd"),
    (re.compile(r"\b(in\s+)?thousands?\b", re.I), "thousands_usd"),
    (re.compile(r"\bin\s+thousand\b", re.I), "thousands_usd"),
    # Scale-only (no currency scope — applied inside context)
    (re.compile(r"\b(in\s+)?millions?\b", re.I), "millions_usd"),
    (re.compile(r"\b(in\s+)?billions?\b", re.I), "billions_usd"),
    # Raw dollars, no scale
    (re.compile(r"^\s*\(?dollars?\)?\s*$", re.I), "dollars"),
    # Percent / index / counts
    (re.compile(r"\bpercent(age)?\b|\bper\s*cent\b|%", re.I), "percent"),
    (re.compile(r"\bindex\b", re.I), "index"),
    (re.compile(r"\bnumber\b", re.I), "count"),
]


# ── Schema ───────────────────────────────────────────────────────────────────

SCHEMA_DDL = """
CREATE TABLE IF NOT EXISTS tables (
  id             INTEGER PRIMARY KEY,
  file           TEXT    NOT NULL,
  element_seq    INTEGER NOT NULL,  -- 0-based document-order index (unique within file)
  element_id     INTEGER,           -- original element id (not unique within file!)
  page_id        INTEGER,
  file_year      INTEGER,
  file_month     INTEGER,
  title          TEXT,
  section        TEXT,
  caption        TEXT,
  unit           TEXT,       -- 'millions_usd' | 'billions_usd' | 'thousands_usd' | 'percent' | 'index' | null
  period         TEXT,       -- 'fiscal' | 'calendar' | 'unknown'
  table_kind     TEXT,       -- 'data' | 'toc' | 'unknown'
  n_rows         INTEGER,
  n_cols         INTEGER,
  signature      TEXT,       -- sha1[:16] of lower(title|section|caption)
  source_sha256  TEXT,       -- sha256 of the raw HTML content (for drift detection)
  raw_html       TEXT,       -- populated only for parse failures / disagreements, null otherwise
  parse_method   TEXT,       -- 'dual_agree' | 'dual_disagree' | 'lxml_only' | 'pandas_only' | 'fail'
  parse_ok       INTEGER,    -- 0 or 1
  parse_error    TEXT,
  continues_from_table_id INTEGER REFERENCES tables(id),  -- Phase 4: multi-page continuation
  UNIQUE (file, element_seq)
);

CREATE TABLE IF NOT EXISTS table_columns (
  id              INTEGER PRIMARY KEY,
  table_id        INTEGER NOT NULL REFERENCES tables(id),
  col_index       INTEGER NOT NULL,
  col_path        TEXT,     -- 'Budget > 1940 > Expenditures'
  col_leaf        TEXT,     -- 'Expenditures'
  year_extracted  INTEGER,
  month_extracted INTEGER
);

CREATE TABLE IF NOT EXISTS table_rows (
  id                 INTEGER PRIMARY KEY,
  table_id           INTEGER NOT NULL REFERENCES tables(id),
  row_index          INTEGER NOT NULL,
  row_path           TEXT,
  row_leaf           TEXT,
  indent_level       INTEGER DEFAULT 0,
  year_extracted     INTEGER,
  month_extracted    INTEGER,
  is_section_header  INTEGER DEFAULT 0,
  metric_slug        TEXT               -- normalize_metric_slug(row_path), populated in build_metric_layer
);

CREATE TABLE IF NOT EXISTS cells (
  table_id       INTEGER NOT NULL,
  row_id         INTEGER NOT NULL,
  col_id         INTEGER NOT NULL,
  raw_value      TEXT,
  numeric_value  REAL,
  is_missing     INTEGER DEFAULT 0,
  is_zero        INTEGER DEFAULT 0,
  is_percent     INTEGER DEFAULT 0,  -- cell raw had a % suffix
  has_footnote   INTEGER DEFAULT 0,
  is_revised     INTEGER DEFAULT 0,
  is_preliminary INTEGER DEFAULT 0,
  is_estimated   INTEGER DEFAULT 0,
  parse_status   TEXT,     -- 'ok' | 'missing' | 'text' | 'unparseable'
  PRIMARY KEY (table_id, row_id, col_id)
);

-- Per-table invariant check: does the cell count reconcile with the raw HTML?
CREATE TABLE IF NOT EXISTS table_invariants (
  table_id       INTEGER PRIMARY KEY REFERENCES tables(id),
  raw_html_cells INTEGER,   -- count of <th> + <td> in source HTML
  n_origin_cells INTEGER,   -- non-extension grid positions (should equal raw_html_cells)
  n_column_cells INTEGER,   -- contributions to table_columns
  n_row_labels   INTEGER,   -- contributions to table_rows row_leaf
  n_section_hdr  INTEGER,   -- row-level section headers
  n_data_cells   INTEGER,   -- entries in cells table
  delta          INTEGER,   -- raw_html_cells - (column + row_label + section + data)
  ok             INTEGER    -- 1 if delta == 0
);

CREATE INDEX IF NOT EXISTS idx_tables_file_year ON tables(file_year, file_month);
CREATE INDEX IF NOT EXISTS idx_tables_signature ON tables(signature);
CREATE INDEX IF NOT EXISTS idx_rows_table       ON table_rows(table_id);
CREATE INDEX IF NOT EXISTS idx_cols_table       ON table_columns(table_id);
CREATE INDEX IF NOT EXISTS idx_cols_year        ON table_columns(year_extracted);
CREATE INDEX IF NOT EXISTS idx_rows_year        ON table_rows(year_extracted);
CREATE INDEX IF NOT EXISTS idx_cells_numeric    ON cells(numeric_value) WHERE numeric_value IS NOT NULL;

-- Prose elements (type='text' in the upstream parser). Currently dropped.
-- Carries the surrounding narrative — section recap, paragraph descriptions,
-- explanatory notes — that contextualizes adjacent tables.
CREATE TABLE IF NOT EXISTS prose (
  id              INTEGER PRIMARY KEY,
  file            TEXT NOT NULL,
  element_seq     INTEGER NOT NULL,
  page_id         INTEGER,
  file_year       INTEGER,
  file_month      INTEGER,
  section         TEXT,           -- latched current_section at time of element
  title           TEXT,           -- latched current_title
  content         TEXT NOT NULL,
  near_table_id   INTEGER REFERENCES tables(id),
  UNIQUE (file, element_seq)
);
CREATE INDEX IF NOT EXISTS idx_prose_file_page ON prose(file, page_id);
CREATE INDEX IF NOT EXISTS idx_prose_near      ON prose(near_table_id);

-- Footnote elements (type='footnote'). Currently dropped.
-- Finally gives meaning to the orphaned `cells.has_footnote` flags: at
-- query time we can join cells to footnotes via (file, page_id) proximity.
CREATE TABLE IF NOT EXISTS footnotes (
  id                   INTEGER PRIMARY KEY,
  file                 TEXT NOT NULL,
  element_seq          INTEGER NOT NULL,
  page_id              INTEGER,
  file_year            INTEGER,
  file_month           INTEGER,
  marker               TEXT,           -- '1/', '2/', '*', etc.
  content              TEXT NOT NULL,
  attached_to_table_id INTEGER REFERENCES tables(id),
  UNIQUE (file, element_seq)
);
CREATE INDEX IF NOT EXISTS idx_footnotes_file_page ON footnotes(file, page_id);
CREATE INDEX IF NOT EXISTS idx_footnotes_attached  ON footnotes(attached_to_table_id);

-- Per-page metadata: page headers, footers, image URIs. Cheap, used to
-- carry section context across page breaks and for downstream references
-- back to the source PDF.
CREATE TABLE IF NOT EXISTS page_metadata (
  file            TEXT NOT NULL,
  page_id         INTEGER NOT NULL,
  header_text     TEXT,
  footer_text     TEXT,
  image_uri       TEXT,
  PRIMARY KEY (file, page_id)
);

-- Phase 3: metrics is a VIEW over cells/table_rows/table_columns/tables.
-- Synthesized CY/FY totals (~10K rows) live in metrics_synthesized and
-- are UNION ALL'd into the VIEW. This saves ~5GB vs a materialized table.
-- The metric_slug column on table_rows is populated by build_metric_layer.
CREATE TABLE IF NOT EXISTS metrics_synthesized (
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
);
"""

VIEWS_DDL = """
DROP VIEW IF EXISTS facts;
CREATE VIEW facts AS
SELECT
  c.table_id, c.row_id, c.col_id,
  t.file, t.file_year, t.file_month, t.element_id, t.page_id,
  t.signature AS table_signature,
  t.title, t.section, t.caption, t.unit, t.period,
  r.row_path, r.row_leaf, r.year_extracted AS row_year, r.month_extracted AS row_month, r.indent_level,
  col.col_path, col.col_leaf, col.year_extracted AS col_year, col.month_extracted AS col_month,
  c.raw_value, c.numeric_value,
  c.is_missing, c.is_zero, c.has_footnote, c.is_revised, c.is_preliminary,
  c.is_estimated, c.parse_status,
  COALESCE(col.year_extracted, r.year_extracted, t.file_year) AS data_year,
  COALESCE(col.month_extracted, r.month_extracted) AS data_month
FROM cells c
JOIN tables t      ON c.table_id = t.id
JOIN table_rows r  ON c.row_id   = r.id
JOIN table_columns col ON c.col_id = col.id
WHERE t.parse_ok = 1;

DROP VIEW IF EXISTS facts_ranked;
CREATE VIEW facts_ranked AS
SELECT
  f.*,
  ROW_NUMBER() OVER (
    PARTITION BY f.table_signature,
                 LOWER(TRIM(COALESCE(f.row_path, ''))),
                 LOWER(TRIM(COALESCE(f.col_path, ''))),
                 f.data_year
    ORDER BY f.file_year DESC, f.file_month DESC
  ) AS dedup_rank
FROM facts f
WHERE f.parse_status IN ('ok', 'missing')
  AND f.table_id IN (SELECT id FROM tables WHERE table_kind = 'data');

DROP VIEW IF EXISTS canonical_facts;
CREATE VIEW canonical_facts AS
SELECT * FROM facts_ranked WHERE dedup_rank = 1;

DROP VIEW IF EXISTS supersessions;
CREATE VIEW supersessions AS
SELECT * FROM facts_ranked WHERE dedup_rank > 1;
"""


def init_db(path: Path, rebuild: bool = False) -> sqlite3.Connection:
    if rebuild and path.exists():
        path.unlink()
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.executescript(SCHEMA_DDL)
    return conn


# ── Primary parser: lxml grid walker with rowspan/colspan ────────────────────


def parse_table_lxml(html_str: str) -> dict:
    """Parse a <table> HTML string into a rectangular grid, preserving spans.

    Returns a dict with:
      parse_ok     : bool
      error        : str | None
      n_rows       : int
      n_cols       : int
      grid         : list[list[str]]   — [row][col] → cell text
      is_origin    : list[list[bool]]  — True for positions that came directly
                                         from a <th>/<td>, False for extension
                                         positions created by rowspan/colspan
      is_header    : list[bool]        — per-row flag, True when every cell was <th>
      n_th_raw     : int               — count of <th> elements in source
      n_td_raw     : int               — count of <td> elements in source
      n_origins    : int               — count of origin grid positions (should
                                         equal n_th_raw + n_td_raw)
    """
    empty = {
        "parse_ok": False,
        "error": "empty html",
        "n_rows": 0,
        "n_cols": 0,
        "grid": [],
        "is_origin": [],
        "is_header": [],
        "row_indents": [],
        "n_th_raw": 0,
        "n_td_raw": 0,
        "n_origins": 0,
    }
    if not html_str or not html_str.strip():
        return empty

    try:
        root = lhtml.fromstring(html_str)
    except Exception as e:
        return {**empty, "error": f"lxml parse: {e}"}

    table = root if root.tag == "table" else root.find(".//table")
    if table is None:
        return {**empty, "error": "no <table> element"}

    trs = table.findall(".//tr")
    if not trs:
        return {**empty, "error": "no <tr> elements"}

    # Pass 1: determine max column width, accounting for colspan on each row.
    max_cols = 0
    for tr in trs:
        width = 0
        for cell in tr:
            if cell.tag not in ("th", "td"):
                continue
            try:
                width += int(cell.get("colspan", "1") or "1")
            except ValueError:
                width += 1
        if width > max_cols:
            max_cols = width

    n_rows = len(trs)
    grid: list[list[str | None]] = [[None] * max_cols for _ in range(n_rows)]
    is_origin: list[list[bool]] = [[False] * max_cols for _ in range(n_rows)]
    is_header = [False] * n_rows
    row_indents: list[int] = [0] * n_rows
    n_th_raw = 0
    n_td_raw = 0
    n_origins = 0

    # Pass 2: fill the grid, respecting rowspan + colspan, skipping cells
    # already occupied by a prior rowspan.
    for ri, tr in enumerate(trs):
        ci = 0
        while ci < max_cols and grid[ri][ci] is not None:
            ci += 1

        saw_th = False
        saw_td = False
        first_cell_style: str | None = None
        for cell in tr:
            if cell.tag not in ("th", "td"):
                continue
            if cell.tag == "th":
                saw_th = True
                n_th_raw += 1
            else:
                saw_td = True
                n_td_raw += 1

            # Capture the first cell's style string so we can recover
            # indent information from padding-left: that's how the upstream
            # parser encodes hierarchy in text rows.
            if first_cell_style is None:
                first_cell_style = cell.get("style") or ""

            try:
                rs = max(1, int(cell.get("rowspan", "1") or "1"))
                cs = max(1, int(cell.get("colspan", "1") or "1"))
            except ValueError:
                rs, cs = 1, 1

            text = " ".join(cell.text_content().split()).strip()
            _origin_r, _origin_c = ri, ci  # noqa: F841

            for dr in range(rs):
                for dc in range(cs):
                    r, c = ri + dr, ci + dc
                    if 0 <= r < n_rows and 0 <= c < max_cols:
                        if dr == 0 and dc == 0:
                            if grid[r][c] is None:
                                grid[r][c] = text
                                is_origin[r][c] = True
                                n_origins += 1
                        else:
                            if grid[r][c] is None:
                                grid[r][c] = ""
                                # extension — is_origin stays False

            ci += cs
            while ci < max_cols and grid[ri][ci] is not None:
                ci += 1

        is_header[ri] = saw_th and not saw_td

        # Derive indent level from padding-left (px), capped at 6. Treasury
        # tables use ~20px per logical indent level; 10px is also common.
        if first_cell_style:
            m = re.search(r"padding-left\s*:\s*(\d+)\s*px", first_cell_style)
            if m:
                px = int(m.group(1))
                row_indents[ri] = min(6, px // 10)

    # Any leftover None → empty string (rare; happens if a row has fewer cells
    # than max_cols and no rowspan filled the gap). These are not origins.
    for r in range(n_rows):
        for c in range(max_cols):
            if grid[r][c] is None:
                grid[r][c] = ""

    # Post-fill: propagate row-label text through column-0 rowspan extensions.
    # When a row label like "National defense" spans N rows, extension rows get
    # grid[r][0] = "" which produces empty row_path — breaking retrieval.
    # Fix: forward-fill column 0 for extension cells (is_origin[r][0] == False)
    # that follow a non-empty origin. Stop at the next origin cell.
    # Scope: column 0 ONLY. Numeric data cells (other columns) must NOT be copied.
    if max_cols > 0:
        last_label = ""
        for r in range(n_rows):
            if is_origin[r][0]:
                # This row has its own label — update the running label.
                last_label = grid[r][0]  # type: ignore[assignment]
            else:
                # Extension cell: fill from the spanning label above if non-empty.
                if last_label:
                    grid[r][0] = last_label

    return {
        "parse_ok": True,
        "error": None,
        "n_rows": n_rows,
        "n_cols": max_cols,
        "grid": grid,  # type: ignore[return-value]
        "is_origin": is_origin,
        "is_header": is_header,
        "row_indents": row_indents,
        "n_th_raw": n_th_raw,
        "n_td_raw": n_td_raw,
        "n_origins": n_origins,
    }


# ── Secondary parser: pd.read_html cross-check ──────────────────────────────


def parse_table_pandas(html_str: str) -> dict:
    """Shape-only cross-check via pd.read_html. Returns data row/col counts
    plus the raw DataFrame so the ingest can rescue lxml failures by rebuilding
    a grid from the DataFrame instead of dropping the table entirely.
    """
    if not html_str or not html_str.strip():
        return {
            "parse_ok": False,
            "error": "empty html",
            "n_data_rows": 0,
            "n_data_cols": 0,
            "df": None,
        }
    try:
        dfs = pd.read_html(io.StringIO(html_str), flavor="lxml")
    except Exception as e:
        return {
            "parse_ok": False,
            "error": f"pandas: {e}",
            "n_data_rows": 0,
            "n_data_cols": 0,
            "df": None,
        }
    if not dfs:
        return {
            "parse_ok": False,
            "error": "pandas returned no tables",
            "n_data_rows": 0,
            "n_data_cols": 0,
            "df": None,
        }
    df = dfs[0]
    return {
        "parse_ok": True,
        "error": None,
        "n_data_rows": int(df.shape[0]),
        "n_data_cols": int(df.shape[1]),
        "df": df,
    }


def pandas_df_to_grid(df: "pd.DataFrame") -> dict:
    """Rebuild the lxml-shaped result from a pandas DataFrame.

    Used as a rescue path when lxml fails but pandas succeeded: rather than
    dropping the whole table (losing every cell), we reconstruct a flat grid
    where every cell is an origin and the first row(s) are flagged as headers.
    Multi-level column headers become multiple header rows.
    """
    n_rows_data = int(df.shape[0])
    n_cols = int(df.shape[1])
    cols = df.columns

    header_rows: list[list[str]] = []
    if hasattr(cols, "nlevels") and cols.nlevels > 1:
        for lv in range(cols.nlevels):
            header_rows.append(
                [
                    ""
                    if str(cols.get_level_values(lv)[c]).startswith("Unnamed")
                    else str(cols.get_level_values(lv)[c])
                    for c in range(n_cols)
                ]
            )
    else:
        header_rows.append(["" if str(c).startswith("Unnamed") else str(c) for c in cols])

    data_rows: list[list[str]] = []
    for r in range(n_rows_data):
        row = []
        for c in range(n_cols):
            v = df.iat[r, c]
            if pd.isna(v):
                row.append("")
            else:
                row.append(str(v))
        data_rows.append(row)

    grid = header_rows + data_rows
    n_rows = len(grid)
    is_origin = [[True] * n_cols for _ in range(n_rows)]
    is_header = [True] * len(header_rows) + [False] * len(data_rows)
    row_indents = [0] * n_rows  # pandas rescue loses style info
    n_origins = n_rows * n_cols

    return {
        "parse_ok": True,
        "error": None,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "grid": grid,
        "is_origin": is_origin,
        "is_header": is_header,
        "row_indents": row_indents,
        "n_th_raw": len(header_rows) * n_cols,
        "n_td_raw": len(data_rows) * n_cols,
        "n_origins": n_origins,
    }


def cross_check(lxml_result: dict, pandas_result: dict) -> str:
    """Return the parse_method label based on the two parsers agreeing.

    pandas reports data-only row count (header stripped); lxml reports total
    row count. We compare pandas.n_data_rows against lxml.(n_rows - n_header_rows).
    """
    if not lxml_result["parse_ok"] and not pandas_result["parse_ok"]:
        return "fail"
    if not lxml_result["parse_ok"]:
        return "pandas_only"
    if not pandas_result["parse_ok"]:
        return "lxml_only"

    # Only count CONTIGUOUS top header rows (consistent with build_column_paths)
    n_header_rows = 0
    for h in lxml_result["is_header"]:
        if h:
            n_header_rows += 1
        else:
            break
    lxml_data_rows = max(0, lxml_result["n_rows"] - n_header_rows)
    pandas_data_rows = pandas_result["n_data_rows"]

    rows_close = abs(lxml_data_rows - pandas_data_rows) <= 1
    cols_close = abs(lxml_result["n_cols"] - pandas_result["n_data_cols"]) <= 1
    return "dual_agree" if (rows_close and cols_close) else "dual_disagree"


# ── Header/row path extraction from grid ────────────────────────────────────


def build_column_paths(grid: list[list[str]], is_header: list[bool]) -> list[dict]:
    """Build column paths by joining header-row cells in each column with ' > '.

    For each column index c, walks down all header rows and concatenates the
    non-empty cell values into a path like 'Federal Government > Civilian'.
    Empty cells (from spanned extensions, or genuinely blank) are skipped.
    """
    if not grid:
        return []
    n_cols = len(grid[0]) if grid else 0
    # Only use CONTIGUOUS header rows from the top of the table.
    # A <th> row buried in the middle (e.g. a sub-section header) should
    # NOT be joined into the column path — it may have shifted alignment
    # or different semantics.
    header_row_indices: list[int] = []
    for i, h in enumerate(is_header):
        if h:
            header_row_indices.append(i)
        else:
            break  # stop at first non-header row

    # If no header rows detected, treat the first row as the header.
    if not header_row_indices and grid:
        header_row_indices = [0]

    # Pre-compute per-row "filled" headers: propagate the last non-empty cell
    # value left-to-right within each header row.  This handles colspan spans
    # where the lxml parser writes the text only into the origin cell and fills
    # extension cells with "".  Without propagation, "1940 > Jan." through
    # "1940 > Dec." would lose the "1940 >" prefix for all but the first column.
    filled: list[list[str]] = []
    for r in header_row_indices:
        row: list[str] = []
        last_val = ""
        for c in range(n_cols):
            cell = (grid[r][c] if c < len(grid[r]) else "") or ""
            cell = cell.strip()
            if cell:
                last_val = cell
            row.append(last_val)
        filled.append(row)

    cols: list[dict] = []
    for c in range(n_cols):
        parts: list[str] = []
        last_part: str | None = None
        for filled_row in filled:
            cell = filled_row[c] if c < len(filled_row) else ""
            if cell and cell != last_part:
                parts.append(cell)
                last_part = cell
        col_path = " > ".join(parts) if parts else ""
        col_leaf = parts[-1] if parts else ""
        year, month = extract_year_month(col_path)
        cols.append(
            {
                "col_index": c,
                "col_path": col_path,
                "col_leaf": col_leaf,
                "year_extracted": year,
                "month_extracted": month,
            }
        )
    return cols


def build_row_entries(
    grid: list[list[str]],
    is_header: list[bool],
    row_indents: list[int] | None = None,
) -> list[dict]:
    """Build row entries with row_path + indent level.

    First column is treated as the row label. Indent level comes from the
    optional `row_indents` list (derived from padding-left in the raw HTML).
    row_path is built by walking the ancestry stack at shallower indents,
    so a data row at indent=2 inherits the path segments at indent=0 and
    indent=1 that precede it.

    A row with an alphabetic label and no numeric data in other cells is
    treated as a section header. Section headers push into the ancestry
    stack at their own depth (preserving shallower siblings) rather than
    wiping the stack like the previous simple implementation.
    """
    rows: list[dict] = []
    n_rows = len(grid)
    if n_rows == 0:
        return rows

    if row_indents is None:
        row_indents = [0] * n_rows

    # Ancestry stack: one label slot per indent depth. Section headers
    # overwrite their depth and truncate deeper entries; data rows read
    # the prefix at depths < their own.
    ancestry: list[str] = []

    # Find where contiguous top headers end
    first_non_header = 0
    for i, h in enumerate(is_header):
        if h:
            first_non_header = i + 1
        else:
            break

    for r in range(n_rows):
        # Only skip contiguous top header rows. Mid-table <th> rows
        # (sub-section headers deep in the body) should be kept and
        # treated as section headers, not silently dropped.
        if is_header[r] and r < first_non_header:
            continue
        row_cells = grid[r]
        label = (row_cells[0] if row_cells else "").strip()
        if not label:
            # Empty label row — still record for cell alignment, but with no path
            rows.append(
                {
                    "row_index": r,
                    "row_path": "",
                    "row_leaf": "",
                    "indent_level": 0,
                    "year_extracted": None,
                    "month_extracted": None,
                    "is_section_header": 0,
                }
            )
            continue

        indent_level = row_indents[r] if r < len(row_indents) else 0

        # Section header heuristic: label is non-empty, contains at least
        # one alphabetic character (so "1940" alone doesn't qualify), and
        # all other cells in the row are empty or missing tokens.
        other_cells = [(c or "").strip() for c in row_cells[1:]]
        has_any_data = any(c and c.lower() not in MISSING_TOKENS for c in other_cells)
        label_has_alpha = any(ch.isalpha() for ch in label)
        is_section = 1 if (not has_any_data and label_has_alpha) else 0

        # Depth-aware ancestry: section header pushes into slot N (and
        # truncates deeper slots); data row reads the ancestry prefix.
        # When we have no padding-left info (indent_level == 0), fall back
        # to the flat-stack behaviour from before Phase 4 so hierarchy
        # isn't lost on tables where the upstream parser didn't emit style
        # attributes.
        if is_section:
            if indent_level > 0:
                if len(ancestry) <= indent_level:
                    ancestry.extend([""] * (indent_level - len(ancestry) + 1))
                ancestry[indent_level] = label
                del ancestry[indent_level + 1 :]
            else:
                ancestry = [label]
            row_path = " > ".join(a for a in ancestry if a)
        else:
            if indent_level > 0:
                effective = [a for a in ancestry[:indent_level] if a]
            else:
                effective = [a for a in ancestry if a]
            row_path = " > ".join(effective + [label]) if effective else label

        # Year/month extraction: try the leaf label first so a local year
        # mention wins, then fall back to walking the ancestry chain via
        # row_path. This is what makes "February" under "Fiscal year 1940"
        # inherit year=1940 — without it, every monthly row in a year-
        # labelled section loses its year and the ledger can't answer
        # month-level queries for any such table.
        year, month = extract_year_month(label)
        if year is None or month is None:
            path_year, path_month = extract_year_month(row_path)
            if year is None:
                year = path_year
            if month is None:
                month = path_month
        rows.append(
            {
                "row_index": r,
                "row_path": row_path,
                "row_leaf": label,
                "indent_level": indent_level,
                "year_extracted": year,
                "month_extracted": month,
                "is_section_header": is_section,
            }
        )
    return rows


def propagate_row_years_long_format(rows: list[dict]) -> None:
    """Fill year_extracted on bare-month rows from the latest explicit anchor.

    Walk order is the list order (same as ``row_index`` ascending from
    :func:`build_row_entries`). Matches ``migrate_propagate_year.walk_table``:
    track ``current_year`` from non-null ``year_extracted``, clear it on
    truthy ``is_section_header`` rows, and set ``year_extracted`` when it is
    currently null, ``month_extracted`` is set, and ``current_year`` is known.
    """
    current_year: int | None = None
    for row in rows:
        if row.get("is_section_header"):
            current_year = None
            continue
        if row.get("year_extracted") is not None:
            current_year = row["year_extracted"]
            continue
        if row.get("month_extracted") is not None and current_year is not None:
            row["year_extracted"] = current_year


def extract_year_month(text: str) -> tuple[int | None, int | None]:
    if not text:
        return None, None
    years = YEAR_RE.findall(text)
    year = int(years[0]) if years else None
    month_match = MONTH_RE.search(text)
    month = None
    if month_match:
        m = month_match.group(1).lower()
        month = MONTH_TO_INT.get(m)
    return year, month


# Pattern: "Month YYYY" pairs for title date-range parsing
_MONTH_YEAR_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december|jan|feb|mar|apr|may|jun|jul|aug|sept?|oct|nov|dec)"
    r"[\s.,]*(\d{4})\b",
    re.IGNORECASE,
)


def _title_date_range(title: str) -> tuple[int, int, int, int] | None:
    """Parse a rolling-series title like 'March 1979 through February 1980'.

    Returns (start_month, start_year, end_month, end_year) when the title
    contains at least two Month+Year pairs, else None.
    """
    matches = _MONTH_YEAR_RE.findall(title)
    if len(matches) < 2:
        return None
    start_m = MONTH_TO_INT.get(matches[0][0].lower())
    start_y = int(matches[0][1])
    end_m = MONTH_TO_INT.get(matches[-1][0].lower())
    end_y = int(matches[-1][1])
    if not start_m or not end_m:
        return None
    # Sanity: end must be after start, within a 24-month window
    total_months = (end_y - start_y) * 12 + (end_m - start_m)
    if not (0 < total_months <= 24):
        return None
    return start_m, start_y, end_m, end_y


def _fill_column_years(cols: list[dict], title: str) -> list[dict]:
    """For Type-B rolling-series tables, infer year_extracted on columns
    that have month_extracted but no year_extracted.

    Treasury rolling tables have titles like "March 1979 through February 1980"
    with column headers that are just month abbreviations ("Mar.", "Apr.", ...).
    The year isn't in the column header — it's deducible from the title range
    and the column's sequential position in the month sequence.

    Algorithm:
      1. Parse start and end (month, year) from title.
      2. Generate the ordered (month, year) sequence for that range.
      3. Walk columns in order; for each month-only column, assign the
         matching year from the sequence (advancing through it).
    """
    date_range = _title_date_range(title)
    if not date_range:
        return cols
    start_m, start_y, end_m, end_y = date_range

    # Build expected (month, year) sequence
    sequence: list[tuple[int, int]] = []
    y, m = start_y, start_m
    while (y < end_y) or (y == end_y and m <= end_m):
        sequence.append((m, y))
        m += 1
        if m > 12:
            m, y = 1, y + 1
        if len(sequence) > 25:
            break

    # Walk columns in order; for each month-only col, consume next matching
    # entry from the sequence.
    seq_pos = 0
    for col in cols:
        if col["year_extracted"] is not None or col["month_extracted"] is None:
            continue
        col_month = col["month_extracted"]
        # Advance sequence to find next occurrence of this month
        found = False
        for i in range(seq_pos, len(sequence)):
            if sequence[i][0] == col_month:
                col["year_extracted"] = sequence[i][1]
                seq_pos = i + 1
                found = True
                break
        if not found:
            # Month not in remaining sequence — stop to avoid wrong assignments
            break

    return cols


# ── Cell normalization ──────────────────────────────────────────────────────


def normalize_cell(raw: str | None) -> dict:
    """Classify and parse a single cell's string content.

    parse_status values:
      'ok'          — parsed to a numeric value
      'missing'     — empty/dash/nan/n.a./(x) etc.
      'text'        — descriptive content, not a numeric cell
      'unparseable' — looks numeric but couldn't be converted

    Flags preserved regardless of status: has_footnote, is_revised,
    is_preliminary, is_percent, is_zero.
    """
    out = {
        "raw_value": raw,
        "numeric_value": None,
        "is_missing": 0,
        "is_zero": 0,
        "is_percent": 0,
        "has_footnote": 0,
        "is_revised": 0,
        "is_preliminary": 0,
        "is_estimated": 0,
        "parse_status": "ok",
    }
    if raw is None:
        out["is_missing"] = 1
        out["parse_status"] = "missing"
        return out

    s = raw.strip()

    # Fast path: known missing tokens
    if s.lower() in MISSING_TOKENS:
        out["is_missing"] = 1
        out["parse_status"] = "missing"
        return out

    # Preserve text-heavy flag for the fallback at the end — Phase 4
    # inverts the order so the numeric extraction path runs first even
    # on alphanumeric cells. Only on numeric failure do we fall back to
    # the "text" classification.
    letter_count = sum(1 for c in s if c.isalpha())
    digit_count = sum(1 for c in s if c.isdigit())
    text_heavy = letter_count > 0 and letter_count > digit_count

    work = s

    # Trailing footnote markers: "1/", "2/", "*", "†"
    # For digit footnotes, strip a SINGLE trailing digit+/ first (handles
    # OCR-fused cells like "10,0653/" → "10,065" with footnote 3/).
    # Then also handle space-separated multi-digit markers like " 12/".
    if re.search(r"(\d/\s*$)|(\s\d{1,2}/\s*$)|(\s*\*+\s*$)|(\s*†\s*$)", work):
        out["has_footnote"] = 1
        work = re.sub(r"(\s+\d{1,2}/\s*$)|(\d/\s*$)|(\s*\*+\s*$)|(\s*†\s*$)", "", work).strip()

    # Leading footnote markers (less common but exists)
    if re.match(r"^\s*\d+/", work) and not re.match(r"^\s*\d+/\d+\s*$", work):
        out["has_footnote"] = 1
        work = re.sub(r"^\s*\d+/\s*", "", work).strip()

    # Revised / preliminary / estimated
    if re.search(r"\br/\s*$|\sr\s*$", work):
        out["is_revised"] = 1
        work = re.sub(r"\s*r/\s*$|\s*r\s*$", "", work).strip()
    if re.search(r"\bp/\s*$|\sp\s*$", work):
        out["is_preliminary"] = 1
        work = re.sub(r"\s*p/\s*$|\s*p\s*$", "", work).strip()
    if re.search(r"\be/\s*$", work):
        out["is_estimated"] = 1
        work = re.sub(r"\s*e/\s*$", "", work).strip()

    # Phase 4 item 4: inline parenthetical markers like "(revised)",
    # "(preliminary)", "(estimated)". The numeric extractor then runs on
    # the remaining digits, so "$2.5 million (revised)" now sets is_revised
    # and produces numeric_value=2.5 (the scale "million" is handled at
    # the table unit level, not per cell).
    for token, flag in (
        ("revised", "is_revised"),
        ("preliminary", "is_preliminary"),
        ("estimated", "is_estimated"),
        ("estimate", "is_estimated"),
    ):
        if re.search(rf"\(\s*{token}\s*\)", work, flags=re.I):
            out[flag] = 1
            work = re.sub(rf"\s*\(\s*{token}\s*\)\s*", " ", work, flags=re.I).strip()

    # Strip an inline scale word ("million", "billion", "thousand") when
    # the cell is a bare number + scale — the scale is recorded at table
    # level. Without this, "$2.5 million" never parses.
    work = re.sub(r"\bmillion[s]?\b|\bbillion[s]?\b|\bthousand[s]?\b", "", work, flags=re.I).strip()

    # Percent suffix: parse the number and flag it
    if work.endswith("%"):
        out["is_percent"] = 1
        work = work[:-1].strip()

    # After stripping markers, re-check empties
    if not work or work.lower() in MISSING_TOKENS:
        out["is_missing"] = 1
        out["parse_status"] = "missing"
        return out

    # Parentheses for negative
    neg = False
    if work.startswith("(") and work.endswith(")"):
        neg = True
        work = work[1:-1].strip()

    # Mixed fraction notation: "1,844 1/2" → 1844.5 (Treasury convention)
    mixed = re.match(r"^\s*([\d,]+)\s+(\d+)/(\d+)\s*$", work)
    if mixed:
        try:
            whole = float(mixed.group(1).replace(",", ""))
            num = float(mixed.group(2))
            den = float(mixed.group(3))
            if den != 0:
                val = whole + num / den
                if neg:
                    val = -val
                out["numeric_value"] = val
                if val == 0.0:
                    out["is_zero"] = 1
                return out
        except ValueError:
            pass

    # Bare fraction notation: "1/2" → 0.5 (bond series labels are caught
    # by the text classifier above, so anything reaching here is likely
    # a legitimate numeric fraction)
    bare_frac = re.match(r"^\s*(\d+)/(\d+)\s*$", work)
    if bare_frac:
        try:
            num = float(bare_frac.group(1))
            den = float(bare_frac.group(2))
            if den != 0:
                val = num / den
                if neg:
                    val = -val
                out["numeric_value"] = val
                if val == 0.0:
                    out["is_zero"] = 1
                return out
        except ValueError:
            pass

    # Strip formatting
    cleaned = work.replace(",", "").replace("$", "").replace(" ", "")
    if not cleaned or cleaned.lower() in MISSING_TOKENS:
        out["is_missing"] = 1
        out["parse_status"] = "missing"
        return out

    try:
        val = float(cleaned)
        if neg:
            val = -val
        out["numeric_value"] = val
        if val == 0.0:
            out["is_zero"] = 1
    except ValueError:
        # Numeric extraction failed. Fall back to 'text' if the raw cell was
        # dominated by alphabetic content (descriptive), otherwise mark it
        # as genuinely unparseable.
        out["parse_status"] = "text" if text_heavy else "unparseable"

    return out


def detect_unit_period_from_grid(
    grid: list[list[str]], is_header: list[bool], *external_texts: str
) -> tuple[str | None, str]:
    """Find the unit and period by scanning the first header rows of the grid,
    then falling back to external title/section/caption.

    Treasury bulletins overwhelmingly declare the unit as a colspan'd header
    row like '<th colspan="6">(In millions of dollars)</th>' rather than in
    an external caption element, so we look inside the table first.
    """
    # Gather strings from the first ~3 header rows and the first data row
    candidates: list[str] = []
    for r in range(min(len(grid), 4)):
        for cell in grid[r]:
            cell = (cell or "").strip()
            if cell:
                candidates.append(cell)

    blob_grid = " ".join(candidates)

    unit = detect_unit(blob_grid)
    if unit is None:
        unit = detect_unit(*external_texts)

    period = detect_period(blob_grid)
    if period == "unknown":
        period = detect_period(*external_texts)

    return unit, period


def detect_table_kind(title: str, section: str, caption: str) -> str:
    """Classify a table as 'toc' | 'data' based on its topical identity.

    Only the literal phrase "table of contents" counts as TOC — the looser
    `\\bcontents\\b` check mis-flagged thousands of real data tables whose
    title or section merely contained the word (e.g. "Contents of the trust
    fund portfolio").
    """
    blob = f"{title or ''} {section or ''} {caption or ''}".lower()
    if "table of contents" in blob:
        return "toc"
    return "data"


# ── Unit / period / signature ───────────────────────────────────────────────


def detect_unit(*texts: str) -> str | None:
    blob = " ".join(t or "" for t in texts)
    for pattern, label in UNIT_PATTERNS:
        if pattern.search(blob):
            return label
    return None


def detect_period(*texts: str) -> str:
    blob = " ".join(t or "" for t in texts).lower()
    has_fiscal = "fiscal year" in blob or re.search(r"\bfy\b", blob) is not None
    has_calendar = "calendar year" in blob or "calendar mo" in blob or "cal. yr" in blob
    if has_fiscal and not has_calendar:
        return "fiscal"
    if has_calendar and not has_fiscal:
        return "calendar"
    return "unknown"


def table_signature(title: str | None, section: str | None, caption: str | None) -> str:
    parts = [
        (title or "").strip().lower(),
        (section or "").strip().lower(),
        (caption or "").strip().lower(),
    ]
    s = "|".join(parts)
    s = re.sub(r"\s+", " ", s).strip()
    return hashlib.sha1(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


# ── File walker ─────────────────────────────────────────────────────────────


def _page_id_of(element: dict) -> int | None:
    bbox = element.get("bbox")
    if isinstance(bbox, list) and bbox and isinstance(bbox[0], dict):
        pid = bbox[0].get("page_id")
        try:
            return int(pid)
        except (TypeError, ValueError):
            return None
    return None


def _content_text(element: dict) -> str:
    c = element.get("content")
    if isinstance(c, str):
        return c
    return ""


_FOOTNOTE_MARKER_RE = re.compile(r"^\s*((?:\d+/)|(?:\*+)|(?:†))")


def _parse_footnote_marker(content: str) -> str | None:
    """Extract the leading footnote marker (e.g. '1/', '2/', '*') from a
    footnote element's content. Treasury convention is a leading digit+'/'."""
    if not content:
        return None
    m = _FOOTNOTE_MARKER_RE.match(content)
    return m.group(1) if m else None


def process_file(conn: sqlite3.Connection, path: Path) -> dict:
    """Parse one corpus_json file and insert all its tables into the ledger.

    Returns per-file stats for instrumentation.
    """
    stats = {
        "file": path.name,
        "tables_seen": 0,
        "tables_parsed_ok": 0,
        "tables_parse_fail": 0,
        "tables_toc": 0,
        "dual_agree": 0,
        "dual_disagree": 0,
        "lxml_only": 0,
        "pandas_only": 0,
        "cells_total": 0,
        "cells_ok": 0,
        "cells_missing": 0,
        "cells_text": 0,
        "cells_unparseable": 0,
        "cells_zero": 0,
        "cells_percent": 0,
        "invariant_ok": 0,
        "invariant_mismatch": 0,
        "invariant_total_delta": 0,
    }

    try:
        data = json.loads(path.read_text())
    except Exception as e:
        stats["read_error"] = str(e)
        return stats

    m = FILENAME_RE.search(path.name)
    file_year = int(m.group(1)) if m else None
    file_month = int(m.group(2)) if m else None

    doc = data.get("document") or {}
    elements = doc.get("elements") or []

    current_title = ""
    current_section = ""
    current_caption = ""
    current_caption_page: int | None = None

    # Phase 2: non-table element queues. We buffer these during the main
    # loop and resolve table attachments afterward, once every table in the
    # file has a db id.
    prose_queue: list[dict] = []
    footnote_queue: list[dict] = []
    page_headers: dict[int, str] = {}
    page_footers: dict[int, str] = {}
    tables_in_file: list[tuple[int, int | None, int]] = []  # (element_seq, page_id, table_id)

    # Phase 4 item 3: track the last table's identity for multi-page
    # continuation detection. Same signature on the next page → treat
    # the new table as a continuation of the prior one.
    prev_table: dict | None = None

    cur = conn.cursor()

    for idx, el in enumerate(elements):
        etype = el.get("type")
        el_page = _page_id_of(el)
        if etype == "title":
            current_title = _content_text(el) or current_title
            continue
        if etype == "section_header":
            current_section = _content_text(el) or current_section
            continue
        if etype == "caption":
            # Caption applies until the next table on the same page (or until
            # the page changes).
            current_caption = _content_text(el) or ""
            current_caption_page = el_page
            continue

        if etype != "table":
            # Phase 2 ingest: capture previously-dropped element types.
            if etype == "text":
                text = _content_text(el).strip()
                if text:
                    prose_queue.append(
                        {
                            "element_seq": idx,
                            "page_id": el_page,
                            "section": current_section,
                            "title": current_title,
                            "content": text,
                        }
                    )
            elif etype == "footnote":
                text = _content_text(el).strip()
                if text:
                    footnote_queue.append(
                        {
                            "element_seq": idx,
                            "page_id": el_page,
                            "marker": _parse_footnote_marker(text),
                            "content": text,
                        }
                    )
            elif etype == "page_header":
                text = _content_text(el).strip()
                if text and el_page is not None:
                    # Concatenate duplicate page_header elements on the same
                    # page (Treasury sometimes has multiple header strips).
                    prior = page_headers.get(el_page, "")
                    page_headers[el_page] = f"{prior}\n{text}".strip() if prior else text
            elif etype == "page_footer":
                text = _content_text(el).strip()
                if text and el_page is not None:
                    prior = page_footers.get(el_page, "")
                    page_footers[el_page] = f"{prior}\n{text}".strip() if prior else text

            # Drop any pending caption if we've crossed the page boundary
            # it was anchored to. Using page boundary rather than a fixed
            # element-count window lets long narratives between caption
            # and table live on the same page without losing the caption,
            # and prevents cross-page leakage.
            if (
                current_caption
                and current_caption_page is not None
                and el_page is not None
                and el_page != current_caption_page
            ):
                current_caption = ""
                current_caption_page = None
            continue

        stats["tables_seen"] += 1

        html_str = _content_text(el)
        page_id = _page_id_of(el)

        # If the pending caption was anchored to a different page, drop it
        # before attaching it to this table.
        if (
            current_caption
            and current_caption_page is not None
            and page_id is not None
            and page_id != current_caption_page
        ):
            current_caption = ""
            current_caption_page = None

        lxml_result = parse_table_lxml(html_str)
        pandas_result = parse_table_pandas(html_str)
        parse_method = cross_check(lxml_result, pandas_result)

        # Pick the authoritative result. Pandas-only rescue: when lxml fails
        # but pd.read_html succeeded, reshape the DataFrame back into a grid
        # rather than dropping hundreds of tables silently.
        if lxml_result["parse_ok"]:
            auth = lxml_result
        elif pandas_result["parse_ok"] and pandas_result.get("df") is not None:
            try:
                auth = pandas_df_to_grid(pandas_result["df"])
            except Exception as e:
                auth = None
                lxml_result["error"] = (
                    f"{lxml_result.get('error') or 'lxml failed'}; pandas rescue failed: {e}"
                )
        else:
            auth = None

        parse_ok = 1 if auth is not None else 0
        parse_error = lxml_result.get("error") if not parse_ok else None
        n_rows = auth["n_rows"] if auth else 0
        n_cols = auth["n_cols"] if auth else 0

        sig = table_signature(current_title, current_section, current_caption)
        table_kind = detect_table_kind(current_title, current_section, current_caption)

        # Multi-page continuation: same signature, consecutive pages, and
        # there was no intervening table with a different signature. We
        # compare against prev_table which is always the immediately prior
        # table in document order.
        continues_from = None
        if (
            prev_table is not None
            and prev_table["signature"] == sig
            and page_id is not None
            and prev_table["page_id"] is not None
            and page_id == prev_table["page_id"] + 1
        ):
            continues_from = prev_table["table_id"]

        # Source drift detection
        source_sha256 = (
            hashlib.sha256((html_str or "").encode("utf-8")).hexdigest() if html_str else None
        )

        # Store raw HTML only for problematic tables — failed parses, parser
        # disagreements, or pandas-only fallbacks. Successful dual_agree tables
        # can always be re-read from corpus_json if needed.
        store_html = parse_method in ("dual_disagree", "lxml_only", "pandas_only", "fail")
        raw_html_stored = html_str if store_html else None

        # Unit and period: prefer grid-internal declarations over external
        # metadata, since Treasury puts them as colspan'd header rows.
        if auth is not None:
            unit, period = detect_unit_period_from_grid(
                auth["grid"],
                auth["is_header"],
                current_caption,
                current_section,
                current_title,
            )
        else:
            unit = detect_unit(current_caption, current_section, current_title)
            period = detect_period(current_caption, current_section, current_title)

        cur.execute(
            """
            INSERT INTO tables
              (file, element_seq, element_id, page_id, file_year, file_month,
               title, section, caption, unit, period, table_kind,
               n_rows, n_cols, signature, source_sha256, raw_html,
               parse_method, parse_ok, parse_error, continues_from_table_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            RETURNING id
            """,
            (
                path.name,
                idx,
                el.get("id"),
                page_id,
                file_year,
                file_month,
                current_title,
                current_section,
                current_caption,
                unit,
                period,
                table_kind,
                n_rows,
                n_cols,
                sig,
                source_sha256,
                raw_html_stored,
                parse_method,
                parse_ok,
                parse_error,
                continues_from,
            ),
        )
        row = cur.fetchone()
        if row is None:
            print(f"    ! could not insert table {path.name}#{idx}", flush=True)
            current_caption = ""
            continue
        table_id = row[0]
        tables_in_file.append((idx, page_id, table_id))
        prev_table = {
            "signature": sig,
            "page_id": page_id,
            "table_id": table_id,
        }

        if table_kind == "toc":
            stats["tables_toc"] += 1

        # Consume the caption — it only applies to one table
        current_caption = ""
        current_caption_page = None

        if parse_method == "dual_agree":
            stats["dual_agree"] += 1
        elif parse_method == "dual_disagree":
            stats["dual_disagree"] += 1
        elif parse_method == "lxml_only":
            stats["lxml_only"] += 1
        elif parse_method == "pandas_only":
            stats["pandas_only"] += 1

        if not parse_ok:
            stats["tables_parse_fail"] += 1
            continue

        stats["tables_parsed_ok"] += 1

        # Build columns and rows
        cols = build_column_paths(auth["grid"], auth["is_header"])
        # For rolling-series tables (Type B), infer year_extracted on
        # month-only columns from the title date range.
        cols = _fill_column_years(cols, current_title or "")
        rows = build_row_entries(
            auth["grid"],
            auth["is_header"],
            auth.get("row_indents"),
        )
        propagate_row_years_long_format(rows)

        col_id_by_index: dict[int, int] = {}
        for col in cols:
            cur.execute(
                """
                INSERT INTO table_columns
                  (table_id, col_index, col_path, col_leaf, year_extracted, month_extracted)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    table_id,
                    col["col_index"],
                    col["col_path"],
                    col["col_leaf"],
                    col["year_extracted"],
                    col["month_extracted"],
                ),
            )
            col_id_by_index[col["col_index"]] = cur.lastrowid  # type: ignore[assignment]

        row_id_by_index: dict[int, int] = {}
        for row in rows:
            cur.execute(
                """
                INSERT INTO table_rows
                  (table_id, row_index, row_path, row_leaf, indent_level,
                   year_extracted, month_extracted, is_section_header)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    table_id,
                    row["row_index"],
                    row["row_path"],
                    row["row_leaf"],
                    row["indent_level"],
                    row["year_extracted"],
                    row["month_extracted"],
                    row["is_section_header"],
                ),
            )
            row_id_by_index[row["row_index"]] = cur.lastrowid  # type: ignore[assignment]

        # Insert cells: skip column 0 (that's the row label) and skip cells
        # in rows flagged as section headers (they have no data anyway).
        grid = auth["grid"]
        cell_batch: list[tuple] = []
        for row in rows:
            r = row["row_index"]
            if row["is_section_header"]:
                continue
            if r not in row_id_by_index:
                continue
            row_id = row_id_by_index[r]
            for c in range(1, len(grid[r])):
                if c not in col_id_by_index:
                    continue
                col_id = col_id_by_index[c]
                raw_cell = grid[r][c]
                norm = normalize_cell(raw_cell)
                cell_batch.append(
                    (
                        table_id,
                        row_id,
                        col_id,
                        norm["raw_value"],
                        norm["numeric_value"],
                        norm["is_missing"],
                        norm["is_zero"],
                        norm["is_percent"],
                        norm["has_footnote"],
                        norm["is_revised"],
                        norm["is_preliminary"],
                        norm["is_estimated"],
                        norm["parse_status"],
                    )
                )
                stats["cells_total"] += 1
                if norm["parse_status"] == "ok":
                    stats["cells_ok"] += 1
                elif norm["parse_status"] == "missing":
                    stats["cells_missing"] += 1
                elif norm["parse_status"] == "text":
                    stats["cells_text"] += 1
                else:
                    stats["cells_unparseable"] += 1
                if norm["is_zero"]:
                    stats["cells_zero"] += 1
                if norm["is_percent"]:
                    stats["cells_percent"] += 1

        if cell_batch:
            cur.executemany(
                """
                INSERT OR IGNORE INTO cells
                  (table_id, row_id, col_id, raw_value, numeric_value,
                   is_missing, is_zero, is_percent, has_footnote, is_revised,
                   is_preliminary, is_estimated, parse_status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                cell_batch,
            )

        # ─── 1-to-1 structural invariant check for this table ──────────
        # The key structural invariant: every <th>/<td> in the source HTML
        # becomes exactly one origin grid position. This verifies the lxml
        # walker is correct, independent of downstream accounting.
        raw_html_cells = auth["n_th_raw"] + auth["n_td_raw"]
        n_origin = auth["n_origins"]

        # Downstream accounting: how many cells ended up in the ledger,
        # split by destination. Not strictly equal to raw_html_cells because
        # some row-label cells overlap with origin column 0 that we already
        # counted in table_rows (not cells).
        n_column_cells = len(cols)  # col_paths inserted
        n_row_labels = sum(1 for r in rows if r["row_leaf"])
        n_section_headers = sum(1 for r in rows if r["is_section_header"])
        n_data_cells = len(cell_batch)
        _accounted = n_column_cells + n_row_labels + n_section_headers + n_data_cells  # noqa: F841

        # The structural delta is origins vs HTML cells — this must be 0.
        delta = raw_html_cells - n_origin
        ok = 1 if delta == 0 else 0

        cur.execute(
            """
            INSERT INTO table_invariants
              (table_id, raw_html_cells, n_origin_cells, n_column_cells,
               n_row_labels, n_section_hdr, n_data_cells, delta, ok)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                table_id,
                raw_html_cells,
                n_origin,
                n_column_cells,
                n_row_labels,
                n_section_headers,
                n_data_cells,
                delta,
                ok,
            ),
        )
        if ok:
            stats["invariant_ok"] += 1
        else:
            stats["invariant_mismatch"] += 1
        stats["invariant_total_delta"] += abs(delta)

    # ─── Phase 2: insert non-table elements ─────────────────────────
    # Attach prose to the nearest table on the same page (within a ±5
    # element-seq window), and attach footnotes to the most recent prior
    # table on the same page (footnotes typically follow the table they
    # annotate). Elements with no suitable table neighbor get NULL.

    def _nearest_table_same_page(el_seq: int, el_page: int | None, window: int = 5) -> int | None:
        if el_page is None:
            return None
        best: tuple[int, int] | None = None  # (distance, table_id)
        for t_seq, t_page, t_id in tables_in_file:
            if t_page != el_page:
                continue
            dist = abs(t_seq - el_seq)
            if dist > window:
                continue
            if best is None or dist < best[0]:
                best = (dist, t_id)
        return best[1] if best else None

    def _prior_table_same_page(el_seq: int, el_page: int | None) -> int | None:
        if el_page is None:
            return None
        best: tuple[int, int] | None = None  # (seq, table_id)
        for t_seq, t_page, t_id in tables_in_file:
            if t_page != el_page or t_seq > el_seq:
                continue
            if best is None or t_seq > best[0]:
                best = (t_seq, t_id)
        return best[1] if best else None

    for p in prose_queue:
        near_id = _nearest_table_same_page(p["element_seq"], p["page_id"])
        cur.execute(
            """
            INSERT OR IGNORE INTO prose
              (file, element_seq, page_id, file_year, file_month,
               section, title, content, near_table_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                path.name,
                p["element_seq"],
                p["page_id"],
                file_year,
                file_month,
                p["section"],
                p["title"],
                p["content"],
                near_id,
            ),
        )
        stats["prose_inserted"] = stats.get("prose_inserted", 0) + 1

    for fn in footnote_queue:
        attached = _prior_table_same_page(fn["element_seq"], fn["page_id"])
        cur.execute(
            """
            INSERT OR IGNORE INTO footnotes
              (file, element_seq, page_id, file_year, file_month,
               marker, content, attached_to_table_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                path.name,
                fn["element_seq"],
                fn["page_id"],
                file_year,
                file_month,
                fn["marker"],
                fn["content"],
                attached,
            ),
        )
        stats["footnotes_inserted"] = stats.get("footnotes_inserted", 0) + 1

    # Page metadata: merge header/footer collections with pages[] image_uri.
    pages_info: dict[int, dict] = {}
    for pg in doc.get("pages") or []:
        pid = pg.get("id")
        try:
            pid = int(pid) if pid is not None else None
        except (TypeError, ValueError):
            pid = None
        if pid is None:
            continue
        pages_info[pid] = {"image_uri": pg.get("image_uri")}

    all_page_ids = set(page_headers) | set(page_footers) | set(pages_info)
    for pid in all_page_ids:
        cur.execute(
            """
            INSERT OR REPLACE INTO page_metadata
              (file, page_id, header_text, footer_text, image_uri)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                path.name,
                pid,
                page_headers.get(pid),
                page_footers.get(pid),
                (pages_info.get(pid) or {}).get("image_uri"),
            ),
        )
        stats["page_metadata_inserted"] = stats.get("page_metadata_inserted", 0) + 1

    conn.commit()
    return stats


# ── Sanity checks ────────────────────────────────────────────────────────────


_METRIC_SLUG_FOOTNOTE_RE = re.compile(r"\s*\d+/")
_METRIC_SLUG_PUNCT_RE = re.compile(r"[^\w\s\-]")
_METRIC_SLUG_WS_RE = re.compile(r"\s+")


def normalize_metric_slug(metric: str | None) -> str:
    """Normalize a row/metric label the way the arena's master_ledger did.

    Lowercase, strip footnote markers like '1/', strip punctuation except
    hyphens, collapse whitespace. Matches the arena's historical
    _normalize_metric_slug regex so slugs are comparable to its vocabulary.
    """
    if not metric:
        return ""
    s = metric.lower().strip()
    s = _METRIC_SLUG_FOOTNOTE_RE.sub("", s)
    s = _METRIC_SLUG_PUNCT_RE.sub("", s)
    s = _METRIC_SLUG_WS_RE.sub(" ", s).strip()
    return s


def build_metric_layer(conn: sqlite3.Connection) -> None:
    """Build the metrics VIEW and lookup tables.

    The metrics VIEW joins cells/table_rows/table_columns/tables on the fly,
    avoiding a 5GB materialized table. Synthesized CY/FY totals (~10K rows)
    live in metrics_synthesized and are UNION ALL'd into the VIEW.

    Steps:
    1. Populate metric_slug on table_rows (normalize_metric_slug of row_path).
    2. Build row_label_lookup and col_label_lookup from the join query.
    3. Create the metrics VIEW (join + UNION ALL metrics_synthesized).
    4. Synthesize CY totals into metrics_synthesized.
    """
    print("\nBuilding metric layer...", flush=True)
    t0 = time.time()

    conn.create_function("norm_slug", 1, normalize_metric_slug, deterministic=True)

    # Drop old tables/views that we're rebuilding
    conn.executescript(
        """
        DROP VIEW IF EXISTS metrics;
        DELETE FROM metrics_synthesized;
        DROP TABLE IF EXISTS row_label_lookup;
        DROP TABLE IF EXISTS col_label_lookup;
        """
    )

    # Step 1: Populate metric_slug on table_rows
    print("  populating table_rows.metric_slug...", flush=True)
    conn.execute("UPDATE table_rows SET metric_slug = norm_slug(row_path)")
    conn.commit()

    # Count how many rows the VIEW will produce (for reporting)
    (n_view_rows,) = conn.execute(
        """
        SELECT COUNT(*)
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
        """
    ).fetchone()
    print(f"  metrics VIEW base rows: {n_view_rows:,}", flush=True)

    # Step 2: Lookup tables — dedup substrings for fast retrieval.
    # Built from the same join that the VIEW uses, but without materializing.
    conn.execute(
        """
        CREATE TABLE row_label_lookup AS
        SELECT DISTINCT r.table_id, r.metric_slug, r.row_path
          FROM table_rows r
          JOIN tables t ON r.table_id = t.id
         WHERE t.parse_ok = 1
           AND t.table_kind = 'data'
           AND r.is_section_header = 0
           AND r.row_path IS NOT NULL
           AND LENGTH(TRIM(r.row_path)) > 0
           AND r.metric_slug != ''
        """
    )
    conn.executescript(
        """
        CREATE INDEX idx_rll_slug  ON row_label_lookup(metric_slug);
        CREATE INDEX idx_rll_table ON row_label_lookup(table_id);
        """
    )

    conn.execute(
        """
        CREATE TABLE col_label_lookup AS
        SELECT DISTINCT
               col.table_id,
               col.col_path,
               norm_slug(col.col_path) AS col_norm
          FROM table_columns col
          JOIN tables t ON col.table_id = t.id
         WHERE t.parse_ok = 1
           AND t.table_kind = 'data'
           AND col.col_path IS NOT NULL
           AND col.col_path != ''
        """
    )
    conn.executescript(
        """
        CREATE INDEX idx_cll_norm  ON col_label_lookup(col_norm);
        CREATE INDEX idx_cll_table ON col_label_lookup(table_id);
        """
    )
    conn.commit()

    (n_rll,) = conn.execute("SELECT COUNT(*) FROM row_label_lookup").fetchone()
    (n_cll,) = conn.execute("SELECT COUNT(*) FROM col_label_lookup").fetchone()
    print(
        f"  row_label_lookup: {n_rll:,}   col_label_lookup: {n_cll:,}",
        flush=True,
    )

    # Step 3: Create the metrics VIEW
    conn.execute(
        """
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
    )
    conn.commit()
    print("  metrics VIEW created", flush=True)

    # Step 4: CY total synthesis. For any (table, metric, year) triple that
    # has 12 distinct months of numeric data, synthesize a row with the sum.
    # The arena's research report says 16 benchmark questions need this.
    t1 = time.time()
    conn.execute(
        """
        INSERT INTO metrics_synthesized
          (table_id, metric_slug, row_path, col_path, time_key,
           year, month, period_basis, value, value_raw, unit,
           file, page_id,
           has_footnote, is_revised, is_preliminary, is_estimated,
           is_synthesized)
        SELECT
            t.id,
            r.metric_slug,
            MAX(r.row_path),
            'CY total (synthesized)',
            printf('%04d', COALESCE(col.year_extracted, r.year_extracted, t.file_year)),
            COALESCE(col.year_extracted, r.year_extracted, t.file_year),
            NULL,
            'calendar',
            SUM(c.numeric_value),
            NULL,
            MAX(t.unit),
            MAX(t.file),
            MAX(t.page_id),
            MAX(c.has_footnote), MAX(c.is_revised), MAX(c.is_preliminary), MAX(c.is_estimated),
            1
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
          AND COALESCE(col.month_extracted, r.month_extracted) IS NOT NULL
          AND c.numeric_value IS NOT NULL
          AND r.metric_slug != ''
          AND COALESCE(col.year_extracted, r.year_extracted, t.file_year) IS NOT NULL
        GROUP BY t.id, r.metric_slug, COALESCE(col.year_extracted, r.year_extracted, t.file_year)
        HAVING COUNT(DISTINCT COALESCE(col.month_extracted, r.month_extracted)) = 12
        """
    )
    conn.commit()
    (n_synth,) = conn.execute("SELECT COUNT(*) FROM metrics_synthesized").fetchone()
    print(
        f"  CY synthesized rows: {n_synth:,}  ({time.time() - t1:.1f}s)",
        flush=True,
    )
    print(f"  metric layer built in {time.time() - t0:.1f}s", flush=True)


def build_fts_index(conn: sqlite3.Connection) -> None:
    """Build an FTS5 index over table-level context.

    Indexed content per row = concatenation of title, section, caption,
    column header text, and row label text — the same search blob used by
    retrieve.py's keyword filter. The rowid mirrors tables.id so lookups
    are a trivial JOIN. Rebuilt from scratch each time, which is cheap
    compared to the main ingest (~seconds on 94K tables).
    """
    print("\nBuilding FTS5 text index...", flush=True)
    t0 = time.time()
    conn.executescript(
        """
        DROP TABLE IF EXISTS tables_fts;
        CREATE VIRTUAL TABLE tables_fts USING fts5(
            title, section, caption, columns, rows,
            content='',
            tokenize='porter unicode61 remove_diacritics 2'
        );
        """
    )
    # Pull aggregated column/row text per table in one pass, then insert.
    cur = conn.execute(
        """
        SELECT
            t.id,
            COALESCE(t.title, ''),
            COALESCE(t.section, ''),
            COALESCE(t.caption, ''),
            COALESCE((SELECT GROUP_CONCAT(col_path, ' | ')
                      FROM table_columns tc WHERE tc.table_id = t.id), ''),
            COALESCE((SELECT GROUP_CONCAT(row_path, ' | ')
                      FROM table_rows tr WHERE tr.table_id = t.id), '')
        FROM tables t
        WHERE t.table_kind = 'data'
        """
    )
    conn.execute("BEGIN")
    n = 0
    for row in cur:
        conn.execute(
            "INSERT INTO tables_fts(rowid, title, section, caption, columns, rows) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            row,
        )
        n += 1
    conn.commit()
    dt = time.time() - t0
    print(f"  indexed {n:,} data tables in {dt:.1f}s", flush=True)

    # Phase 2: prose + footnotes FTS
    print("Building FTS5 prose + footnotes indices...", flush=True)
    t1 = time.time()
    conn.executescript(
        """
        DROP TABLE IF EXISTS prose_fts;
        CREATE VIRTUAL TABLE prose_fts USING fts5(
            content, section, title,
            content='',
            tokenize='porter unicode61 remove_diacritics 2'
        );
        DROP TABLE IF EXISTS footnotes_fts;
        CREATE VIRTUAL TABLE footnotes_fts USING fts5(
            content,
            content='',
            tokenize='porter unicode61 remove_diacritics 2'
        );
        """
    )
    p_rows = conn.execute(
        "SELECT id, COALESCE(content,''), COALESCE(section,''), COALESCE(title,'') FROM prose"
    )
    n_p = 0
    conn.execute("BEGIN")
    for rowid, content, section, title in p_rows:
        conn.execute(
            "INSERT INTO prose_fts(rowid, content, section, title) VALUES (?, ?, ?, ?)",
            (rowid, content, section, title),
        )
        n_p += 1
    conn.commit()

    f_rows = conn.execute("SELECT id, COALESCE(content,'') FROM footnotes")
    n_f = 0
    conn.execute("BEGIN")
    for rowid, content in f_rows:
        conn.execute(
            "INSERT INTO footnotes_fts(rowid, content) VALUES (?, ?)",
            (rowid, content),
        )
        n_f += 1
    conn.commit()
    print(
        f"  indexed {n_p:,} prose + {n_f:,} footnotes in {time.time() - t1:.1f}s",
        flush=True,
    )


def build_family_tables(conn: sqlite3.Connection) -> None:
    """Build table_family* tables and families_fts index.

    Groups tables by LOWER(title) as family_id (same grouping as original
    migration). Populates family_years from facts.data_year so year coverage
    is broad (includes row-extracted years, not just col-extracted years).
    """
    import time as _time

    t0 = _time.time()
    print("\n─── Family tables ──────────────────────────────────────────")

    # Ensure tables.family_id column exists
    cols = [r[1] for r in conn.execute("PRAGMA table_info(tables)").fetchall()]
    if "family_id" not in cols:
        conn.execute("ALTER TABLE tables ADD COLUMN family_id TEXT")
    conn.execute("UPDATE tables SET family_id = LOWER(TRIM(COALESCE(title, '')))")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_tables_family ON tables(family_id)")

    conn.executescript("""
        DROP TABLE IF EXISTS table_families;
        DROP TABLE IF EXISTS table_family_columns;
        DROP TABLE IF EXISTS table_family_rows;
        DROP TABLE IF EXISTS table_family_years;
        DROP TABLE IF EXISTS families_fts;

        CREATE TABLE table_families(
            family_id TEXT, primary_title TEXT,
            member_count INTEGER, latest_ver INTEGER, latest_table_id INTEGER
        );
        CREATE INDEX idx_families_fid    ON table_families(family_id);
        CREATE INDEX idx_families_latest ON table_families(latest_table_id);

        CREATE TABLE table_family_columns(family_id TEXT, col_leaf_lower TEXT);
        CREATE INDEX idx_family_columns_fid   ON table_family_columns(family_id);
        CREATE INDEX idx_family_columns_lower ON table_family_columns(col_leaf_lower);

        CREATE TABLE table_family_rows(family_id TEXT, metric_slug TEXT);
        CREATE INDEX idx_family_rows_fid  ON table_family_rows(family_id);
        CREATE INDEX idx_family_rows_slug ON table_family_rows(metric_slug);

        CREATE TABLE table_family_years(family_id TEXT, year INT, best_table_id INT);
        CREATE INDEX idx_family_years_fid      ON table_family_years(family_id);
        CREATE INDEX idx_family_years_year     ON table_family_years(year);
        CREATE INDEX idx_family_years_fid_year ON table_family_years(family_id, year);
        CREATE INDEX idx_family_years_best     ON table_family_years(best_table_id);
    """)

    conn.execute("""
        INSERT INTO table_families(family_id, primary_title, member_count, latest_ver, latest_table_id)
        SELECT family_id,
            (SELECT title FROM tables t2 WHERE t2.family_id = t.family_id
             ORDER BY t2.file_year DESC, t2.file_month DESC, t2.id DESC LIMIT 1),
            COUNT(*),
            MAX(file_year * 100 + COALESCE(file_month, 0)),
            (SELECT id FROM tables t3 WHERE t3.family_id = t.family_id
             ORDER BY t3.file_year DESC, t3.file_month DESC, t3.id DESC LIMIT 1)
        FROM tables t WHERE family_id IS NOT NULL AND family_id != '' GROUP BY family_id
    """)

    conn.execute("""
        INSERT INTO table_family_columns(family_id, col_leaf_lower)
        SELECT DISTINCT t.family_id, LOWER(TRIM(tc.col_leaf))
        FROM table_columns tc JOIN tables t ON t.id = tc.table_id
        WHERE t.family_id IS NOT NULL AND t.family_id != ''
          AND tc.col_leaf IS NOT NULL AND TRIM(tc.col_leaf) != ''
    """)

    conn.execute("""
        INSERT INTO table_family_rows(family_id, metric_slug)
        SELECT DISTINCT t.family_id, tr.metric_slug
        FROM table_rows tr JOIN tables t ON t.id = tr.table_id
        WHERE t.family_id IS NOT NULL AND t.family_id != ''
          AND tr.metric_slug IS NOT NULL AND TRIM(tr.metric_slug) != ''
    """)

    conn.execute("""
        INSERT INTO table_family_years(family_id, year, best_table_id)
        SELECT tf.family_id, f.data_year,
            (SELECT t2.id FROM tables t2 JOIN facts f2 ON f2.table_id = t2.id
             WHERE t2.family_id = tf.family_id AND f2.data_year = f.data_year
             ORDER BY t2.file_year DESC, t2.file_month DESC, t2.id DESC LIMIT 1)
        FROM facts f JOIN tables tf ON tf.id = f.table_id
        WHERE tf.family_id IS NOT NULL AND tf.family_id != '' AND f.data_year IS NOT NULL
        GROUP BY tf.family_id, f.data_year
    """)

    conn.executescript("""
        CREATE VIRTUAL TABLE families_fts USING fts5(
            family_id UNINDEXED, primary_title, tokenize = 'unicode61'
        );
        INSERT INTO families_fts(family_id, primary_title)
        SELECT family_id, COALESCE(primary_title, '') FROM table_families;
    """)

    fam = conn.execute("SELECT COUNT(*) FROM table_families").fetchone()[0]
    yr = conn.execute("SELECT COUNT(*) FROM table_family_years").fetchone()[0]
    print(f"  {fam:,} families, {yr:,} family-year pairs in {_time.time() - t0:.1f}s", flush=True)


def run_sanity_checks(conn: sqlite3.Connection) -> None:
    print("\n─── Sanity checks ─────────────────────────────────────────")
    queries = [
        ("total tables", "SELECT COUNT(*) FROM tables"),
        ("tables parsed ok", "SELECT COUNT(*) FROM tables WHERE parse_ok = 1"),
        ("tables parse fail", "SELECT COUNT(*) FROM tables WHERE parse_ok = 0"),
        ("tables kind=data", "SELECT COUNT(*) FROM tables WHERE table_kind = 'data'"),
        ("tables kind=toc", "SELECT COUNT(*) FROM tables WHERE table_kind = 'toc'"),
        ("distinct signatures", "SELECT COUNT(DISTINCT signature) FROM tables"),
        ("dual_agree tables", "SELECT COUNT(*) FROM tables WHERE parse_method = 'dual_agree'"),
        (
            "dual_disagree tables",
            "SELECT COUNT(*) FROM tables WHERE parse_method = 'dual_disagree'",
        ),
        ("lxml_only tables", "SELECT COUNT(*) FROM tables WHERE parse_method = 'lxml_only'"),
        ("pandas_only tables", "SELECT COUNT(*) FROM tables WHERE parse_method = 'pandas_only'"),
        ("total table_columns", "SELECT COUNT(*) FROM table_columns"),
        ("total table_rows", "SELECT COUNT(*) FROM table_rows"),
        ("total cells", "SELECT COUNT(*) FROM cells"),
        ("cells ok", "SELECT COUNT(*) FROM cells WHERE parse_status = 'ok'"),
        ("cells missing", "SELECT COUNT(*) FROM cells WHERE parse_status = 'missing'"),
        ("cells text (descriptive)", "SELECT COUNT(*) FROM cells WHERE parse_status = 'text'"),
        ("cells unparseable", "SELECT COUNT(*) FROM cells WHERE parse_status = 'unparseable'"),
        ("cells zero (real zeros)", "SELECT COUNT(*) FROM cells WHERE is_zero = 1"),
        ("cells percent", "SELECT COUNT(*) FROM cells WHERE is_percent = 1"),
        ("cells revised", "SELECT COUNT(*) FROM cells WHERE is_revised = 1"),
        ("cells with footnote", "SELECT COUNT(*) FROM cells WHERE has_footnote = 1"),
        ("cells estimated", "SELECT COUNT(*) FROM cells WHERE is_estimated = 1"),
        ("cells preliminary", "SELECT COUNT(*) FROM cells WHERE is_preliminary = 1"),
        ("prose elements", "SELECT COUNT(*) FROM prose"),
        ("prose attached to table", "SELECT COUNT(*) FROM prose WHERE near_table_id IS NOT NULL"),
        ("footnotes", "SELECT COUNT(*) FROM footnotes"),
        (
            "footnotes attached to table",
            "SELECT COUNT(*) FROM footnotes WHERE attached_to_table_id IS NOT NULL",
        ),
        ("page_metadata rows", "SELECT COUNT(*) FROM page_metadata"),
        (
            "metrics VIEW base rows (approx)",
            "SELECT COUNT(*) FROM cells c JOIN tables t ON c.table_id=t.id JOIN table_rows r ON c.row_id=r.id WHERE t.parse_ok=1 AND t.table_kind='data' AND r.is_section_header=0 AND r.row_path IS NOT NULL AND LENGTH(TRIM(r.row_path))>0 AND c.parse_status IN ('ok','missing')",
        ),
        ("metrics_synthesized rows", "SELECT COUNT(*) FROM metrics_synthesized"),
        (
            "distinct metric_slug (row_label_lookup)",
            "SELECT COUNT(DISTINCT metric_slug) FROM row_label_lookup",
        ),
        (
            "tables with multi-page continuation",
            "SELECT COUNT(*) FROM tables WHERE continues_from_table_id IS NOT NULL",
        ),
        (
            "rows with hierarchical path (a > b)",
            "SELECT COUNT(*) FROM table_rows WHERE row_path LIKE '% > %'",
        ),
        (
            "cols with year extracted",
            "SELECT COUNT(*) FROM table_columns WHERE year_extracted IS NOT NULL",
        ),
        (
            "rows with year extracted",
            "SELECT COUNT(*) FROM table_rows   WHERE year_extracted IS NOT NULL",
        ),
        ("tables w/ unit detected", "SELECT COUNT(*) FROM tables WHERE unit IS NOT NULL"),
        ("tables w/ period detected", "SELECT COUNT(*) FROM tables WHERE period != 'unknown'"),
        (
            "exclusivity (zero AND missing)",
            "SELECT COUNT(*) FROM cells WHERE is_zero = 1 AND is_missing = 1",
        ),
        ("invariant: tables ok (delta<=2)", "SELECT COUNT(*) FROM table_invariants WHERE ok = 1"),
        (
            "invariant: tables with loss/excess",
            "SELECT COUNT(*) FROM table_invariants WHERE ok = 0",
        ),
        ("invariant: sum abs(delta)", "SELECT COALESCE(SUM(ABS(delta)), 0) FROM table_invariants"),
        (
            "invariant: total raw html cells",
            "SELECT COALESCE(SUM(raw_html_cells), 0) FROM table_invariants",
        ),
        (
            "invariant: total accounted cells",
            "SELECT COALESCE(SUM(n_column_cells + n_row_labels + n_section_hdr + n_data_cells), 0) FROM table_invariants",
        ),
    ]
    for label, sql in queries:
        (n,) = conn.execute(sql).fetchone()
        print(f"  {label:48s} {n:>12,}")
    print()

    print("─── View counts ────────────────────────────────────────────")
    for label, sql in [
        ("facts rows", "SELECT COUNT(*) FROM facts"),
        ("canonical_facts", "SELECT COUNT(*) FROM canonical_facts"),
        ("supersessions", "SELECT COUNT(*) FROM supersessions"),
    ]:
        (n,) = conn.execute(sql).fetchone()
        print(f"  {label:48s} {n:>12,}")
    print()


# ── Build driver ─────────────────────────────────────────────────────────────


def build(limit: int = 0, rebuild: bool = False, ledger_path: Path | None = None) -> None:
    files = sorted(CORPUS_JSON.glob("treasury_bulletin_*.json"))
    if not files:
        print(f"No files in {CORPUS_JSON}", file=sys.stderr)
        sys.exit(1)
    if limit:
        files = files[:limit]

    out = ledger_path if ledger_path is not None else LEDGER_PATH

    print(f"Building ledger from {len(files)} files...")
    print(f"  schema: {out}")
    print(f"  rebuild: {rebuild}\n")

    conn = init_db(out, rebuild=rebuild)

    totals: dict = {}
    t0 = time.time()

    for i, f in enumerate(files, 1):
        t_start = time.time()
        stats = process_file(conn, f)
        dt = time.time() - t_start
        for k, v in stats.items():
            if isinstance(v, (int, float)):
                totals[k] = totals.get(k, 0) + v

        unp_pct = (
            stats["cells_unparseable"] / stats["cells_total"] * 100 if stats["cells_total"] else 0.0
        )
        flag = " ⚠" if unp_pct > 5.0 else ""
        print(
            f"  [{i:3d}/{len(files)}] {f.name:45s}  "
            f"tables={stats['tables_seen']:3d} ok={stats['tables_parsed_ok']:3d}  "
            f"cells={stats['cells_total']:5,d} unp={stats['cells_unparseable']:4,d}({unp_pct:4.1f}%)"
            f"{flag}  {dt * 1000:5.0f}ms",
            flush=True,
        )

    elapsed = time.time() - t0
    print(f"\n─── Totals ({elapsed:.1f}s) ───────────────────────────────")
    for k in sorted(totals):
        print(f"  {k:40s} {totals[k]:>12,}")

    # Create views now that all cells are in place
    conn.executescript(VIEWS_DDL)
    conn.commit()

    build_metric_layer(conn)
    build_fts_index(conn)
    build_family_tables(conn)
    run_sanity_checks(conn)
    conn.close()

    size_mb = out.stat().st_size / (1024 * 1024)
    print(f"Wrote {out} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="process only the first N files (0 = all)")
    ap.add_argument(
        "--rebuild", action="store_true", help="drop and recreate the ledger before ingesting"
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=LEDGER_PATH,
        metavar="PATH",
        help="SQLite output path (default: ledger.sqlite in cwd)",
    )
    args = ap.parse_args()
    out = args.output
    if not out.is_absolute():
        out = (Path(__file__).resolve().parent / out).resolve()
    build(limit=args.limit, rebuild=args.rebuild, ledger_path=out)
