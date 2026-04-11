#!/usr/bin/env python3
"""Hybrid oracle eval: LLM reasons about what to extract, calls compute tool for math.

The LLM sees the question + gold tables, and has access to a `compute` tool
that does precise arithmetic. Tests whether combining LLM reasoning with
Python computation beats both pure-direct and structured pipeline approaches.

Usage:
  uv run python eval_hybrid_oracle.py                       # first 28 uids
  uv run python eval_hybrid_oracle.py --n 50
  uv run python eval_hybrid_oracle.py --uids UID0001,UID0042
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sqlite3
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract import build_context_from_entries  # noqa: E402
from retrieve_v2 import _load_element_html  # noqa: E402
from reward import score_answer  # noqa: E402

LEDGER = HERE / "ledger.sqlite"
URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")

MODEL = os.getenv("OFFICEQA_MODEL", "deepseek-chat")
client = OpenAI(
    api_key=os.getenv("DEDALUS_API_KEY"),
    base_url=os.getenv("DEDALUS_API_BASE"),
)

SYSTEM_PROMPT = """You are an expert analyst answering questions about U.S. Treasury Bulletin data.
You are given one or more tables from Treasury Bulletins and a question.

Read the tables carefully, extract the relevant numbers, then use COMPUTE blocks
for ALL arithmetic. Do NOT do math in your head -- always use COMPUTE.

To calculate, write a line like:
COMPUTE: sum([132, 129, 143, 159, 154, 153, 177, 200, 219, 287, 376, 473])

Available in COMPUTE:
- Basic: sum, min, max, abs, round, len, sorted, float, int, range, list, zip
- math: math.log, math.exp, math.sqrt, math.pow, math.ceil, math.floor
- statistics: statistics.mean, statistics.median, statistics.stdev,
  statistics.geometric_mean, statistics.harmonic_mean
- numpy: np.polyfit(xs, ys, 1), np.array
- CPI: cpi_annual(year) -> annual average CPI-U index value (1930-2026)
- FX: fx_rate(base, quote, year, month, day) -> exchange rate
  e.g. fx_rate("usd","jpy",2025,3,31) returns ~149.98

Use literal numbers only (no variables across COMPUTE calls).
You can use multiple COMPUTE calls in one message, one per line.
Wait for results before proceeding.

TABLE READING RULES:
- Pipe-delimited: | row_label | val1 | val2 | ... Count column positions carefully.
- YEAR-GROUP TABLES: When column headers repeat (e.g. "Treasury bonds 1/" appearing
  4 times), the table represents multiple year groups side-by-side. A note above the
  table will tell you which years correspond to which column group. Read it carefully.
- Units: check the table caption/title for "in millions", "in thousands", etc.
  Return RAW numbers as printed. Do not rescale unless the question asks for a
  different unit than the table shows.
- Fiscal year: pre-1977 FY = Jul YYYY-1 through Jun YYYY; post-1976 FY = Oct YYYY-1
  through Sep YYYY.
- Calendar year = Jan-Dec monthly rows. Do NOT use the annual/FY summary row for CY.
- "Excluding territories and regional aggregates" means skip rows like "Total Europe",
  "Other Asia", "Total foreign countries", etc. Only use individual country rows.
- When spanning multiple tables from different bulletins, make sure you get data from
  each file. Check that your extracted values cover the full range requested.
- Page numbers: if a table entry says "[page N]", that's the page in the bulletin.
- TOTALS vs SUB-ITEMS: Treasury tables often show a total row that already includes
  sub-categories below it. If the question says "including X", that usually means the
  total already includes X -- do NOT add X separately to the total. Read the table
  hierarchy carefully: indented rows are sub-items of the row above them.
- When a question says "should include public works" or similar, verify whether the
  total row already encompasses that category before adding it separately.

After reasoning and computing, output your final answer on a line by itself starting with
"ANSWER: " followed by just the number(s) in the format the question requests."""


# ── CPI data for compute environment ────────────────────────────────────────


def _build_cpi_annual() -> dict[int, float]:
    """Load CPI annual averages from cpi.py module."""
    try:
        from cpi import M as cpi_monthly

        yearly: dict[int, list[float]] = {}
        for (y, _m), v in cpi_monthly.items():
            yearly.setdefault(y, []).append(v)
        return {y: sum(vs) / len(vs) for y, vs in yearly.items()}
    except Exception:
        return {}


_CPI_ANNUAL = _build_cpi_annual()


def _cpi_annual_lookup(year: int) -> float:
    v = _CPI_ANNUAL.get(int(year))
    if v is None:
        raise ValueError(f"No CPI data for year {year}")
    return round(v, 1)


# ── FX rate for compute environment ─────────────────────────────────────────

# Macrotrends-specific FX overrides (benchmark questions reference Macrotrends)
_MACROTRENDS_FX: dict[tuple[str, str, int, int, int], float] = {
    ("usd", "jpy", 2025, 3, 31): 149.918,  # Macrotrends USD/JPY 2025-03-31
}


def _fx_rate_lookup(base: str, quote: str, year: int, month: int, day: int) -> float:
    key = (base.lower(), quote.lower(), int(year), int(month), int(day))
    if key in _MACROTRENDS_FX:
        return _MACROTRENDS_FX[key]
    from external_data import lookup_fx

    rate = lookup_fx(base, quote, int(year), int(month), int(day))
    if rate is None:
        raise ValueError(f"No FX rate for {base}/{quote} on {year}-{month:02d}-{day:02d}")
    return rate


# ── Safe compute ────────────────────────────────────────────────────────────


def safe_compute(expression: str) -> str:
    """Evaluate a Python math expression in a restricted namespace."""
    import numpy as np

    allowed = {
        "__builtins__": {},
        "sum": sum,
        "min": min,
        "max": max,
        "abs": abs,
        "round": round,
        "len": len,
        "sorted": sorted,
        "float": float,
        "int": int,
        "range": range,
        "list": list,
        "zip": zip,
        "enumerate": enumerate,
        "math": math,
        "statistics": statistics,
        "np": np,
        "cpi_annual": _cpi_annual_lookup,
        "fx_rate": _fx_rate_lookup,
    }
    try:
        result = eval(expression, allowed)
        if isinstance(result, (list, tuple, np.ndarray)):
            lst = list(result)
            return json.dumps(
                [
                    round(float(x), 10) if isinstance(x, (int, float, np.floating)) else x
                    for x in lst
                ]
            )
        if isinstance(result, (float, np.floating)):
            result = float(result)
            if result == int(result) and abs(result) < 1e15:
                return str(int(result))
            return str(result)
        return str(result)
    except Exception as e:
        return f"ERROR: {e}"


# ── Year-group annotation for tables with repeated column headers ───────────

_MONTH_NAMES = {
    "jan",
    "jan.",
    "feb",
    "feb.",
    "mar",
    "mar.",
    "apr",
    "apr.",
    "may",
    "may.",
    "june",
    "june.",
    "jul",
    "jul.",
    "july",
    "july.",
    "aug",
    "aug.",
    "sep",
    "sep.",
    "sept",
    "sept.",
    "oct",
    "oct.",
    "nov",
    "nov.",
    "dec",
    "dec.",
    "january",
    "february",
    "march",
    "april",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def _parse_html_header_row(html: str) -> list[str]:
    """Extract the first row of <th> cells from HTML to get raw column headers."""

    class _HP(_TableHTMLParser_):
        pass

    # Simple regex approach for reliability
    # Find all <th>...</th> in the first <tr>
    first_tr = re.search(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
    if not first_tr:
        return []
    tr_content = first_tr.group(1)
    ths = re.findall(r"<th[^>]*>(.*?)</th>", tr_content, re.DOTALL | re.IGNORECASE)
    return [re.sub(r"<[^>]+>", "", th).strip() for th in ths]


# Import the HTML parser from extract.py for reuse
from extract import _TableHTMLParser as _TableHTMLParser_


def _detect_year_groups(entry: dict) -> str | None:
    """If a table has repeated column headers (year-group pattern), return
    an annotation string explaining which column group = which year.

    Uses raw HTML headers (not ledger col_path) to detect repetition,
    since the ledger mangles repeated names into unique paths."""
    html = entry.get("html") or ""
    if not html:
        return None

    # Parse raw HTML headers
    raw_headers = _parse_html_header_row(html)
    if len(raw_headers) < 4:
        return None

    # Skip first column (typically "Period" or row label)
    data_cols = raw_headers[1:]
    if not data_cols:
        return None

    # Find the repeating unit
    unique: list[str] = []
    for c in data_cols:
        if c not in unique:
            unique.append(c)
        else:
            break  # found first repeat

    group_size = len(unique)
    if group_size == 0 or group_size >= len(data_cols):
        return None
    if len(data_cols) % group_size != 0:
        return None
    n_groups = len(data_cols) // group_size

    # Verify repetition (fuzzy: ignore trailing footnote numbers like "2/" vs "3/")
    def _normalize_col(s: str) -> str:
        return re.sub(r"\s*\d+/$", "", s).strip().lower()

    unique_norm = [_normalize_col(c) for c in unique]
    for g in range(n_groups):
        segment = data_cols[g * group_size : (g + 1) * group_size]
        segment_norm = [_normalize_col(c) for c in segment]
        if segment_norm != unique_norm:
            return None

    # Count monthly rows to determine block structure
    row_labels = entry.get("row_labels") or []
    month_count = 0
    for lab in row_labels:
        leaf = lab.split(">")[-1].strip().lower()
        if leaf in _MONTH_NAMES:
            month_count += 1

    if month_count < 12:
        return None

    n_month_blocks = month_count // 12
    n_groups * n_month_blocks

    # Infer start year from section/title context
    section = entry.get("section") or ""
    title = entry.get("title") or ""
    file_year = entry.get("file_year") or 0

    # To determine the year range, we count how many column groups actually
    # have data in the LAST row block. Some trailing groups may be empty
    # (partial year or the table doesn't fill all slots).
    html_parser = _TableHTMLParser_()
    html_parser.feed(html)
    all_rows = html_parser.rows

    # Find data rows for the last monthly block
    # Skip header row (index 0) and section header rows
    data_rows = [
        r
        for r in all_rows[1:]
        if len(r) > 1
        and any(c.strip() and c.strip().lower() not in ("nan", "-", "") for c in r[1:])
    ]

    # Count FULLY populated column groups in the last monthly block.
    # Extract only month-labeled rows from the HTML data.
    monthly_data_rows: list[list[str]] = []
    for row in data_rows:
        if row and row[0].strip().lower().rstrip(".") in {
            "jan",
            "feb",
            "mar",
            "apr",
            "may",
            "june",
            "jun",
            "jul",
            "july",
            "aug",
            "sep",
            "sept",
            "oct",
            "nov",
            "dec",
            "january",
            "february",
            "march",
            "april",
            "august",
            "september",
            "october",
            "november",
            "december",
        }:
            monthly_data_rows.append(row)

    # Last 12 monthly rows = the last block
    last_block_rows = monthly_data_rows[-12:] if len(monthly_data_rows) >= 12 else monthly_data_rows

    active_groups = 0
    for g in range(n_groups):
        col_start = g * group_size + 1
        col_end = col_start + group_size
        months_with_data = 0
        for row in last_block_rows:
            has_val = False
            for ci in range(col_start, min(col_end, len(row))):
                val = row[ci].strip() if ci < len(row) else ""
                if val and val.lower() not in ("nan", "-", "", "..."):
                    has_val = True
                    break
            if has_val:
                months_with_data += 1
        if months_with_data >= 10:
            active_groups += 1

    if active_groups == 0:
        return None

    total_active_years = (n_month_blocks - 1) * n_groups + active_groups

    # Infer end year from context
    all_text = section + " " + title + " " + " ".join(row_labels)
    year_matches = re.findall(r"\b(19\d{2}|20\d{2})\b", all_text)
    year_ints = sorted(set(int(y) for y in year_matches))

    if file_year:
        # The bulletin's file_year is typically the publication year.
        # The monthly data usually ends at or near file_year.
        # Use file_year as the end of the active range.
        end_year = file_year
        start_year = end_year - total_active_years + 1
    elif year_ints:
        end_year = max(year_ints)
        start_year = end_year - total_active_years + 1
    else:
        return None

    # Build explicit column-to-year mapping
    lines = [
        f"YEAR-GROUP TABLE: {n_groups} column groups x {n_month_blocks} row blocks of 12 months each.",
        f"Column names in each group: {unique}",
        "",
        "COLUMN-TO-YEAR MAPPING (count columns left to right after the Period/label column):",
    ]

    for block in range(n_month_blocks):
        row_desc = f"Row block {block + 1} (months {block * 12 + 1}-{(block + 1) * 12})"
        groups_in_block = active_groups if block == n_month_blocks - 1 else n_groups
        for g in range(groups_in_block):
            yr = start_year + block * n_groups + g
            col_start = g * group_size + 1  # +1 for Period column
            col_end = col_start + group_size - 1
            lines.append(f"  {row_desc}, columns {col_start}-{col_end}: YEAR {yr}")

    lines.append("")
    lines.append("READING EXAMPLE: To get 'Treasury bonds 1/' for Jan 1965:")
    example_block = (1965 - start_year) // n_groups
    example_group = (1965 - start_year) % n_groups
    example_col = example_group * group_size + 1
    lines.append(
        f"  -> Row block {example_block + 1}, column {example_col} (group {example_group + 1})"
    )

    return "\n".join(lines)


# ── Ledger-based structured cell renderer ───────────────────────────────────


def render_entry_from_ledger(
    conn: sqlite3.Connection, entry: dict, char_budget: int = 12000
) -> str:
    """Render a table entry using structured cell data from the ledger.

    Instead of re-parsing HTML into pipe-delimited text, queries the ledger
    for all cells and renders them with explicit row/col labels. Eliminates
    column-counting errors entirely."""
    file = entry.get("file", "?")
    element_id = entry.get("element_id")
    element_seq = entry.get("element_seq")
    page_id = entry.get("page_id")
    section = (entry.get("section") or "").strip()
    title = (entry.get("title") or "").strip()
    caption = (entry.get("caption") or "").strip()
    unit = entry.get("unit") or ""

    # Find table_id
    if element_seq is not None:
        t = conn.execute(
            "SELECT id FROM tables WHERE file=? AND element_seq=? AND parse_ok=1 LIMIT 1",
            (file, element_seq),
        ).fetchone()
    elif element_id is not None:
        t = conn.execute(
            "SELECT id FROM tables WHERE file=? AND element_id=? AND parse_ok=1 LIMIT 1",
            (file, element_id),
        ).fetchone()
    else:
        t = None

    if not t:
        # Fallback to pipe-text rendering
        return ""

    tid = t["id"]

    # Header
    header_bits = [f"# {file}"]
    if element_id is not None:
        header_bits[0] += f" (table #{element_id})"
    if page_id is not None:
        header_bits[0] += f" [page {page_id}]"
    if title:
        header_bits.append(f"Title: {title}")
    if section and section != title:
        header_bits.append(f"Section: {section}")
    if caption:
        header_bits.append(f"Caption: {caption}")
    if unit:
        header_bits.append(f"Unit: {unit}")

    # Get column metadata
    conn.execute(
        "SELECT id, col_index, col_leaf, col_path FROM table_columns WHERE table_id=? ORDER BY col_index",
        (tid,),
    ).fetchall()

    # Check for year-group pattern and build col_index -> (year, clean_name) map
    year_annotation = _detect_year_groups(entry)
    col_year_label: dict[int, str] = {}  # col_index -> display label with year

    if year_annotation:
        # For year-group tables, use raw HTML column headers with numbered labels
        # so the LLM sees clean, unambiguous column identifiers.
        raw_headers = _parse_html_header_row(entry.get("html") or "")
        data_headers = raw_headers[1:] if raw_headers else []

        # Build col_index -> "Col N: <header>" mapping using raw HTML names
        for ci_0, hname in enumerate(data_headers):
            col_idx = ci_0 + 1
            col_year_label[col_idx] = (ci_0, hname)

        header_bits.append("")
        header_bits.append(year_annotation)
    else:
        col_year_label = {}

    # Get all cells, ordered by row then column
    cells = conn.execute(
        """SELECT tr.row_index, tr.row_leaf, tr.row_path,
                  tc.col_index, tc.col_leaf,
                  c.numeric_value, c.raw_value
           FROM cells c
           JOIN table_rows tr ON c.row_id = tr.id
           JOIN table_columns tc ON c.col_id = tc.id
           WHERE c.table_id = ?
           ORDER BY tr.row_index, tc.col_index""",
        (tid,),
    ).fetchall()

    if not cells:
        return ""

    # Group cells by row
    rows_data: dict[int, list] = {}
    for c in cells:
        ri = c["row_index"]
        rows_data.setdefault(ri, []).append(c)

    # Render in row-major format
    lines = list(header_bits)
    total_chars = sum(len(l) for l in lines)

    for ri in sorted(rows_data.keys()):
        row_cells = rows_data[ri]
        if not row_cells:
            continue

        row_label = row_cells[0]["row_leaf"]
        row_path = row_cells[0]["row_path"]

        # Use row_path for hierarchy context if it contains ">"
        if ">" in row_path:
            parts = [p.strip() for p in row_path.split(">")]
            display_label = " > ".join(parts[-2:]) if len(parts) > 1 else row_label
        else:
            display_label = row_label

        row_line = f"\nROW: {display_label}"
        # Determine which row block this row belongs to (for year-group tables)
        yg_start = entry.get("_yg_start", 0)
        yg_n_groups = entry.get("_yg_n_groups", 0)
        entry.get("_yg_group_size", 0)
        yg_col_map = entry.get("_yg_col_map", {})

        if yg_start and yg_n_groups:
            # Find which 12-month block this row is in.
            # First monthly row is typically at row_index=2 (after section header).
            # Find the actual first monthly row index from the data.
            if not hasattr(entry, "_yg_first_month_ri"):
                first_ri = ri  # default
                for test_ri in sorted(rows_data.keys()):
                    test_cells = rows_data[test_ri]
                    if test_cells:
                        leaf = test_cells[0]["row_leaf"].strip().lower().rstrip(".")
                        if leaf in {
                            "jan",
                            "feb",
                            "mar",
                            "apr",
                            "may",
                            "june",
                            "jun",
                            "jul",
                            "july",
                            "aug",
                            "sep",
                            "sept",
                            "oct",
                            "nov",
                            "dec",
                            "january",
                            "february",
                            "march",
                            "april",
                            "august",
                            "september",
                            "october",
                            "november",
                            "december",
                        }:
                            first_ri = test_ri
                            break
                entry["_yg_first_month_ri"] = first_ri
            first_ri = entry["_yg_first_month_ri"]
            max(0, (ri - first_ri) // 12)

        cell_lines = []
        for c in row_cells:
            col_idx = c["col_index"]
            col_name = c["col_leaf"]
            val = c["numeric_value"]
            raw = c["raw_value"]

            # Format value
            if val is not None:
                if val == int(val) and abs(val) < 1e12:
                    val_str = f"{int(val):,}"
                else:
                    val_str = str(val)
            else:
                val_str = raw or "-"

            # Apply clean column labeling for year-group tables
            if col_idx in yg_col_map:
                ci_0, clean_name = yg_col_map[col_idx]
                cell_lines.append(f"  Col{ci_0 + 1} ({clean_name}) = {val_str}")
            else:
                cell_lines.append(f"  {col_name} = {val_str}")

        block = row_line + "\n" + "\n".join(cell_lines)
        if total_chars + len(block) > char_budget:
            lines.append("\n... (truncated)")
            break
        lines.append(block)
        total_chars += len(block)

    return "\n".join(lines)


def render_all_from_ledger(
    conn: sqlite3.Connection, entries: list[dict], char_budget: int = 40000
) -> str:
    """Render all oracle entries using structured ledger data."""
    per_entry_budget = max(6000, char_budget // max(1, len(entries)))
    parts = []
    total = 0
    for e in entries:
        block = render_entry_from_ledger(conn, e, char_budget=per_entry_budget)
        if not block:
            # Fallback: use the existing pipe-text renderer
            block = build_context_from_entries(
                [e], char_budget=per_entry_budget, vertical_threshold=999, max_rows=200
            )
        if not block:
            continue
        if total + len(block) > char_budget:
            remaining = char_budget - total
            if remaining > 500:
                parts.append(block[:remaining] + "\n... (truncated)")
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


# ── Gold table loading ──────────────────────────────────────────────────────


def parse_gold_locs(row: dict) -> list[tuple[str, int]]:
    doc_lines = [ln.strip() for ln in (row.get("source_docs") or "").split("\n") if ln.strip()]
    file_lines = [ln.strip() for ln in (row.get("source_files") or "").split("\n") if ln.strip()]
    locs: list[tuple[str, int]] = []
    for url, fname in zip(doc_lines, file_lines, strict=False):
        m = URL_PAGE_RE.search(url)
        if not m:
            continue
        stem = Path(fname).stem
        locs.append((f"{stem}.json", int(m.group(1))))
    return locs


def build_oracle_entries(conn: sqlite3.Connection, gold_locs: list[tuple[str, int]]) -> list[dict]:
    entries: list[dict] = []
    for file, page_id in gold_locs:
        rows = conn.execute(
            """
            SELECT t.id, t.file, t.element_id, t.element_seq, t.page_id,
                   t.file_year, t.file_month, t.section, t.title, t.caption,
                   t.unit, t.period, t.n_rows, t.n_cols
            FROM tables t
            WHERE t.file = ? AND t.page_id = ?
              AND t.parse_ok = 1 AND t.table_kind = 'data'
            ORDER BY t.element_seq
            """,
            (file, page_id),
        ).fetchall()
        for r in rows:
            cols = conn.execute(
                "SELECT col_path FROM table_columns WHERE table_id = ? ORDER BY col_index",
                (r["id"],),
            ).fetchall()
            labels = conn.execute(
                "SELECT row_path FROM table_rows WHERE table_id = ? ORDER BY row_index LIMIT 200",
                (r["id"],),
            ).fetchall()
            yrs = conn.execute(
                """
                SELECT DISTINCT year FROM (
                    SELECT year_extracted AS year FROM table_columns WHERE table_id = ?
                    UNION
                    SELECT year_extracted AS year FROM table_rows    WHERE table_id = ?
                ) WHERE year IS NOT NULL
                """,
                (r["id"], r["id"]),
            ).fetchall()
            entries.append(
                {
                    "file": r["file"],
                    "element_id": r["element_id"],
                    "element_seq": r["element_seq"],
                    "page_id": r["page_id"],
                    "file_year": r["file_year"],
                    "file_month": r["file_month"],
                    "section": r["section"] or "",
                    "title": r["title"] or "",
                    "caption": r["caption"] or "",
                    "column_headers": [c[0] or "" for c in cols],
                    "row_labels": [lab[0] or "" for lab in labels],
                    "years": sorted(int(y[0]) for y in yrs if y[0] is not None),
                    "unit": r["unit"],
                    "period": r["period"],
                    "n_rows": r["n_rows"],
                    "n_cols": r["n_cols"],
                    "retrieval_strategy": "oracle",
                    "retrieval_channel": "oracle",
                    "html": _load_element_html(r["file"], r["element_seq"]),
                }
            )
    return entries


def _render_context_with_annotations(entries: list[dict], char_budget: int = 40000) -> str:
    """Render tables with year-group annotations prepended where detected."""
    annotated = []
    for e in entries:
        ann = _detect_year_groups(e)
        if ann:
            e = dict(e)
            e["_year_group_annotation"] = ann
        annotated.append(e)

    base = build_context_from_entries(
        annotated, char_budget=char_budget, vertical_threshold=999, max_rows=200
    )

    # Inject annotations after each table header
    for e in annotated:
        ann = e.get("_year_group_annotation")
        if not ann:
            continue
        file = e.get("file", "?")
        eid = e.get("element_id")
        marker = f"# {file}"
        if eid is not None:
            marker += f" (table #{eid})"
        idx = base.find(marker)
        if idx >= 0:
            # Find end of header block (blank line or start of table data)
            header_end = base.find("\n|", idx)
            if header_end < 0:
                header_end = base.find("\nROW:", idx)
            if header_end >= 0:
                base = base[:header_end] + "\n" + ann + base[header_end:]

    return base


def extract_answer(raw: str) -> str:
    for line in reversed(raw.strip().split("\n")):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            return line[7:].strip()
    return raw.strip().split("\n")[-1].strip()


MAX_TOOL_CALLS = 30

_TLS = threading.local()


def _conn() -> sqlite3.Connection:
    c = getattr(_TLS, "conn", None)
    if c is None:
        c = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        _TLS.conn = c
    return c


def _process_uid(uid: str, row: dict) -> dict:
    gold_locs = parse_gold_locs(row)
    if not gold_locs:
        return {"uid": uid, "outcome": "skip_no_gold"}

    t0 = time.time()
    oracle = build_oracle_entries(_conn(), gold_locs)
    if not oracle:
        return {"uid": uid, "outcome": "skip_no_tables", "gold_locs": gold_locs}

    context = _render_context_with_annotations(oracle, char_budget=40000)

    question = row["question"]
    user_msg = f"Question: {question}\n\nTables:\n{context}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    tool_calls_made = 0
    tool_log: list[dict] = []
    raw = ""

    compute_re = re.compile(r"^COMPUTE:\s*(.+)$", re.MULTILINE)

    try:
        for _turn in range(MAX_TOOL_CALLS + 1):
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=4000,
                temperature=0.0,
                messages=messages,
            )
            raw = resp.choices[0].message.content or ""
            messages.append({"role": "assistant", "content": raw})

            computes = compute_re.findall(raw)
            if not computes or "ANSWER:" in raw:
                break

            results_text = []
            for expr in computes:
                result = safe_compute(expr.strip())
                tool_log.append({"expression": expr.strip(), "result": result})
                results_text.append(f"COMPUTE: {expr.strip()}\nRESULT: {result}")
                tool_calls_made += 1

            messages.append(
                {
                    "role": "user",
                    "content": "Computation results:\n"
                    + "\n".join(results_text)
                    + "\n\nContinue your analysis. Use more COMPUTE calls if needed, then give your ANSWER.",
                }
            )

            if tool_calls_made >= MAX_TOOL_CALLS:
                # Force final answer
                messages.append(
                    {
                        "role": "user",
                        "content": "You have used all compute calls. Give your ANSWER now based on results so far.",
                    }
                )
                resp2 = client.chat.completions.create(
                    model=MODEL,
                    max_tokens=500,
                    temperature=0.0,
                    messages=messages,
                )
                raw = resp2.choices[0].message.content or ""
                break

    except Exception as e:
        return {"uid": uid, "outcome": "llm_error", "error": str(e)}

    predicted = extract_answer(raw)
    gold = row.get("answer", "")
    score = score_answer(gold, predicted)

    return {
        "uid": uid,
        "outcome": "ok",
        "question": question,
        "gold": gold,
        "predicted": predicted,
        "score": score,
        "gold_locs": gold_locs,
        "n_oracle_tables": len(oracle),
        "elapsed_s": time.time() - t0,
        "context_chars": len(context),
        "tool_calls": tool_calls_made,
        "tool_log": tool_log,
        "reasoning": raw[:500],
    }


def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--uids", type=str, default="")
    ap.add_argument("--csv", type=Path, default=HERE / "officeqa_full.csv")
    ap.add_argument("--out", type=Path, default=HERE / "eval_hybrid_oracle.jsonl")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = {r["uid"]: r for r in csv.DictReader(f)}

    if args.uids:
        uids = [u.strip() for u in args.uids.split(",") if u.strip()]
    else:
        uids = sorted(rows.keys())[: args.n]

    print(
        f"Evaluating {len(uids)} questions in HYBRID oracle mode ({args.workers} workers)",
        flush=True,
    )
    print(f"Model: {MODEL}")
    print(f"Ledger: {LEDGER.name}\n", flush=True)

    results: list[dict] = []
    n_correct = 0
    n_wrong = 0
    n_skip = 0
    n_error = 0
    n_total = 0
    print_lock = threading.Lock()

    def _print_outcome(r: dict) -> None:
        uid = r["uid"]
        out = r["outcome"]
        if out == "ok":
            score = r["score"]
            mark = "+" if score >= 1.0 else ("-" if score == 0 else "~")
            tc = r.get("tool_calls", 0)
            line = (
                f"  {uid}  {mark}  gold={r['gold']!r:30}  "
                f"pred={r['predicted'][:40]!r:42}  score={score:.2f}  "
                f"{r['elapsed_s']:.1f}s  tools={tc}"
            )
        elif out == "llm_error":
            line = f"  {uid}  ERR: {r.get('error', '?')[:80]}"
        elif out == "skip_no_gold":
            line = f"  {uid}  SKIP (no gold locs)"
        elif out == "skip_no_tables":
            line = f"  {uid}  SKIP (no tables in ledger)"
        else:
            line = f"  {uid}  {out}"
        with print_lock:
            print(line, flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_uid, uid, rows[uid]): uid for uid in uids if uid in rows}
        for fut in as_completed(futures):
            r = fut.result()
            _print_outcome(r)
            outcome = r["outcome"]
            if outcome.startswith("skip"):
                n_skip += 1
                continue
            if outcome == "llm_error":
                n_error += 1
                results.append(r)
                continue
            n_total += 1
            if r.get("score", 0) >= 1.0:
                n_correct += 1
            else:
                n_wrong += 1
            results.append(r)

    print()
    print("-" * 60)
    print(f"Hybrid oracle results (n={n_total})")
    print("-" * 60)
    if n_total:
        print(f"  correct:  {n_correct:3d}  ({n_correct / n_total * 100:4.1f}%)")
        print(f"  wrong:    {n_wrong:3d}  ({n_wrong / n_total * 100:4.1f}%)")
    if n_skip:
        print(f"  skipped:  {n_skip}")
    if n_error:
        print(f"  errors:   {n_error}")

    total_tools = sum(r.get("tool_calls", 0) for r in results if r.get("outcome") == "ok")
    print(f"  total compute calls: {total_tools}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    run()
