"""sql_solve.py — End-to-end SQL-agent solver for the officeqa benchmark.

A model with four tools (search_tables, describe_table, run_sql, submit_answer)
reasons freely over ledger.sqlite and returns a final answer. No decompose,
retrieve_v2, or extract stages — the model decides the strategy.

Usage:
    uv run python sql_solve.py "What were total expenditures for national defense in 1940?"
    uv run python sql_solve.py --variant think "<question>"
    uv run python sql_solve.py --eval --n 10 [--variant plain|think|both]
    uv run python sql_solve.py --eval --uids UID0001,UID0021,UID0111 [--variant plain|think]
    uv run python sql_solve.py --eval --n 10 --model gpt-4o  # stronger-model escape hatch
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from ledger_paths import get_ledger_sqlite_path
from reward import fuzzy_match_answer

load_dotenv()

LEDGER_PATH = get_ledger_sqlite_path()
BENCHMARK_CSV = Path(__file__).parent / "officeqa_full.csv"
RUNS_DIR = Path(__file__).parent / "runs"

# Prefer OFFICEQA_AGENT_MODEL; fall back to OFFICEQA_FIND_MODEL (the model
# find.py uses for tool calls, known to work); then OFFICEQA_MODEL as last resort.
DEFAULT_MODEL = os.getenv(
    "OFFICEQA_AGENT_MODEL",
    os.getenv("OFFICEQA_FIND_MODEL", os.getenv("OFFICEQA_MODEL", "deepseek/deepseek-chat")),
)
MAX_ITERS = 12

_client: OpenAI | None = None
_TLS = threading.local()


def _openai(model: str | None = None) -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEDALUS_API_KEY"),
            base_url=os.getenv("DEDALUS_API_BASE"),
        )
    return _client


def _conn() -> sqlite3.Connection:
    conn = getattr(_TLS, "sql_conn", None)
    if conn is None:
        if not LEDGER_PATH.exists():
            raise FileNotFoundError(f"{LEDGER_PATH} — run build_ledger.py first")
        # Read-only URI — physical write block
        conn = sqlite3.connect(f"file:{LEDGER_PATH}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        _TLS.sql_conn = conn
    return conn


# ── System prompts ──────────────────────────────────────────────────────────

_SELF_AUDIT = """You have full authority to explore. Use your tool budget freely.

Before calling submit_answer, ask yourself:
- Which file and table did this value come from?
- Is that the earliest bulletin that covers this data year? Later bulletins revise
  historical figures — prefer the edition published closest to (and after) the data year.
- Does the magnitude make sense for this era and question?
- Have I cross-checked against at least one other source or table member?

If any answer is "I'm not sure", use another tool call before submitting.

"""

_SCHEMA_SECTION = """
== SCHEMA ==

=== Table Families (primary navigation layer) ===

Treasury Bulletin publishes the same table every month with updated data.
A "family" groups all versions of the same recurring table across bulletins.
Use families to navigate to the RIGHT TABLE for a given data year.

  table_families(family_id TEXT, primary_title, member_count, latest_table_id)
    family_id = lowercase of primary_title (e.g. 'table 3.- expenditures for national defense...')

  table_family_years(family_id, year INT, best_table_id INT)
    best_table_id = the table_id of the version that best covers this data year.

  table_family_rows(family_id, metric_slug TEXT)
    canonical row slugs that appear in this family.

  families_fts — FTS5 index over family_id + primary_title. Query: families_fts MATCH 'keywords'

=== Data view ===

  metrics(table_id, metric_slug, row_path, col_path, time_key, year, month,
          period_basis, value, value_raw, unit, file, page_id,
          has_footnote, is_revised, is_preliminary, is_estimated, is_synthesized)

  - metric_slug: lowercase normalized row label. ALWAYS use LIKE '%keyword%' to match.
  - year / month: DATA year/month (from column or row headers), not bulletin publish date.
  - period_basis: 'fiscal', 'calendar', 'monthly', 'annual', or NULL.
  - value: float. NULL for missing cells.
  - unit: 'millions_usd', 'billions_usd', 'thousands_usd', 'percent', 'index', or NULL.
  - is_synthesized: 1 for precomputed CY/FY totals, 0 for raw cells.

=== Other tables ===

  tables(id, file, file_year, file_month, title, section, caption, unit, period, n_rows, n_cols)
  table_rows(id, table_id, row_path, row_leaf, metric_slug, year_extracted, month_extracted)
  table_columns(id, table_id, col_path, col_leaf, year_extracted, month_extracted)
  tables_fts — FTS5 over title/section/caption/columns/rows (fallback, use families_fts first)

== UNIT SCALING ==

The benchmark answer is in whatever unit the source table uses. Match it:
  millions_usd  → value is in millions. Return as-is for most questions.
  billions_usd  → value is in billions. Multiply × 1000 if benchmark expects millions.
  thousands_usd → value is in thousands. Divide by 1000 if benchmark expects millions.
  percent       → value is a percentage (3.5 = 3.5%).

== WORKFLOW ==

Step 1 — ALWAYS start with lookup_metric(keywords, data_year, file_year_min, file_year_max).
          Use the CORE METRIC TERM as keywords — short and specific, not the full question phrase.
          "national defense" not "total expenditures for national defense".
          "veterans" not "payments to veterans administration".
          Set file_year window to data_year through data_year+5 for original publications.

          If the question asks for a CALENDAR YEAR total: set monthly=true and SUM the returned values.
          Monthly rows have month != null. Annual rows have month = null.
          If monthly=false returns a suspiciously low value for a CY question, retry with monthly=true.

Step 2 — Inspect the results. The n_months and file_year fields together tell you what you have:
          - n_months=12, file_year=data_year:   fiscal year total (Oct thru Sep). NOT a CY total.
          - n_months=12, file_year=data_year+1: CALENDAR year total (Jan thru Dec). Use this for CY questions.
          - n_months < 12: partial year — skip.
          For CALENDAR year questions, pick the n_months=12 entry with the SMALLEST file_year > data_year.
          For revised/retrospective comparisons, prefer the earliest file_year overall.

Step 3 — The value field in lookup_metric results IS the answer. Submit it directly.
          Do NOT re-derive it with run_sql — you will get wrong results if you use metric_slug
          for a column-based table or vice versa. The result also includes metrics_filter
          which is the correct WHERE clause if you do need custom SQL.

If lookup_metric returns nothing, fall back to browse_rows, then retry lookup_metric.
Use run_sql only for custom aggregations (e.g. multi-year trends) not covered by lookup_metric.

== SQL PATTERNS ==

IMPORTANT: Always use table_id from describe_family. NEVER scan all of metrics without table_id.
A full metrics scan times out. Every run_sql MUST include: WHERE table_id = <id>

-- Step 1 — Resolve best_table_id (fast indexed lookup):
SELECT best_table_id FROM table_family_years
WHERE family_id = 'table 3.- expenditures for national defense and related activities'
  AND year = 1940
LIMIT 1

-- Step 2 — Query that table for the row you need:
SELECT value, value_raw, unit, metric_slug, year, month, file
FROM metrics
WHERE table_id = 16501   -- always bind to a specific table_id
  AND metric_slug LIKE '%national defense%'
  AND year = 1940
ORDER BY value DESC
LIMIT 20

-- See all rows in a table (use to explore structure):
SELECT DISTINCT metric_slug, year, month, value, unit
FROM metrics
WHERE table_id = 16501
ORDER BY year, month
LIMIT 50

-- Monthly sum for a specific year (e.g. all months of FY1940):
SELECT SUM(value) AS total, unit, COUNT(*) as n_months
FROM metrics
WHERE table_id = 16501
  AND metric_slug LIKE '%national defense%'
  AND year = 1940
  AND month IS NOT NULL
GROUP BY unit

-- Multi-year range across family versions (look up each year's best_table_id first):
SELECT tfy.year, m.value, m.unit, m.metric_slug
FROM metrics m
JOIN table_family_years tfy ON tfy.best_table_id = m.table_id
  AND tfy.family_id = '...'
WHERE m.metric_slug LIKE '%total%'
  AND tfy.year BETWEEN 1960 AND 1968
ORDER BY tfy.year
LIMIT 30

-- FTS fallback (when families miss):
SELECT id, file, title, file_year FROM tables
WHERE id IN (SELECT rowid FROM tables_fts WHERE tables_fts MATCH 'veterans expenditures 1960')
ORDER BY file_year DESC LIMIT 15

== RULES ==

- EVERY run_sql query MUST include WHERE table_id = <specific id>. No exceptions.
  A full metrics scan takes 30+ seconds and times out.
- Get table_id from describe_family or from a table_family_years lookup first.
- ALWAYS use metric_slug for row matching. NEVER use row_leaf or col_leaf text.
- ALWAYS add ORDER BY and LIMIT.
- PUBLICATION LAG: data year != bulletin publish year. Filter by year column in metrics, not file_year.
- Max 6 tool calls total, then submit_answer.
- If a table has years as rows (metric_slug = year string like '1940'), look for the answer in columns.
  Check all distinct metric_slugs in the table first with DISTINCT query.
"""

_THINK_PREFIX = _SELF_AUDIT

_BASE_SYSTEM = (
    "You are a research agent answering U.S. Treasury Bulletin questions "
    "from a structured SQLite ledger. Use the tools to find and verify the "
    "data before committing an answer. Cite the source table in your reasoning." + _SCHEMA_SECTION
)

SYSTEM_MINIMAL = (
    """You are answering U.S. Treasury Bulletin questions from a SQLite database.

"""
    + _SELF_AUDIT
    + """Schema:
- metrics view: table_id, metric_slug, year, month, value, unit, file (joins cells+rows+cols+tables)
- metric_slug: lowercase with spaces, e.g. 'national defense', 'veterans administration'
- unit values: 'millions_usd', 'billions_usd', 'thousands_usd', 'percent' (not English phrases)
- year/month: data year, NOT bulletin publish year. CY questions need month IS NOT NULL rows.
- table_families(family_id, primary_title, member_count), families_fts (FTS5 over titles)
- table_family_years(family_id, year, best_table_id) — maps data year to best table_id
- tables(id, file, file_year, title, section, unit, period)
- table_rows(id, table_id, metric_slug, year_extracted, month_extracted)
- table_columns(id, table_id, col_leaf, year_extracted, month_extracted)
- cells(table_id, row_id, col_id, raw_value, numeric_value, is_missing)
- tables_fts, families_fts: FTS5 indexes

IMPORTANT: Always filter by table_id. Full metrics scans time out. Get table_id from table_family_years or tables_fts first.

Use run_sql to find the answer. Call submit_answer when done."""
)

SYSTEM_PLAIN = _BASE_SYSTEM
SYSTEM_THINK = _THINK_PREFIX + _BASE_SYSTEM


# ── Tool schemas ────────────────────────────────────────────────────────────

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_rows",
            "description": (
                "Find table families that contain a specific row label (metric_slug). "
                "Use this when you need to find tables where your metric appears as a ROW — "
                "e.g. 'national defense' as a line item inside a broader budget classification table. "
                "Complements search_families (which searches by table title). "
                "Returns family_id, primary_title, earliest_file_year, and data year range."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Substring to match against row labels (metric_slug). E.g. 'national defense', 'veterans'.",
                    },
                    "data_year": {
                        "type": "integer",
                        "description": "Filter to families that have data for this year (optional).",
                    },
                    "max_file_year": {
                        "type": "integer",
                        "description": "Only return families with at least one member published on or before this year. Use to find original (non-revised) publications.",
                    },
                    "limit": {"type": "integer", "description": "Max results (default 10)."},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_metric",
            "description": (
                "Directly retrieve cell values for a metric in a specific data year from bulletins "
                "published in a given year window. Searches both row labels AND column labels — "
                "no need to know whether data is in rows or columns. "
                "Use this as the PRIMARY tool: give it the metric keywords, the data year you need, "
                "and the publication window. It returns actual values you can inspect and aggregate. "
                "Set monthly=true for calendar year totals (sum 12 monthly rows). "
                "Set monthly=false for annual rows."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "Keywords matching the metric label (row or column). E.g. 'national defense'.",
                    },
                    "data_year": {
                        "type": "integer",
                        "description": "The year the data refers to (not the publication year).",
                    },
                    "file_year_min": {
                        "type": "integer",
                        "description": "Earliest bulletin publication year to search.",
                    },
                    "file_year_max": {
                        "type": "integer",
                        "description": "Latest bulletin publication year to search.",
                    },
                    "monthly": {
                        "type": "boolean",
                        "description": "True to get monthly rows (for calendar year SUM). False for annual rows.",
                    },
                    "limit": {"type": "integer", "description": "Max rows (default 50)."},
                },
                "required": ["keywords", "data_year", "file_year_min", "file_year_max"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "browse_rows",
            "description": (
                "Discover what row labels (metric_slugs) actually exist in Treasury Bulletin tables "
                "published in a given era. The FIRST tool to call for any question. "
                "Because terminology changes across decades, pass multiple synonyms — "
                "e.g. for 'national defense' in 1940 also try 'army', 'navy', 'military', 'war'. "
                "Returns each matching label with: source (row/col), example_family_id to pass to describe_family, "
                "and metrics_filter — the exact WHERE clause fragment to use in run_sql against the metrics view. "
                "Always use metrics_filter from the result; do not guess metric_slug vs col_path yourself. "
                "You must provide data_year OR file_year_min/max (unfiltered scans time out)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Keyword variants to match against row labels (OR logic). Include both modern and era-appropriate terms.",
                    },
                    "data_year": {
                        "type": "integer",
                        "description": "The year the data refers to. Auto-sets file_year window to data_year through data_year+5.",
                    },
                    "file_year_min": {
                        "type": "integer",
                        "description": "Only look in bulletins published on or after this year (overrides data_year if set).",
                    },
                    "file_year_max": {
                        "type": "integer",
                        "description": "Only look in bulletins published on or before this year (overrides data_year if set).",
                    },
                    "limit": {"type": "integer", "description": "Max slugs returned (default 30)."},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_families",
            "description": (
                "Search for recurring table families by title keywords. "
                "A 'family' groups all bulletin editions of the same table. "
                "Returns family_id, primary_title, member_count. "
                "Use when the metric IS the table title. For metrics hidden as rows inside tables, use browse_rows instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "FTS5 keywords matching the table title (e.g. 'national defense expenditures').",
                    },
                    "limit": {"type": "integer", "description": "Max results (default 10)."},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_family",
            "description": (
                "Return details about a table family: "
                "canonical metric_slugs (row labels), "
                "and which data years are covered with the best table_id for each. "
                "Use the best_table_id to write targeted SQL against metrics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "family_id": {
                        "type": "string",
                        "description": "family_id from search_families result.",
                    },
                },
                "required": ["family_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": (
                "Return the structure of one specific table: metadata (title, unit, period), "
                "all distinct metric_slugs (row labels), and all data years/months present. "
                "Use after describe_family to confirm this member has the rows and years you need "
                "before writing a run_sql query."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {
                        "type": "integer",
                        "description": "table_id from describe_family members list.",
                    },
                },
                "required": ["table_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": (
                "Execute a read-only SELECT against ledger.sqlite. "
                "Query the `metrics` view for cell values. "
                "Always include WHERE table_id = <specific id>. Never scan all of metrics. "
                "Always include ORDER BY and LIMIT."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A single SELECT or WITH...SELECT statement. No semicolons.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max rows returned (default 100, max 200).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": (
                "Terminate the agent loop and record your final answer. "
                "'value' should be a string matching the benchmark format "
                "(e.g. '2602', '2602.3', '3.5%'). "
                "'reasoning' is one sentence citing the source family, row label, and year."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "Final answer as a string.",
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit from the table (e.g. 'millions_usd', 'percent').",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence: family name, row label, year, and source file.",
                    },
                },
                "required": ["value", "reasoning"],
            },
        },
    },
]


# ── Tool implementations ────────────────────────────────────────────────────

_SQL_BLACKLIST = re.compile(
    r"\b(insert|update|delete|drop|create|alter|attach|detach|pragma|vacuum|replace)\b",
    re.IGNORECASE,
)


def _tool_browse_rows(
    keywords: list[str],
    file_year_min: int | None,
    file_year_max: int | None,
    data_year: int | None,
    limit: int,
) -> dict:
    """Browse distinct row labels (metric_slugs) from tables published in a given bulletin year window.
    Searches for ANY of the provided keyword variants (OR logic). Returns matching slugs with example table context."""
    # Auto-derive file_year window from data_year if not provided explicitly
    if data_year is not None:
        if file_year_min is None:
            file_year_min = data_year
        if file_year_max is None:
            file_year_max = data_year + 5  # original publication usually within 5 years
    if file_year_min is None and file_year_max is None:
        return {
            "error": "Provide data_year, file_year_min, or file_year_max — unfiltered scan is too slow."
        }
    conn = _conn()
    try:
        # Word-boundary matching: pad slug with spaces so 'war' doesn't match 'delaware'
        kw_params = [f"% {kw.lower()} %" for kw in keywords]
        like_row = " OR ".join("(' ' || lower(tr.metric_slug) || ' ') LIKE ?" for _ in keywords)
        like_col = " OR ".join("(' ' || lower(tc.col_leaf) || ' ') LIKE ?" for _ in keywords)

        file_year_filter = ""
        fy_params: list = []
        if file_year_min is not None:
            file_year_filter += " AND t.file_year >= ?"
            fy_params.append(file_year_min)
        if file_year_max is not None:
            file_year_filter += " AND t.file_year <= ?"
            fy_params.append(file_year_max)

        # param order: [row_kw...] [fy...] [col_kw...] [fy...] [limit]
        params: list = kw_params + fy_params + kw_params + fy_params + [min(limit, 100)]

        sql = f"""
            SELECT label, source, n_tables, earliest_file_year, latest_file_year,
                   example_table_title, example_family_id
            FROM (
                SELECT tr.metric_slug AS label,
                       'row' AS source,
                       COUNT(DISTINCT t.id) AS n_tables,
                       MIN(t.file_year) AS earliest_file_year,
                       MAX(t.file_year) AS latest_file_year,
                       MIN(t.title) AS example_table_title,
                       MIN(tfy.family_id) AS example_family_id
                FROM table_rows tr
                JOIN tables t ON t.id = tr.table_id
                LEFT JOIN table_family_years tfy ON tfy.best_table_id = t.id
                WHERE ({like_row})
                  AND tr.metric_slug IS NOT NULL
                  {file_year_filter}
                GROUP BY tr.metric_slug
                UNION ALL
                SELECT tc.col_leaf AS label,
                       'col' AS source,
                       COUNT(DISTINCT t.id) AS n_tables,
                       MIN(t.file_year) AS earliest_file_year,
                       MAX(t.file_year) AS latest_file_year,
                       MIN(t.title) AS example_table_title,
                       MIN(tfy.family_id) AS example_family_id
                FROM table_columns tc
                JOIN tables t ON t.id = tc.table_id
                LEFT JOIN table_family_years tfy ON tfy.best_table_id = t.id
                WHERE ({like_col})
                  AND tc.col_leaf IS NOT NULL
                  {file_year_filter}
                GROUP BY tc.col_leaf
            )
            ORDER BY earliest_file_year, n_tables DESC
            LIMIT ?
        """
        raw = conn.execute(sql, params).fetchall()
        slugs = []
        for r in raw:
            d = dict(r)
            # Tell the model exactly how to filter metrics for this result.
            # Rows: filter by metric_slug. Columns: filter by col_path.
            # This abstracts away the row/col schema detail.
            if d["source"] == "row":
                d["metrics_filter"] = f"metric_slug = '{d['label']}'"
            else:
                d["metrics_filter"] = f"col_path LIKE '%{d['label']}%'"
            slugs.append(d)
        return {"slugs": slugs, "n": len(slugs)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tool_lookup_metric(
    keywords: str,
    data_year: int,
    file_year_min: int,
    file_year_max: int,
    monthly: bool,
    limit: int,
) -> dict:
    """Look up actual cell values for a metric in a given data year, from bulletins published in
    a given file_year window. Searches both row labels (metric_slug) and column labels (col_path).
    Uses FTS to find candidate tables quickly. Returns matching rows with values — ready to aggregate.
    Set monthly=true to get individual monthly rows (for CY sums). monthly=false returns annual rows."""
    conn = _conn()
    try:
        # Step 1: use FTS to find table_ids that mention these keywords anywhere
        # tables_fts covers title/section/caption/columns/rows — fast O(matches)
        fts_rows = conn.execute(
            """
            SELECT t.id FROM tables t
            WHERE t.id IN (SELECT rowid FROM tables_fts WHERE tables_fts MATCH ?)
              AND t.file_year BETWEEN ? AND ?
            LIMIT 200
            """,
            (keywords, file_year_min, file_year_max),
        ).fetchall()
        table_ids = [r[0] for r in fts_rows]
        if not table_ids:
            return {
                "rows": [],
                "n": 0,
                "note": "No tables matched in that publication window. Try different keywords or wider file_year range.",
            }

        placeholders = ",".join("?" * len(table_ids))
        placeholders2 = placeholders  # same table_ids, needed twice for UNION
        like_kw = f"%{keywords.lower()}%"

        # Collect candidates from all 4 paths, then deduplicate per table_id:
        # keep the row with the most months (most complete CY data), break ties by value desc.
        raw = conn.execute(
            f"""
            SELECT label, match_field, unit, value_or_total AS value, n_months,
                   file, file_year, table_id, title
            FROM (
                SELECT m.metric_slug AS label, 'row_annual' AS match_field, m.unit,
                       m.value AS value_or_total, 0 AS n_months,
                       t.file, t.file_year, t.id AS table_id, t.title
                FROM metrics m JOIN tables t ON t.id = m.table_id
                WHERE m.table_id IN ({placeholders})
                  AND m.year = ? AND m.value IS NOT NULL
                  AND lower(m.metric_slug) LIKE ? AND m.month IS NULL
                UNION ALL
                SELECT m.col_path AS label, 'col_annual' AS match_field, m.unit,
                       m.value AS value_or_total, 0 AS n_months,
                       t.file, t.file_year, t.id AS table_id, t.title
                FROM metrics m JOIN tables t ON t.id = m.table_id
                WHERE m.table_id IN ({placeholders2})
                  AND m.year = ? AND m.value IS NOT NULL
                  AND lower(m.col_path) LIKE ? AND m.month IS NULL
                UNION ALL
                SELECT m.metric_slug AS label, 'row_monthly_sum' AS match_field, m.unit,
                       SUM(m.value) AS value_or_total, COUNT(*) AS n_months,
                       t.file, t.file_year, t.id AS table_id, t.title
                FROM metrics m JOIN tables t ON t.id = m.table_id
                WHERE m.table_id IN ({placeholders2})
                  AND m.year = ? AND m.value IS NOT NULL
                  AND lower(m.metric_slug) LIKE ? AND m.month IS NOT NULL
                GROUP BY m.table_id, m.metric_slug
                UNION ALL
                SELECT m.col_path AS label, 'col_monthly_sum' AS match_field, m.unit,
                       SUM(m.value) AS value_or_total, COUNT(*) AS n_months,
                       t.file, t.file_year, t.id AS table_id, t.title
                FROM metrics m JOIN tables t ON t.id = m.table_id
                WHERE m.table_id IN ({placeholders2})
                  AND m.year = ? AND m.value IS NOT NULL
                  AND lower(m.col_path) LIKE ? AND m.month IS NOT NULL
                GROUP BY m.table_id, m.col_path
            )
            ORDER BY file_year, n_months DESC, value DESC, table_id
            LIMIT 400
            """,
            (
                table_ids
                + [data_year, like_kw]
                + table_ids
                + [data_year, like_kw]
                + table_ids
                + [data_year, like_kw]
                + table_ids
                + [data_year, like_kw]
            ),
        ).fetchall()
        # Deduplicate: one row per table_id, preferring highest n_months then highest value
        seen: dict[int, dict] = {}
        for r in raw:
            d = dict(r)
            tid = d["table_id"]
            if tid not in seen or d["n_months"] > seen[tid]["n_months"]:
                # Add metrics_filter hint so the model knows how to query this table
                if "col" in d.get("match_field", ""):
                    d["metrics_filter"] = f"col_path LIKE '%{d['label']}%'"
                else:
                    d["metrics_filter"] = f"metric_slug = '{d['label']}'"
                # Tag CY-complete entries: n_months=12 AND published AFTER data year ended
                if d["n_months"] == 12 and d["file_year"] > data_year:
                    d["period_note"] = (
                        "CY_complete: 12 months, bulletin published after year ended — use for calendar year totals"
                    )
                elif d["n_months"] == 12 and d["file_year"] <= data_year:
                    d["period_note"] = (
                        "FY_likely: 12 months but published during data year — probably fiscal year (Oct-Sep), not CY"
                    )
                elif d["n_months"] > 0:
                    d["period_note"] = f"partial: only {d['n_months']} months"
                seen[tid] = d
        # Sort: most complete year first (n_months DESC), then earliest publication (file_year ASC)
        deduped = sorted(
            seen.values(), key=lambda x: (-x["n_months"], x["file_year"], x["table_id"])
        )
        return {"rows": deduped[: min(limit, 100)], "n": len(deduped)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tool_search_rows(
    keywords: str, data_year: int | None, max_file_year: int | None, limit: int
) -> dict:
    """Find families that contain a row/column matching the keywords, optionally near a data year."""
    conn = _conn()
    try:
        params: list = [f"%{keywords.lower()}%"]
        year_join = ""
        year_filter = ""
        file_year_filter = ""

        if data_year is not None:
            year_join = "JOIN table_family_years tfy ON tfy.family_id = tfr.family_id"
            year_filter = "AND tfy.year = ?"
            params.append(data_year)

        if max_file_year is not None:
            if not year_join:
                year_join = "JOIN table_family_years tfy ON tfy.family_id = tfr.family_id"
                year_filter = "AND tfy.year = ?" if data_year else ""
                if data_year:
                    params.append(data_year)
            file_year_filter = "AND t.file_year <= ?"
            params.append(max_file_year)

        file_year_join = ""
        if max_file_year is not None or data_year is not None:
            file_year_join = "JOIN tables t ON t.id = tfy.best_table_id"

        sql = f"""
            SELECT DISTINCT tfr.family_id, tf.primary_title,
                   MIN(t2.file_year) AS earliest_file_year,
                   MAX(tfy2.year) AS max_data_year,
                   MIN(tfy2.year) AS min_data_year
            FROM table_family_rows tfr
            JOIN table_families tf ON tf.family_id = tfr.family_id
            {year_join}
            {file_year_join}
            LEFT JOIN table_family_years tfy2 ON tfy2.family_id = tfr.family_id
            LEFT JOIN tables t2 ON t2.id = tfy2.best_table_id
            WHERE lower(tfr.metric_slug) LIKE ?
            {year_filter}
            {file_year_filter}
            GROUP BY tfr.family_id
            ORDER BY earliest_file_year
            LIMIT ?
        """
        params.append(min(limit, 30))
        rows = conn.execute(sql, params).fetchall()
        return {"families": [dict(r) for r in rows], "n": len(rows)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tool_search_families(keywords: str, limit: int) -> dict:
    conn = _conn()
    try:
        rows = conn.execute(
            """
            SELECT f.family_id, f.primary_title, f.member_count,
                   MIN(tfy.year) AS year_min, MAX(tfy.year) AS year_max
            FROM table_families f
            LEFT JOIN table_family_years tfy ON tfy.family_id = f.family_id
            WHERE f.family_id IN (
                SELECT family_id FROM families_fts WHERE families_fts MATCH ?
            )
            GROUP BY f.family_id
            ORDER BY f.member_count DESC
            LIMIT ?
            """,
            (keywords, min(limit, 30)),
        ).fetchall()
        return {"families": [dict(r) for r in rows], "n": len(rows)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tool_describe_family(family_id: str) -> dict:
    conn = _conn()
    try:
        meta = conn.execute(
            "SELECT family_id, primary_title, member_count, latest_table_id "
            "FROM table_families WHERE family_id = ?",
            (family_id,),
        ).fetchone()
        if not meta:
            # Try case-insensitive fallback
            meta = conn.execute(
                "SELECT family_id, primary_title, member_count, latest_table_id "
                "FROM table_families WHERE lower(family_id) = lower(?)",
                (family_id,),
            ).fetchone()
        if not meta:
            return {"error": f"family_id not found: {family_id!r}"}

        fid = meta["family_id"]

        slugs = conn.execute(
            "SELECT metric_slug FROM table_family_rows WHERE family_id = ? "
            "ORDER BY metric_slug LIMIT 60",
            (fid,),
        ).fetchall()

        # All member tables with the full range of data years each covers.
        # No best_table_id heuristic — caller picks the right member.
        members = conn.execute(
            """
            SELECT t.id AS table_id, t.file, t.file_year, t.file_month,
                   t.title, t.unit, t.period, t.n_rows, t.n_cols,
                   MIN(tfy.year) AS data_year_min, MAX(tfy.year) AS data_year_max
            FROM tables t
            JOIN table_family_years tfy ON tfy.best_table_id = t.id
            WHERE tfy.family_id = ?
            GROUP BY t.id
            ORDER BY t.file_year, t.file_month
            """,
            (fid,),
        ).fetchall()

        member_list = [dict(r) for r in members]
        earliest_file_year = min(
            (m["file_year"] for m in member_list if m["file_year"]), default=None
        )
        min_data_year = min(
            (m["data_year_min"] for m in member_list if m["data_year_min"]), default=None
        )

        warning = None
        if earliest_file_year and min_data_year and (earliest_file_year - min_data_year) > 3:
            warning = (
                f"WARNING: earliest member is from {earliest_file_year}, but data starts at {min_data_year}. "
                f"This family only has retrospective/revised data — no original publications from that era. "
                f"Use search_rows(keywords, data_year={min_data_year}, max_file_year={min_data_year + 3}) "
                f"to find tables published closer to the data year."
            )

        return {
            "family_id": fid,
            "primary_title": meta["primary_title"],
            "member_count": meta["member_count"],
            "metric_slugs": [r[0] for r in slugs],
            "members": member_list,
            **({"warning": warning} if warning else {}),
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tool_search_tables(
    keywords: str, year_min: int | None, year_max: int | None, limit: int
) -> dict:
    conn = _conn()
    try:
        # FTS base query
        params: list[Any] = [keywords, min(limit, 50)]
        year_join = ""
        if year_min is not None or year_max is not None:
            year_join = """
            JOIN (
                SELECT DISTINCT table_id FROM table_columns
                WHERE year_extracted IS NOT NULL
                  {yr_min} {yr_max}
                UNION
                SELECT DISTINCT table_id FROM table_rows
                WHERE year_extracted IS NOT NULL
                  {yr_min2} {yr_max2}
            ) yf ON t.id = yf.table_id
            """.format(
                yr_min=f"AND year_extracted >= {year_min}" if year_min else "",
                yr_max=f"AND year_extracted <= {year_max}" if year_max else "",
                yr_min2=f"AND year_extracted >= {year_min}" if year_min else "",
                yr_max2=f"AND year_extracted <= {year_max}" if year_max else "",
            )
        sql = f"""
            SELECT t.id, t.file, t.title, t.section, t.unit, t.period, t.file_year, t.file_month
            FROM tables t
            {year_join}
            WHERE t.id IN (
                SELECT rowid FROM tables_fts WHERE tables_fts MATCH ?
            )
            ORDER BY t.file_year DESC
            LIMIT ?
        """
        rows = conn.execute(sql, params).fetchall()
        return {"tables": [dict(r) for r in rows], "n": len(rows)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tool_describe_table(table_id: int) -> dict:
    conn = _conn()
    try:
        meta = conn.execute(
            "SELECT id, file, file_year, file_month, title, section, caption, unit, period, n_rows, n_cols "
            "FROM tables WHERE id = ?",
            (table_id,),
        ).fetchone()
        if not meta:
            return {"error": f"table_id {table_id} not found"}

        slugs = conn.execute(
            "SELECT DISTINCT metric_slug FROM table_rows WHERE table_id = ? AND metric_slug IS NOT NULL "
            "ORDER BY row_index LIMIT 60",
            (table_id,),
        ).fetchall()

        years = conn.execute(
            """
            SELECT DISTINCT year FROM (
                SELECT year_extracted AS year FROM table_columns WHERE table_id = ?
                UNION
                SELECT year_extracted AS year FROM table_rows WHERE table_id = ?
            ) WHERE year IS NOT NULL ORDER BY year
            """,
            (table_id, table_id),
        ).fetchall()

        months = conn.execute(
            """
            SELECT DISTINCT month FROM (
                SELECT month_extracted AS month FROM table_columns WHERE table_id = ?
                UNION
                SELECT month_extracted AS month FROM table_rows WHERE table_id = ?
            ) WHERE month IS NOT NULL ORDER BY month
            """,
            (table_id, table_id),
        ).fetchall()

        return {
            "metadata": dict(meta),
            "metric_slugs": [r[0] for r in slugs],
            "years": [r[0] for r in years],
            "months": [r[0] for r in months],
        }
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tool_run_sql(query: str, limit: int) -> dict:
    query = query.strip().rstrip(";")

    # Reject non-SELECT and blacklisted keywords
    if not re.match(r"^\s*(select|with)\b", query, re.IGNORECASE):
        return {"error": "only SELECT or WITH...SELECT queries are allowed"}
    if _SQL_BLACKLIST.search(query):
        return {"error": "query contains a disallowed keyword"}
    if ";" in query:
        return {"error": "semicolons are not allowed — one query at a time"}

    # Enforce LIMIT
    effective_limit = min(max(1, limit), 200)
    if not re.search(r"\blimit\b", query, re.IGNORECASE):
        query = f"{query}\nLIMIT {effective_limit}"

    conn = _conn()
    try:
        cur = conn.execute(query)
        cols = [d[0] for d in cur.description] if cur.description else []
        raw_rows = cur.fetchmany(effective_limit)
        rows = []
        for r in raw_rows:
            row = {}
            for col, val in zip(cols, r, strict=False):
                if isinstance(val, str) and len(val) > 200:
                    val = val[:200] + "…"
                row[col] = val
            rows.append(row)
        truncated = len(raw_rows) >= effective_limit
        return {"columns": cols, "rows": rows, "n": len(rows), "truncated": truncated}
    except sqlite3.OperationalError as exc:
        return {"error": f"SQLite error: {exc}"}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _dispatch(name: str, args: dict) -> dict:
    if name == "lookup_metric":
        kw = str(args.get("keywords") or "")
        if not kw:
            return {"error": "keywords is required"}
        dy = args.get("data_year")
        fy_min = args.get("file_year_min")
        fy_max = args.get("file_year_max")
        if dy is None or fy_min is None or fy_max is None:
            return {"error": "data_year, file_year_min, and file_year_max are all required"}
        return _tool_lookup_metric(
            keywords=kw,
            data_year=int(dy),
            file_year_min=int(fy_min),
            file_year_max=int(fy_max),
            monthly=bool(args.get("monthly", False)),
            limit=int(args.get("limit", 50)),
        )
    if name == "browse_rows":
        kws = args.get("keywords")
        if not kws:
            return {"error": "keywords list is required"}
        if isinstance(kws, str):
            kws = [kws]
        return _tool_browse_rows(
            keywords=kws,
            file_year_min=int(args["file_year_min"]) if args.get("file_year_min") else None,
            file_year_max=int(args["file_year_max"]) if args.get("file_year_max") else None,
            data_year=int(args["data_year"]) if args.get("data_year") else None,
            limit=int(args.get("limit", 30)),
        )
    if name == "search_rows":
        kw = str(args.get("keywords") or "")
        if not kw:
            return {"error": "keywords parameter is required"}
        return _tool_search_rows(
            keywords=kw,
            data_year=int(args["data_year"]) if args.get("data_year") else None,
            max_file_year=int(args["max_file_year"]) if args.get("max_file_year") else None,
            limit=int(args.get("limit", 10)),
        )
    if name == "search_families":
        kw = str(args.get("keywords") or args.get("query") or "")
        if not kw:
            return {"error": "keywords parameter is required"}
        return _tool_search_families(keywords=kw, limit=int(args.get("limit", 10)))
    if name == "describe_family":
        return _tool_describe_family(str(args.get("family_id", "")))
    if name == "describe_table":
        tid = args.get("table_id") or args.get("id")
        if tid is None:
            return {"error": "table_id is required"}
        return _tool_describe_table(int(tid))
    if name == "search_tables":
        return _tool_search_tables(
            keywords=str(args.get("keywords", "")),
            year_min=args.get("year_min"),
            year_max=args.get("year_max"),
            limit=int(args.get("limit", 15)),
        )
    if name == "describe_table":
        tid = args.get("table_id") or args.get("id")
        if tid is None:
            return {"error": "table_id is required"}
        return _tool_describe_table(int(tid))
    if name == "run_sql":
        query = str(args.get("query") or args.get("sql") or "")
        return _tool_run_sql(query, int(args.get("limit", 100)))
    return {"error": f"unknown tool: {name}"}


# ── Agent loop ──────────────────────────────────────────────────────────────


def solve(
    question: str, variant: str = "plain", model: str | None = None, verbose: bool = False
) -> dict:
    """Run the SQL agent on a single question. Returns a result dict."""
    model = model or DEFAULT_MODEL
    if variant == "think":
        system = SYSTEM_THINK
    elif variant == "minimal":
        system = SYSTEM_MINIMAL
    else:
        system = SYSTEM_PLAIN
    client = _openai()
    t0 = time.time()

    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    total_tokens = 0

    for step in range(MAX_ITERS):
        # Force submit_answer on the last iteration.
        # Dedalus endpoint uses {"type": "tool"} not {"type": "function"}.
        tool_choice: Any = (
            {"type": "tool", "name": "submit_answer"} if step == MAX_ITERS - 1 else "auto"
        )

        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,  # type: ignore[arg-type]
            tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
            tool_choice=tool_choice,  # type: ignore[arg-type]
        )
        msg = resp.choices[0].message
        if resp.usage:
            total_tokens += resp.usage.total_tokens

        # Serialize assistant message (keep tool_calls if present)
        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,  # type: ignore[union-attr]
                        "arguments": tc.function.arguments,  # type: ignore[union-attr]
                    },
                }
                for tc in msg.tool_calls
            ]
        # omit tool_calls entirely when absent — null is rejected by the API
        messages.append(assistant_msg)

        if not msg.tool_calls:
            # Model returned text instead of a tool call. If there's still
            # budget, nudge it to call submit_answer. Otherwise give up.
            if step < MAX_ITERS - 2:
                messages.append(
                    {
                        "role": "user",
                        "content": "You must call submit_answer to record your final answer.",
                    }
                )
                continue
            return {
                "answer": msg.content or None,  # last-resort: use the text
                "error": "model returned no tool call",
                "trace": trace,
                "elapsed_s": time.time() - t0,
                "token_count": total_tokens,
            }

        submitted: dict | None = None
        for tc in msg.tool_calls:
            name = tc.function.name  # type: ignore[union-attr]
            raw_args = tc.function.arguments or "{}"  # type: ignore[union-attr]
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                # Return parse error back to model instead of silently dropping
                args = {}
                result: dict = {"error": f"invalid JSON in tool arguments: {raw_args[:200]}"}
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)}
                )
                trace.append({"step": step, "tool": name, "args": args, "result": result})
                continue

            trace.append({"step": step, "tool": name, "args": args})

            if verbose:
                if name == "run_sql":
                    q = (args.get("query") or args.get("sql") or "").replace("\n", " ")[:300]
                    print(f"  [{step}] run_sql: {q}", flush=True)
                else:
                    print(f"  [{step}] {name}({json.dumps(args)[:100]})", flush=True)

            if name == "submit_answer":
                submitted = {
                    "answer": str(args.get("value") or args.get("answer") or ""),
                    "unit": args.get("unit"),
                    "reasoning": args.get("reasoning", ""),
                    "trace": trace,
                    "elapsed_s": time.time() - t0,
                    "token_count": total_tokens,
                }
                break

            result = _dispatch(name, args)
            trace[-1]["result_n"] = result.get("n") or ("error" if "error" in result else "ok")
            if verbose:
                if "error" in result:
                    print(f"       → ERROR: {result['error'][:120]}", flush=True)
                else:
                    print(f"       → {result.get('n', 'ok')} rows", flush=True)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str)[:8000],
                }
            )

        if submitted:
            return submitted

    return {
        "answer": None,
        "error": "AGENT_BUDGET_EXCEEDED",
        "trace": trace,
        "elapsed_s": time.time() - t0,
        "token_count": total_tokens,
    }


# ── Eval harness ────────────────────────────────────────────────────────────


def load_benchmark(
    uids: list[str] | None = None, n: int | None = None, offset: int = 0
) -> list[dict]:
    rows = []
    with open(BENCHMARK_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if uids and row["uid"] not in uids:
                continue
            rows.append(row)
    if not uids:
        rows = rows[offset:]
        if n:
            rows = rows[:n]
    return rows


def run_eval(
    variant: str = "plain",
    n: int | None = None,
    offset: int = 0,
    uids: list[str] | None = None,
    model: str | None = None,
    verbose: bool = False,
) -> dict:
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    questions = load_benchmark(uids=uids, n=n, offset=offset)
    if not questions:
        print("No questions matched.", flush=True)
        return {}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RUNS_DIR.mkdir(exist_ok=True)
    out_path = RUNS_DIR / f"sql_solve_{variant}_{ts}.jsonl"
    print(f"Writing to {out_path}", flush=True)

    correct = 0
    total = 0
    with open(out_path, "w") as out_f:
        for row in questions:
            uid = row["uid"]
            question = row["question"]
            gold = row["answer"]

            result = solve(question, variant=variant, model=model, verbose=verbose)
            answer = result.get("answer") or ""

            try:
                is_correct, _ = fuzzy_match_answer(gold, answer) if answer else (False, "no answer")
            except Exception:
                is_correct = False

            total += 1
            if is_correct:
                correct += 1

            record = {
                "uid": uid,
                "question": question,
                "gold": gold,
                "answer": answer,
                "correct": is_correct,
                "unit": result.get("unit"),
                "reasoning": result.get("reasoning")
                or (result.get("trace") or [{}])[-1:][0].get("args", {}).get("reasoning"),
                "tool_trace": result.get("trace", []),
                "elapsed_s": result.get("elapsed_s"),
                "token_count": result.get("token_count"),
                "error": result.get("error"),
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            mark = "✓" if is_correct else "✗"
            print(
                f"{mark} [{uid}] {answer!r:<20} gold={gold!r} "
                f"({result.get('elapsed_s', 0):.1f}s, {result.get('token_count', 0)} tok)",
                flush=True,
            )

    accuracy = correct / total if total else 0
    summary = {
        "n": total,
        "correct": correct,
        "accuracy": accuracy,
        "variant": variant,
        "model": model or DEFAULT_MODEL,
    }
    summary_path = out_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n{variant}: {correct}/{total} = {accuracy:.1%}", flush=True)
    return summary


# ── CLI ─────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="SQL-agent solver for officeqa")
    parser.add_argument("question", nargs="?", help="Single question to solve")
    parser.add_argument("--eval", action="store_true", help="Run benchmark eval")
    parser.add_argument("--n", type=int, default=None, help="Number of questions")
    parser.add_argument("--offset", type=int, default=0, help="Offset into benchmark")
    parser.add_argument("--uids", type=str, default=None, help="Comma-separated UIDs to eval")
    parser.add_argument(
        "--variant",
        choices=["plain", "think", "minimal", "both"],
        default="plain",
        help="plain (default), think, minimal (bare schema only), or both",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=f"Model override (default: {DEFAULT_MODEL}). Use 'gpt-4o' as capability escape hatch.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print tool calls during single-question mode"
    )
    args = parser.parse_args()

    uids = [u.strip() for u in args.uids.split(",")] if args.uids else None

    if args.eval:
        if args.variant == "both":
            for v in ("plain", "think", "minimal"):
                run_eval(
                    variant=v,
                    n=args.n,
                    offset=args.offset,
                    uids=uids,
                    model=args.model,
                    verbose=args.verbose,
                )
        else:
            run_eval(
                variant=args.variant,
                n=args.n,
                offset=args.offset,
                uids=uids,
                model=args.model,
                verbose=args.verbose,
            )
        return

    if not args.question:
        parser.error("provide a question or use --eval")

    result = solve(args.question, variant=args.variant, model=args.model)
    if args.verbose:
        for t in result.get("trace", []):
            print(f"  [{t['step']}] {t['tool']}({json.dumps(t['args'])[:120]})")
    print(f"\nAnswer: {result.get('answer')}")
    if result.get("unit"):
        print(f"Unit:   {result['unit']}")
    if result.get("reasoning"):
        print(
            f"Reason: {result.get('reasoning') or result.get('trace', [{}])[-1].get('args', {}).get('reasoning')}"
        )
    if result.get("error"):
        print(f"Error:  {result['error']}", file=sys.stderr)
    print(f"\n{result.get('elapsed_s', 0):.1f}s, {result.get('token_count', 0)} tokens")


if __name__ == "__main__":
    main()
