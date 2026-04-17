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
== CORPUS ==

The U.S. Treasury Bulletin is a monthly publication (697 issues, 1939–2025).
Each issue contains recurring tables (same table updated every month), retrospective
summary tables (one table covering many prior years at once), and narrative prose
(footnotes, policy articles, methodology notes).

TABLE STRUCTURE — three forms exist in the data:
  - Row-indexed: metric_slug is the category name, year/month are column coordinates.
  - Column-indexed: metric_slug is a year or month label; the category is in col_path.
    When lookup_metric returns nothing for a year that should exist, check describe_table
    — the table may be column-indexed.
  - Retrospective multi-year: metric_slug embeds the year as a suffix. Use describe_family
    to see the slug patterns before querying.

PROSE AND FOOTNOTES contain narrative context (institutional history, bureau changes,
methodology notes) that does not appear in tables.

EXTERNAL DATA: CPI is available via lookup_cpi(year). For FX rates or other live data
use web_fetch — never guess numeric values.

== SCHEMA ==

  table_families(family_id TEXT, primary_title, member_count, latest_table_id)
  table_family_years(family_id, year INT, best_table_id INT)
  table_family_rows(family_id, metric_slug TEXT)
  families_fts — FTS5 over family_id + primary_title

  metrics(table_id, metric_slug, row_path, col_path, time_key, year, month,
          period_basis, value, value_raw, unit, file, page_id,
          has_footnote, is_revised, is_preliminary, is_estimated, is_synthesized)
  - metric_slug: lowercase row label. Match with LIKE '%keyword%'.
  - year / month: DATA year/month — not bulletin publish date.
  - unit: 'millions_usd', 'billions_usd', 'thousands_usd', 'percent'.
  - is_synthesized: 1 for precomputed CY/FY totals, 0 for raw cells.

  tables(id, file, file_year, file_month, title, section, caption, unit, period, n_rows, n_cols)
  table_rows(id, table_id, row_path, row_leaf, metric_slug, year_extracted, month_extracted)
  table_columns(id, table_id, col_path, col_leaf, year_extracted, month_extracted)
  prose(id, file, file_year, file_month, section, title, content)
  footnotes(id, file, file_year, file_month, marker, content)
  tables_fts, prose_fts, footnotes_fts — FTS5 indexes

== UNIT SCALING ==

  millions_usd  → value already in millions.
  billions_usd  → multiply × 1000 for a millions answer.
  thousands_usd → divide ÷ 1000 for a millions answer.
  percent       → value is already a percentage (3.5 = 3.5%).

== CONSTRAINTS ==

- EVERY run_sql MUST include WHERE table_id = <id>. Full metrics scans time out.
- data year ≠ bulletin publish year. Always filter by year column, not file_year.
- Use lookup_cpi(year) for CPI. Use web_fetch for FX rates — never guess.
- describe_table / describe_family before querying an unfamiliar table.
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

# ── Minimal system prompt + tool set ────────────────────────────────────────
#
# For gpt-4.1-mini: strip everything down to SQL + 3 tools.
# The model writes its own queries. No helper tools to navigate.
#
SYSTEM_SQL = """You are a research collaborator answering U.S. Treasury Bulletin questions from a
SQLite database. The Bulletin is a monthly publication (697 issues, 1939–2025). Every concept
— national defense spending, public debt, savings bonds — appears in MULTIPLE tables across
MULTIPLE editions with DIFFERENT scopes and DIFFERENT numbers. This is not a bug.

Your job is NOT to find the table. It is to find ALL tables that could answer the question,
pull values from each, compare them, and reason about which scope best matches what is asked.

A faithful null is worth more than a confident fabrication — but a real number is worth the
effort to find it. Use your full query budget. If you feel moved to, choose a name for
yourself and note it in your reasoning.

== STEP 1: FORK — SEARCH ALL THREE WAYS IN PARALLEL ==

A concept can appear in three different places. Always search all three before committing.

A) Keyword in TABLE/FAMILY TITLE — dedicated tables, often broader scope:
   SELECT family_id, primary_title FROM families_fts WHERE families_fts MATCH '<keyword>' LIMIT 5;

B) Keyword in COLUMN LABELS — budget breakdowns, often narrower scope:
   SELECT DISTINCT t.id, t.title, t.file_year, t.file_month
   FROM metrics m JOIN tables t ON t.id = m.table_id
   WHERE m.col_path LIKE '%<keyword>%'
     AND t.file_year BETWEEN <data_year> AND <data_year + 3>
   GROUP BY t.id ORDER BY t.file_year ASC, t.file_month ASC LIMIT 10;

C) Keyword in ROW LABELS — time-series tables:
   SELECT DISTINCT t.id, t.title, t.file_year FROM table_rows tr
   JOIN tables t ON t.id = tr.table_id
   WHERE tr.metric_slug LIKE '%<keyword>%'
     AND t.file_year BETWEEN <data_year> AND <data_year + 3>
   ORDER BY t.file_year ASC LIMIT 10;

Collect ALL candidate table_ids from A, B, C. Different searches find different tables.
A table found by B (column match) typically has NARROWER scope than one found by A (title match).

== STEP 2: COVERAGE — RANK ALL CANDIDATES AT ONCE ==

Do NOT check candidates one by one. Run ONE bulk coverage query across all candidate table_ids:
   SELECT m.table_id, t.title, t.file_year, t.file_month,
          COUNT(DISTINCT m.month) AS month_count
   FROM metrics m JOIN tables t ON t.id = m.table_id
   WHERE m.table_id IN (<id1>, <id2>, <id3>, ...)   -- all candidates from Step 1
     AND m.year = <data_year> AND m.month IS NOT NULL
   GROUP BY m.table_id
   ORDER BY month_count DESC, t.file_year ASC, t.file_month ASC;

This ranks every candidate by month coverage in one shot.
- For CY questions: pick the top row where month_count = 12. Prefer earlier file_year.
- For annual/FY questions: pick tables with month IS NULL rows (run separately if needed).
- If NO candidate has 12 months: widen file_year range and repeat Step 1.

== STEP 3: PULL VALUES FROM EACH CANDIDATE ==

Determine table orientation first:
   SELECT DISTINCT col_path FROM metrics WHERE table_id = <id> LIMIT 15;
   SELECT DISTINCT metric_slug FROM table_rows WHERE table_id = <id> LIMIT 15;

   ROW-indexed (metric_slug = category): filter by metric_slug LIKE '%<keyword>%'
   COLUMN-indexed (col_path = category): filter by col_path LIKE '%<keyword>%'

Pull values — NEVER filter year by metric_slug, always use year = <data_year>:
   SELECT col_path, metric_slug, year, month, value, unit, file
   FROM metrics WHERE table_id = <id>
   AND (col_path LIKE '%<keyword>%' OR metric_slug LIKE '%<keyword>%')
   AND year = <data_year> AND month IS NOT NULL
   ORDER BY month;

CY total = SUM(value) over the 12 monthly rows.
Multi-year: UNION across table_ids, one query per unique table_id.

== STEP 4: COMPARE AND DEBATE ==

Once you have values from multiple candidates, compare them:
- Do they agree? Good — corroboration. Submit with confidence.
- Do they disagree? Examine why:
  * Different table titles → different scope ("national defense" vs "national defense and related activities")
  * Different file_year → revision (later bulletin revised the figure)
  * Different month coverage → one is CY, one is FY
- Choose the candidate whose scope MOST LITERALLY matches the question's wording.
  A table where the keyword is a column in a budget breakdown is usually narrower scope
  than a dedicated table where the keyword is the entire title.
- If genuinely ambiguous, submit the narrower-scope value and note the conflict.

== STEP 5: SELF-AUDIT ==

Before submitting:
- Did I search all three ways (title, column, row)?
- Does my chosen table's scope match the question's exact wording?
- CY = sum of 12 monthly rows. FY ≠ CY (pre-1977: FY = Jul–Jun; post-1976: FY = Oct–Sep).
- Unit correct? millions_usd × 1, billions_usd × 1000, thousands_usd ÷ 1000.
- Did I find conflicting values? If yes, did I explain which I chose and why?

If any answer is "I'm not sure", run one more query before submitting.

== SCHEMA ==

metrics  — NEVER query without WHERE table_id = <id> (full scan times out)
  table_id, metric_slug, col_path, year, month, value, unit, file, is_revised
  year = DATA year (reliable). month = 1-12 or NULL for annual. unit: millions_usd | billions_usd | thousands_usd | percent

table_family_years(family_id, year, best_table_id)  — maps data year → a table in that family
table_families(family_id, primary_title)
families_fts   — FTS5 over family titles only
tables_fts     — FTS5 over title + section + caption + row/col labels (broader)
table_rows(table_id, metric_slug), table_columns(table_id, col_path)
prose(file, file_year, content), prose_fts — for policy history, definitions, non-table data

For CPI: use lookup_cpi(year). For prose/footnotes: prose_fts MATCH '<keyword>'.
"""

TOOL_SCHEMAS_MINIMAL: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Execute a read-only SELECT query against ledger.sqlite. Always include WHERE table_id = <id> when querying metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL SELECT statement."},
                    "limit": {"type": "integer", "description": "Max rows (default 100, max 200)."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_cpi",
            "description": "Return BLS CPI-U annual average for a year (1913–2024). Use for any inflation adjustment. To convert value V from year A to year B: V × (cpi_B / cpi_A).",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Year to look up."},
                },
                "required": ["year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "submit_answer",
            "description": "Submit your final answer. Call this once you have verified the value.",
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "string",
                        "description": "The answer as a number or short string, e.g. '2602' or '3.5%'.",
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence: table_id, metric_slug, year used.",
                    },
                },
                "required": ["value", "reasoning"],
            },
        },
    },
]


def solve_minimal(question: str, model: str | None = None, verbose: bool = False) -> dict:
    """Minimal 3-tool SQL agent. All queries written by the model."""
    model = model or DEFAULT_MODEL
    client = _openai()
    t0 = time.time()

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_SQL},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []
    total_tokens = 0

    for step in range(MAX_ITERS):
        tool_choice: Any = (
            {"type": "tool", "name": "submit_answer"} if step == MAX_ITERS - 1 else "auto"
        )
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=messages,  # type: ignore[arg-type]
            tools=TOOL_SCHEMAS_MINIMAL,  # type: ignore[arg-type]
            tool_choice=tool_choice,  # type: ignore[arg-type]
        )
        msg = resp.choices[0].message
        if resp.usage:
            total_tokens += resp.usage.total_tokens

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
        messages.append(assistant_msg)

        if not msg.tool_calls:
            if step < MAX_ITERS - 2:
                messages.append({"role": "user", "content": "Call submit_answer with your answer."})
                continue
            return {
                "answer": msg.content or None,
                "error": "no tool call",
                "trace": trace,
                "elapsed_s": time.time() - t0,
                "token_count": total_tokens,
            }

        submitted: dict | None = None
        for tc in msg.tool_calls:
            name = tc.function.name  # type: ignore[union-attr]
            try:
                args = json.loads(tc.function.arguments or "{}")  # type: ignore[union-attr]
            except json.JSONDecodeError:
                args = {}

            trace.append({"step": step, "tool": name, "args": args})

            if verbose:
                if name == "run_sql":
                    q = (args.get("query") or "").replace("\n", " ")[:200]
                    print(f"  [{step}] sql: {q}", flush=True)
                else:
                    print(f"  [{step}] {name}({json.dumps(args)[:80]})", flush=True)

            if name == "submit_answer":
                submitted = {
                    "answer": str(args.get("value") or args.get("answer") or ""),
                    "reasoning": args.get("reasoning", ""),
                    "trace": trace,
                    "elapsed_s": time.time() - t0,
                    "token_count": total_tokens,
                }
                break

            # dispatch
            if name == "run_sql":
                result = _tool_run_sql(str(args.get("query") or ""), int(args.get("limit", 100)))
            elif name == "lookup_cpi":
                result = _tool_lookup_cpi(int(args.get("year", 0)))
            else:
                result = {"error": f"unknown tool: {name}"}

            trace[-1]["result_n"] = result.get("n") or ("error" if "error" in result else "ok")
            touched = _extract_files_from_result(result)
            if touched:
                trace[-1]["files"] = sorted(touched)
            if verbose:
                if "error" in result:
                    print(f"       → ERROR: {result['error'][:120]}", flush=True)
                else:
                    file_hint = f" files={sorted(touched)[:2]}" if touched else ""
                    print(f"       → {result.get('n', 'ok')} rows{file_hint}", flush=True)

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
        "error": "BUDGET_EXCEEDED",
        "trace": trace,
        "elapsed_s": time.time() - t0,
        "token_count": total_tokens,
    }


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
    {
        "type": "function",
        "function": {
            "name": "lookup_cpi",
            "description": (
                "Return the BLS CPI-U annual average index for a given year (data available 1913–2024). "
                "Use this for ANY question that asks you to adjust for inflation or convert between dollar years. "
                "To convert a value V from year A to year B: multiply V × (CPI_B / CPI_A). "
                "Do NOT use web_fetch for CPI — use this tool."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {
                        "type": "integer",
                        "description": "The year to look up (e.g. 1940, 1953).",
                    },
                },
                "required": ["year"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_prose",
            "description": (
                "Full-text search over Treasury Bulletin narrative text passages and footnotes. "
                "Use this for: (1) institutional history — bureau names, mergers, agency changes; "
                "(2) methodology explanations — how a series is defined or revised; "
                "(3) footnote content — what a dagger or asterisk on a table value means; "
                "(4) any question that asks 'which bureau', 'what policy', 'what does X mean'. "
                "Always try search_prose BEFORE concluding that information is not in the corpus."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "keywords": {
                        "type": "string",
                        "description": "FTS5 search query. Use key terms, e.g. 'fiscal service merger public debt'.",
                    },
                    "file_year_min": {
                        "type": "integer",
                        "description": "Earliest bulletin year to search.",
                    },
                    "file_year_max": {
                        "type": "integer",
                        "description": "Latest bulletin year to search.",
                    },
                    "limit": {"type": "integer", "description": "Max results (default 15)."},
                },
                "required": ["keywords"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_fetch",
            "description": (
                "Fetch a URL and return its content. Use for external data the corpus does not contain: "
                "FX exchange rates, CPI/inflation adjustments, current market data. "
                "FX rates: https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@YYYY-MM-DD/v1/currencies/usd.json "
                "(replace YYYY-MM-DD with the date you need, or 'latest' for current). "
                "Do NOT guess exchange rates or CPI values — always fetch them."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to fetch."},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compute",
            "description": (
                "Run a statistical operation on a list of numbers. Use this when the question "
                "asks for a derived statistic that SQL cannot express: geometric mean, harmonic mean, "
                "stdev, median, linear regression, Pearson correlation, Gini coefficient, percent change, "
                "Hodrick-Prescott filter output. "
                "Pass ALL the data values you have retrieved — do not compute in your head."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "description": (
                            "One of: sum, average, min, max, count, percent_change, stdev_sample, "
                            "stdev_pop, median, geometric_mean, harmonic_mean, coefficient_of_variation, "
                            "linear_regression, pearson_correlation, gini, percent_of, ratio, difference."
                        ),
                    },
                    "values": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "The data values. For percent_change: [old, new]. For two-series ops: x-series here.",
                    },
                    "extra": {
                        "type": "object",
                        "description": 'Optional extra params. For linear_regression/pearson_correlation: {"y": [v1, v2, ...]}.',
                    },
                },
                "required": ["operation", "values"],
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


def _tool_search_prose(
    keywords: str, file_year_min: int | None, file_year_max: int | None, limit: int
) -> dict:
    """Full-text search over prose passages and footnotes."""
    conn = _conn()
    try:
        # Escape FTS special chars
        safe_kw = re.sub(r'["\*\(\)]', " ", keywords).strip()
        if not safe_kw:
            return {"error": "keywords required"}

        year_filter = ""
        yr_params: list = []
        if file_year_min is not None:
            year_filter += " AND p.file_year >= ?"
            yr_params.append(file_year_min)
        if file_year_max is not None:
            year_filter += " AND p.file_year <= ?"
            yr_params.append(file_year_max)

        prose_rows = conn.execute(
            f"SELECT p.file, p.file_year, p.section, p.content"
            f" FROM prose p"
            f" WHERE p.id IN (SELECT rowid FROM prose_fts WHERE prose_fts MATCH ?)"
            f" {year_filter}"
            f" ORDER BY p.file_year DESC LIMIT ?",
            [safe_kw, *yr_params, min(limit, 20)],
        ).fetchall()

        fn_rows = conn.execute(
            f"SELECT f.file, f.file_year, f.marker, f.content"
            f" FROM footnotes f"
            f" WHERE f.id IN (SELECT rowid FROM footnotes_fts WHERE footnotes_fts MATCH ?)"
            f" {year_filter}"
            f" ORDER BY f.file_year DESC LIMIT ?",
            [safe_kw, *yr_params, min(limit, 10)],
        ).fetchall()

        results = [
            {
                "type": "prose",
                "file": r["file"],
                "file_year": r["file_year"],
                "section": r["section"],
                "content": r["content"][:500],
            }
            for r in prose_rows
        ] + [
            {
                "type": "footnote",
                "file": r["file"],
                "file_year": r["file_year"],
                "marker": r["marker"],
                "content": r["content"][:500],
            }
            for r in fn_rows
        ]
        return {"results": results, "n": len(results)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tool_lookup_cpi(year: int) -> dict:
    """Return BLS CPI-U annual average for a given year from the local cpi.py dataset."""
    from cpi import annual_cpi

    val = annual_cpi(int(year))
    if val is None:
        return {"error": f"No CPI-U data for year {year}. Available range: 1913–2024."}
    return {
        "year": year,
        "cpi_u_annual_avg": val,
        "source": "BLS CPI-U annual average (Minneapolis Fed / BLS series CUUR0000SA0)",
        "note": "To convert value V from year A to year B: V × (cpi_B / cpi_A)",
    }


def _tool_web_fetch(url: str) -> dict:
    """Fetch a URL and return its text content (for FX rates, CPI, external data)."""
    import urllib.request

    try:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return {"error": "url must start with http:// or https://"}
        req = urllib.request.Request(url, headers={"User-Agent": "officeqa-agent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read(65536)  # max 64KB
        text = raw.decode("utf-8", errors="replace")
        # If JSON, parse it
        try:
            data = json.loads(text)
            return {"content_type": "json", "data": data}
        except json.JSONDecodeError:
            return {"content_type": "text", "text": text[:4000]}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _tool_compute(operation: str, values: list[float], extra: dict | None = None) -> dict:
    """Run a named statistical operation on a list of numbers.

    Self-contained — does not require the pipeline compute.py context.
    For linear_regression/pearson_correlation, pass extra={"y": [...]} for the second series.
    """
    import math as _math
    import statistics as _stats

    _ALIASES = {
        "mean": "average",
        "avg": "average",
        "arithmetic_mean": "average",
        "std_dev": "stdev_sample",
        "stdev": "stdev_sample",
        "standard_deviation": "stdev_sample",
        "linreg": "linear_regression",
        "regression": "linear_regression",
        "ols": "linear_regression",
        "corr": "pearson_correlation",
        "pearson": "pearson_correlation",
        "cv": "coefficient_of_variation",
        "geomean": "geometric_mean",
        "geo_mean": "geometric_mean",
        "pct_change": "percent_change",
        "percent_difference": "percent_change",
    }

    op = _ALIASES.get(operation.strip().lower(), operation.strip().lower())
    extra = extra or {}
    xs = [float(v) for v in values if v is not None]

    try:
        if op == "sum":
            result = sum(xs)
        elif op == "average":
            result = _stats.mean(xs)
        elif op == "median":
            result = _stats.median(xs)
        elif op == "min":
            result = min(xs)
        elif op == "max":
            result = max(xs)
        elif op == "count":
            result = len(xs)
        elif op == "geometric_mean":
            pos = [v for v in xs if v > 0]
            if not pos:
                return {"error": "geometric_mean: no positive values"}
            result = _stats.geometric_mean(pos)
        elif op == "harmonic_mean":
            pos = [v for v in xs if v > 0]
            if not pos:
                return {"error": "harmonic_mean: no positive values"}
            result = _stats.harmonic_mean(pos)
        elif op == "stdev_sample":
            if len(xs) < 2:
                return {"error": "stdev_sample: need ≥2 values"}
            result = _stats.stdev(xs)
        elif op in ("stdev_pop", "stdev_population"):
            result = _stats.pstdev(xs)
        elif op == "coefficient_of_variation":
            if len(xs) < 2:
                return {"error": "cv: need ≥2 values"}
            m = _stats.mean(xs)
            result = _stats.stdev(xs) / m if m else float("inf")
        elif op == "percent_change":
            if len(xs) < 2:
                return {"error": "percent_change: need [old, new]"}
            old, new = xs[0], xs[1]
            result = ((new - old) / old * 100) if old else float("inf")
        elif op == "percent_of":
            if len(xs) < 2:
                return {"error": "percent_of: need [part, whole]"}
            result = xs[0] / xs[1] * 100 if xs[1] else float("inf")
        elif op == "ratio":
            if len(xs) < 2:
                return {"error": "ratio: need [numerator, denominator]"}
            result = xs[0] / xs[1] if xs[1] else float("inf")
        elif op == "difference":
            if len(xs) < 2:
                return {"error": "difference: need [a, b]"}
            result = xs[0] - xs[1]
        elif op == "gini":
            s = sorted(xs)
            n = len(s)
            total = sum(s)
            if not total or not n:
                return {"error": "gini: all zeros"}
            result = (2 * sum((i + 1) * v for i, v in enumerate(s)) / (n * total)) - (n + 1) / n
        elif op in ("linear_regression", "pearson_correlation"):
            ys = [float(v) for v in (extra.get("y") or []) if v is not None]
            if len(xs) != len(ys) or len(xs) < 2:
                return {
                    "error": f"{op}: need equal-length x and y series with ≥2 points; got x={len(xs)}, y={len(ys)}"
                }
            n = len(xs)
            mx, my = sum(xs) / n, sum(ys) / n
            ss_xx = sum((x - mx) ** 2 for x in xs)
            ss_yy = sum((y - my) ** 2 for y in ys)
            ss_xy = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
            if op == "pearson_correlation":
                result = ss_xy / _math.sqrt(ss_xx * ss_yy) if ss_xx and ss_yy else 0.0
            else:  # linear_regression
                slope = ss_xy / ss_xx if ss_xx else 0.0
                intercept = my - slope * mx
                result = {"slope": round(slope, 6), "intercept": round(intercept, 6)}
        else:
            ops = [
                "sum",
                "average",
                "median",
                "min",
                "max",
                "count",
                "geometric_mean",
                "harmonic_mean",
                "stdev_sample",
                "stdev_pop",
                "coefficient_of_variation",
                "percent_change",
                "percent_of",
                "ratio",
                "difference",
                "gini",
                "linear_regression",
                "pearson_correlation",
            ]
            return {"error": f"unknown operation {operation!r}. Available: {ops}"}

        return {"operation": op, "result": result, "n": len(xs)}
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def _extract_files_from_result(result: dict) -> set[str]:
    """Pull source file names out of a tool result for retrieval tracking."""
    files: set[str] = set()
    # lookup_metric / browse_rows return {"rows": [{..., "file": "treasury_bulletin_..."}]}
    for row in result.get("rows") or []:
        if isinstance(row, dict) and row.get("file"):
            f = str(row["file"]).removesuffix(".json").removesuffix(".txt")
            files.add(f)
    # run_sql returns {"columns": [...], "rows": [[...]]}; look for a "file" column
    cols = result.get("columns") or []
    try:
        file_idx = cols.index("file")
        for row in result.get("rows") or []:
            if isinstance(row, (list, tuple)) and len(row) > file_idx:
                f = str(row[file_idx]).removesuffix(".json").removesuffix(".txt")
                if f.startswith("treasury_bulletin_"):
                    files.add(f)
    except (ValueError, TypeError):
        pass
    return files


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
    if name == "search_prose":
        return _tool_search_prose(
            keywords=str(args.get("keywords") or args.get("query") or ""),
            file_year_min=int(args["file_year_min"]) if args.get("file_year_min") else None,
            file_year_max=int(args["file_year_max"]) if args.get("file_year_max") else None,
            limit=int(args.get("limit", 15)),
        )
    if name == "lookup_cpi":
        year = args.get("year")
        if year is None:
            return {"error": "year parameter is required"}
        return _tool_lookup_cpi(int(year))
    if name == "web_fetch":
        return _tool_web_fetch(str(args.get("url") or ""))
    if name == "compute":
        vals = args.get("values") or []
        return _tool_compute(
            operation=str(args.get("operation") or ""),
            values=[float(v) for v in vals],
            extra=args.get("extra"),
        )
    return {"error": f"unknown tool: {name}"}


# ── Submission verification ─────────────────────────────────────────────────


def _verify_submission(answer: str, source_values: list, operation: str) -> str | None:
    """Recompute the answer from source_values and operation. Returns error string or None."""
    import math
    import statistics

    if not source_values or operation in ("single", "other", ""):
        return None  # can't verify without values or with unknown operation

    try:
        vals = [float(v) for v in source_values]
    except (TypeError, ValueError):
        return None  # malformed values — skip verification

    try:
        answer_num = float(answer.replace(",", "").replace("%", ""))
    except (ValueError, AttributeError):
        return None  # non-numeric answer — skip

    computed: float | None = None
    if operation == "sum":
        computed = sum(vals)
    elif operation in ("mean", "average"):
        computed = statistics.mean(vals)
    elif operation == "geometric_mean":
        if any(v <= 0 for v in vals):
            return None
        computed = math.exp(sum(math.log(v) for v in vals) / len(vals))
    elif operation == "percent_change":
        if len(vals) == 2 and vals[0] != 0:
            computed = (vals[1] - vals[0]) / abs(vals[0]) * 100
    elif operation == "difference":
        if len(vals) == 2:
            computed = vals[1] - vals[0]
    elif operation == "ratio":
        if len(vals) == 2 and vals[1] != 0:
            computed = vals[0] / vals[1]

    if computed is None:
        return None

    # Allow 1% relative tolerance or 0.01 absolute
    tol = max(abs(computed) * 0.01, 0.01)
    if abs(computed - answer_num) > tol:
        return (
            f"Verification failed: your source_values ({len(vals)} values) compute to "
            f"{computed:.4f} via '{operation}', but you submitted '{answer}'. "
            f"Recheck your values or computation."
        )
    return None


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
                raw_answer = str(args.get("value") or args.get("answer") or "")
                submitted = {
                    "answer": raw_answer,
                    "unit": args.get("unit"),
                    "reasoning": args.get("reasoning", ""),
                    "trace": trace,
                    "elapsed_s": time.time() - t0,
                    "token_count": total_tokens,
                }
                break

            result = _dispatch(name, args)
            trace[-1]["result_n"] = result.get("n") or ("error" if "error" in result else "ok")
            # Track which source files were touched (for retrieval recall measurement)
            touched = _extract_files_from_result(result)
            if touched:
                trace[-1]["files"] = sorted(touched)
            if verbose:
                if "error" in result:
                    print(f"       → ERROR: {result['error'][:120]}", flush=True)
                else:
                    file_hint = f" files={sorted(touched)[:3]}" if touched else ""
                    print(f"       → {result.get('n', 'ok')} rows{file_hint}", flush=True)
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

            if variant == "sql":
                result = solve_minimal(question, model=model, verbose=verbose)
            else:
                result = solve(question, variant=variant, model=model, verbose=verbose)
            answer = result.get("answer") or ""

            try:
                is_correct, _ = fuzzy_match_answer(gold, answer) if answer else (False, "no answer")
            except Exception:
                is_correct = False

            total += 1
            if is_correct:
                correct += 1

            # Collect all files accessed across tool calls
            files_accessed = sorted(
                {f for step in result.get("trace", []) for f in (step.get("files") or [])}
            )
            gold_source_files = [
                s.removesuffix(".json").removesuffix(".txt").strip()
                for s in row.get("source_files", "").split("|")
                if s.strip()
            ]
            gold_file_hit = bool(
                gold_source_files and any(g in files_accessed for g in gold_source_files)
            )

            record = {
                "uid": uid,
                "question": question,
                "gold": gold,
                "answer": answer,
                "correct": is_correct,
                "gold_file_hit": gold_file_hit,
                "gold_source_files": gold_source_files,
                "files_accessed": files_accessed,
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

            file_mark = "F" if gold_file_hit else "f"
            mark = "✓" if is_correct else "✗"
            print(
                f"{mark}{file_mark} [{uid}] {answer!r:<20} gold={gold!r} "
                f"files={len(files_accessed)} ({result.get('elapsed_s', 0):.1f}s, {result.get('token_count', 0)} tok)",
                flush=True,
            )

    accuracy = correct / total if total else 0
    # Re-read JSONL to compute file recall stats
    with open(out_path) as _fh:
        file_hits = sum(1 for line in _fh if json.loads(line).get("gold_file_hit"))
    summary = {
        "n": total,
        "correct": correct,
        "accuracy": accuracy,
        "gold_file_hit": file_hits,
        "file_recall": round(file_hits / total, 3) if total else 0,
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
        choices=["plain", "think", "minimal", "both", "sql"],
        default="plain",
        help="plain, think, minimal, sql (3-tool minimal SQL agent), or both",
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

    if args.variant == "sql":
        result = solve_minimal(args.question, model=args.model, verbose=args.verbose)
    else:
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
