"""Ledger search tools for LLM-driven table discovery.

`find_tables(query, year, month, period)` runs a hard year/month/period
filter over `table_columns` + `table_rows` first, then ranks the
survivors with FTS5 over `tables_fts`. The LLM only ever sees tables
that can structurally contain the requested value.

`describe_table(table_id)` returns the structural shape of one table
(column leaves + row leaves) so the LLM can confirm its pick.

`llm_find(question)` wraps both tools in an OpenAI tool-calling loop
against `groq/openai/gpt-oss-120b` by default (tool-calling is broken
on DeepSeek via Dedalus) and returns the table_id the model commits to.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import threading
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

LEDGER_PATH = Path(__file__).parent / "ledger.sqlite"

load_dotenv()

FIND_MODEL = os.getenv("OFFICEQA_FIND_MODEL", "groq/openai/gpt-oss-120b")

_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEDALUS_API_KEY"),
            base_url=os.getenv("DEDALUS_API_BASE"),
        )
    return _client


_TLS = threading.local()


def _conn() -> sqlite3.Connection:
    conn = getattr(_TLS, "conn", None)
    if conn is None:
        if not LEDGER_PATH.exists():
            raise FileNotFoundError(f"{LEDGER_PATH} not found — run build_ledger.py")
        conn = sqlite3.connect(str(LEDGER_PATH))
        conn.row_factory = sqlite3.Row
        _TLS.conn = conn
    return conn


_STOP = {
    "the",
    "a",
    "an",
    "of",
    "for",
    "in",
    "on",
    "at",
    "to",
    "and",
    "or",
    "was",
    "were",
    "is",
    "are",
    "be",
    "been",
    "what",
    "which",
    "who",
    "how",
    "much",
    "many",
    "did",
    "do",
    "does",
    "had",
    "has",
    "have",
    "by",
    "from",
    "with",
    "as",
    "that",
    "this",
    "these",
    "those",
    "year",
    "fiscal",
    "calendar",
    "total",
    "value",
    "amount",
}


def _tokens(query: str) -> list[str]:
    raw = re.findall(r"[A-Za-z][A-Za-z\-]{2,}", query.lower())
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        if t in _STOP or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:12]


def _fts_match(tokens: list[str]) -> str:
    safe = [f'"{t}"' for t in tokens if '"' not in t and "'" not in t]
    return " OR ".join(safe) if safe else ""


def _normalize_years(year: int | list[int] | None) -> list[int]:
    if year is None:
        return []
    if isinstance(year, int):
        return [year]
    return [int(y) for y in year]


def find_tables(
    query: str,
    year: int | list[int] | None = None,
    month: int | None = None,
    period: str | None = None,
    granularity: str | None = None,
    limit: int = 20,
) -> list[dict]:
    """Strict-filter table search.

    Returns ranked tables where:
      - `year` (if given) appears as `year_extracted` on at least one
        column or row of the table.
      - `month` (if given) also appears alongside the year on a column
        or row (monthly data).
      - `period` (if given) matches `tables.period`, with 'unknown'
        always allowed since ingest leaves many tables unlabeled.
      - FTS5 MATCH against `query` tokens over title/section/caption/
        columns/rows.
    """
    conn = _conn()
    tokens = _tokens(query)
    fts_q = _fts_match(tokens)
    if not fts_q:
        return []

    years = _normalize_years(year)

    year_filter_sql = ""
    year_filter_params: list = []
    if years:
        yph = ",".join("?" * len(years))
        month_col = "AND month_extracted = ?" if month is not None else ""
        month_params = [month] if month is not None else []
        year_filter_sql = f"""
            AND t.id IN (
                SELECT table_id FROM table_columns
                 WHERE year_extracted IN ({yph}) {month_col}
                UNION
                SELECT table_id FROM table_rows
                 WHERE year_extracted IN ({yph}) {month_col}
            )
        """
        year_filter_params = [*years, *month_params, *years, *month_params]

    period_sql = ""
    period_params: list = []
    if period in ("fiscal", "calendar"):
        period_sql = "AND t.period IN (?, 'unknown')"
        period_params = [period]

    sql = f"""
        SELECT
            t.id AS table_id,
            t.file,
            t.file_year,
            t.file_month,
            t.title,
            t.section,
            t.caption,
            t.period,
            t.unit,
            t.n_rows,
            t.n_cols,
            bm25(tables_fts, 2.0, 1.5, 1.5, 2.0, 2.5) AS score
        FROM tables_fts
        JOIN tables t ON t.id = tables_fts.rowid
        WHERE tables_fts MATCH ?
          AND t.table_kind = 'data'
          {year_filter_sql}
          {period_sql}
        ORDER BY score
        LIMIT ?
    """
    # Overfetch so we have headroom after dedup.
    fetch_limit = max(limit * 10, 200)
    params = [fts_q, *year_filter_params, *period_params, fetch_limit]
    raw_rows = conn.execute(sql, params).fetchall()

    # Compute coverage of the target year(s) per candidate table.
    # `year_rows` = number of distinct rows tagged with target year
    # (annual or monthly). `month_rows` = number of distinct months
    # of the target year covered (rows + columns). Higher = more
    # comprehensive coverage.
    table_ids = [r["table_id"] for r in raw_rows]
    coverage: dict[int, dict] = {tid: {"year_rows": 0, "months": 0} for tid in table_ids}
    if years and table_ids:
        ph_tids = ",".join("?" * len(table_ids))
        ph_yrs = ",".join("?" * len(years))
        # rows tagged with target year (annual + monthly)
        for row in conn.execute(
            f"""SELECT table_id, COUNT(*) AS n
                  FROM table_rows
                 WHERE table_id IN ({ph_tids})
                   AND year_extracted IN ({ph_yrs})
                 GROUP BY table_id""",
            (*table_ids, *years),
        ):
            coverage[row[0]]["year_rows"] = row[1]
        # months covered by row labels
        row_months: dict[int, set] = {tid: set() for tid in table_ids}
        for row in conn.execute(
            f"""SELECT table_id, month_extracted
                  FROM table_rows
                 WHERE table_id IN ({ph_tids})
                   AND year_extracted IN ({ph_yrs})
                   AND month_extracted IS NOT NULL""",
            (*table_ids, *years),
        ):
            row_months[row[0]].add(row[1])
        # months covered by column labels
        col_months: dict[int, set] = {tid: set() for tid in table_ids}
        for row in conn.execute(
            f"""SELECT table_id, month_extracted
                  FROM table_columns
                 WHERE table_id IN ({ph_tids})
                   AND year_extracted IN ({ph_yrs})
                   AND month_extracted IS NOT NULL""",
            (*table_ids, *years),
        ):
            col_months[row[0]].add(row[1])
        for tid in table_ids:
            coverage[tid]["months"] = len(row_months[tid] | col_months[tid])

    # Dedup by section (fall back to title, then file). Within each
    # group, ranking depends on granularity:
    #   - 'monthly' (or month set): prefer tables with the most months
    #     of the target year covered, since the question needs a wide
    #     monthly span.
    #   - 'annual' / unset: prefer best bm25 score (year filter already
    #     guarantees the table has the year somewhere). Don't penalize
    #     tables that happen to have low monthly coverage.
    monthly_mode = granularity == "monthly" or month is not None

    def _rank_key(r: sqlite3.Row) -> tuple:
        cov = coverage[r["table_id"]]
        if monthly_mode:
            return (
                -cov["months"],
                r["score"],
                r["file_year"] or 9999,
                r["file_month"] or 99,
            )
        return (
            r["score"],
            r["file_year"] or 9999,
            r["file_month"] or 99,
        )

    best: dict[str, sqlite3.Row] = {}
    for r in raw_rows:
        key = (r["section"] or "").strip() or (r["title"] or "").strip() or r["file"]
        prev = best.get(key)
        if prev is None or _rank_key(r) < _rank_key(prev):
            best[key] = r

    rows = sorted(best.values(), key=_rank_key)[:limit]

    return [
        {
            "table_id": r["table_id"],
            "file": r["file"],
            "file_year": r["file_year"],
            "file_month": r["file_month"],
            "title": r["title"],
            "section": r["section"],
            "caption": r["caption"],
            "period": r["period"],
            "unit": r["unit"],
            "n_rows": r["n_rows"],
            "n_cols": r["n_cols"],
            "score": round(r["score"], 3),
            "year_rows": coverage[r["table_id"]]["year_rows"],
            "months_covered": coverage[r["table_id"]]["months"],
        }
        for r in rows
    ]


def describe_table(table_id: int, max_labels: int = 60) -> dict:
    """Return the structural shape of a table: column leaves, row leaves
    (excluding section headers), plus header metadata and any footnote
    markers attached to the table. No cell values."""
    conn = _conn()
    t = conn.execute(
        """
        SELECT id, file, file_year, file_month, title, section, caption,
               period, unit, n_rows, n_cols
          FROM tables WHERE id = ?
        """,
        (table_id,),
    ).fetchone()
    if not t:
        return {"error": f"table {table_id} not found"}

    cols = conn.execute(
        """
        SELECT col_index, col_leaf, col_path, year_extracted, month_extracted
          FROM table_columns WHERE table_id = ? ORDER BY col_index
        """,
        (table_id,),
    ).fetchall()

    rows = conn.execute(
        """
        SELECT row_index, row_leaf, row_path, indent_level,
               year_extracted, month_extracted, is_section_header
          FROM table_rows WHERE table_id = ? ORDER BY row_index
        """,
        (table_id,),
    ).fetchall()

    footnotes = conn.execute(
        """
        SELECT marker, content FROM footnotes
         WHERE attached_to_table_id = ?
         ORDER BY id
        """,
        (table_id,),
    ).fetchall()

    def _shorten(items: list, limit: int) -> tuple[list, int]:
        if len(items) <= limit:
            return items, 0
        return items[:limit], len(items) - limit

    col_labels = [
        {
            "i": c["col_index"],
            "leaf": c["col_leaf"],
            "path": c["col_path"],
            "year": c["year_extracted"],
            "month": c["month_extracted"],
        }
        for c in cols
    ]
    row_labels_all = [
        {
            "i": r["row_index"],
            "leaf": r["row_leaf"],
            "path": r["row_path"],
            "indent": r["indent_level"],
            "year": r["year_extracted"],
            "month": r["month_extracted"],
        }
        for r in rows
        if not r["is_section_header"]
    ]
    section_headers = [
        {"i": r["row_index"], "leaf": r["row_leaf"]} for r in rows if r["is_section_header"]
    ]

    col_labels, col_truncated = _shorten(col_labels, max_labels)
    row_labels, row_truncated = _shorten(row_labels_all, max_labels)

    return {
        "table_id": t["id"],
        "file": t["file"],
        "file_year": t["file_year"],
        "file_month": t["file_month"],
        "title": t["title"],
        "section": t["section"],
        "caption": t["caption"],
        "period": t["period"],
        "unit": t["unit"],
        "n_rows": t["n_rows"],
        "n_cols": t["n_cols"],
        "columns": col_labels,
        "columns_truncated": col_truncated,
        "rows": row_labels,
        "rows_truncated": row_truncated,
        "section_headers": section_headers,
        "footnotes": [{"marker": f["marker"], "content": f["content"]} for f in footnotes],
    }


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "find_tables",
            "description": (
                "Search the Treasury Bulletin table ledger. Hard-filters by "
                "`year` (must appear on a column or row of the table), `month` "
                "(monthly-granularity data), and `period` ('fiscal' or "
                "'calendar' — unknown-period tables are always allowed through). "
                "Within the filtered set, ranks by FTS5 bm25 over title/section/"
                "caption/columns/rows. Returns candidate metadata, no cell "
                "values."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Keyword query. Use distinctive nouns from the "
                            "question — e.g. 'national defense expenditures', "
                            "'veterans administration', 'silver imports'."
                        ),
                    },
                    "year": {
                        "type": "integer",
                        "description": (
                            "Year that must appear as an extracted column or "
                            "row label. Omit if the question has no year "
                            "anchor."
                        ),
                    },
                    "month": {
                        "type": "integer",
                        "description": "Month 1-12 for monthly-resolution data.",
                    },
                    "period": {
                        "type": "string",
                        "enum": ["fiscal", "calendar"],
                    },
                    "granularity": {
                        "type": "string",
                        "enum": ["annual", "monthly"],
                        "description": (
                            "Pass 'monthly' for any question that needs "
                            "month-level data — calendar-year sums, monthly "
                            "ranges, specific-month lookups. Ranking will "
                            "prefer tables with the most monthly rows for "
                            "the target year. Pass 'annual' (default) for "
                            "fiscal-year totals or any single annual lookup."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max candidates to return (default 15).",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": (
                "Return the full column and row labels of a specific table so "
                "you can confirm which exact cells contain the values you need."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "integer"},
                },
                "required": ["table_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "commit_cells",
            "description": (
                "Commit the answer. Provide the table_id you verified with "
                "describe_table, a list of cells from that table, and a "
                "Python expression that combines the named cells into the "
                "final answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {
                        "type": "integer",
                        "description": "Table that contains all the cells.",
                    },
                    "cells": {
                        "type": "array",
                        "description": (
                            "List of cell references from the single table. "
                            "For a single-value lookup, one cell. For a sum, "
                            "one cell per term. Each cell MUST have row_leaf "
                            "(verbatim from describe_table), col_leaf "
                            "(verbatim), and name (a short Python identifier "
                            "like 'm01', 'fy', 'a' that you reference in the "
                            "expression)."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "row_leaf": {
                                    "type": "string",
                                    "description": "Row label, verbatim.",
                                },
                                "col_leaf": {
                                    "type": "string",
                                    "description": "Column label, verbatim.",
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Python identifier for this cell.",
                                },
                            },
                            "required": ["row_leaf", "col_leaf", "name"],
                        },
                    },
                    "expression": {
                        "type": "string",
                        "description": (
                            "Python expression over the cell names. "
                            "Examples: 'fy' for a single lookup; "
                            "'m01+m02+m03+m04+m05+m06+m07+m08+m09+m10+m11+m12' "
                            "for a calendar-year sum; '(a-b)/b*100' for a "
                            "percent change."
                        ),
                    },
                    "answer_unit": {
                        "type": "string",
                        "description": (
                            "What unit the question asks the answer in: "
                            "'millions', 'billions', 'thousands', 'percent', "
                            "'dollars', 'count', or 'raw'."
                        ),
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "One sentence: why these cells.",
                    },
                },
                "required": [
                    "table_id",
                    "cells",
                    "expression",
                    "answer_unit",
                    "reasoning",
                ],
            },
        },
    },
]


LLM_FIND_SYSTEM = """You answer quantitative questions about U.S. Treasury Bulletins (1930s–2025) by finding the right cells in a ledger of 94k parsed tables and specifying a computation over them.

═══════════════════════════════════════════════════════════════════════
CORPUS
═══════════════════════════════════════════════════════════════════════
One file per bulletin issue: treasury_bulletin_YYYY_MM.json.
- Published MONTHLY through 1982 (12 issues/year)
- Published QUARTERLY from 1983 (Mar/Jun/Sep/Dec typically)

Bulletins publish data retrospectively. A table in the 1941-10 issue contains FY1940 and monthly 1940-Sep, 1940-Oct, etc. data. A table in the 1947-10 issue contains summary rows for 1938-1946 plus monthly rows for the recent year. When searching for a year, the best file is usually the issue published ~6-18 months after that year's data was finalized.

Common table categories:
- Budget Receipts and Expenditures (by function: National Defense, Veterans, Net Interest, ...)
- Federal Debt (public debt, interest-bearing, marketable securities)
- International Capital (claims, liabilities, by country)
- Currency in Circulation (by denomination)
- Foreign Currency Positions (JPY, GBP, INR, DEM, CAD)
- Savings Bonds sales (by series, by state)
- Treasury auctions (notes, bills, bonds)

═══════════════════════════════════════════════════════════════════════
FISCAL YEAR vs CALENDAR YEAR — CRITICAL
═══════════════════════════════════════════════════════════════════════
CY YYYY = Jan 1 YYYY – Dec 31 YYYY (sum of 12 monthly rows Jan–Dec YYYY)
FY YYYY pre-1977  = Jul 1 YYYY−1 – Jun 30 YYYY
FY YYYY post-1976 = Oct 1 YYYY−1 – Sep 30 YYYY

The same metric for CY YYYY and FY YYYY are DIFFERENT numbers. Do not confuse them.

- "calendar year YYYY" / "CY YYYY" / "the year YYYY" (no FY modifier):
    → You need a table with monthly rows for YYYY (Jan, Feb, …, Dec).
    → Commit 12 cells, one per month, and sum them in the expression.
    → Do NOT use an annual row labeled "YYYY" — that is almost always fiscal year.
- "fiscal year YYYY" / "FY YYYY":
    → Use the annual row labeled "YYYY" in a fiscal-year table (period='fiscal').
    → Commit a single cell.

═══════════════════════════════════════════════════════════════════════
ROWS AND COLUMNS
═══════════════════════════════════════════════════════════════════════
Treasury tables often put TIME on the rows:
  row 1: 1938
  row 2: 1939
  row 3: 1940               ← fiscal-year summary row
  row 4: 1941
  row 5: 1940-September     ← monthly rows
  row 6: 1940-October
  ...

And METRIC BREAKDOWNS on the columns:
  col 0: Fiscal year or month
  col 1: Total
  col 2: War Department
  col 3: Navy Department
  ...

When you `describe_table`, check BOTH rows and columns for the data you need. Don't assume time is on columns.

Row labels from older bulletins often have OCR errors ("Piecals" for "Fiscal", "Havv" for "Navy"). Match them verbatim from describe_table — don't correct them in your commit.

"Total X" rows already include their children. Never ask for "Total Defense" when you want a specific sub-component like "War Department".

═══════════════════════════════════════════════════════════════════════
UNITS
═══════════════════════════════════════════════════════════════════════
Tables report values at their own scale (usually millions of dollars for budget tables; thousands for debt; percent for rates). The table's unit is shown in describe_table's `unit` field when known. The raw cell value you'll work with is at that native scale — do NOT multiply/divide to convert. Set `answer_unit` to whatever the question asks for, and the formatter will handle conversion.

═══════════════════════════════════════════════════════════════════════
WORKFLOW
═══════════════════════════════════════════════════════════════════════
1. Read the question. Decide: CY or FY? Single value, monthly sum, multi-year, ratio?
2. Call `find_tables` with a distinctive keyword query + year. For CY questions needing monthly data, pass `month=1` in a first call to bias toward tables that have monthly rows, or just pass `year` and inspect candidates.
3. Inspect top 3–5 candidates. Pick the most specific-titled one.
4. Call `describe_table` on it. Confirm the rows/columns contain what you need.
5. If the table is wrong, try another candidate or refine the query.
6. Call `commit_cells` with the exact row_leaf/col_leaf labels and a Python expression. Use short names like `m01`, `m02`, ..., `m12` for monthly cells; `fy` or `v1` for single lookups; `a`, `b` for differences.

RULES:
- You MUST finish with a `commit_cells` call. Do not answer in plain text.
- Copy row_leaf and col_leaf VERBATIM from describe_table, including any OCR quirks.
- For CY questions, commit 12 monthly cells, not an annual row.
- If find_tables returns zero with a year filter, retry with a broader query or without period."""


def _run_tool(name: str, args: dict) -> dict:
    if name == "find_tables":
        hits = find_tables(
            query=args.get("query", ""),
            year=args.get("year"),
            month=args.get("month"),
            period=args.get("period"),
            granularity=args.get("granularity"),
            limit=args.get("limit", args.get("top_n", 15)),
        )
        return {"hits": hits, "n": len(hits)}
    if name == "describe_table":
        return describe_table(int(args["table_id"]))
    return {"error": f"unknown tool {name}"}


def llm_find(question: str, max_iters: int = 8, verbose: bool = False) -> dict:
    """Drive the LLM tool-calling loop to pick a table for `question`.

    Returns `{table_id, row_leaf, col_leaf, reasoning, trace}` where
    `trace` is the list of tool calls the model made (for debugging).
    """
    client = _openai()
    messages: list[dict] = [
        {"role": "system", "content": LLM_FIND_SYSTEM},
        {"role": "user", "content": question},
    ]
    trace: list[dict] = []

    for step in range(max_iters):
        if step == max_iters - 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have one tool call left. Call commit_cells now "
                        "with your best current answer, even if uncertain. "
                        "Do not call any other tool."
                    ),
                }
            )
            tool_choice = {"type": "function", "function": {"name": "commit_cells"}}
        else:
            tool_choice = "auto"

        resp = client.chat.completions.create(
            model=FIND_MODEL,
            temperature=0,
            messages=messages,  # type: ignore[arg-type]
            tools=TOOL_SCHEMAS,  # type: ignore[arg-type]
            tool_choice=tool_choice,  # type: ignore[arg-type]
        )
        msg = resp.choices[0].message
        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,  # type: ignore[union-attr]
                            "arguments": tc.function.arguments,  # type: ignore[union-attr]
                        },
                    }
                    for tc in (msg.tool_calls or [])
                ]
                or None,
            }
        )

        if not msg.tool_calls:
            return {
                "error": "model returned no tool call",
                "content": msg.content,
                "trace": trace,
            }

        for tc in msg.tool_calls:
            name = tc.function.name  # type: ignore[union-attr]
            try:
                args = json.loads(tc.function.arguments or "{}")  # type: ignore[union-attr]
            except json.JSONDecodeError:
                args = {}
            trace.append({"step": step, "tool": name, "args": args})
            if verbose:
                print(f"[{step}] {name}({args})")

            if name == "commit_cells":
                return {
                    "table_id": args.get("table_id"),
                    "cells": args.get("cells") or [],
                    "expression": args.get("expression"),
                    "answer_unit": args.get("answer_unit"),
                    "reasoning": args.get("reasoning"),
                    "trace": trace,
                }

            result = _run_tool(name, args)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str)[:8000],
                }
            )

    return {"error": "max_iters exhausted", "trace": trace}


_SAFE_EVAL_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "round": round,
    "float": float,
    "int": int,
    "pow": pow,
}

_EXPR_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_BUILTIN_NAMES = set(_SAFE_EVAL_BUILTINS)


def _expression_names(expression: str) -> list[str]:
    """Ordered unique list of identifiers in the expression, minus builtins."""
    seen: list[str] = []
    for m in _EXPR_NAME_RE.finditer(expression or ""):
        tok = m.group(1)
        if tok in _BUILTIN_NAMES or tok in seen:
            continue
        seen.append(tok)
    return seen


def _cell_name(cell: dict, fallback: str) -> str:
    """Fetch the cell's identifier, tolerating schema drift from the LLM
    (model has been seen emitting 'name', 'value', 'value_name', 'id',
    'var', 'label' as the key)."""
    for key in ("name", "value_name", "value", "id", "var", "label", "key"):
        v = cell.get(key)
        if isinstance(v, str) and v.isidentifier():
            return v
    return fallback


def resolve_cells(table_id: int, cells: list[dict], expression: str = "") -> dict:
    """Look up the numeric value of each committed cell.

    All cells come from the single `table_id`. If the LLM's cell names
    don't match the expression identifiers (because it used a
    non-standard key), we fall back to positional assignment — the k-th
    cell becomes the k-th identifier in the expression.

    Returns `{values: {name: v_or_None}, debug: {name: info}}`. Uses
    correct foreign keys: `cells.row_id` → `table_rows.id`,
    `cells.col_id` → `table_columns.id`.
    """
    conn = _conn()
    values: dict[str, float | None] = {}
    debug: dict[str, dict] = {}

    expr_names = _expression_names(expression)
    use_positional = len(expr_names) == len(cells) and all(
        _cell_name(c, "") not in expr_names for c in cells
    )

    for i, c in enumerate(cells):
        name = expr_names[i] if use_positional else _cell_name(c, f"c{i}")
        tid = int(c.get("table_id", table_id))
        rleaf = c.get("row_leaf", "")
        cleaf = c.get("col_leaf", "")

        row = conn.execute(
            "SELECT id FROM table_rows WHERE table_id=? AND row_leaf=? LIMIT 1",
            (tid, rleaf),
        ).fetchone()
        col = conn.execute(
            "SELECT id FROM table_columns WHERE table_id=? AND col_leaf=? LIMIT 1",
            (tid, cleaf),
        ).fetchone()

        if not row or not col:
            values[name] = None
            debug[name] = {
                "status": "label_miss",
                "row_found": bool(row),
                "col_found": bool(col),
                "row_leaf": rleaf,
                "col_leaf": cleaf,
            }
            continue

        cell = conn.execute(
            """SELECT raw_value, numeric_value, parse_status
                 FROM cells
                WHERE table_id=? AND row_id=? AND col_id=?""",
            (tid, row["id"], col["id"]),
        ).fetchone()

        if not cell:
            values[name] = None
            debug[name] = {"status": "cell_missing", "row_id": row["id"], "col_id": col["id"]}
            continue

        values[name] = cell["numeric_value"]
        debug[name] = {
            "status": "ok",
            "raw": cell["raw_value"],
            "num": cell["numeric_value"],
            "parse": cell["parse_status"],
        }

    return {"values": values, "debug": debug}


def eval_expression(expression: str, values: dict) -> float:
    """Evaluate a Python expression over named cell values. Restricted
    builtins — no imports, no attribute access on arbitrary objects."""
    code = compile(expression, "<commit_cells>", "eval")
    for name in code.co_names:
        if name not in _SAFE_EVAL_BUILTINS and name not in values:
            raise ValueError(f"expression references unknown name: {name!r}")
    return eval(code, {"__builtins__": _SAFE_EVAL_BUILTINS}, dict(values))


def solve(question: str, verbose: bool = False) -> dict:
    """End-to-end: LLM tool loop → cell resolution → expression eval.

    Returns a dict with `answer`, `commit`, `resolved`, and `trace`.
    """
    commit = llm_find(question, verbose=verbose)
    if "error" in commit:
        return {"answer": None, "error": commit["error"], "commit": commit}

    tid = commit.get("table_id")
    resolved = resolve_cells(
        tid if tid is not None else 0,  # type: ignore[arg-type]
        commit["cells"],
        commit.get("expression", ""),
    )
    missing = [k for k, v in resolved["values"].items() if v is None]
    if missing:
        return {
            "answer": None,
            "error": f"unresolved cells: {missing}",
            "commit": commit,
            "resolved": resolved,
        }

    try:
        answer = eval_expression(commit["expression"], resolved["values"])
    except Exception as e:
        return {
            "answer": None,
            "error": f"expression failed: {e}",
            "commit": commit,
            "resolved": resolved,
        }

    return {
        "answer": answer,
        "expression": commit["expression"],
        "answer_unit": commit.get("answer_unit"),
        "cells": commit["cells"],
        "resolved": resolved["values"],
        "reasoning": commit.get("reasoning"),
        "trace": commit.get("trace"),
    }


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="?")
    ap.add_argument("--year", type=int, action="append")
    ap.add_argument("--month", type=int)
    ap.add_argument("--period", choices=["fiscal", "calendar"])
    ap.add_argument("--granularity", choices=["annual", "monthly"])
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--describe", type=int, help="describe a table_id")
    ap.add_argument("--solve", action="store_true", help="run LLM tool loop on query")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.describe is not None:
        print(json.dumps(describe_table(args.describe), indent=2))
        return

    if args.solve:
        if not args.query:
            ap.error("--solve requires a question")
        result = solve(args.query, verbose=args.verbose)
        print(json.dumps(result, indent=2, default=str))
        return

    if not args.query:
        ap.error("query required unless --describe is used")

    hits = find_tables(
        args.query,
        year=args.year,
        month=args.month,
        period=args.period,
        granularity=args.granularity,
        limit=args.limit,
    )

    if args.json:
        print(json.dumps(hits, indent=2))
        return

    if not hits:
        print("(no hits)")
        return

    for i, h in enumerate(hits, 1):
        date = f"{h['file_year']}-{h['file_month'] or '??':02}" if h["file_year"] else "?"
        title = (h["title"] or "").strip() or "(no title)"
        section = (h["section"] or "").strip()
        caption = (h["caption"] or "").strip()
        print(f"[{i}] id={h['table_id']} {date} period={h['period']} unit={h['unit']}")
        print(f"    file: {h['file']}")
        print(f"    title: {title[:120]}")
        if section:
            print(f"    section: {section[:120]}")
        if caption:
            print(f"    caption: {caption[:120]}")
        print(
            f"    {h['n_rows']}r x {h['n_cols']}c  score={h['score']}"
            f"  year_rows={h.get('year_rows', 0)}"
            f"  months={h.get('months_covered', 0)}"
        )
        print()


if __name__ == "__main__":
    _cli()
