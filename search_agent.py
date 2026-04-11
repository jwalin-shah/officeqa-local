"""Agentic retrieval fallback.

When the deterministic retrieve_v2 funnel fails to surface any candidate
table whose row labels actually match the decomposed row_hint (i.e. every
entry has `probe_matched_rows == 0`), run an LLM tool-loop that can query
the ledger schema directly and return its own top-K table_ids. Those get
hydrated into the same entry-dict shape the rest of the pipeline expects.

Tools exposed to the agent:
  - search_tables(query, year?, month?, period?, limit?) → ranked list of
    tables with title/section/file_year, via find.find_tables.
  - describe_table(table_id) → full row/col shape of one table.
  - return_candidates(table_ids, reasoning) → terminal.
"""

from __future__ import annotations

import json
import os
import sqlite3

from find import _openai, describe_table, find_tables

# Use DeepSeek for the search agent — it follows tool schemas reliably.
# Fall back to FIND_MODEL only if explicitly overridden.
AGENT_MODEL = os.getenv(
    "OFFICEQA_AGENT_MODEL", os.getenv("OFFICEQA_MODEL", "deepseek/deepseek-chat")
)
from retrieve_v2 import _ledger_conn, _load_element_html

SEARCH_AGENT_SYSTEM = """You find Treasury Bulletin tables that answer a user question.
Your goal is to identify the right TABLE IDs — not to extract or verify cell values.
Call return_candidates as soon as you have good candidates. Do NOT spin on cell verification.

Schema (ledger.sqlite):
- tables(id, file, file_year, file_month, title, section, caption, unit, period, n_rows, n_cols)
  file_year/file_month = PUBLICATION date of the bulletin, NOT the data year inside.
- table_rows(id, table_id, row_index, row_path, row_leaf, indent_level, metric_slug)
  ALWAYS use metric_slug (normalized) for row matching, NEVER row_leaf.
- table_columns(id, table_id, col_index, col_path, col_leaf, year_extracted, month_extracted)
  year_extracted/month_extracted = the data year/month of the column. NEVER filter by col_leaf text.
- FTS index: tables_fts (title, section, caption, columns, rows) — search with MATCH

Tools:
1. sql(query) — read-only SELECT against ledger.sqlite. Returns up to 50 rows.
2. search_tables(query, year?, month?, period?, limit?) — FTS wrapper, year filter uses year_extracted.
3. return_candidates(table_ids, reasoning) — TERMINAL. Return up to 10 table_ids best-first.

RULES — follow exactly:
- NEVER filter by file_year for data year lookups. ALWAYS use year_extracted on table_columns or table_rows.
- NEVER filter by col_leaf text (e.g. col_leaf LIKE '%1940%'). Use year_extracted = 1940.
- ALWAYS use metric_slug for row matching, not row_leaf. metric_slug is lowercase and normalized.
- ALWAYS add ORDER BY to sql() queries. Result is capped at 50 rows — unsorted queries miss gold tables.
- Do NOT call describe_table or check_cell. Your job is retrieval, not extraction.
- Max 3 sql()/search_tables() calls, then call return_candidates with your best candidates.
- PUBLICATION LAG: data for month M is published in the bulletin issued in month M+1 or later.
  A question about "March 1977" data will be answered by the April 1977 (or later) bulletin.
  Search for tables with year_extracted=1977 AND month_extracted=3, then ORDER BY t.file_year DESC, t.file_month DESC
  to surface the bulletin that first published complete data for that period.
- DIVERSITY: return table_ids from DIFFERENT files. Do not return 10 tables from the same bulletin.

STRATEGY — 3 steps, no more:
  Step 1: sql() joining tables + table_rows + table_columns, filtering on metric_slug LIKE and year_extracted.
  Step 2: If step 1 returns <3 results, call search_tables() for broad FTS fallback.
  Step 3: return_candidates() with the best table_ids from steps 1-2.

SQL TEMPLATES:

  -- Metric + year, years as COLUMNS (Type A — most common):
  SELECT DISTINCT t.id, t.file, t.title, t.file_year FROM tables t
    JOIN table_rows r ON r.table_id = t.id
    JOIN table_columns c ON c.table_id = t.id
    WHERE r.metric_slug LIKE '%national defense%'
      AND c.year_extracted = 1940
    ORDER BY t.file_year DESC LIMIT 10

  -- Metric + year, years as ROWS (Type C — summary/historical tables):
  -- Use this when the Type A query returns few/no results.
  -- These tables have the metric as a column header and years as row labels.
  SELECT DISTINCT t.id, t.file, t.title, t.file_year FROM tables t
    JOIN table_rows r ON r.table_id = t.id
    WHERE r.metric_slug LIKE '%national defense%'
      AND r.year_extracted = 1940
    ORDER BY t.file_year DESC LIMIT 10

  -- ALWAYS try BOTH Type A and Type C queries. Many gold tables use years-as-rows.

  -- Month-specific data:
  SELECT DISTINCT t.id, t.file, t.title FROM tables t
    JOIN table_columns c ON c.table_id = t.id
    JOIN table_rows r ON r.table_id = t.id
    WHERE c.year_extracted = 1979 AND c.month_extracted = 12
      AND r.metric_slug LIKE '%currency and coin in circulation%'
    ORDER BY t.file_year DESC LIMIT 10

  -- Multi-year range (try both cols and rows):
  SELECT DISTINCT t.id, t.file, t.title FROM tables t
    JOIN table_rows r ON r.table_id = t.id
    WHERE r.metric_slug LIKE '%budget expenditures%'
      AND (t.id IN (SELECT table_id FROM table_columns WHERE year_extracted BETWEEN 1942 AND 1948)
        OR t.id IN (SELECT table_id FROM table_rows WHERE year_extracted BETWEEN 1942 AND 1948))
    ORDER BY t.file_year DESC LIMIT 10

  -- FTS fallback:
  SELECT id, file, title, file_year FROM tables_fts
    WHERE tables_fts MATCH 'veterans expenditures' ORDER BY rank LIMIT 15
"""


SEARCH_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "sql",
            "description": "Run a read-only SQL SELECT against ledger.sqlite. Returns up to 50 rows as a list of objects. Use for precise lookups: metric_slug LIKE, year filters, FTS MATCH via tables_fts, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A SELECT statement. Must be read-only.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_tables",
            "description": "FTS5 search over parsed Treasury Bulletin tables. Returns ranked matches with title, section, file_year, file_month, period.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Free-text query; tokens are ANDed via FTS5.",
                    },
                    "year": {
                        "type": "integer",
                        "description": "Strict filter: table must carry this year on a column or row.",
                    },
                    "month": {
                        "type": "integer",
                        "description": "Strict filter: used with year for monthly data.",
                    },
                    "period": {
                        "type": "string",
                        "description": "Filter by period type: 'fiscal' or 'calendar'.",
                    },
                    "limit": {"type": "integer", "default": 15},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "describe_table",
            "description": "Return structural shape of a single table: column leaves, row leaves, header metadata, footnotes. No cell values.",
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
            "name": "check_cell",
            "description": (
                "Verify that a candidate table actually contains the cell you need. "
                "Returns the matching cell values (raw + numeric) and the column/row labels. "
                "Use this BEFORE committing a table — if it returns no rows, the table does not "
                "have the data and you should try the next candidate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "integer"},
                    "year": {
                        "type": "integer",
                        "description": "Data year to check (year_extracted). Omit for annual data.",
                    },
                    "month": {
                        "type": "integer",
                        "description": "Data month 1-12 (month_extracted). Omit for annual data.",
                    },
                    "metric_slug_contains": {
                        "type": "string",
                        "description": "Substring to match against row metric_slug, e.g. 'currency and coin in circulation'.",
                    },
                },
                "required": ["table_id", "metric_slug_contains"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "return_candidates",
            "description": "Terminal call. Return best-first list of table_ids to send to extract.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_ids": {
                        "type": "array",
                        "items": {"type": "integer"},
                        "description": "Up to 10 table_ids, most relevant first.",
                    },
                    "reasoning": {"type": "string"},
                },
                "required": ["table_ids"],
            },
        },
    },
]


_SQL_BLACKLIST = ("insert", "update", "delete", "drop", "create", "alter", "attach", "pragma")


def _run_search_tool(name: str, args: dict) -> dict:
    if name == "sql":
        query = (args.get("query") or "").strip()
        if not query.lower().startswith("select"):
            return {"error": "only SELECT queries are allowed"}
        if any(kw in query.lower() for kw in _SQL_BLACKLIST):
            return {"error": "query contains disallowed keyword"}
        try:
            conn = _ledger_conn()
            cur = conn.execute(query)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchmany(50)
            result = [dict(zip(cols, row, strict=False)) for row in rows]
            return {"rows": result, "n": len(result)}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    if name == "check_cell":
        tid = int(args["table_id"])
        year = args.get("year")
        month = args.get("month")
        slug = args.get("metric_slug_contains", "")
        try:
            conn = _ledger_conn()
            col_filter = ""
            col_params: list = []
            if year is not None and month is not None:
                col_filter = "AND col.year_extracted = ? AND col.month_extracted = ?"
                col_params = [year, month]
            elif year is not None:
                col_filter = "AND col.year_extracted = ?"
                col_params = [year]
            rows = conn.execute(
                f"""
                SELECT r.row_leaf, col.col_leaf, col.year_extracted, col.month_extracted,
                       c.raw_value, c.numeric_value
                FROM cells c
                JOIN table_rows r ON r.id = c.row_id AND r.table_id = c.table_id
                JOIN table_columns col ON col.id = c.col_id AND col.table_id = c.table_id
                WHERE c.table_id = ?
                  AND r.metric_slug LIKE ?
                  {col_filter}
                  AND c.is_missing = 0
                ORDER BY col.col_index
                LIMIT 10
                """,
                [tid, f"%{slug}%"] + col_params,
            ).fetchall()
            [d[0] for d in (rows[0].keys() if rows else [])]
            result = [
                dict(
                    zip(
                        [
                            "row_leaf",
                            "col_leaf",
                            "year_extracted",
                            "month_extracted",
                            "raw_value",
                            "numeric_value",
                        ],
                        row, strict=False,
                    )
                )
                for row in rows
            ]
            return {"found": len(result) > 0, "cells": result, "n": len(result)}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
    if name == "search_tables":
        hits = find_tables(
            query=args.get("query", ""),
            year=args.get("year"),
            month=args.get("month"),
            period=args.get("period"),
            limit=args.get("limit", 15),
        )
        return {"hits": hits, "n": len(hits)}
    if name == "describe_table":
        return describe_table(int(args["table_id"]))
    return {"error": f"unknown tool {name}"}


def _harvest_from_trace(trace: list[dict], verbose: bool = False) -> dict:
    """Extract best-effort table_ids from the accumulated tool trace when the
    agent exits without calling return_candidates.

    Priority: (1) tables the agent explicitly described (deliberate), then
    (2) tables from the last search_tables result (replayed cheaply).
    """
    harvested: list[int] = []
    seen: set[int] = set()

    for t in trace:
        if t["tool"] == "describe_table":
            tid = int(t["args"].get("table_id", 0))
            if tid and tid not in seen:
                harvested.append(tid)
                seen.add(tid)

    for t in reversed(trace):
        if t["tool"] == "search_tables" and len(harvested) < 10:
            try:
                hits = find_tables(
                    query=t["args"].get("query", ""),
                    year=t["args"].get("year"),
                    month=t["args"].get("month"),
                    period=t["args"].get("period"),
                    limit=10,
                )
                for h in hits:
                    tid = int(h.get("table_id", 0))
                    if tid and tid not in seen:
                        harvested.append(tid)
                        seen.add(tid)
            except Exception:
                pass
            break

    if verbose and harvested:
        print(
            f"  [search_agent] harvested {len(harvested)} table_ids from trace",
            flush=True,
        )
    return {"table_ids": harvested[:10]}


def run_search_agent(
    question: str,
    plan: dict | None = None,
    max_iters: int = 12,
    verbose: bool = False,
) -> dict:
    """Drive the LLM tool loop; return {'table_ids': [...], 'trace': [...]}."""
    client = _openai()

    user_content = question
    if plan:
        hint_bits = []
        for dr in plan.get("data_requests", []) or []:
            bits = []
            if dr.get("row_hint"):
                bits.append(f"row_hint={dr['row_hint']!r}")
            if dr.get("column_hint"):
                bits.append(f"column_hint={dr['column_hint']!r}")
            if dr.get("years"):
                bits.append(f"years={dr['years']}")
            if bits:
                hint_bits.append("  - " + ", ".join(bits))
        if hint_bits:
            user_content += (
                "\n\nDecompose hints (use as a starting hypothesis, verify with "
                "describe_table before committing):\n" + "\n".join(hint_bits)
            )

    messages: list[dict] = [
        {"role": "system", "content": SEARCH_AGENT_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    trace: list[dict] = []

    for step in range(max_iters):
        if step == max_iters - 1:
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You have one tool call left. Call return_candidates "
                        "now with your best table_ids — do not call any other "
                        "tool."
                    ),
                }
            )

        try:
            resp = client.chat.completions.create(
                model=AGENT_MODEL,
                temperature=0,
                messages=messages,  # type: ignore[arg-type]
                tools=SEARCH_TOOL_SCHEMAS,  # type: ignore[arg-type]
                tool_choice="auto",  # type: ignore[arg-type]
            )
        except Exception as api_exc:
            # API error mid-loop — harvest whatever we have so far
            if verbose:
                print(f"  [search_agent] API error at step {step}: {api_exc}", flush=True)
            return {
                **_harvest_from_trace(trace, verbose=verbose),
                "trace": trace,
                "error": f"api_error: {api_exc}",
            }
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
                **_harvest_from_trace(trace, verbose=verbose),
                "error": "model returned no tool call",
            }

        for tc in msg.tool_calls:
            name = tc.function.name  # type: ignore[union-attr]
            try:
                args = json.loads(tc.function.arguments or "{}")  # type: ignore[union-attr]
            except json.JSONDecodeError:
                args = {}
            trace.append({"step": step, "tool": name, "args": args})
            if verbose:
                print(f"[search_agent:{step}] {name}({args})", flush=True)

            if name == "return_candidates":
                return {
                    "table_ids": [int(x) for x in (args.get("table_ids") or [])][:10],
                    "reasoning": args.get("reasoning"),
                    "trace": trace,
                }

            try:
                result = _run_search_tool(name, args)
            except Exception as exc:  # defensive: bad SQL / bad args → feed back
                result = {"error": f"{type(exc).__name__}: {exc}"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": json.dumps(result, default=str)[:8000],
                }
            )

    return {
        **_harvest_from_trace(trace, verbose=verbose),
        "trace": trace,
        "error": "max_iters exhausted",
    }


def _hydrate_entries(
    conn: sqlite3.Connection,
    table_ids: list[int],
    load_html: bool = True,
    max_per_file: int = 2,
) -> list[dict]:
    """Materialize table_ids into retrieve_v2-shaped entry dicts.

    Deduplicates by file: at most max_per_file entries per bulletin file so
    that a single file cannot fill all return slots and crowd out other files.
    """
    entries: list[dict] = []
    file_counts: dict[str, int] = {}
    for tid in table_ids:
        r = conn.execute(
            """
            SELECT id, file, element_id, element_seq, page_id, file_year,
                   file_month, title, section, caption, unit, period,
                   n_rows, n_cols
              FROM tables WHERE id = ?
            """,
            (tid,),
        ).fetchone()
        if not r:
            continue

        # Skip if this file already has max_per_file entries
        file_key = r["file"]
        if file_counts.get(file_key, 0) >= max_per_file:
            continue
        file_counts[file_key] = file_counts.get(file_key, 0) + 1

        cols = conn.execute(
            "SELECT col_path FROM table_columns WHERE table_id = ? ORDER BY col_index",
            (tid,),
        ).fetchall()
        labels = conn.execute(
            "SELECT row_path FROM table_rows WHERE table_id = ? ORDER BY row_index LIMIT 80",
            (tid,),
        ).fetchall()
        year_rows = conn.execute(
            """
            SELECT DISTINCT year FROM (
                SELECT year_extracted AS year FROM table_columns WHERE table_id = ?
                UNION
                SELECT year_extracted AS year FROM table_rows    WHERE table_id = ?
            ) WHERE year IS NOT NULL
            """,
            (tid, tid),
        ).fetchall()
        years = sorted(int(y[0]) for y in year_rows if y[0] is not None)

        entries.append(
            {
                "probe_matched_rows": 0,
                "probe_best_cells": 0,
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
                "years": years,
                "unit": r["unit"],
                "period": r["period"],
                "n_rows": r["n_rows"],
                "n_cols": r["n_cols"],
                "retrieval_strategy": "search_agent",
                "retrieval_channel": "agent",
                "html": _load_element_html(r["file"], r["element_seq"]) if load_html else "",
            }
        )
    return entries


def _has_probe_hit(entries: list[dict]) -> bool:
    return any((e.get("probe_matched_rows") or 0) > 0 for e in entries)


def _plan_has_row_hints(plan: dict | None) -> bool:
    if not plan:
        return False
    return any((dr.get("row_hint") or "").strip() for dr in plan.get("data_requests", []) or [])


def retrieve_with_agent_fallback(
    plan: dict | None,
    question: str,
    top_k: int = 10,
    verbose: bool = False,
    **retrieve_kwargs,
) -> list[dict]:
    """Run deterministic retrieve_v2 first; fall back to the search agent
    when retrieval has nothing useful.

    Trigger:
      - `len(entries) == 0`, OR
      - plan carried a row_hint but no returned entry has a probe hit
        (meaning no candidate table has a row matching the decomposed
        row_hint — classic hard-miss failure mode).

    When the plan has no row_hint the probe is uninformative, so we only
    fall back on empty results.
    """
    from retrieve_v2 import retrieve as retrieve_v2_retrieve

    entries = retrieve_v2_retrieve(plan, question, top_k=top_k, verbose=verbose, **retrieve_kwargs)

    probe_meaningful = _plan_has_row_hints(plan)
    should_fallback = (not entries) or (probe_meaningful and not _has_probe_hit(entries))
    if not should_fallback:
        return entries

    if verbose:
        reason = "empty" if not entries else "no probe hit"
        print(f"  [fallback] triggering search agent ({reason})", flush=True)

    agent_result = run_search_agent(question, plan=plan, verbose=verbose)
    table_ids = agent_result.get("table_ids") or []
    if not table_ids:
        if verbose:
            print("  [fallback] agent returned no candidates", flush=True)
        return entries  # keep whatever deterministic gave us (possibly [])

    conn = _ledger_conn()
    agent_entries = _hydrate_entries(
        conn, table_ids, load_html=retrieve_kwargs.get("load_html", True)
    )
    if verbose:
        print(f"  [fallback] agent returned {len(agent_entries)} entries", flush=True)
    return agent_entries[:top_k]


# ── CLI: quick sanity run ───────────────────────────────────────────────────


def _cli() -> None:
    import sys

    if len(sys.argv) < 2:
        print("Usage: uv run python search_agent.py '<question>'")
        sys.exit(1)
    q = sys.argv[1]
    result = run_search_agent(q, verbose=True)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    _cli()
