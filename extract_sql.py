#!/usr/bin/env python3
"""SQL-based extraction for oracle mode.

Instead of parsing HTML tables, the LLM writes SQL queries to extract data
directly from the ledger. Much more reliable and precise.
"""

import json
import sqlite3

from ledger_paths import get_ledger_sqlite_path
from llm_client import MODEL, THINKING_EXTRA_BODY, client, strip_thinking

LEDGER_SCHEMA = """
-- Treasury Bulletin Data Schema

tables(id, file, title, section, file_year, file_month, unit, period)
  - Each row is a table in a bulletin file
  - file: 'treasury_bulletin_1941_01.json'
  - unit: 'millions_usd', 'billions_usd', 'percent', etc.
  - period: 'fiscal', 'calendar', 'unknown'

table_columns(id, table_id, col_index, col_path, col_leaf, year_extracted, month_extracted)
  - Column headers with extracted year/month
  - col_path: full header hierarchy 'Budget > 1940 > Expenditures'
  - col_leaf: just the leaf name 'Expenditures'
  - year_extracted: what year this column represents (if detectable)

table_rows(id, table_id, row_index, row_path, row_leaf, year_extracted, month_extracted, metric_slug)
  - Row labels with extracted year/month
  - metric_slug: normalized name like 'national_defense' for searching
  - year_extracted: what year this row represents (if detectable)

cells(table_id, row_id, col_id, raw_value, numeric_value, ...)
  - Cell values at intersections
  - numeric_value: parsed number, or NULL if not numeric

Example queries:
  -- Find 1940 national defense monthly values
  SELECT c.numeric_value, tc.month_extracted, t.file
  FROM cells c
  JOIN table_rows tr ON c.row_id = tr.id
  JOIN table_columns tc ON c.col_id = tc.id
  JOIN tables t ON t.id = c.table_id
  WHERE tr.metric_slug LIKE '%defense%'
    AND tc.year_extracted = 1940
    AND t.file = 'treasury_bulletin_1941_01.json'
  ORDER BY tc.month_extracted;

  -- Find total debt for year-end 2015
  SELECT c.numeric_value, t.file
  FROM cells c
  JOIN table_rows tr ON c.row_id = tr.id
  JOIN table_columns tc ON c.col_id = tc.id
  JOIN tables t ON t.id = c.table_id
  WHERE tr.row_leaf LIKE '%public debt%'
    AND tc.year_extracted = 2015
    AND tc.month_extracted IS NULL  -- year-end (annual, no month)
    AND t.file = 'treasury_bulletin_2016_12.json'
  LIMIT 1;
"""


def extract_sql(
    question: str,
    file: str,
    spec: dict,
    verbose: bool = False,
) -> dict | None:
    """Extract values by having the LLM write SQL queries.

    Args:
      question: The original question
      file: Gold file name (e.g., 'treasury_bulletin_1941_01.json')
      spec: Decompose spec with data_requests
      verbose: Print debug info

    Returns:
      {'extractions': {dr_id: {'values': [...], ...}}} or None on failure
    """
    data_requests = spec.get("data_requests") or []
    if not data_requests:
        return None

    # Build context for the LLM
    drs_json = json.dumps(data_requests, indent=2)

    user_msg = f"""You have access to a Treasury Bulletin SQLite database.

Schema:
{LEDGER_SCHEMA}

File: {file}
Question: {question}

Data needed:
{drs_json}

Write SQL queries to find each value. Return ONLY a JSON object with this format:
{{
  "queries": [
    {{"dr_id": "v1", "query": "SELECT ... WHERE ..."}}
  ]
}}

Each query should return rows with the values needed for that data_request.
"""

    if verbose:
        print(f"  Extract(SQL): asking LLM to write queries for {len(data_requests)} DRs")
        print(
            f"\n  === LLM PROMPT ===\n{user_msg[:800]}...\n"
            if len(user_msg) > 800
            else f"\n  === LLM PROMPT ===\n{user_msg}\n"
        )

    # Call LLM to generate SQL
    raw = client.chat.completions.create(
        model=MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_msg}],
        extra_body=THINKING_EXTRA_BODY,
    )
    response_text = strip_thinking(raw.choices[0].message.content or "")

    if verbose:
        print(f"  === LLM RESPONSE ===\n{response_text}\n")

    # Parse the response
    import re

    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response_text).strip()
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            result = json.loads(cleaned[: cleaned.rfind("}") + 1])
        except Exception:
            return None

    queries = result.get("queries") or []

    # Execute queries against ledger
    ledger_path = get_ledger_sqlite_path()
    if not ledger_path:
        return None

    extractions = {}
    try:
        conn = sqlite3.connect(ledger_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        for q_spec in queries:
            dr_id = q_spec.get("dr_id")
            query = q_spec.get("query")
            if not dr_id or not query:
                continue

            if verbose:
                print(f"  Executing SQL for {dr_id}: {query[:80]}...")

            try:
                cursor.execute(query)
                rows = cursor.fetchall()

                # Extract numeric values from results
                values = []
                for row in rows:
                    # Try to find a numeric_value column, else any numeric column
                    val = None
                    for col_name in ["numeric_value", "value", "total", "count", "amount"]:
                        if col_name in row.keys():
                            val = row[col_name]
                            break
                    if val is None and len(row.keys()) > 0:
                        # Try the first column
                        val = row[list(row.keys())[0]]
                    if val is not None:
                        try:
                            values.append(float(val))
                        except (ValueError, TypeError):
                            pass

                if values:
                    extractions[dr_id] = {
                        "values": values,
                        "labels": [str(v) for v in values],
                        "source": "sql",
                    }
                else:
                    extractions[dr_id] = {"values": [], "source": "sql_no_results"}

            except Exception as e:
                if verbose:
                    print(f"  SQL error for {dr_id}: {e}")
                extractions[dr_id] = {"values": [], "error": str(e)}

        conn.close()
    except Exception as e:
        if verbose:
            print(f"  Ledger error: {e}")
        return None

    return {"extractions": extractions}
