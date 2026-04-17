#!/usr/bin/env python3
"""Extract with injected tool results from SQLite + JSON file access."""

import json
import sqlite3
from pathlib import Path

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.sqlite"
CORPUS = HERE / "corpus_json"


def query_sqlite(query: str) -> list[dict]:
    """Execute SQL query on ledger.sqlite, return rows."""
    conn = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(query).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def read_json_file(file_stem: str) -> dict:
    """Load corpus JSON file, return full content."""
    path = CORPUS / f"{file_stem}.json"
    if not path.exists():
        return {"error": f"File not found: {file_stem}"}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


def build_tool_context(gold_files: list[str], spec: dict) -> str:
    """Build context by running tools on gold files based on spec.

    Returns formatted context string with:
    - Table metadata from ledger
    - Table data from files
    """
    lines = ["AVAILABLE DATA FROM GOLD FILES:", ""]

    data_requests = spec.get("data_requests", [])

    for dr in data_requests:
        dr_id = dr.get("id", "?")
        row_hint = dr.get("row_hint", "")
        col_hint = dr.get("column_hint", "")
        years = dr.get("years", [])

        lines.append(f"DATA REQUEST: {dr_id}")
        lines.append(f"  Looking for: {row_hint} (year: {years})")
        lines.append("")

        # Query ledger for matching tables in gold files
        for file_stem in gold_files:
            file_json = f"{file_stem}.json"

            # Find tables in this file that match the criteria
            query = """
            SELECT DISTINCT t.title, t.caption, t.file, GROUP_CONCAT(r.row_path) as rows
            FROM tables t
            LEFT JOIN table_rows r ON r.table_id = t.id
            WHERE t.file = ? AND t.table_kind = 'data'
            GROUP BY t.id
            LIMIT 10
            """

            try:
                rows = query_sqlite(query, (file_json,))
                if rows:
                    lines.append(f"  File: {file_stem}")
                    for row in rows:
                        title = row.get("title", "?")
                        lines.append(f"    TABLE: {title}")
                        if row.get("rows"):
                            row_labels = row["rows"].split(",")[:5]
                            lines.append(f"      Rows: {', '.join(row_labels)}")
                    lines.append("")
            except:
                pass

        # Also load and summarize the JSON file content
        for file_stem in gold_files:
            try:
                data = read_json_file(file_stem)
                if "error" not in data:
                    lines.append(f"  Raw file: {file_stem}.json")
                    tables = [e for e in data.get("elements", []) if e.get("type") == "table"]
                    lines.append(f"    Contains {len(tables)} tables")
                    for t in tables[:3]:
                        lines.append(f"      - {t.get('title', '?')}")
                    lines.append("")
            except:
                pass

    return "\n".join(lines)


# Monkey-patch to add tool context injection
_original_extract_structured = None


def extract_structured_with_tools(
    spec: dict,
    per_dr_entries: dict,
    question: str,
    gold_files: list[str] = None,
    verbose: bool = False,
) -> dict | None:
    """Wrapper around extract_structured that injects tool context."""
    from extract import extract_structured as orig_extract

    # Build tool context from gold files
    tool_context = ""
    if gold_files:
        tool_context = build_tool_context(gold_files, spec)

    # Modify entries to include tool context
    enhanced_entries = {}
    for dr_id, entries in per_dr_entries.items():
        if tool_context and entries:
            # Add tool context as a pseudo-entry
            enhanced_entries[dr_id] = entries + [
                {
                    "file": "TOOL_CONTEXT",
                    "element_id": "tool_results",
                    "title": "Available Data from Gold Files",
                    "content": tool_context,
                    "retrieval_strategy": "oracle_tools",
                    "retrieval_channel": "oracle_tools",
                }
            ]
        else:
            enhanced_entries[dr_id] = entries

    # Call original extract with enhanced entries
    return orig_extract(spec, enhanced_entries, question, verbose=verbose)


if __name__ == "__main__":
    # Test
    spec = {
        "data_requests": [
            {
                "id": "test_dr",
                "row_hint": "National Defense",
                "column_hint": "1940",
                "years": [1940],
            }
        ]
    }

    gold_files = ["treasury_bulletin_1940_01"]
    context = build_tool_context(gold_files, spec)
    print(context)
