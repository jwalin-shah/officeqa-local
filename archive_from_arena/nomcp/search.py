#!/usr/bin/env python3
"""Treasury Bulletin search tool — zero dependencies, works on raw TXT files.

Usage:
    python3 /installed-agent/search.py find "national defense" 1940
    python3 /installed-agent/search.py find "customs duties" 1953
    python3 /installed-agent/search.py table <file> <line_number>
    python3 /installed-agent/search.py files 1940
    python3 /installed-agent/search.py compute "a - b" '{"a": 500.3, "b": 120.1}'

Commands:
    find <query> [year]    — fuzzy search across all TXT files, returns matches with context
    table <file> <line>    — extract full markdown table around a given line number
    files <year>           — list bulletin files that likely contain data for a year
    compute <expr> <vars>  — evaluate math expression with variables
"""

import json
import math
import os
import re
import sys
from pathlib import Path

CORPUS_DIR = os.environ.get("CORPUS_DIR", "/app/corpus")


def _find_files(year=None):
    """List TXT files, optionally filtered to those likely containing data for a year."""
    corpus = Path(CORPUS_DIR)
    if not corpus.exists():
        # Try alternate locations
        for alt in ["/app/corpus", "/installed-agent/corpus"]:
            if Path(alt).exists():
                corpus = Path(alt)
                break

    files = sorted(corpus.glob("treasury_bulletin_*.txt"))

    if year is not None:
        year = int(year)
        # Prioritize: bulletins from year and year+1 (most likely to have data),
        # but also include year+2 through year+10 for historical tables
        primary = []
        secondary = []
        for f in files:
            m = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", f.name)
            if m:
                pub_year = int(m.group(1))
                if year <= pub_year <= year + 2:
                    primary.append(f)
                elif year - 1 <= pub_year <= year + 15:
                    secondary.append(f)
        # Search primary first (most likely), then secondary
        return primary + secondary
    return files


def _normalize(text):
    """Normalize text for fuzzy matching."""
    return re.sub(r"[^a-z0-9 ]", " ", text.lower())


def _fuzzy_match(query_terms, line):
    """Score a line against query terms. Returns (score, matched_terms)."""
    line_norm = _normalize(line)
    matched = []
    for term in query_terms:
        if term in line_norm:
            matched.append(term)
    return len(matched), matched


def cmd_find(query, year=None):
    """Search across TXT files for a query, return matches with table context."""
    query_terms = _normalize(query).split()
    files = _find_files(year)

    if not files:
        # Fallback: search all files
        files = _find_files(None)
    if not files:
        print(f"No files found in {CORPUS_DIR}")
        return

    results = []

    for fpath in files:
        try:
            lines = fpath.read_text(errors="replace").splitlines()
        except Exception:
            continue

        for i, line in enumerate(lines):
            if "|" not in line:
                continue  # Only search table rows

            score, matched = _fuzzy_match(query_terms, line)
            if score < max(1, len(query_terms) - 1):
                continue  # Need most terms to match

            # Get context: find table header (units) by scanning upward
            units = ""
            table_title = ""
            for j in range(max(0, i - 20), i):
                ctx_line = lines[j].strip()
                if re.search(r"\(.*(?:millions|thousands|dollars|percent).*\)", ctx_line, re.I):
                    units = ctx_line
                if (
                    ctx_line
                    and not ctx_line.startswith("|")
                    and not ctx_line.startswith("---")
                    and len(ctx_line) > 10
                ):
                    if not re.match(r"^\d+$", ctx_line.strip()):
                        table_title = ctx_line

            # Get column header (first row with |---|)
            col_header = ""
            for j in range(max(0, i - 15), i):
                if lines[j].strip().startswith("|") and "---" not in lines[j]:
                    col_header = lines[j].strip()
                    break

            results.append(
                {
                    "file": fpath.name,
                    "line": i + 1,
                    "score": score,
                    "match": line.strip()[:200],
                    "units": units,
                    "table_title": table_title[:100],
                    "col_header": col_header[:200] if col_header else "",
                }
            )

    # Sort by score descending, then by file name descending (prefer later bulletins)
    results.sort(key=lambda r: (-r["score"], r["file"]), reverse=False)
    results.sort(key=lambda r: (-r["score"], r["file"]), reverse=True)
    # Re-sort: highest score first, within same score prefer later files
    results.sort(key=lambda r: (-r["score"], [-ord(c) for c in r["file"]]))

    # Deduplicate: same row label in different files → keep latest bulletin
    seen = set()
    deduped = []
    for r in results:
        # Extract row label (first cell in the pipe-delimited row)
        cells = r["match"].split("|")
        row_label = cells[1].strip().lower() if len(cells) > 1 else r["match"][:40]
        key = row_label[:60]
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    # Print top results
    for r in deduped[:10]:
        print(f"\n=== {r['file']}:{r['line']} (score={r['score']}) ===")
        if r["table_title"]:
            print(f"Table: {r['table_title']}")
        if r["units"]:
            print(f"Units: {r['units']}")
        if r["col_header"]:
            print(f"Headers: {r['col_header'][:150]}")
        print(f"Match: {r['match']}")

    if not deduped:
        print(f"No matches for '{query}' (year={year})")
        # Suggest broader search
        if len(query_terms) > 1:
            print(f'Try: python3 /installed-agent/search.py find "{query_terms[0]}" {year or ""}')


def cmd_table(filepath, line_number):
    """Extract the full markdown table surrounding a given line number."""
    fpath = Path(filepath)
    if not fpath.exists():
        fpath = Path(CORPUS_DIR) / filepath

    if not fpath.exists():
        print(f"File not found: {filepath}")
        return

    lines = fpath.read_text(errors="replace").splitlines()
    target = int(line_number) - 1  # Convert to 0-indexed

    if target < 0 or target >= len(lines):
        print(f"Line {line_number} out of range (file has {len(lines)} lines)")
        return

    # Scan upward to find table start
    start = target
    for i in range(target - 1, max(0, target - 30), -1):
        line = lines[i].strip()
        if line.startswith("|") or "---" in line:
            start = i
        elif line and not line.startswith("|"):
            # Found non-table content — check if it's units/title
            if re.search(r"\(.*(?:millions|thousands|dollars|percent).*\)", line, re.I):
                start = i  # Include units line
            elif i < start - 1:
                break

    # Include 3 lines of context above table for title/units
    start = max(0, start - 3)

    # Scan downward to find table end
    end = target
    for i in range(target + 1, min(len(lines), target + 50)):
        line = lines[i].strip()
        if line.startswith("|"):
            end = i
        elif line.startswith("Source:") or line.startswith("Note"):
            end = i  # Include source/footnotes
            break
        elif not line:
            # Empty line — include a few more for footnotes
            if i < end + 3:
                continue
            break

    # Include footnotes after table
    for i in range(end + 1, min(len(lines), end + 15)):
        line = lines[i].strip()
        if (
            re.match(r"^[0-9*/]+[/)]", line)
            or line.startswith("Source:")
            or line.startswith("Note")
            or line.startswith("|")
        ):
            end = i
        elif not line:
            continue
        else:
            break

    print(f"--- {fpath.name} lines {start + 1}-{end + 1} ---")
    for i in range(start, end + 1):
        print(f"{i + 1:5d} | {lines[i]}")


def cmd_files(year):
    """List files likely containing data for a given year."""
    files = _find_files(int(year))
    if files:
        print(f"Files likely containing {year} data:")
        for f in files:
            size_kb = f.stat().st_size // 1024
            print(f"  {f.name}  ({size_kb}KB)")
    else:
        print(f"No files found for year {year}")


def cmd_compute(expr_str, vars_str=None):
    """Evaluate a math expression with optional variables."""
    variables = {}
    if vars_str:
        variables = json.loads(vars_str)

    ns = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "pow": pow,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pi": math.pi,
        "e": math.e,
    }
    ns.update(variables)

    try:
        result = eval(expr_str, {"__builtins__": {}}, ns)
        print(f"{result}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == "find":
        query = sys.argv[2] if len(sys.argv) > 2 else ""
        year = sys.argv[3] if len(sys.argv) > 3 else None
        cmd_find(query, year)
    elif cmd == "table":
        filepath = sys.argv[2] if len(sys.argv) > 2 else ""
        line_num = sys.argv[3] if len(sys.argv) > 3 else "1"
        cmd_table(filepath, line_num)
    elif cmd == "files":
        year = sys.argv[2] if len(sys.argv) > 2 else ""
        cmd_files(year)
    elif cmd == "compute":
        expr = sys.argv[2] if len(sys.argv) > 2 else ""
        vars_json = sys.argv[3] if len(sys.argv) > 3 else None
        cmd_compute(expr, vars_json)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
