#!/usr/bin/env python3
"""Zero-dependency MCP stdio server for v14 — backup tools for the Python intern."""

import ast
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

# --- Anti-spin & caching ---
_call_count = 0
_cache = {}
WARN_AT, STOP_AT = 10, 15

# --- CPI-U annual averages (1913-2024) ---
CPI = {
    1913: 9.9,
    1914: 10.0,
    1915: 10.1,
    1916: 10.9,
    1917: 12.8,
    1918: 15.1,
    1919: 17.3,
    1920: 20.0,
    1921: 17.9,
    1922: 16.8,
    1923: 17.1,
    1924: 17.1,
    1925: 17.5,
    1926: 17.7,
    1927: 17.4,
    1928: 17.2,
    1929: 17.2,
    1930: 16.7,
    1931: 15.2,
    1932: 13.6,
    1933: 12.9,
    1934: 13.4,
    1935: 13.7,
    1936: 13.9,
    1937: 14.4,
    1938: 14.1,
    1939: 13.9,
    1940: 14.0,
    1941: 14.7,
    1942: 16.3,
    1943: 17.3,
    1944: 17.6,
    1945: 18.0,
    1946: 19.5,
    1947: 22.3,
    1948: 24.1,
    1949: 23.8,
    1950: 24.1,
    1951: 26.0,
    1952: 26.5,
    1953: 26.7,
    1954: 26.9,
    1955: 26.8,
    1956: 27.2,
    1957: 28.1,
    1958: 28.9,
    1959: 29.1,
    1960: 29.6,
    1961: 29.9,
    1962: 30.2,
    1963: 30.6,
    1964: 31.0,
    1965: 31.5,
    1966: 32.4,
    1967: 33.4,
    1968: 34.8,
    1969: 36.7,
    1970: 38.8,
    1971: 40.5,
    1972: 41.8,
    1973: 44.4,
    1974: 49.3,
    1975: 53.8,
    1976: 56.9,
    1977: 60.6,
    1978: 65.2,
    1979: 72.6,
    1980: 82.4,
    1981: 90.9,
    1982: 96.5,
    1983: 99.6,
    1984: 103.9,
    1985: 107.6,
    1986: 109.6,
    1987: 113.6,
    1988: 118.3,
    1989: 124.0,
    1990: 130.7,
    1991: 136.2,
    1992: 140.3,
    1993: 144.5,
    1994: 148.2,
    1995: 152.4,
    1996: 156.9,
    1997: 160.5,
    1998: 163.0,
    1999: 166.6,
    2000: 172.2,
    2001: 177.1,
    2002: 179.9,
    2003: 184.0,
    2004: 188.9,
    2005: 195.3,
    2006: 201.6,
    2007: 207.3,
    2008: 215.3,
    2009: 214.5,
    2010: 218.1,
    2011: 224.9,
    2012: 229.6,
    2013: 233.0,
    2014: 236.7,
    2015: 237.0,
    2016: 240.0,
    2017: 245.1,
    2018: 251.1,
    2019: 255.7,
    2020: 258.8,
    2021: 271.0,
    2022: 292.7,
    2023: 304.7,
    2024: 314.2,
}


# --- Tool: search_tables ---
def search_tables(query, year=None):
    keywords = query.lower().split()
    results = []
    dirs = [
        Path(os.environ.get("RESOURCES_DIR", "/app/resources")),
        Path(os.environ.get("CORPUS_DIR", "/app/corpus")),
    ]
    for d in dirs:
        if not d.exists():
            continue
        for fp in sorted(d.iterdir()):
            if not fp.is_file():
                continue
            try:
                text = fp.read_text(errors="replace")
            except Exception:
                continue
            lower = text.lower()
            score = sum(lower.count(kw) for kw in keywords)
            if score == 0:
                continue
            # Extract metadata
            title = ""
            units = ""
            years_found = sorted(set(int(m) for m in re.findall(r"\b(19\d{2}|20\d{2})\b", text)))
            for line in text.split("\n")[:10]:
                if line.strip() and not title:
                    title = line.strip()[:120]
                if (
                    re.search(r"(million|billion|thousand|percent|dollar)", line, re.I)
                    and not units
                ):
                    m = re.search(r"(millions?|billions?|thousands?|percent|dollars?)", line, re.I)
                    if m:
                        units = m.group(0)
            period_signals = []
            for sig in ["fiscal year", "calendar year", "quarterly", "monthly", "annual"]:
                if sig in lower:
                    period_signals.append(sig)
            yr_range = f"{years_found[0]}-{years_found[-1]}" if years_found else ""
            if year and years_found and year not in years_found:
                score -= 2
            results.append(
                (
                    score,
                    {
                        "file": str(fp),
                        "table_title": title,
                        "units": units,
                        "year_range": yr_range,
                        "period_signals": period_signals,
                    },
                )
            )
    results.sort(key=lambda x: -x[0])
    return [r[1] for r in results[:5]]


# --- Tool: query_table_rows ---
def query_table_rows(file_path, row_filter=None, year=None):
    fp = Path(file_path)
    if not fp.exists():
        return f"ERROR: file not found: {file_path}"
    text = fp.read_text(errors="replace")
    lines = text.split("\n")
    # Detect delimiter: tab or |
    delim = "\t" if "\t" in text else "|"
    # Find header row (first row with multiple columns)
    header_idx = None
    for i, line in enumerate(lines):
        parts = [c.strip() for c in line.split(delim) if c.strip()]
        if len(parts) >= 2:
            header_idx = i
            break
    if header_idx is None:
        return text[:3000]
    headers = [c.strip() for c in lines[header_idx].split(delim)]
    # Filter columns by year if requested
    col_indices = list(range(len(headers)))
    if year:
        yr_str = str(year)
        col_indices = [0]  # always keep label column
        for j in range(1, len(headers)):
            if yr_str in headers[j]:
                col_indices.append(j)
        if len(col_indices) == 1:
            col_indices = list(range(len(headers)))  # fallback
    # Vertical serialization
    out = []
    for i in range(header_idx + 1, len(lines)):
        row = [c.strip() for c in lines[i].split(delim)]
        if not any(c for c in row):
            continue
        label = row[0] if row else ""
        if row_filter and row_filter.lower() not in label.lower():
            continue
        for j in col_indices[1:]:
            if j < len(row) and row[j]:
                col_name = headers[j] if j < len(headers) else f"col{j}"
                out.append(f"{label}, {col_name}: {row[j]}")
    return "\n".join(out) if out else text[:3000]


# --- Tool: compute_expression ---
def _safe_eval(expression, variables=None):
    variables = variables or {}

    def _collect(args):
        r = []
        for a in args:
            if isinstance(a, ast.List):
                r.extend(_ev(e) for e in a.elts)
            else:
                r.append(_ev(a))
        return r

    def _ev(node):
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id in variables:
                return float(variables[node.id])
            raise ValueError(f"unknown variable: {node.id}")
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -_ev(node.operand)
            if isinstance(node.op, ast.UAdd):
                return _ev(node.operand)
        if isinstance(node, ast.BinOp):
            L, R = _ev(node.left), _ev(node.right)
            op = node.op
            if isinstance(op, ast.Add):
                return L + R
            if isinstance(op, ast.Sub):
                return L - R
            if isinstance(op, ast.Mult):
                return L * R
            if isinstance(op, ast.Div):
                if R == 0:
                    raise ValueError("division by zero")
                return L / R
            if isinstance(op, ast.Pow):
                return L**R
            if isinstance(op, ast.Mod):
                return L % R
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn = node.func.id
            if fn == "abs":
                return abs(_ev(node.args[0]))
            if fn == "round":
                a = [_ev(x) for x in node.args]
                return round(a[0], int(a[1])) if len(a) == 2 else round(a[0])
            if fn in ("min", "max"):
                return min(_collect(node.args)) if fn == "min" else max(_collect(node.args))
            if fn == "sqrt":
                return math.sqrt(_ev(node.args[0]))
            if fn == "log":
                a = [_ev(x) for x in node.args]
                return math.log(*a)
            if fn == "exp":
                return math.exp(_ev(node.args[0]))
            if fn == "sum":
                return sum(_collect(node.args))
            if fn == "mean":
                v = _collect(node.args)
                return sum(v) / len(v)
            if fn == "stdev":
                return statistics.stdev(_collect(node.args))
            if fn == "pstdev":
                return statistics.pstdev(_collect(node.args))
            if fn == "median":
                return statistics.median(_collect(node.args))
            if fn == "pct_change":
                a = [_ev(x) for x in node.args]
                if a[0] == 0:
                    raise ValueError("division by zero")
                return (a[1] - a[0]) / a[0] * 100
            if fn == "cagr":
                a = [_ev(x) for x in node.args]
                return ((a[1] / a[0]) ** (1.0 / a[2]) - 1) * 100
            if fn == "variance":
                return statistics.variance(_collect(node.args))
            if fn == "pvariance":
                return statistics.pvariance(_collect(node.args))
            if fn == "geometric_mean":
                return statistics.geometric_mean(_collect(node.args))
            if fn == "harmonic_mean":
                return statistics.harmonic_mean(_collect(node.args))
            if fn == "correlation":
                a = _collect(node.args)
                n = len(a) // 2
                return statistics.correlation(a[:n], a[n:])
            if fn == "linear_regression":
                a = _collect(node.args)
                n = len(a) // 2
                r = statistics.linear_regression(a[:n], a[n:])
                return r.slope
        raise ValueError(f"unsupported: {ast.dump(node)}")

    expr = expression.strip().replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    return _ev(tree)


def compute_expression(expression, variables=None):
    try:
        result = _safe_eval(expression, variables)
        return json.dumps({"result": result, "expression": expression})
    except Exception as e:
        return json.dumps({"error": str(e), "expression": expression})


# --- Tool: get_cpi_index ---
def get_cpi_index(year):
    if year in CPI:
        return json.dumps({"year": year, "cpi_u": CPI[year], "base": "1982-84=100"})
    return json.dumps({"error": f"no CPI data for {year}", "range": "1913-2024"})


# --- Tool: verify_answer ---
def verify_answer(question, proposed_answer):
    warnings = []
    valid = True
    # Strip and clean
    ans = proposed_answer.strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        num = float(ans)
    except ValueError:
        return json.dumps({"valid": False, "warnings": ["answer is not a valid number"]})
    # Suspiciously round
    if num != 0 and num == int(num):
        mag = abs(num)
        if mag >= 1_000_000 and mag % 1_000_000 == 0:
            warnings.append("exact millions — check if units are correct")
        if mag >= 1_000_000_000 and mag % 1_000_000_000 == 0:
            warnings.append("exact billions — check if units are correct")
    # Zero answer
    if num == 0:
        warnings.append("answer is zero — double-check")
    # Percentage sanity
    qlower = question.lower()
    if "percent" in qlower or "%" in question:
        if abs(num) > 1000:
            warnings.append("percentage over 1000% — likely wrong units")
    return json.dumps({"valid": valid, "warnings": warnings, "parsed_value": num})


# --- Tool schemas ---
TOOLS = [
    {
        "name": "search_tables",
        "description": "Search for tables in Treasury Bulletin files by keyword",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search keywords"},
                "year": {"type": "integer", "description": "Target year (optional)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_table_rows",
        "description": "Extract rows from a file using vertical serialization (ROW_LABEL, COLUMN: VALUE)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {"type": "string", "description": "Path to the file"},
                "row_filter": {
                    "type": "string",
                    "description": "Only rows matching this substring",
                },
                "year": {"type": "integer", "description": "Only columns for this year"},
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "compute_expression",
        "description": "Safe math evaluator: +,-,*,/,**,%, abs, round, min, max, sum, mean, sqrt, log, exp, stdev, pstdev, median, pct_change, cagr, geometric_mean, harmonic_mean, correlation, linear_regression",
        "inputSchema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate"},
                "variables": {"type": "object", "description": "Variable name-to-value mapping"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_cpi_index",
        "description": "Return CPI-U annual average index for a given year (1913-2024, base 1982-84=100)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "year": {"type": "integer", "description": "Year to look up"},
            },
            "required": ["year"],
        },
    },
    {
        "name": "verify_answer",
        "description": "Verify a proposed numeric answer: checks validity, suspicious rounding, unit plausibility",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The original question"},
                "proposed_answer": {
                    "type": "string",
                    "description": "The proposed answer to verify",
                },
            },
            "required": ["question", "proposed_answer"],
        },
    },
]

# --- Dispatch ---
DISPATCH = {
    "search_tables": lambda args: json.dumps(search_tables(args["query"], args.get("year"))),
    "query_table_rows": lambda args: query_table_rows(
        args["file_path"], args.get("row_filter"), args.get("year")
    ),
    "compute_expression": lambda args: compute_expression(
        args["expression"], args.get("variables")
    ),
    "get_cpi_index": lambda args: get_cpi_index(args["year"]),
    "verify_answer": lambda args: verify_answer(args["question"], args["proposed_answer"]),
}


# --- JSON-RPC handler ---
def handle(msg):
    global _call_count
    method = msg.get("method", "")
    mid = msg.get("id")

    # Notifications (no id) get no response
    if mid is None:
        return None

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "officeqa-v14-mcp", "version": "0.1.0"},
            },
        }

    if method == "ping":
        return {"jsonrpc": "2.0", "id": mid, "result": {}}

    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}

    if method == "tools/call":
        params = msg.get("params", {})
        tool_name = params.get("name", "")
        arguments = params.get("arguments", {})

        # Anti-spin
        _call_count += 1
        if _call_count > STOP_AT:
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": "HARD STOP: 15 tool calls reached. Write answer.txt NOW with your best answer.",
                        }
                    ],
                    "isError": True,
                },
            }

        # Cache check
        cache_key = json.dumps([tool_name, arguments], sort_keys=True)
        if cache_key in _cache:
            text = _cache[cache_key]
            if _call_count >= WARN_AT:
                text += f"\n\nWARNING: {_call_count}/{STOP_AT} calls used. Wrap up and write answer.txt."
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "content": [{"type": "text", "text": text}],
                },
            }

        fn = DISPATCH.get(tool_name)
        if not fn:
            return {
                "jsonrpc": "2.0",
                "id": mid,
                "result": {
                    "content": [{"type": "text", "text": f"unknown tool: {tool_name}"}],
                    "isError": True,
                },
            }

        try:
            text = fn(arguments)
        except Exception as e:
            text = json.dumps({"error": str(e)})

        _cache[cache_key] = text

        if _call_count >= WARN_AT:
            text += (
                f"\n\nWARNING: {_call_count}/{STOP_AT} calls used. Wrap up and write answer.txt."
            )

        return {
            "jsonrpc": "2.0",
            "id": mid,
            "result": {
                "content": [{"type": "text", "text": text}],
            },
        }

    return {
        "jsonrpc": "2.0",
        "id": mid,
        "error": {"code": -32601, "message": f"unknown method: {method}"},
    }


# --- Main loop ---
def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = handle(msg)
        if resp:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
