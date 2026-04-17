#!/usr/bin/env python3
"""Extract with LLM-decided tool calls (SQLite queries, JSON reads)."""

import json
import re
from pathlib import Path

from cpi import annual_cpi
from external_data import lookup_fx
from extract import llm

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.sqlite"
CORPUS = HERE / "corpus_json"


def search_json_table(file_stem: str, search_terms: str) -> str:
    """Search JSON file for rows/tables matching search terms.

    Returns matching tables as readable pipe-delimited format.
    """
    try:
        path = CORPUS / f"{file_stem}.json"
        if not path.exists():
            return json.dumps({"error": f"File not found: {file_stem}"})
        with open(path) as f:
            data = json.load(f)

        search_lower = search_terms.lower()
        results = []

        # Search through elements
        for elem in data.get("elements", []):
            if elem.get("type") != "table":
                continue

            # Check if table title/caption matches
            title_lower = (elem.get("title", "") + " " + elem.get("caption", "")).lower()
            if search_lower not in title_lower:
                continue

            # Found matching table - format as readable text
            output = []
            output.append(f"TABLE: {elem.get('title', 'Untitled')}")
            if elem.get("caption"):
                output.append(f"Caption: {elem.get('caption')}")
            output.append("")

            # Extract column headers
            if elem.get("rows"):
                first_row = elem["rows"][0]
                if first_row.get("cells"):
                    headers = [cell.get("text", "") for cell in first_row["cells"]]
                    output.append(" | ".join(headers))
                    output.append("-" * 60)

                    # Extract all rows
                    for row in elem.get("rows", [])[:30]:  # Limit to 30 rows
                        row_data = [cell.get("text", "") for cell in row.get("cells", [])]
                        output.append(" | ".join(row_data))

            results.append("\n".join(output))

        return "\n\n".join(results)[:4000]  # Limit output
    except Exception as e:
        return f"Error: {str(e)}"


def fetch_cpi(year: int) -> str:
    """Fetch CPI-U value for a given year."""
    try:
        value = annual_cpi(year)
        if value is None:
            return json.dumps({"error": f"CPI not available for {year}"})
        return json.dumps({"year": year, "cpi": value})
    except Exception as e:
        return json.dumps({"error": str(e)})


def fetch_fx(currency: str, year: int) -> str:
    """Fetch FX rate for a currency in a given year."""
    try:
        # lookup_fx expects (base, quote, year, month, day) format
        # For year-level queries, use June 30 as middle of year
        rate = lookup_fx("USD", currency, year, 6, 30)
        if rate is None:
            return json.dumps({"error": f"FX rate not available for {currency} in {year}"})
        return json.dumps({"currency": currency, "year": year, "rate": rate})
    except Exception as e:
        return json.dumps({"error": str(e)})


def run_compute(template: str, values: str) -> str:
    """Execute a compute template with given values.

    values is JSON string like: {"dr_id": {"values": [1, 2, 3]}}
    """
    try:
        values_dict = json.loads(values) if isinstance(values, str) else values
        # compute_execute expects (spec, extractions) but we can evaluate directly
        result = eval(template, {"__builtins__": {}}, values_dict)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


def grep_file(file_stem: str, pattern: str) -> str:
    """Execute grep on a .txt file."""
    try:
        txt_path = CORPUS.parent / "corpus" / f"{file_stem}.txt"
        if not txt_path.exists():
            return f"File not found: {file_stem}"
        # Use grep to find matching lines
        import subprocess

        result = subprocess.run(
            ["grep", "-i", pattern, str(txt_path)], capture_output=True, text=True, timeout=5
        )
        lines = result.stdout.split("\n")[:20]  # Limit to 20 matching lines
        return "\n".join(lines) if lines else f"No matches for '{pattern}'"
    except subprocess.TimeoutExpired:
        return "grep timed out"
    except Exception as e:
        return f"Error: {str(e)}"


def parse_tool_calls(response: str) -> list[tuple[str, str]]:
    """Parse tool calls from LLM response.

    Looks for patterns like:
      grep("pattern", "treasury_bulletin_1941_01")
      fetch_cpi(1940)
      fetch_fx('USD', 1940)
      compute(template, values)
    """
    tools = []

    # Parse grep calls: grep("pattern", "filename")
    grep_calls = re.findall(r"grep\(['\"]([^'\"]+)['\"],\s*['\"]([^'\"]+)['\"]\)", response)
    for pattern, filename in grep_calls:
        tools.append(("grep", f"{filename.strip()}|{pattern.strip()}"))

    # Parse CPI calls: fetch_cpi(1940)
    cpi_calls = re.findall(r"fetch_cpi\((\d+)\)", response)
    for year in cpi_calls:
        tools.append(("fetch_cpi", year))

    # Parse FX calls: fetch_fx('USD', 1940)
    fx_calls = re.findall(r"fetch_fx\(['\"](.*?)['\"],\s*(\d+)\)", response)
    for currency, year in fx_calls:
        tools.append(("fetch_fx", f"{currency}:{year}"))

    # Parse compute calls: compute('template', '{"dr_id": ...}')
    compute_calls = re.findall(r"compute\((.*?)\)", response, re.DOTALL)
    for call in compute_calls:
        tools.append(("run_compute", call.strip()))

    return tools


def extract_structured_with_tools(
    spec: dict,
    per_dr_entries: dict,
    question: str,
    gold_files: list[str],
    verbose: bool = False,
) -> dict | None:
    """Extract with LLM-driven tool calls.

    Flow:
    1. LLM sees gold file names + available tools
    2. LLM outputs tool calls (SQL queries, file reads)
    3. We execute those tools
    4. Inject results back
    5. LLM extracts from results
    """
    data_requests = spec.get("data_requests", [])
    if not data_requests:
        return None

    gold_files_str = ", ".join(gold_files)

    # Step 1: LLM decides what to grep
    system_prompt = f"""You are extracting Treasury Bulletin data using grep.

Gold files available:
{gold_files_str}

You can grep these files using: grep("search_pattern", "filename")

Output your grep commands to find data for each request."""

    user_msg = f"""Question: {question}

Find these values:
{json.dumps([{"id": dr.get("id"), "label": dr.get("label"), "keywords": dr.get("keywords")} for dr in data_requests], indent=2)}

What grep commands would you use?"""

    print("  [planning_greps] LLM deciding what to grep...", flush=True)
    grep_response = llm(system_prompt, user_msg, max_tokens=1000)

    # Parse and execute grep commands
    tool_calls = parse_tool_calls(grep_response)
    tool_results = {}

    print(f"  [greping]        Found {len(tool_calls)} grep calls", flush=True)

    for tool_name, tool_arg in tool_calls:
        if tool_name != "grep":
            continue
        filename, pattern = tool_arg.split("|", 1)
        print(f"    - grep({pattern[:40]}..., {filename})", flush=True)
        result = grep_file(filename, pattern)
        tool_results[f"grep({pattern[:30]})"] = result

    # Step 2: LLM extracts from grep results
    tool_context = "\n\n".join(f"{name}:\n{result}" for name, result in tool_results.items())

    extraction_system = """You are extracting numerical values from grep results.

For each data request ID, output:
REQUEST_ID: value_number [unit] [year]

Example:
request_1: 2602 millions 1940
request_2: 12500

Only output lines where you found values."""

    extraction_user = f"""Question: {question}

Data requests:
{json.dumps([{"id": dr.get("id"), "label": dr.get("label"), "keywords": dr.get("keywords")} for dr in data_requests], indent=2)}

Search results:
{tool_context}

Extract the values."""

    print("  [extraction]     Extracting from grep results...", flush=True)
    extraction_response = llm(extraction_system, extraction_user, max_tokens=500)

    # Parse simple line-based format: REQUEST_ID: value [unit] [year]
    extraction_dict = {}
    for line in extraction_response.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        try:
            req_id, rest = line.split(":", 1)
            req_id = req_id.strip()
            parts = rest.strip().split()
            if parts:
                # Try to parse: value unit year
                value = None
                try:
                    value = float(parts[0])
                except ValueError:
                    continue

                extraction_dict[req_id] = {
                    "values": [value],
                    "unit": parts[1] if len(parts) > 1 else None,
                    "year": int(parts[2]) if len(parts) > 2 else None,
                }
        except Exception:
            pass

    return {
        "extractions": extraction_dict,
    }


if __name__ == "__main__":
    # Test tool parsing
    response = """
Let me search for the relevant table:
search_json_table('treasury_bulletin_1940_01', 'national defense expenditures')

And also check another file:
search_json_table('treasury_bulletin_1941_01', 'veterans administration')

Plus get CPI: fetch_cpi(1940)
"""

    tools = parse_tool_calls(response)
    print("Parsed tools:")
    for tool_name, arg in tools:
        print(f"  {tool_name}: {arg}")
