#!/usr/bin/env python3
"""Minimal agent loop for extraction with tool-calling.

LLM loops: plan tools → execute → observe → loop until extracted.
No external framework, just conversation state + tool execution.
"""

import json
import re

from extract import llm
from extract_with_tools import fetch_cpi, fetch_fx, grep_file


def execute_tool(tool_name: str, **kwargs) -> dict:
    """Execute a single tool call."""
    try:
        if tool_name == "grep":
            result = grep_file(kwargs["filename"], kwargs["pattern"])
            lines = result.split("\n")[:20]
            return {"tool": "grep", "result": "\n".join(lines), "status": "ok"}
        elif tool_name == "fetch_cpi":
            value = fetch_cpi(kwargs["year"])
            return {"tool": "fetch_cpi", "result": str(value), "status": "ok"}
        elif tool_name == "fetch_fx":
            value = fetch_fx(kwargs["currency"], kwargs["year"])
            return {"tool": "fetch_fx", "result": str(value), "status": "ok"}
        else:
            return {"tool": tool_name, "result": "", "status": "unknown_tool"}
    except Exception as e:
        return {"tool": tool_name, "result": "", "status": "error", "error": str(e)}


def parse_tool_calls_from_llm(response: str) -> list[tuple[str, dict]]:
    """Parse tool calls from LLM response.

    Expected formats:
    - grep("pattern", "filename")
    - fetch_cpi(1940)
    - fetch_fx("GBP", 1940)
    """
    tools = []

    # grep("pattern", "filename")
    grep_calls = re.findall(r'grep\(["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\)', response)
    for pattern, filename in grep_calls:
        tools.append(("grep", {"pattern": pattern, "filename": filename}))

    # fetch_cpi(1940)
    cpi_calls = re.findall(r"fetch_cpi\((\d+)\)", response)
    for year in cpi_calls:
        tools.append(("fetch_cpi", {"year": int(year)}))

    # fetch_fx("GBP", 1940)
    fx_calls = re.findall(r'fetch_fx\(["\']([A-Z]+)["\']\s*,\s*(\d+)\)', response)
    for currency, year in fx_calls:
        tools.append(("fetch_fx", {"currency": currency, "year": int(year)}))

    return tools


def extract_with_agent_loop(
    spec: dict,
    question: str,
    gold_files: list[str],
    max_iterations: int = 5,
    verbose: bool = False,
) -> dict | None:
    """Extract values using agent loop with tool calling.

    Flow:
    1. LLM sees question + data_requests + available tools
    2. LLM generates tool calls (grep, fetch_cpi, fetch_fx)
    3. Execute tools, collect observations
    4. Feed observations back to LLM (same conversation)
    5. Repeat up to max_iterations
    6. LLM extracts final values
    """
    data_requests = spec.get("data_requests", [])
    if not data_requests:
        return None

    gold_files_str = ", ".join(gold_files)

    # System prompt for agent
    system_prompt = f"""You are extracting Treasury Bulletin data using available tools.

Gold files (search these):
{gold_files_str}

Available tools:
- grep("pattern", "filename") - search .txt files
- fetch_cpi(year) - get inflation index
- fetch_fx("CURRENCY", year) - get exchange rate

Data to find:
{json.dumps([{"id": dr.get("id"), "label": dr.get("label"), "keywords": dr.get("keywords")} for dr in data_requests], indent=2)}

Loop strategy:
1. Plan what searches you need
2. Call tools: grep("pattern", "file"), fetch_cpi(year), etc.
3. Look at results
4. If you have values, output:
   EXTRACTED:
   request_id: value unit year

5. Otherwise, continue searching

Be efficient - plan all searches at once, not one at a time."""

    conversation = []

    # Initial user message
    user_msg = f"""Question: {question}

Find the values for each data_request. Start by calling grep() on the gold files to search for relevant data."""

    print("[agent_loop]  Starting extraction loop", flush=True)

    for iteration in range(max_iterations):
        print(f"[iteration {iteration + 1}/{max_iterations}]", flush=True)

        # Add user message to conversation
        conversation.append({"role": "user", "content": user_msg})

        # Get LLM response using the llm() wrapper
        # Build conversation as a single prompt (not multi-turn messages)
        conv_text = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in conversation)
        full_prompt = f"{conv_text}\nASSISTANT:"
        response_text = llm(system_prompt, full_prompt, max_tokens=2000)

        # Check if LLM extracted values
        if "EXTRACTED:" in response_text:
            print("[done]       LLM extracted values", flush=True)
            conversation.append({"role": "assistant", "content": response_text})

            # Parse extraction
            extraction_dict = {}
            for line in response_text.split("\n"):
                if ":" not in line or "EXTRACTED" in line:
                    continue
                try:
                    req_id, rest = line.split(":", 1)
                    req_id = req_id.strip()
                    parts = rest.strip().split()
                    if parts:
                        value = float(parts[0])
                        extraction_dict[req_id] = {
                            "values": [value],
                            "unit": parts[1] if len(parts) > 1 else None,
                            "year": int(parts[2]) if len(parts) > 2 else None,
                        }
                except Exception:
                    pass

            return {"extractions": extraction_dict}

        # Parse tool calls from response
        tool_calls = parse_tool_calls_from_llm(response_text)

        if not tool_calls:
            print("[no_tools]   No tool calls found, stopping", flush=True)
            # Return empty if no tools and no extraction
            return {"extractions": {}}

        print(f"[tools]      Found {len(tool_calls)} tool calls", flush=True)

        # Execute tools
        tool_results = []
        for tool_name, kwargs in tool_calls:
            result = execute_tool(tool_name, **kwargs)
            tool_results.append(result)
            print(f"  - {tool_name}: {kwargs} → {result['status']}", flush=True)

        # Add assistant response and tool results to conversation
        conversation.append({"role": "assistant", "content": response_text})

        # Create observation message
        obs_msg = "Tool results:\n" + json.dumps(tool_results, indent=2)
        user_msg = (
            obs_msg
            + "\n\nContinue: do you have enough data to extract values? If yes, output EXTRACTED: ... If no, call more tools."
        )

    print("[done]       Max iterations reached, no extraction", flush=True)
    return {"extractions": {}}


if __name__ == "__main__":
    # Test
    spec = {
        "data_requests": [
            {"id": "dr1", "label": "defense", "keywords": ["national defense"]},
        ]
    }
    result = extract_with_agent_loop(
        spec,
        "What were total expenditures for national defense in 1940?",
        ["treasury_bulletin_1941_01"],
    )
    print("Result:", result)
