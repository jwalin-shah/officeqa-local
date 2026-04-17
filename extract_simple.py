#!/usr/bin/env python3
"""Simple extraction: load gold .txt files, LLM extracts values directly."""

import json
from pathlib import Path

from extract import llm

CORPUS = Path(__file__).resolve().parent / "corpus_json"


def extract_from_files(
    spec: dict, question: str, gold_files: list[str], verbose: bool = False
) -> dict | None:
    """Load gold .txt files, pass to LLM for extraction.

    No tools, no loops - just direct extraction from file content.
    """
    data_requests = spec.get("data_requests", [])
    if not data_requests:
        return None

    # Load gold .txt files
    file_contents = {}
    for file_stem in gold_files:
        txt_path = CORPUS.parent / "corpus" / f"{file_stem}.txt"
        if txt_path.exists():
            try:
                with open(txt_path) as f:
                    content = f.read()
                file_contents[file_stem] = content
                if verbose:
                    print(f"  Loaded {file_stem}: {len(content)} chars", flush=True)
            except Exception as e:
                if verbose:
                    print(f"  Error loading {file_stem}: {e}", flush=True)

    if not file_contents:
        return None

    # System prompt
    system_prompt = """You are extracting Treasury Bulletin data from document text.

For each data request ID, find the value in the documents and output:
REQUEST_ID: value_number [unit] [year]

Example:
dr_1: 2602 millions 1940
dr_2: 12500

Only output lines for requests where you found exact values in the text.
Be literal - search for the actual numbers."""

    # User prompt with documents
    docs_text = "\n\n---\n\n".join(
        f"=== {fname} ===\n{content}" for fname, content in file_contents.items()
    )

    user_prompt = f"""Question: {question}

Find these values:
{json.dumps([{"id": dr.get("id"), "label": dr.get("label"), "keywords": dr.get("keywords")} for dr in data_requests], indent=2)}

Document text:
{docs_text}

Extract the values."""

    # Get LLM extraction
    extraction_response = llm(system_prompt, user_prompt, max_tokens=500)

    # Parse response
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
                try:
                    value = float(parts[0])
                    extraction_dict[req_id] = {
                        "values": [value],
                        "unit": parts[1] if len(parts) > 1 else None,
                        "year": int(parts[2]) if len(parts) > 2 else None,
                    }
                except ValueError:
                    continue
        except Exception:
            pass

    return {"extractions": extraction_dict}


if __name__ == "__main__":
    # Test
    spec = {
        "data_requests": [
            {"id": "dr_1", "label": "defense", "keywords": ["national defense"]},
        ]
    }
    result = extract_from_files(
        spec,
        "What were total expenditures for national defense in 1940?",
        ["treasury_bulletin_1941_01"],
    )
    print("Result:", result)
