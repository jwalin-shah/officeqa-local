#!/usr/bin/env python3
"""End-to-end test: Fresh decompose → Orchestrator with LLM → Final answer"""

import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from solve_decompose import (
    compute,
    decompose,
    extract_value,
    search_data,
    search_tables,
    select_table,
)

from orchestrator import SearchTaskOrchestrator

# Check API key
API_KEY = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("LLM_API_KEY")
if not API_KEY:
    print("ERROR: No API key set (OPENROUTER_API_KEY or LLM_API_KEY)", file=sys.stderr)
    sys.exit(1)

print(f"✓ API Key set: {API_KEY[:20]}...", file=sys.stderr)


def process_piece_with_llm_extraction(piece):
    """Process one decomposition piece: search → select → extract (with LLM).

    This is the core worker function for the orchestrator. Includes retry on first failure.
    """
    piece_id = piece.get("piece_id", "?")

    try:
        # Phase 1: Search tables
        print(f"    [Piece {piece_id}] Searching...", end=" ", flush=True, file=sys.stderr)
        candidates = search_tables(piece)
        if not candidates:
            print("no tables", file=sys.stderr)
            return {
                "piece_id": piece_id,
                "value": None,
                "confidence": 0.0,
                "success": False,
                "notes": "No tables found",
            }
        print(f"found {len(candidates)}", file=sys.stderr)

        # Phase 2: Select best table
        print(f"    [Piece {piece_id}] Selecting...", end=" ", flush=True, file=sys.stderr)
        selected = select_table(piece, candidates)
        if not selected:
            print("select failed", file=sys.stderr)
            return {
                "piece_id": piece_id,
                "value": None,
                "confidence": 0.0,
                "success": False,
                "notes": "Select failed",
            }
        print("selected", file=sys.stderr)

        # Phase 3: Extract focused data
        print(f"    [Piece {piece_id}] Reading data...", end=" ", flush=True, file=sys.stderr)
        focused = search_data(piece, selected)
        if not focused:
            print("no data", file=sys.stderr)
            # Retry with alternate candidate
            for alt in candidates[1:3]:
                print(
                    f"    [Piece {piece_id}] Retrying with alternate...",
                    end=" ",
                    flush=True,
                    file=sys.stderr,
                )
                focused = search_data(piece, alt)
                if focused:
                    print("got data", file=sys.stderr)
                    break
            else:
                print("all failed", file=sys.stderr)
                return {
                    "piece_id": piece_id,
                    "value": None,
                    "confidence": 0.0,
                    "success": False,
                    "notes": "No data (all candidates)",
                }
        print(f"{len(focused)} chars", file=sys.stderr)

        # Phase 4: LLM-based extraction
        print(f"    [Piece {piece_id}] Extracting (LLM)...", end=" ", flush=True, file=sys.stderr)
        extraction = extract_value(piece, focused)
        if extraction.get("values") is None:
            print(f"failed: {extraction.get('notes', 'unknown')[:40]}", file=sys.stderr)
            return {
                "piece_id": piece_id,
                "value": None,
                "confidence": 0.0,
                "success": False,
                "notes": extraction.get("notes", "LLM failed"),
            }

        print(f"value={extraction.get('values')}", file=sys.stderr)

        # Convert confidence
        conf_str = str(extraction.get("confidence", "low")).lower()
        if "high" in conf_str or "very" in conf_str:
            conf_float = 0.95
        elif "medium" in conf_str:
            conf_float = 0.70
        elif "low" in conf_str:
            conf_float = 0.40
        else:
            conf_float = 0.5

        return {
            "piece_id": piece_id,
            "value": extraction.get("values"),
            "confidence": conf_float,
            "success": True,
            "notes": "",
        }

    except Exception as e:
        print(f"exception: {str(e)[:40]}", file=sys.stderr)
        return {
            "piece_id": piece_id,
            "value": None,
            "confidence": 0.0,
            "success": False,
            "notes": str(e)[:60],
        }


def run_e2e(question):
    """Run end-to-end: decompose → orchestrate → compute.

    Args:
        question: Original question string

    Returns:
        Dict with result and success status
    """
    print(f"\n{'─' * 80}")
    print(f"Q: {question[:75]}...")
    print(f"{'─' * 80}")

    # Phase 1: Decompose fresh
    print("  [1/4] Decomposing...", end=" ", flush=True)
    plan = decompose(question)
    if not plan:
        print("FAIL")
        return {"success": False, "error": "Decompose failed"}

    sub_queries = plan.get("sub_queries", [])
    if not sub_queries:
        print("FAIL (no sub-queries)")
        return {"success": False, "error": "No sub-queries"}

    print(f"OK ({len(sub_queries)} pieces)")

    # Convert sub_queries to pieces format
    pieces = []
    for i, sq in enumerate(sub_queries):
        piece = {"piece_id": sq.get("id", i), **{k: v for k, v in sq.items() if k != "id"}}
        pieces.append(piece)

    decomposition = {
        "question_id": 0,
        "question": question,
        "pieces": pieces,
        "computation": plan.get("computation", {}).get("type", "direct"),
    }

    # Phase 2-4: Orchestrator with LLM extraction in parallel
    print("  [2/4] Orchestrating...", end=" ", flush=True)
    orch = SearchTaskOrchestrator(
        search_task_processor=process_piece_with_llm_extraction,
        max_workers=3,
        timeout_per_piece=180,
    )
    result = orch.execute(decomposition)

    piece_results = result.get("piece_results", [])
    successful = [p for p in piece_results if p.get("success")]
    print(f"OK ({len(successful)}/{len(piece_results)} pieces)")

    if not successful:
        print("  [3/4] Computing... FAIL (no data)")
        return {"success": False, "error": "No pieces succeeded"}

    # Phase 5: Compute final answer
    print("  [3/4] Computing...", end=" ", flush=True)

    # Build extracted dict for compute
    extracted = {}
    for p in piece_results:
        extracted[p.get("piece_id")] = {
            "values": p.get("value"),
            "confidence": p.get("confidence"),
            "success": p.get("success"),
        }

    try:
        answer = compute(plan, extracted)
        print("OK")
        print(f"  [4/4] ANSWER: {answer}")
        return {
            "success": True,
            "answer": answer,
            "pieces": len(piece_results),
            "successful": len(successful),
        }
    except Exception as e:
        print(f"FAIL: {str(e)[:40]}")
        return {"success": False, "error": f"Compute failed: {str(e)[:40]}"}


def main():
    print(f"\n{'=' * 80}")
    print("END-TO-END TEST: Fresh Decompose → Orchestrator → LLM Extract → Answer")
    print(f"{'=' * 80}")

    # Load sample questions
    decomp_file = "/Users/jwalinshah/projects/officeqa-arena/decomposition_results_v3.json"
    if not Path(decomp_file).exists():
        print(f"ERROR: {decomp_file} not found")
        return

    with open(decomp_file) as f:
        all_items = json.load(f)

    questions = [item["question"] for item in all_items]
    sample = random.sample(questions, min(5, len(questions)))

    results = defaultdict(int)

    for question in sample:
        result = run_e2e(question)
        if result.get("success"):
            results["success"] += 1
        else:
            results["failed"] += 1

    # Summary
    total = len(sample)
    success = results["success"]

    print(f"\n{'=' * 80}")
    print(f"SUMMARY: {success}/{total} successful ({success * 100 / total:.0f}%)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
