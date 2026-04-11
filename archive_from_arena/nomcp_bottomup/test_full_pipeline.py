#!/usr/bin/env python3
"""Full end-to-end pipeline: Decompose → Orchestrator → LLM Extract → Compute"""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# Add parent dirs to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import orchestrator and solve_decompose components
from solve_decompose import (
    compute,
    extract_value,
    search_data,
    search_tables,
    select_table,
)

from orchestrator import SearchTaskOrchestrator

# Config
CORPUS_DIR = os.environ.get("CORPUS_DIR", "/app/corpus")
DECOMP_FILE = "/Users/jwalinshah/projects/officeqa-arena/decomposition_results_v3.json"


def process_piece_with_extract(piece):
    """Process one piece: search → select → search_data → extract_value (LLM).

    This integrates the orchestrator with the LLM-based extraction from solve_decompose.
    """
    piece_id = piece.get("piece_id")
    description = piece.get("description", "")[:80]

    try:
        # Phase 1: Search tables by keywords
        candidates = search_tables(piece)
        if not candidates:
            return {
                "piece_id": piece_id,
                "value": None,
                "confidence": 0.0,
                "success": False,
                "notes": "No tables found",
            }

        # Phase 2: Select best table
        selected = select_table(piece, candidates)
        if not selected:
            return {
                "piece_id": piece_id,
                "value": None,
                "confidence": 0.0,
                "success": False,
                "notes": "Table selection failed",
            }

        # Phase 3: Extract focused data from table
        focused_data = search_data(piece, selected)
        if not focused_data:
            return {
                "piece_id": piece_id,
                "value": None,
                "confidence": 0.0,
                "success": False,
                "notes": "No data rows found",
            }

        # Phase 4: Use LLM to extract value from focused data
        # This is the key step - calls OpenRouter API with minimax model
        extraction = extract_value(piece, focused_data)

        if extraction.get("values") is None:
            return {
                "piece_id": piece_id,
                "value": None,
                "confidence": 0.0,
                "success": False,
                "notes": f"LLM extraction failed: {extraction.get('notes', 'unknown')}",
            }

        # Convert confidence string to float
        conf_str = extraction.get("confidence", "low").lower()
        if "high" in conf_str or "very" in conf_str:
            conf_float = 0.95
        elif "medium" in conf_str or "moderate" in conf_str:
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
            "notes": extraction.get("notes", ""),
        }

    except Exception as e:
        return {
            "piece_id": piece_id,
            "value": None,
            "confidence": 0.0,
            "success": False,
            "notes": f"Exception: {str(e)[:50]}",
        }


def execute_full_pipeline(question, decomp_result):
    """Execute full pipeline: pre-computed decomposition → orchestrator → LLM extraction → compute.

    Args:
        question: Original question string
        decomp_result: Pre-computed decomposition result dict

    Returns:
        Dict with final answer and confidence
    """
    print(f"\n{'=' * 80}")
    print(f"Question: {question[:70]}...")
    print(f"{'=' * 80}")

    if not decomp_result:
        return {"success": False, "error": "No decomposition provided"}

    # Extract decomposition info
    decomp_data = decomp_result.get("result", {})
    if not decomp_data:
        return {"success": False, "error": "No decomposition data"}

    # Create a single piece from the decomposition result
    # (v3 decompositions are single-piece)
    piece = {
        "piece_id": 1,
        "description": decomp_data.get("topic", ""),
        "data_year": decomp_data.get("data_year"),
        "period_type": decomp_data.get("period_type", "calendar"),
        "search_terms": decomp_data.get("search_terms", []),
        "column_hint": decomp_data.get("column_hint", ""),
    }

    pieces = [piece]
    computation = decomp_data.get("computation", "direct")

    print(f"Computation rule: {computation}")
    print(f"Pieces: {len(pieces)}")

    # Create decomposition dict for orchestrator
    decomposition = {
        "question_id": decomp_result.get("id", 0),
        "question": question,
        "pieces": pieces,
        "computation": computation,
    }

    # Phase 2-4: Use Orchestrator to parallelize piece processing with LLM extraction
    print("\nProcessing pieces in parallel...")
    orchestrator = SearchTaskOrchestrator(
        processor=process_piece_with_extract,
        max_workers=3,
        timeout_per_piece=180,
    )

    orch_result = orchestrator.execute(decomposition)

    if not orch_result.get("success"):
        return {
            "success": False,
            "error": f"Orchestrator failed: {orch_result.get('notes', 'unknown')}",
        }

    # Extract piece results
    piece_results = orch_result.get("piece_results", [])
    successful_pieces = [p for p in piece_results if p.get("success")]
    failed_pieces = [p for p in piece_results if not p.get("success")]

    print(f"\n✅ Successful: {len(successful_pieces)}/{len(piece_results)}")
    for p in successful_pieces:
        print(f"  [Piece {p['piece_id']}] value={p['value']:.0f}, conf={p['confidence']:.2f}")

    if failed_pieces:
        print(f"❌ Failed: {len(failed_pieces)}")
        for p in failed_pieces[:3]:
            print(f"  [Piece {p['piece_id']}] {p.get('notes', 'unknown')[:60]}")

    # Phase 5: Compute final answer if we have data
    if not successful_pieces:
        return {
            "success": False,
            "error": "No pieces could be extracted",
        }

    # Build extracted dict for compute function
    extracted = {}
    for p in piece_results:
        extracted[p.get("piece_id")] = {
            "values": p.get("value"),
            "confidence": p.get("confidence"),
            "success": p.get("success"),
        }

    # Call compute to get final answer
    print(f"\nComputing final answer ({computation})...")
    final_answer = compute(decomposition, extracted)

    print(f"\n✅ Final answer: {final_answer}")

    return {
        "success": True,
        "answer": final_answer,
        "pieces_processed": len(piece_results),
        "pieces_successful": len(successful_pieces),
    }


def main():
    """Run full pipeline on sample decompositions."""
    print(f"\n{'=' * 80}")
    print("FULL PIPELINE TEST: Decompose → Orchestrator → LLM Extract → Compute")
    print(f"{'=' * 80}")

    # Load questions from decomposition results
    if not Path(DECOMP_FILE).exists():
        print(f"ERROR: {DECOMP_FILE} not found")
        return

    with open(DECOMP_FILE) as f:
        all_questions = json.load(f)

    # Sample a few questions
    import random

    sample = random.sample(all_questions, min(5, len(all_questions)))

    results = defaultdict(int)

    for item in sample:
        question = item.get("question", "")

        result = execute_full_pipeline(question)

        if result.get("success"):
            results["success"] += 1
        else:
            results["failed"] += 1
            print(f"❌ Error: {result.get('error', 'unknown')}")

    # Summary
    total = len(sample)
    success = results["success"]

    print(f"\n{'=' * 80}")
    print("PIPELINE SUMMARY:")
    print(f"  ✅ Success: {success}/{total} ({success * 100 / total:.1f}%)")
    print(f"  ❌ Failed:  {results['failed']}/{total} ({results['failed'] * 100 / total:.1f}%)")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
