#!/usr/bin/env python3
"""Integration example: Using Orchestrator with solve_decompose pipeline.

This demonstrates how to refactor the existing solve() pipeline to use
the Orchestrator for parallelizing multiple decomposition pieces.

Before (current solve_decompose.py):
  solve(question)
    → decompose() → sub_queries
    → ThreadPoolExecutor: _process_subquery(sq) in parallel
    → collect results
    → compute()
    → return answer

After (with Orchestrator):
  solve_with_orchestrator(question)
    → decompose() → sub_queries
    → convert to pieces (same data, new structure)
    → Orchestrator.execute(pieces)
      → SearchTaskOrchestrator parallelizes piece processing
      → combines results per computation rule
    → return answer

Benefits:
  1. Cleaner separation of concerns
  2. Reusable orchestration logic
  3. Better composition of pipelines
  4. Easier to test and debug
  5. Flexible computation rules per question
"""

import sys
from typing import Any

# This assumes solve_decompose.py is in the same directory
# In a real integration, adjust imports as needed
# NOTE: In production, you would do:
#   from solve_decompose import decompose, search_tables, select_table, search_data, extract_value, compute, call_llm
# For now, we'll document the integration pattern
from orchestrator import SearchTaskOrchestrator

# ============================================================================
# Helper: Convert decompose sub_queries to orchestrator pieces
# ============================================================================


def convert_subquery_to_piece(subquery: dict[str, Any], index: int) -> dict[str, Any]:
    """Convert a decompose sub_query to an orchestrator piece.

    Args:
        subquery: Sub-query dict from decompose()
        index: Index for piece_id if not already present

    Returns:
        Piece dict suitable for Orchestrator
    """
    piece = {
        "piece_id": subquery.get("id", index),
        # Copy over all relevant decomposition fields
        **{k: v for k, v in subquery.items() if k != "id"},
    }
    return piece


def convert_decomposition_to_pieces(
    sub_queries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert a list of sub_queries to pieces.

    Args:
        sub_queries: List of sub-queries from decompose()

    Returns:
        List of pieces for Orchestrator
    """
    return [convert_subquery_to_piece(sq, i) for i, sq in enumerate(sub_queries)]


# ============================================================================
# Integration: ProcessPieceAdapter
# ============================================================================


class ProcessPieceAdapter:
    """Adapter that wraps the existing _process_subquery logic.

    This allows the existing search/extract/verify pipeline to work
    with the Orchestrator without major refactoring.

    Usage:
        adapter = ProcessPieceAdapter(
            search_tables_fn=search_tables,
            select_table_fn=select_table,
            search_data_fn=search_data,
            extract_value_fn=extract_value,
            verify_value_fn=verify_value,
        )
        processor = adapter.create_processor()
        orch = SearchTaskOrchestrator(processor)
        result = orch.execute(decomposition)
    """

    def __init__(
        self,
        search_tables_fn,
        select_table_fn,
        search_data_fn,
        extract_value_fn,
        verify_value_fn=None,
    ):
        """Initialize adapter with pipeline functions.

        Args:
            search_tables_fn: Function(subquery) -> List[Dict] of candidates
            select_table_fn: Function(subquery, candidates) -> Dict selected table
            search_data_fn: Function(subquery, selected) -> str focused data
            extract_value_fn: Function(subquery, focused) -> Dict with values/confidence
            verify_value_fn: Optional function to verify extracted values
        """
        self.search_tables_fn = search_tables_fn
        self.select_table_fn = select_table_fn
        self.search_data_fn = search_data_fn
        self.extract_value_fn = extract_value_fn
        self.verify_value_fn = verify_value_fn

    def create_processor(self):
        """Create a processor function for Orchestrator.

        Returns:
            Function that takes a piece and returns {value, confidence, success, ...}
        """

        def _process_piece_wrapper(piece: dict[str, Any]) -> dict[str, Any]:
            """Wrap _process_subquery logic for Orchestrator.

            Args:
                piece: Decomposition piece (former sub_query)

            Returns:
                Dict with value, confidence, success, etc.
            """
            piece_id = piece.get("piece_id")
            desc = piece.get("description", "")[:80]

            print(f"  [{piece_id}] Searching: {desc}", file=sys.stderr)

            try:
                # Phase 2: Search tables
                candidates = self.search_tables_fn(piece)
                if not candidates:
                    print(f"  [{piece_id}] WARNING: No tables found", file=sys.stderr)
                    return {
                        "value": None,
                        "confidence": 0.0,
                        "success": False,
                        "notes": "No tables",
                    }
                print(
                    f"  [{piece_id}] Found {len(candidates)} candidate(s)",
                    file=sys.stderr,
                )

                # Phase 3: Select table
                selected = self.select_table_fn(piece, candidates)
                if not selected:
                    return {
                        "value": None,
                        "confidence": 0.0,
                        "success": False,
                        "notes": "Select failed",
                    }
                print(
                    f"  [{piece_id}] Selected: {selected.get('title', '')[:80]}",
                    file=sys.stderr,
                )

                # Phase 4: Read focused data
                focused = self.search_data_fn(piece, selected)
                if not focused:
                    return {
                        "value": None,
                        "confidence": 0.0,
                        "success": False,
                        "notes": "No data rows",
                    }
                print(
                    f"  [{piece_id}] Loaded {len(focused)} chars",
                    file=sys.stderr,
                )

                # Phase 5: Extract
                extraction = self.extract_value_fn(piece, focused)

                # Phase 5b: Verify (if provided)
                if self.verify_value_fn and extraction.get("values") is not None:
                    extraction = self.verify_value_fn(piece, extraction, focused)

                print(
                    f"  [{piece_id}] Extracted: "
                    f"values={extraction.get('values')}, "
                    f"conf={extraction.get('confidence', '?')}",
                    file=sys.stderr,
                )

                # Retry with next candidate if extraction failed
                if extraction.get("values") is None and len(candidates) > 1:
                    fail_reason = extraction.get("notes", "unknown")
                    print(
                        f"  [{piece_id}] Retrying (reason: {fail_reason[:80]})",
                        file=sys.stderr,
                    )
                    retries = 0
                    for alt in candidates:
                        if alt is selected:
                            continue
                        if retries >= 2:
                            break
                        retries += 1
                        print(
                            f"  [{piece_id}] Trying: {alt.get('title', '')[:60]}",
                            file=sys.stderr,
                        )
                        alt_data = self.search_data_fn(piece, alt)
                        if not alt_data:
                            continue
                        alt_ext = self.extract_value_fn(piece, alt_data)
                        if alt_ext.get("values") is not None:
                            if self.verify_value_fn:
                                alt_ext = self.verify_value_fn(piece, alt_ext, alt_data)
                            print(
                                f"  [{piece_id}] Retry got: {alt_ext.get('values')}",
                                file=sys.stderr,
                            )
                            extraction = alt_ext
                            break

                # Return orchestrator-compatible format
                return {
                    "value": extraction.get("values"),
                    "confidence": self._confidence_to_float(extraction.get("confidence", 0.0)),
                    "success": extraction.get("values") is not None,
                    "notes": extraction.get("notes", ""),
                    "source": extraction.get("source", ""),
                }

            except Exception as e:
                print(f"  [{piece_id}] Exception: {e}", file=sys.stderr)
                return {
                    "value": None,
                    "confidence": 0.0,
                    "success": False,
                    "error": str(e),
                }

        return _process_piece_wrapper

    @staticmethod
    def _confidence_to_float(conf: Any) -> float:
        """Convert confidence value to float 0.0-1.0.

        Handles string formats from extract_value (high/medium/low).
        """
        if isinstance(conf, (int, float)):
            return float(conf)
        if isinstance(conf, str):
            conf_lower = conf.lower()
            if "high" in conf_lower or "very" in conf_lower:
                return 0.95
            elif "medium" in conf_lower or "moderate" in conf_lower:
                return 0.70
            elif "low" in conf_lower:
                return 0.40
        return 0.5  # Default


# ============================================================================
# Example: Refactored solve_with_orchestrator
# ============================================================================


def solve_with_orchestrator(
    question: str,
    decompose_fn,
    search_tables_fn,
    select_table_fn,
    search_data_fn,
    extract_value_fn,
    compute_fn,
    verify_value_fn=None,
    max_workers: int = 4,
) -> str:
    """Refactored solve using Orchestrator for parallelization.

    This is a demonstration of how to integrate Orchestrator with
    the existing solve_decompose pipeline.

    Args:
        question: The original question
        decompose_fn: Function(question) -> decomposition dict
        search_tables_fn: Function(subquery) -> candidates
        select_table_fn: Function(subquery, candidates) -> selected
        search_data_fn: Function(subquery, selected) -> focused text
        extract_value_fn: Function(subquery, focused) -> extraction dict
        compute_fn: Function(plan, extracted) -> final answer
        verify_value_fn: Optional function(subquery, extraction, focused) -> extraction
        max_workers: Number of parallel workers

    Returns:
        Final answer as string
    """
    import time

    t0 = time.time()

    # Phase 1: Decompose
    print("Phase 1: Decomposing...", file=sys.stderr)
    plan = decompose_fn(question)
    if not plan:
        print("FATAL: Decomposition failed", file=sys.stderr)
        return "N/A"

    sub_queries = plan.get("sub_queries", [])
    if not sub_queries:
        print("FATAL: No sub-queries", file=sys.stderr)
        return "N/A"

    print(f"  Decomposed into {len(sub_queries)} sub-queries", file=sys.stderr)

    # Convert to orchestrator pieces
    pieces = convert_decomposition_to_pieces(sub_queries)
    decomposition = {
        "question_id": plan.get("question_id", 0),
        "question": question,
        "pieces": pieces,
        "computation": plan.get("computation", "direct"),
    }

    # Phases 2-6: Use Orchestrator for parallelization
    print("Phases 2-6: Orchestrator processing...", file=sys.stderr)
    adapter = ProcessPieceAdapter(
        search_tables_fn=search_tables_fn,
        select_table_fn=select_table_fn,
        search_data_fn=search_data_fn,
        extract_value_fn=extract_value_fn,
        verify_value_fn=verify_value_fn,
    )
    processor = adapter.create_processor()
    orch = SearchTaskOrchestrator(processor, max_workers=max_workers)
    orch_result = orch.execute(decomposition)

    # Convert orchestrator results back to extracted format for compute()
    extracted = {}
    for piece_result in orch_result["piece_results"]:
        piece_id = piece_result.get("piece_id")
        extracted[piece_id] = {
            "values": piece_result.get("value"),
            "confidence": piece_result.get("confidence"),
            "notes": piece_result.get("notes", ""),
            "success": piece_result.get("success", False),
        }

    # Phase 7: Compute final answer
    print("Phase 7: Computing...", file=sys.stderr)
    answer = compute_fn(plan, extracted)
    print(f"  Computed: {answer}", file=sys.stderr)

    elapsed = time.time() - t0
    print(f"Total: {answer} ({elapsed:.1f}s)", file=sys.stderr)

    return answer


# ============================================================================
# Documentation: Integration Steps
# ============================================================================

"""
INTEGRATION STEPS:

1. In solve_decompose.py, add at the top:
   from orchestrator_integration import (
       convert_decomposition_to_pieces,
       ProcessPieceAdapter,
       solve_with_orchestrator,
   )

2. Refactor solve() to use Orchestrator:

   def solve(question):
       return solve_with_orchestrator(
           question=question,
           decompose_fn=decompose,
           search_tables_fn=search_tables,
           select_table_fn=select_table,
           search_data_fn=search_data,
           extract_value_fn=extract_value,
           compute_fn=compute,
           verify_value_fn=verify_value,
           max_workers=4,
       )

3. For consensus variant, create solve_consensus_with_orchestrator:

   def solve_consensus(question):
       return solve_with_orchestrator(
           question=question,
           decompose_fn=decompose_consensus,
           search_tables_fn=search_tables_multi,
           select_table_fn=select_table,
           search_data_fn=search_data,
           extract_value_fn=extract_value_consensus,
           compute_fn=compute,
           verify_value_fn=verify_value,
           max_workers=4,
       )

ADVANTAGES:

1. Cleaner code: orchestration logic is separate from search/extract
2. Testability: can mock each pipeline function easily
3. Parallelization: automatic via Orchestrator
4. Composition: easy to swap search/extract strategies
5. Metrics: Orchestrator tracks confidence and success rates
6. Debugging: detailed step logs for each piece

BACKWARD COMPATIBILITY:

The existing _process_subquery and solve() can coexist during migration.
Gradually replace solve() calls with solve_with_orchestrator().

FUTURE ENHANCEMENTS:

1. Add retry logic with exponential backoff at orchestrator level
2. Add timeout per piece (currently only on ThreadPoolExecutor)
3. Add batching support (e.g., process 20 questions in parallel)
4. Add result caching to avoid redundant searches
5. Add ML-based confidence adjustment based on question features
"""

if __name__ == "__main__":
    print("Orchestrator integration module loaded", file=sys.stderr)
