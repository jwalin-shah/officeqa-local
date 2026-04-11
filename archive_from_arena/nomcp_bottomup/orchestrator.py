#!/usr/bin/env python3
"""Orchestrator: Manage extraction of MULTIPLE decomposition pieces in parallel.

Coordinates the processing of decomposition pieces:
  1. Receives full decomposition (list of pieces)
  2. Spawns parallel SearchTasks (one per piece)
  3. Collects results
  4. Combines using computation rule
  5. Returns final answer with confidence

Architecture:
  Input decomposition:
    {
      'question_id': 123,
      'pieces': [
        {'piece_id': 1, 'year': 1995, 'table': 'FFO-1', ...},
        {'piece_id': 2, 'year': 1996, 'table': 'FFO-1', ...}
      ],
      'computation': 'sum'
    }

  Orchestrator:
    - Validates decomposition (unique IDs, required fields)
    - Spawns N parallel SearchTasks (one per piece)
    - Collects results with timeouts
    - Applies computation rule (sum, average, difference, ratio, min, max)
    - Calculates confidence (pessimistic: min of pieces)

  Output:
    {
      'question_id': 123,
      'final_value': 2763732.0,
      'final_confidence': 0.98,
      'success': True,
      'computation': 'sum',
      'piece_results': [
        {'piece_id': 1, 'value': 1350576.0, 'confidence': 0.99, ...},
        {'piece_id': 2, 'value': 1413156.0, 'confidence': 0.98, ...}
      ],
      'steps': [
        'Starting orchestrator for question 123',
        'Validation passed: 2 unique pieces',
        'Piece 1: value=1350576.0, confidence=0.99',
        'Piece 2: value=1413156.0, confidence=0.98',
        'Computation: sum([1350576.0, 1413156.0]) = 2763732.0, confidence=0.98',
        'Final result: 2763732.0 (confidence: 0.98)'
      ]
    }
"""

import concurrent.futures
import logging
import math
import sys
from collections.abc import Callable
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("orchestrator")


class Orchestrator:
    """Coordinate extraction of multiple decomposition pieces in parallel."""

    def __init__(self, max_workers: int = 4, timeout_per_piece: int = 180):
        """Initialize orchestrator.

        Args:
            max_workers: Number of parallel workers for piece processing
            timeout_per_piece: Max seconds per piece before timeout
        """
        self.max_workers = max_workers
        self.timeout_per_piece = timeout_per_piece
        self.steps: list[str] = []

    def execute(self, decomposition: dict[str, Any]) -> dict[str, Any]:
        """Execute extraction for all decomposition pieces.

        Args:
            decomposition: Dict containing:
              - question_id: int
              - question: str
              - pieces: List of decomposition pieces
              - computation: str (direct|sum|average|difference|ratio|product|min|max|
                percent_change|geometric_mean|custom)
              - [optional] computation_params: Dict with extra params
                (e.g. {'formula': '(x0 - x1) / x1'} for custom)

        Returns:
            Dict with:
              - question_id
              - final_value: The computed result
              - final_confidence: Confidence in the result
              - success: bool
              - computation: The computation rule used
              - piece_results: List of individual piece results
              - steps: List of processing steps
              - error: Optional error message
        """
        question_id = decomposition.get("question_id")
        question = decomposition.get("question", "")
        pieces = decomposition.get("pieces", [])
        computation = decomposition.get("computation", "direct")
        self._computation_params = decomposition.get("computation_params", {})

        # Log start
        self.steps = []
        self._log(f"Starting orchestrator for question {question_id}")
        self._log(f"Question: {question[:100]}...")
        self._log(f"Decomposition: {len(pieces)} pieces")

        # Validate decomposition
        validation = self._validate_decomposition(pieces)
        if not validation["valid"]:
            return self._error_result(question_id, validation["error"], computation, pieces)

        # Parallelize piece processing
        piece_results = self._parallel_process(pieces)

        # Collect successful results
        values = [
            r["value"] for r in piece_results if r.get("value") is not None and r.get("success")
        ]

        # Check if all pieces failed
        if not values:
            self._log("ERROR: All pieces failed")
            return self._error_result(
                question_id,
                "All pieces returned None",
                computation,
                piece_results,
            )

        # Combine results (pass total piece count for confidence penalty)
        result = self._combine_results(values, piece_results, computation, total_pieces=len(pieces))

        # Build final output
        output = {
            "question_id": question_id,
            "final_value": result["value"],
            "final_confidence": result["confidence"],
            "success": result["success"],
            "computation": computation,
            "piece_results": piece_results,
            "steps": self.steps,
        }

        self._log(f"Final result: {result['value']} (confidence: {result['confidence']})")
        return output

    def _validate_decomposition(self, pieces: list[dict]) -> dict[str, Any]:
        """Validate decomposition has required fields.

        Args:
            pieces: List of decomposition pieces

        Returns:
            Dict with 'valid': bool, 'error': Optional[str]
        """
        if not pieces:
            return {"valid": False, "error": "No decomposition pieces"}

        if not isinstance(pieces, list):
            return {"valid": False, "error": "Pieces must be a list"}

        seen_ids = set()
        for i, piece in enumerate(pieces):
            if not isinstance(piece, dict):
                return {
                    "valid": False,
                    "error": f"Piece {i} is not a dict",
                }

            piece_id = piece.get("piece_id")
            if piece_id is None:
                return {
                    "valid": False,
                    "error": f"Piece {i} missing piece_id",
                }

            if piece_id in seen_ids:
                return {
                    "valid": False,
                    "error": f"Duplicate piece_id: {piece_id}",
                }
            seen_ids.add(piece_id)

        self._log(f"Validation passed: {len(pieces)} unique pieces")
        return {"valid": True}

    def _parallel_process(self, pieces: list[dict]) -> list[dict[str, Any]]:
        """Process pieces in parallel using ThreadPoolExecutor.

        Args:
            pieces: List of decomposition pieces

        Returns:
            List of results, one per piece
        """
        piece_results = []

        def _worker(piece: dict) -> dict[str, Any]:
            """Process single piece and return result."""
            piece_id = piece.get("piece_id")
            try:
                # Call the piece processor (can be overridden by subclass)
                result = self._process_piece(piece)
                result["piece_id"] = piece_id
                return result
            except Exception as e:
                logger.exception(f"Exception in piece {piece_id}")
                return {
                    "piece_id": piece_id,
                    "value": None,
                    "confidence": 0.0,
                    "success": False,
                    "error": str(e),
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(_worker, piece): piece.get("piece_id") for piece in pieces}

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result(timeout=self.timeout_per_piece)
                    piece_results.append(result)
                    self._log(
                        f"Piece {result['piece_id']}: "
                        f"value={result.get('value')}, "
                        f"confidence={result.get('confidence')}"
                    )
                except concurrent.futures.TimeoutError:
                    piece_id = futures[future]
                    self._log(f"Piece {piece_id}: TIMEOUT")
                    piece_results.append(
                        {
                            "piece_id": piece_id,
                            "value": None,
                            "confidence": 0.0,
                            "success": False,
                            "error": "Timeout",
                        }
                    )
                except Exception as e:
                    piece_id = futures[future]
                    logger.exception(f"Error collecting result for piece {piece_id}")
                    piece_results.append(
                        {
                            "piece_id": piece_id,
                            "value": None,
                            "confidence": 0.0,
                            "success": False,
                            "error": str(e),
                        }
                    )

        return piece_results

    def _process_piece(self, piece: dict) -> dict[str, Any]:
        """Process a single decomposition piece.

        This method should be overridden by subclasses to implement
        the actual search/extract logic.

        Default implementation returns a placeholder.

        Args:
            piece: Single decomposition piece

        Returns:
            Dict with:
              - value: extracted value or None
              - confidence: float 0.0-1.0
              - success: bool
              - source: Optional source info
              - [other fields as appropriate]
        """
        raise NotImplementedError("Subclasses must implement _process_piece()")

    def _combine_results(
        self,
        values: list[float],
        piece_results: list[dict],
        computation: str,
        total_pieces: int | None = None,
    ) -> dict[str, Any]:
        """Combine individual piece results into final answer.

        Args:
            values: List of extracted numeric values
            piece_results: Full result dicts (for confidence metadata)
            computation: Computation rule (direct|sum|average|etc)
            total_pieces: Total number of pieces (for confidence penalty)

        Returns:
            Dict with 'value' and 'confidence'
        """

        def _flatten(vals):
            """Flatten lists in values (for monthly series)."""
            flat = []
            for v in vals:
                if isinstance(v, list):
                    flat.extend(v)
                else:
                    flat.append(v)
            return flat if flat else None

        computation_map = {
            "direct": lambda v: v[0] if v else None,
            "sum": lambda v: sum(_flatten(v)) if _flatten(v) else None,
            "average": lambda v: sum(_flatten(v)) / len(_flatten(v)) if _flatten(v) else None,
            "difference": lambda v: abs(v[0] - v[1]) if len(v) >= 2 else None,
            "ratio": lambda v: v[0] / v[1] if len(v) >= 2 and v[1] != 0 else None,
            "percent_change": lambda v: (
                abs((v[1] - v[0]) / v[0]) * 100 if len(v) >= 2 and v[0] != 0 else None
            ),
            "product": lambda v: math.prod(_flatten(v)) if _flatten(v) else None,
            "geometric_mean": lambda v: (
                math.prod(_flatten(v)) ** (1.0 / len(_flatten(v))) if _flatten(v) else None
            ),
            "min": lambda v: min(_flatten(v)) if _flatten(v) else None,
            "max": lambda v: max(_flatten(v)) if _flatten(v) else None,
        }

        # Get computation function
        compute_fn: Callable | None = computation_map.get(computation)

        # Handle 'custom' computation with formula
        if computation == "custom" and self._computation_params.get("formula"):
            formula = self._computation_params["formula"]

            def _custom_compute(v):
                # Build variable mapping: A=v[0], B=v[1], etc.
                env = {
                    "abs": abs,
                    "round": round,
                    "sqrt": math.sqrt,
                    "log": math.log,
                    "pow": pow,
                    "min": min,
                    "max": max,
                    "sum": sum,
                    "math": math,
                }
                for i, val in enumerate(v):
                    env[chr(65 + i)] = val  # A, B, C, ...
                try:
                    return eval(formula, {"__builtins__": {}}, env)
                except Exception as e:
                    logger.warning(f"Custom formula '{formula}' failed: {e}")
                    return None

            compute_fn = _custom_compute
        elif not compute_fn:
            logger.warning(f"Unknown computation: {computation}")
            compute_fn = computation_map["direct"]

        # Apply computation
        try:
            combined_value = compute_fn(values)
        except Exception as e:
            logger.exception(f"Computation error: {e}")
            combined_value = None

        # Calculate confidence
        confidences = [
            r.get("confidence", 0.0)
            for r in piece_results
            if r.get("success") and r.get("value") is not None
        ]
        if not confidences:
            final_confidence = 0.0
        else:
            # Use minimum confidence (pessimistic)
            final_confidence = min(confidences)

            # Boost if all pieces agree (values within 1% relative tolerance)
            # Only compare scalar values; skip if any value is a list (monthly series)
            scalar_values = [v for v in values if isinstance(v, (int, float))]
            if len(scalar_values) > 1:
                max_val = max(abs(v) for v in scalar_values)
                if max_val > 0:
                    relative_variance = max(
                        abs(v - scalar_values[0]) / max_val for v in scalar_values[1:]
                    )
                    if relative_variance < 0.01:
                        final_confidence = min(1.0, final_confidence + 0.1)

            # Penalize if many pieces failed (compare successful vs total requested)
            effective_total = total_pieces if total_pieces is not None else len(piece_results)
            if len(values) < effective_total / 2:
                final_confidence = max(0.0, final_confidence - 0.2)

        self._log(
            f"Computation: {computation}({values}) = {combined_value}, "
            f"confidence={final_confidence}"
        )

        return {
            "value": combined_value,
            "confidence": final_confidence,
            "success": combined_value is not None,
        }

    def _error_result(
        self,
        question_id: int | None,
        error: str,
        computation: str,
        piece_results: list[dict],
    ) -> dict[str, Any]:
        """Create an error result dict.

        Args:
            question_id: Original question ID
            error: Error message
            computation: Computation rule
            piece_results: Partial results if any

        Returns:
            Error result dict
        """
        self._log(f"ERROR: {error}")
        return {
            "question_id": question_id,
            "final_value": None,
            "final_confidence": 0.0,
            "success": False,
            "computation": computation,
            "piece_results": piece_results,
            "steps": self.steps,
            "error": error,
        }

    def _log(self, message: str) -> None:
        """Add a message to steps log and print to stderr."""
        self.steps.append(message)
        logger.info(message)


class SearchTaskOrchestrator(Orchestrator):
    """Orchestrator that uses a SearchTask processor for pieces.

    Usage:
        processor = lambda piece: search_and_extract(piece)
        orch = SearchTaskOrchestrator(processor)
        result = orch.execute(decomposition_dict)
    """

    def __init__(
        self,
        search_task_processor: Callable[[dict], dict[str, Any]],
        max_workers: int = 4,
        timeout_per_piece: int = 180,
    ):
        """Initialize with a search task processor.

        Args:
            search_task_processor: Function that takes a piece dict
              and returns {value, confidence, success, ...}
            max_workers: Number of parallel workers
            timeout_per_piece: Timeout per piece in seconds
        """
        super().__init__(max_workers=max_workers, timeout_per_piece=timeout_per_piece)
        self.search_task_processor = search_task_processor

    def _process_piece(self, piece: dict) -> dict[str, Any]:
        """Process piece using the search task processor."""
        return self.search_task_processor(piece)


# ============================================================================
# Convenience Functions
# ============================================================================


def execute_decomposition(
    decomposition: dict[str, Any],
    search_task_processor: Callable[[dict], dict[str, Any]],
    max_workers: int = 4,
) -> dict[str, Any]:
    """One-shot decomposition executor.

    Args:
        decomposition: Decomposition dict with pieces
        search_task_processor: Function to process each piece
        max_workers: Number of parallel workers

    Returns:
        Final result dict
    """
    orch = SearchTaskOrchestrator(search_task_processor, max_workers=max_workers)
    return orch.execute(decomposition)


if __name__ == "__main__":
    # Example usage for testing
    print("Orchestrator module loaded", file=sys.stderr)

    # Mock processor for testing
    def mock_processor(piece: dict) -> dict[str, Any]:
        """Mock processor that returns fixed values."""
        piece_id = piece.get("piece_id")
        year = piece.get("year", 0)
        return {
            "value": float(year) * 1000000,
            "confidence": 0.95,
            "success": True,
            "source": f"mock/{piece_id}",
        }

    # Test decomposition
    test_decomp = {
        "question_id": 123,
        "question": "What was total receipts in FY 1995 and FY 1996?",
        "pieces": [
            {"piece_id": 1, "year": 1995, "description": "FY 1995 receipts"},
            {"piece_id": 2, "year": 1996, "description": "FY 1996 receipts"},
        ],
        "computation": "sum",
    }

    # Run orchestrator
    orch = SearchTaskOrchestrator(mock_processor)
    result = orch.execute(test_decomp)

    print("\n=== ORCHESTRATOR OUTPUT ===", file=sys.stderr)
    print(f"Final value: {result['final_value']}", file=sys.stderr)
    print(f"Confidence: {result['final_confidence']}", file=sys.stderr)
    print(f"Success: {result['success']}", file=sys.stderr)
    print(f"Pieces: {len(result['piece_results'])}", file=sys.stderr)
    print(f"Steps ({len(result['steps'])}): ", file=sys.stderr)
    for step in result["steps"]:
        print(f"  {step}", file=sys.stderr)
