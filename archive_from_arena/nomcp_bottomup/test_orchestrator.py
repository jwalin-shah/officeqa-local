#!/usr/bin/env python3
"""Test suite for Orchestrator class.

Tests parallelization, computation rules, confidence calculation,
and error handling.
"""

import sys
import time
import unittest
from pathlib import Path

# Add bottomup directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator, SearchTaskOrchestrator, execute_decomposition


class TestOrchestrator(unittest.TestCase):
    """Test base Orchestrator class."""

    def setUp(self):
        """Set up test orchestrator."""

        class MockOrchestrator(Orchestrator):
            """Mock orchestrator for testing."""

            def _process_piece(self, piece):
                """Return mock data based on piece_id."""
                piece_id = piece.get("piece_id")
                year = piece.get("year", 2000)
                # Return value proportional to year for testing
                return {
                    "value": float(year),
                    "confidence": 0.95,
                    "success": True,
                    "source": f"mock/{piece_id}",
                }

        self.orch = MockOrchestrator(max_workers=2)

    def test_validate_decomposition_valid(self):
        """Test validation with valid decomposition."""
        pieces = [
            {"piece_id": 1, "year": 1995},
            {"piece_id": 2, "year": 1996},
        ]
        result = self.orch._validate_decomposition(pieces)
        self.assertTrue(result["valid"])

    def test_validate_decomposition_empty(self):
        """Test validation with empty decomposition."""
        pieces = []
        result = self.orch._validate_decomposition(pieces)
        self.assertFalse(result["valid"])

    def test_validate_decomposition_missing_piece_id(self):
        """Test validation with missing piece_id."""
        pieces = [{"year": 1995}]
        result = self.orch._validate_decomposition(pieces)
        self.assertFalse(result["valid"])

    def test_validate_decomposition_duplicate_ids(self):
        """Test validation with duplicate piece_ids."""
        pieces = [
            {"piece_id": 1, "year": 1995},
            {"piece_id": 1, "year": 1996},
        ]
        result = self.orch._validate_decomposition(pieces)
        self.assertFalse(result["valid"])

    def test_execute_direct(self):
        """Test direct computation (return first value)."""
        decomp = {
            "question_id": 1,
            "question": "What is the value?",
            "pieces": [
                {"piece_id": 1, "year": 1995},
                {"piece_id": 2, "year": 1996},
            ],
            "computation": "direct",
        }
        result = self.orch.execute(decomp)
        self.assertTrue(result["success"])
        self.assertEqual(result["final_value"], 1995.0)

    def test_execute_sum(self):
        """Test sum computation."""
        decomp = {
            "question_id": 1,
            "question": "What is the total?",
            "pieces": [
                {"piece_id": 1, "year": 1995},
                {"piece_id": 2, "year": 1996},
            ],
            "computation": "sum",
        }
        result = self.orch.execute(decomp)
        self.assertTrue(result["success"])
        self.assertEqual(result["final_value"], 1995.0 + 1996.0)

    def test_execute_average(self):
        """Test average computation."""
        decomp = {
            "question_id": 1,
            "question": "What is the average?",
            "pieces": [
                {"piece_id": 1, "year": 1900},
                {"piece_id": 2, "year": 2000},
            ],
            "computation": "average",
        }
        result = self.orch.execute(decomp)
        self.assertTrue(result["success"])
        self.assertEqual(result["final_value"], 1950.0)

    def test_execute_difference(self):
        """Test difference computation."""
        decomp = {
            "question_id": 1,
            "question": "What is the difference?",
            "pieces": [
                {"piece_id": 1, "year": 2000},
                {"piece_id": 2, "year": 1000},
            ],
            "computation": "difference",
        }
        result = self.orch.execute(decomp)
        self.assertTrue(result["success"])
        self.assertEqual(result["final_value"], 1000.0)

    def test_execute_ratio(self):
        """Test ratio computation."""
        decomp = {
            "question_id": 1,
            "question": "What is the ratio?",
            "pieces": [
                {"piece_id": 1, "year": 2000},
                {"piece_id": 2, "year": 1000},
            ],
            "computation": "ratio",
        }
        result = self.orch.execute(decomp)
        self.assertTrue(result["success"])
        self.assertEqual(result["final_value"], 2.0)

    def test_execute_min(self):
        """Test min computation."""
        decomp = {
            "question_id": 1,
            "question": "What is the minimum?",
            "pieces": [
                {"piece_id": 1, "year": 1995},
                {"piece_id": 2, "year": 1996},
            ],
            "computation": "min",
        }
        result = self.orch.execute(decomp)
        self.assertTrue(result["success"])
        self.assertEqual(result["final_value"], 1995.0)

    def test_execute_max(self):
        """Test max computation."""
        decomp = {
            "question_id": 1,
            "question": "What is the maximum?",
            "pieces": [
                {"piece_id": 1, "year": 1995},
                {"piece_id": 2, "year": 1996},
            ],
            "computation": "max",
        }
        result = self.orch.execute(decomp)
        self.assertTrue(result["success"])
        self.assertEqual(result["final_value"], 1996.0)

    def test_execute_with_failed_piece(self):
        """Test execution when one piece fails."""

        class FailingMockOrchestrator(Orchestrator):
            """Mock orchestrator where piece 2 fails."""

            def _process_piece(self, piece):
                piece_id = piece.get("piece_id")
                if piece_id == 2:
                    return {
                        "value": None,
                        "confidence": 0.0,
                        "success": False,
                        "error": "Failed",
                    }
                return {
                    "value": float(piece_id * 1000),
                    "confidence": 0.95,
                    "success": True,
                }

        orch = FailingMockOrchestrator()
        decomp = {
            "question_id": 1,
            "question": "Mixed success",
            "pieces": [
                {"piece_id": 1},
                {"piece_id": 2},
            ],
            "computation": "sum",
        }
        result = orch.execute(decomp)
        self.assertTrue(result["success"])  # Partial success
        self.assertEqual(result["final_value"], 1000.0)  # Only piece 1

    def test_confidence_calculation(self):
        """Test confidence calculation from piece confidences."""
        decomp = {
            "question_id": 1,
            "question": "Test confidence",
            "pieces": [
                {"piece_id": 1, "year": 1995},
                {"piece_id": 2, "year": 1996},
            ],
            "computation": "sum",
        }
        result = self.orch.execute(decomp)
        # Should use minimum confidence from pieces
        self.assertGreater(result["final_confidence"], 0)
        self.assertLessEqual(result["final_confidence"], 1.0)

    def test_parallel_execution_timing(self):
        """Test that pieces execute in parallel."""

        class TimingMockOrchestrator(Orchestrator):
            """Mock that takes time to execute."""

            def _process_piece(self, piece):
                time.sleep(0.1)  # Simulate work
                return {
                    "value": 1.0,
                    "confidence": 0.95,
                    "success": True,
                }

        orch = TimingMockOrchestrator(max_workers=2)
        decomp = {
            "question_id": 1,
            "question": "Timing test",
            "pieces": [{"piece_id": i} for i in range(4)],
            "computation": "sum",
        }

        t0 = time.time()
        result = orch.execute(decomp)
        elapsed = time.time() - t0

        # With 2 workers and 0.1s per task, should take ~0.2s, not 0.4s
        self.assertLess(elapsed, 0.35)
        self.assertTrue(result["success"])

    def test_steps_logged(self):
        """Test that steps are logged."""
        decomp = {
            "question_id": 1,
            "question": "Test logging",
            "pieces": [
                {"piece_id": 1, "year": 1995},
            ],
            "computation": "direct",
        }
        result = self.orch.execute(decomp)
        self.assertGreater(len(result["steps"]), 0)
        # Should have at least: start, validation, piece processing, final result
        self.assertIn("Starting orchestrator", result["steps"][0])


class TestSearchTaskOrchestrator(unittest.TestCase):
    """Test SearchTaskOrchestrator wrapper."""

    def test_with_mock_processor(self):
        """Test SearchTaskOrchestrator with mock processor."""

        def mock_processor(piece):
            return {
                "value": piece.get("year", 0) * 1000000,
                "confidence": 0.95,
                "success": True,
            }

        orch = SearchTaskOrchestrator(mock_processor)
        decomp = {
            "question_id": 1,
            "question": "Test",
            "pieces": [
                {"piece_id": 1, "year": 1995},
                {"piece_id": 2, "year": 1996},
            ],
            "computation": "sum",
        }
        result = orch.execute(decomp)
        self.assertTrue(result["success"])
        expected = 1995000000.0 + 1996000000.0
        self.assertEqual(result["final_value"], expected)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_execute_decomposition(self):
        """Test execute_decomposition convenience function."""

        def processor(piece):
            return {
                "value": piece.get("year", 0),
                "confidence": 0.95,
                "success": True,
            }

        decomp = {
            "question_id": 1,
            "question": "Test",
            "pieces": [
                {"piece_id": 1, "year": 1995},
                {"piece_id": 2, "year": 1996},
            ],
            "computation": "sum",
        }
        result = execute_decomposition(decomp, processor)
        self.assertTrue(result["success"])
        self.assertEqual(result["final_value"], 3991.0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up orchestrator."""

        class SimpleOrchestrator(Orchestrator):
            def _process_piece(self, piece):
                return {
                    "value": 1.0,
                    "confidence": 0.95,
                    "success": True,
                }

        self.orch = SimpleOrchestrator()

    def test_all_pieces_fail(self):
        """Test when all pieces fail."""

        class FailingOrchestrator(Orchestrator):
            def _process_piece(self, piece):
                return {
                    "value": None,
                    "confidence": 0.0,
                    "success": False,
                }

        orch = FailingOrchestrator()
        decomp = {
            "question_id": 1,
            "question": "All fail",
            "pieces": [
                {"piece_id": 1},
                {"piece_id": 2},
            ],
            "computation": "sum",
        }
        result = orch.execute(decomp)
        self.assertFalse(result["success"])
        self.assertIsNone(result["final_value"])

    def test_invalid_computation(self):
        """Test with unknown computation type."""
        decomp = {
            "question_id": 1,
            "question": "Unknown computation",
            "pieces": [
                {"piece_id": 1},
            ],
            "computation": "unknown_operation",
        }
        result = self.orch.execute(decomp)
        # Should fall back to direct but not fail
        self.assertTrue(result["success"])

    def test_ratio_with_zero_divisor(self):
        """Test ratio computation when divisor is zero."""

        class ZeroOrchestrator(Orchestrator):
            def _process_piece(self, piece):
                # piece 1 returns 5, piece 2 returns 0
                if piece.get("piece_id") == 1:
                    return {
                        "value": 5.0,
                        "confidence": 0.95,
                        "success": True,
                    }
                else:
                    return {
                        "value": 0.0,
                        "confidence": 0.95,
                        "success": True,
                    }

        orch = ZeroOrchestrator()
        decomp = {
            "question_id": 1,
            "question": "Ratio with zero",
            "pieces": [
                {"piece_id": 1},
                {"piece_id": 2},
            ],
            "computation": "ratio",
        }
        result = orch.execute(decomp)
        self.assertIsNone(result["final_value"])

    def test_difference_with_single_piece(self):
        """Test difference computation with only one piece."""

        orch = self.orch
        decomp = {
            "question_id": 1,
            "question": "Difference with one piece",
            "pieces": [
                {"piece_id": 1},
            ],
            "computation": "difference",
        }
        result = orch.execute(decomp)
        self.assertIsNone(result["final_value"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
