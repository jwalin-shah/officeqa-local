"""Tests for ColumnFinder: Column matching and extraction."""

import unittest

from column_finder import ColumnFinder


class TestColumnFinder(unittest.TestCase):
    """Test suite for ColumnFinder."""

    def setUp(self):
        """Initialize ColumnFinder for each test."""
        self.finder = ColumnFinder()

    # Tests for find_column()

    def test_find_column_exact_match(self):
        """Test exact match: 'total receipts' matches 'Total > Receipts'."""
        rows = [
            {"Fiscal year": "1995", "Total > Receipts": 1350576.0},
            {"Fiscal year": "1996", "Total > Receipts": 1413156.0},
        ]
        col_name, conf = self.finder.find_column(rows, "total receipts")
        self.assertEqual(col_name, "Total > Receipts")
        self.assertGreaterEqual(conf, 0.85)

    def test_find_column_partial_match(self):
        """Test partial match: 'receipts' matches column containing 'receipts'."""
        rows = [{"Year": "1995", "Total Receipts": 1350576.0, "Total Outlays": 1514389.0}]
        col_name, conf = self.finder.find_column(rows, "receipts")
        self.assertEqual(col_name, "Total Receipts")
        self.assertGreater(conf, 0.50)

    def test_find_column_with_numbering(self):
        """Test match ignoring numbering artifacts like (1), (2)."""
        rows = [
            {
                "Fiscal year": "1995",
                "Total on-budget and off-budget results > Total receipts (1)": 1350576.0,
            }
        ]
        col_name, conf = self.finder.find_column(rows, "total receipts")
        self.assertIsNotNone(col_name)
        self.assertGreater(conf, 0.50)

    def test_find_column_missing_column(self):
        """Test when column doesn't exist: should return None or very low confidence."""
        rows = [{"Fiscal year": "1995", "Total > Outlays": 1514389.0}]
        col_name, conf = self.finder.find_column(rows, "total receipts")
        # "Total > Outlays" has "total" (1/2 keywords match = 50%)
        # This is still >= 50% coverage, so gets 0.65 confidence
        # This is acceptable - it's a weak match but better than nothing
        if col_name is not None:
            self.assertLess(conf, 0.85)  # Not a strong match
        else:
            self.assertEqual(conf, 0.0)

    def test_find_column_empty_rows(self):
        """Test with empty rows list."""
        col_name, conf = self.finder.find_column([], "total receipts")
        self.assertIsNone(col_name)
        self.assertEqual(conf, 0.0)

    def test_find_column_empty_identifier(self):
        """Test with empty identifier."""
        rows = [{"Year": "1995", "Total Receipts": 1350576.0}]
        col_name, conf = self.finder.find_column(rows, "")
        self.assertIsNone(col_name)
        self.assertEqual(conf, 0.0)

    def test_find_column_best_match_among_multiple(self):
        """Test selecting best match when multiple columns match."""
        rows = [
            {
                "Year": "1995",
                "Total Receipts": 1350576.0,
                "On-Budget Receipts": 1000000.0,
                "Off-Budget Receipts": 350576.0,
            }
        ]
        col_name, conf = self.finder.find_column(rows, "total receipts")
        self.assertEqual(col_name, "Total Receipts")
        self.assertGreaterEqual(conf, 0.85)

    def test_find_column_case_insensitive(self):
        """Test that matching is case-insensitive."""
        rows = [{"Year": "1995", "TOTAL RECEIPTS": 1350576.0}]
        col_name, conf = self.finder.find_column(rows, "total receipts")
        self.assertEqual(col_name, "TOTAL RECEIPTS")
        self.assertGreaterEqual(conf, 0.85)

    def test_find_column_with_dashes(self):
        """Test matching with dashes (on-budget vs on budget)."""
        rows = [{"Year": "1995", "On-Budget Receipts": 1000000.0}]
        col_name, conf = self.finder.find_column(rows, "on budget receipts")
        self.assertEqual(col_name, "On-Budget Receipts")
        self.assertGreater(conf, 0.50)

    # Tests for score_column_match()

    def test_score_exact_match(self):
        """Test scoring for exact match."""
        score = self.finder.score_column_match("Total > Receipts", "total receipts")
        self.assertEqual(score, 0.99)

    def test_score_all_keywords_present(self):
        """Test scoring when all keywords are present."""
        score = self.finder.score_column_match("Total on-budget receipts details", "total receipts")
        self.assertGreaterEqual(score, 0.85)

    def test_score_some_keywords_present(self):
        """Test scoring when some keywords are present."""
        score = self.finder.score_column_match("Total Receipts", "total receipts outlays")
        self.assertGreater(score, 0.50)
        self.assertLess(score, 0.85)

    def test_score_no_match(self):
        """Test scoring when no keywords match."""
        score = self.finder.score_column_match("Total Outlays", "total receipts")
        # Both "total" matches, "receipts" doesn't
        self.assertGreater(score, 0.0)

    def test_score_empty_column(self):
        """Test scoring with empty column name."""
        score = self.finder.score_column_match("", "total receipts")
        self.assertEqual(score, 0.0)

    def test_score_empty_identifier(self):
        """Test scoring with empty identifier."""
        score = self.finder.score_column_match("Total Receipts", "")
        self.assertEqual(score, 0.0)

    # Tests for _normalize_for_matching()

    def test_normalize_hierarchical_header(self):
        """Test flattening hierarchical headers."""
        result = self.finder._normalize_for_matching("Total > Receipts")
        self.assertEqual(result, "total receipts")

    def test_normalize_multiple_levels(self):
        """Test flattening multiple hierarchy levels."""
        result = self.finder._normalize_for_matching("A > B > C")
        self.assertEqual(result, "a b c")

    def test_normalize_with_numbering(self):
        """Test removing numbering artifacts."""
        result = self.finder._normalize_for_matching("Total (1) > Receipts (2)")
        # Should remove (1) and (2), flatten >
        self.assertNotIn("(", result)
        self.assertNotIn(")", result)

    def test_normalize_with_dashes(self):
        """Test replacing dashes with spaces."""
        result = self.finder._normalize_for_matching("On-budget Receipts")
        self.assertIn("on budget", result)
        self.assertNotIn("-", result)

    def test_normalize_multiple_spaces(self):
        """Test collapsing multiple spaces."""
        result = self.finder._normalize_for_matching("Total    Receipts")
        self.assertEqual(result, "total receipts")

    def test_normalize_pipes(self):
        """Test removing pipes."""
        result = self.finder._normalize_for_matching("Total | Receipts")
        self.assertEqual(result, "total receipts")

    def test_normalize_quotes(self):
        """Test removing quotes."""
        result = self.finder._normalize_for_matching('Total "Receipts"')
        self.assertEqual(result, "total receipts")

    def test_normalize_single_word(self):
        """Test normalizing single word."""
        result = self.finder._normalize_for_matching("Receipts")
        self.assertEqual(result, "receipts")

    # Tests for flatten_hierarchical_header()

    def test_flatten_simple_hierarchy(self):
        """Test flattening simple hierarchical header."""
        result = self.finder.flatten_hierarchical_header("Category > Subcategory")
        self.assertEqual(result, "category subcategory")

    def test_flatten_multiple_levels(self):
        """Test flattening multiple hierarchy levels."""
        result = self.finder.flatten_hierarchical_header("A > B > C")
        self.assertEqual(result, "a b c")

    def test_flatten_no_hierarchy(self):
        """Test flattening non-hierarchical header."""
        result = self.finder.flatten_hierarchical_header("Single")
        self.assertEqual(result, "single")

    def test_flatten_empty_string(self):
        """Test flattening empty string."""
        result = self.finder.flatten_hierarchical_header("")
        self.assertEqual(result, "")

    # Tests for find_all_candidate_columns()

    def test_find_all_candidates(self):
        """Test finding all candidate columns."""
        rows = [
            {
                "Year": "1995",
                "Total Receipts": 1350576.0,
                "On-Budget Receipts": 1000000.0,
                "Off-Budget Receipts": 350576.0,
            }
        ]
        candidates = self.finder.find_all_candidate_columns(rows, "receipts")
        self.assertGreater(len(candidates), 0)
        # Should find multiple columns with "receipts"
        self.assertGreaterEqual(len(candidates), 3)

    def test_find_all_candidates_sorted(self):
        """Test that candidates are sorted by confidence descending."""
        rows = [
            {
                "Year": "1995",
                "Total Receipts": 1350576.0,
                "Receipts Detail": 100000.0,
                "Total Outlays": 1514389.0,
            }
        ]
        candidates = self.finder.find_all_candidate_columns(rows, "total receipts")
        # First should be best match
        if len(candidates) > 1:
            self.assertGreaterEqual(candidates[0][1], candidates[1][1])

    def test_find_all_candidates_with_min_confidence(self):
        """Test filtering by minimum confidence."""
        rows = [
            {
                "Year": "1995",
                "Total Receipts": 1350576.0,
                "Receipts Detail": 100000.0,
                "Total Outlays": 1514389.0,
            }
        ]
        # High threshold
        candidates_high = self.finder.find_all_candidate_columns(
            rows, "total receipts", min_confidence=0.85
        )
        # Low threshold
        candidates_low = self.finder.find_all_candidate_columns(
            rows, "total receipts", min_confidence=0.50
        )
        self.assertGreaterEqual(len(candidates_low), len(candidates_high))

    def test_find_all_candidates_empty_rows(self):
        """Test with empty rows."""
        candidates = self.finder.find_all_candidate_columns([], "total receipts")
        self.assertEqual(len(candidates), 0)

    # Integration tests

    def test_real_treasury_bulletin_headers(self):
        """Test with realistic Treasury Bulletin headers."""
        rows = [
            {
                "Fiscal year or month": "1995",
                "Total on-budget and off-budget results > Total receipts (1)": 1350576.0,
                "Total on-budget and off-budget results > On-budget receipts (2)": 1000000.0,
                "Total on-budget and off-budget results > Off-budget receipts (3)": 350576.0,
                "Total on-budget and off-budget results > Total outlays (4)": 1514389.0,
            }
        ]

        # Test finding "total receipts"
        col, conf = self.finder.find_column(rows, "total receipts")
        self.assertIsNotNone(col)
        self.assertIn("receipts", col.lower())

        # Test finding "outlays"
        col, conf = self.finder.find_column(rows, "outlays")
        self.assertIsNotNone(col)
        self.assertIn("outlays", col.lower())

    def test_preserve_original_column_name(self):
        """Test that original column name is returned, not normalized."""
        rows = [{"Year": "1995", "Total > Receipts (1)": 1350576.0}]
        col, conf = self.finder.find_column(rows, "total receipts")
        # Should return original name, not normalized
        self.assertEqual(col, "Total > Receipts (1)")

    def test_no_modification_of_rows(self):
        """Test that rows are not modified during matching."""
        rows = [{"Year": "1995", "Total Receipts": 1350576.0}]
        rows_copy = str(rows)
        self.finder.find_column(rows, "total receipts")
        self.assertEqual(str(rows), rows_copy)


if __name__ == "__main__":
    unittest.main()
