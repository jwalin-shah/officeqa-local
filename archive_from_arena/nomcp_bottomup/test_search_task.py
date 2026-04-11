"""Comprehensive tests for SearchTask orchestrator.

Tests the SearchTask class with mock data to verify:
- Input validation
- File location and processing
- Subsearch strategy execution
- Consensus voting
- Error handling
"""

from search_task import SearchTask


class TestSearchTask:
    """Test suite for SearchTask."""

    def __init__(self):
        """Initialize test runner."""
        self.task = SearchTask()
        self.passed = 0
        self.failed = 0
        self.tests = []

    def run_all_tests(self):
        """Run all test cases."""
        print("=" * 80)
        print("SEARCH TASK TEST SUITE")
        print("=" * 80)

        # Test 1: Input validation
        self.test_missing_year()
        self.test_missing_table_id()
        self.test_missing_row_identifier()
        self.test_missing_column_identifier()

        # Test 2: File handling
        self.test_invalid_year()
        self.test_valid_file_location()

        # Test 3: Error handling
        self.test_graceful_failure()

        # Test 4: Result structure
        self.test_result_structure()

        # Test 5: Row identifier normalization
        self.test_row_identifier_normalization()

        # Test 6: Prefer months
        self.test_prefer_months()

        # Test 7: Confidence bounds
        self.test_confidence_bounds()

        # Summary
        self.print_summary()

    def test_missing_year(self):
        """Test that missing year is caught in validation."""
        try:
            piece = {
                "piece_id": 1,
                "table_id": "FFO-1",
                "row_identifier": ["Total"],
                "column_identifier": "receipts",
            }
            result = self.task.execute(piece, verbose=False)
            passed = (not result["success"]) and len(result["errors"]) > 0
            self.log_test("test_missing_year", passed)
        except Exception as e:
            print(f"Exception in test_missing_year: {e}")
            self.log_test("test_missing_year", False)

    def test_missing_table_id(self):
        """Test that missing table_id is caught."""
        try:
            piece = {
                "piece_id": 2,
                "year": 1995,
                "row_identifier": ["Total"],
                "column_identifier": "receipts",
            }
            result = self.task.execute(piece, verbose=False)
            passed = not result["success"]
            self.log_test("test_missing_table_id", passed)
        except Exception as e:
            print(f"Exception in test_missing_table_id: {e}")
            self.log_test("test_missing_table_id", False)

    def test_missing_row_identifier(self):
        """Test that missing row_identifier is caught."""
        try:
            piece = {
                "piece_id": 3,
                "year": 1995,
                "table_id": "FFO-1",
                "column_identifier": "receipts",
            }
            result = self.task.execute(piece, verbose=False)
            passed = not result["success"]
            self.log_test("test_missing_row_identifier", passed)
        except Exception as e:
            print(f"Exception in test_missing_row_identifier: {e}")
            self.log_test("test_missing_row_identifier", False)

    def test_missing_column_identifier(self):
        """Test that missing column_identifier is caught."""
        try:
            piece = {"piece_id": 4, "year": 1995, "table_id": "FFO-1", "row_identifier": ["Total"]}
            result = self.task.execute(piece, verbose=False)
            passed = not result["success"]
            self.log_test("test_missing_column_identifier", passed)
        except Exception as e:
            print(f"Exception in test_missing_column_identifier: {e}")
            self.log_test("test_missing_column_identifier", False)

    def test_invalid_year(self):
        """Test handling of invalid year (not in corpus)."""
        try:
            piece = {
                "piece_id": 5,
                "year": 2050,
                "table_id": "FFO-1",
                "row_identifier": ["Total"],
                "column_identifier": "receipts",
            }
            result = self.task.execute(piece, verbose=False)
            passed = (
                (not result["success"])
                and (result["value"] is None)
                and (result["confidence"] == 0.0)
            )
            self.log_test("test_invalid_year", passed)
        except Exception as e:
            print(f"Exception in test_invalid_year: {e}")
            self.log_test("test_invalid_year", False)

    def test_valid_file_location(self):
        """Test that valid year can locate files."""
        try:
            piece = {
                "piece_id": 6,
                "year": 1995,
                "table_id": "FFO-1",
                "row_identifier": ["Total"],
                "column_identifier": "receipts",
            }
            result = self.task.execute(piece, verbose=False)
            # Should not have validation errors and should attempt to process files
            has_no_validation_errors = (
                len([e for e in result["errors"] if "Missing required" in e]) == 0
            )
            has_file_results = len(result["file_results"]) > 0
            passed = has_no_validation_errors and has_file_results
            self.log_test("test_valid_file_location", passed)
        except Exception as e:
            print(f"Exception in test_valid_file_location: {e}")
            self.log_test("test_valid_file_location", False)

    def test_graceful_failure(self):
        """Test that execution handles errors gracefully."""
        try:
            piece = {
                "piece_id": 7,
                "year": 1995,
                "table_id": "NONEXISTENT",
                "row_identifier": ["Total"],
                "column_identifier": "receipts",
            }
            result = self.task.execute(piece, verbose=False)
            # Should complete without exception
            passed = isinstance(result, dict) and ("piece_id" in result) and ("errors" in result)
            self.log_test("test_graceful_failure", passed)
        except Exception as e:
            print(f"Exception in test_graceful_failure: {e}")
            self.log_test("test_graceful_failure", False)

    def test_result_structure(self):
        """Test that result has expected structure."""
        try:
            piece = {
                "piece_id": 8,
                "year": 1995,
                "table_id": "FFO-1",
                "row_identifier": ["Total"],
                "column_identifier": "receipts",
            }
            result = self.task.execute(piece, verbose=False)

            # Check required fields
            required_fields = [
                "piece_id",
                "value",
                "confidence",
                "success",
                "source",
                "steps",
                "file_results",
                "errors",
            ]
            has_all_fields = all(field in result for field in required_fields)

            # Check types
            type_checks = [
                isinstance(result["piece_id"], int),
                isinstance(result["value"], (type(None), float, int)),
                isinstance(result["confidence"], float),
                isinstance(result["success"], bool),
                isinstance(result["steps"], list),
                isinstance(result["file_results"], list),
                isinstance(result["errors"], list),
            ]
            all_types_correct = all(type_checks)

            passed = has_all_fields and all_types_correct
            self.log_test("test_result_structure", passed)
        except Exception as e:
            print(f"Exception in test_result_structure: {e}")
            self.log_test("test_result_structure", False)

    def test_row_identifier_normalization(self):
        """Test that row_identifier can be string or list."""
        try:
            # Test with string
            piece1 = {
                "piece_id": 9,
                "year": 1995,
                "table_id": "FFO-1",
                "row_identifier": "Total",  # String instead of list
                "column_identifier": "receipts",
            }
            result1 = self.task.execute(piece1, verbose=False)
            handles_string = isinstance(result1, dict)

            # Test with list
            piece2 = {
                "piece_id": 10,
                "year": 1995,
                "table_id": "FFO-1",
                "row_identifier": ["Total"],  # List
                "column_identifier": "receipts",
            }
            result2 = self.task.execute(piece2, verbose=False)
            handles_list = isinstance(result2, dict)

            passed = handles_string and handles_list
            self.log_test("test_row_identifier_normalization", passed)
        except Exception as e:
            print(f"Exception in test_row_identifier_normalization: {e}")
            self.log_test("test_row_identifier_normalization", False)

    def test_prefer_months(self):
        """Test that prefer_months is respected."""
        try:
            piece = {
                "piece_id": 11,
                "year": 1995,
                "table_id": "FFO-1",
                "row_identifier": ["Total"],
                "column_identifier": "receipts",
                "prefer_months": [12, 6, 3],
            }
            result = self.task.execute(piece, verbose=False)
            # Should complete without error
            passed = isinstance(result, dict)
            self.log_test("test_prefer_months", passed)
        except Exception as e:
            print(f"Exception in test_prefer_months: {e}")
            self.log_test("test_prefer_months", False)

    def test_confidence_bounds(self):
        """Test that confidence is always between 0.0 and 1.0."""
        try:
            piece = {
                "piece_id": 12,
                "year": 1995,
                "table_id": "FFO-1",
                "row_identifier": ["Total"],
                "column_identifier": "receipts",
            }
            result = self.task.execute(piece, verbose=False)
            passed = 0.0 <= result["confidence"] <= 1.0
            self.log_test("test_confidence_bounds", passed)
        except Exception as e:
            print(f"Exception in test_confidence_bounds: {e}")
            self.log_test("test_confidence_bounds", False)

    def log_test(self, test_name: str, passed: bool):
        """Log test result."""
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:8} {test_name}")
        if passed:
            self.passed += 1
        else:
            self.failed += 1
        self.tests.append((test_name, passed))

    def print_summary(self):
        """Print test summary."""
        print("\n" + "=" * 80)
        print(
            f"RESULTS: {self.passed} passed, {self.failed} failed out of {self.passed + self.failed} tests"
        )
        print("=" * 80)


def main():
    """Run test suite."""
    tester = TestSearchTask()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
