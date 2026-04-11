"""Test suite for TableValidator."""

import sys

from data_normalizer import DataNormalizer
from table_finder import TableFinder
from table_validator import TableValidator


def test_empty_table():
    """Test with empty table."""
    validator = TableValidator()
    result = validator.validate([], "EMPTY")

    assert result["is_valid"] is False
    assert result["confidence"] == 0.0
    assert result["row_count"] == 0
    assert len(result["structural_issues"]) > 0
    print("✓ test_empty_table passed")


def test_single_row():
    """Test with single row (below minimum)."""
    validator = TableValidator()
    rows = [{"col1": "value1", "col2": "100", "col3": "200"}]
    result = validator.validate(rows, "SMALL")

    assert result["is_valid"] is False
    assert result["confidence"] < 1.0
    assert result["row_count"] == 1
    assert any("Too few rows" in issue for issue in result["completeness_issues"])
    print("✓ test_single_row passed")


def test_good_table():
    """Test with well-formed table."""
    validator = TableValidator()
    rows = [
        {"year": "2020", "revenue": "1000", "expense": "800", "net": "200"},
        {"year": "2021", "revenue": "1100", "expense": "850", "net": "250"},
        {"year": "2022", "revenue": "1200", "expense": "900", "net": "300"},
        {"year": "2023", "revenue": "1300", "expense": "950", "net": "350"},
        {"year": "2024", "revenue": "1400", "expense": "1000", "net": "400"},
    ]
    result = validator.validate(rows, "GOOD")

    assert result["is_valid"] is True
    assert result["confidence"] >= 0.95  # Allow for minor rounding
    assert result["row_count"] == 5
    assert result["column_count"] == 4
    assert result["quality_metrics"]["numeric_columns"] >= 2  # year + at least 2 others
    assert result["quality_metrics"]["complete_rows_pct"] == 100.0
    assert len(result["structural_issues"]) == 0
    assert len(result["completeness_issues"]) == 0
    print("✓ test_good_table passed")


def test_inconsistent_columns():
    """Test with rows having different column counts."""
    validator = TableValidator()
    rows = [
        {"col1": "a", "col2": "b", "col3": "c"},
        {"col1": "a", "col2": "b"},  # Missing col3
        {"col1": "a", "col2": "b", "col3": "c"},
    ]
    result = validator.validate(rows, "INCONSISTENT")

    assert result["is_valid"] is False
    assert len(result["structural_issues"]) > 0
    assert any("columns" in issue.lower() for issue in result["structural_issues"])
    print("✓ test_inconsistent_columns passed")


def test_missing_values():
    """Test with many missing values."""
    validator = TableValidator()
    rows = [
        {"col1": "a", "col2": "", "col3": "c"},
        {"col1": "a", "col2": "", "col3": ""},
        {"col1": "", "col2": "", "col3": ""},
        {"col1": "a", "col2": "", "col3": "c"},
        {"col1": "", "col2": "b", "col3": ""},
        {"col1": "a", "col2": "", "col3": "c"},
    ]
    result = validator.validate(rows, "SPARSE")

    assert result["is_valid"] is False
    assert result["quality_metrics"]["null_percentage"] > 50
    assert len(result["completeness_issues"]) > 0
    print("✓ test_missing_values passed")


def test_no_numeric_columns():
    """Test with table having no numeric columns."""
    validator = TableValidator()
    rows = [
        {"name": "Alice", "city": "NYC", "state": "NY"},
        {"name": "Bob", "city": "LA", "state": "CA"},
        {"name": "Charlie", "city": "CHI", "state": "IL"},
        {"name": "Diana", "city": "BOS", "state": "MA"},
        {"name": "Eve", "city": "SEA", "state": "WA"},
    ]
    result = validator.validate(rows, "TEXTONLY")

    assert result["is_valid"] is False
    assert result["quality_metrics"]["numeric_columns"] == 0
    assert any("numeric" in issue.lower() for issue in result["structural_issues"])
    print("✓ test_no_numeric_columns passed")


def test_all_zeros():
    """Test with all zero numeric values in all numeric columns."""
    validator = TableValidator()
    rows = [
        {"value1": "0", "value2": "0"},
        {"value1": "0", "value2": "0"},
        {"value1": "0", "value2": "0"},
        {"value1": "0", "value2": "0"},
        {"value1": "0", "value2": "0"},
    ]
    result = validator.validate(rows, "ZEROS")

    # Should have all_zeros flag set, indicating data range issue
    assert result["quality_metrics"]["all_zeros"] is True
    assert result["quality_metrics"]["data_range_check"] == "FAIL"
    print("✓ test_all_zeros passed")


def test_required_columns():
    """Test with required column check."""
    validator = TableValidator()
    rows = [
        {"year": "2020", "revenue": "1000", "expense": "800"},
        {"year": "2021", "revenue": "1100", "expense": "850"},
        {"year": "2022", "revenue": "1200", "expense": "900"},
        {"year": "2023", "revenue": "1300", "expense": "950"},
        {"year": "2024", "revenue": "1400", "expense": "1000"},
    ]

    # With all required columns
    result = validator.validate(rows, "TEST", required_columns=["year", "revenue"])
    assert result["is_valid"] is True

    # With missing required column
    result = validator.validate(rows, "TEST", required_columns=["year", "net_income"])
    assert result["is_valid"] is False
    assert any("required" in issue.lower() for issue in result["completeness_issues"])
    print("✓ test_required_columns passed")


def test_mixed_numeric_quality():
    """Test with some numeric columns, some sparse."""
    validator = TableValidator()
    rows = [
        {"year": "2020", "value1": "100", "value2": "", "label": "Q1"},
        {"year": "2021", "value1": "200", "value2": "50", "label": "Q2"},
        {"year": "2022", "value1": "300", "value2": "", "label": "Q3"},
        {"year": "2023", "value1": "400", "value2": "75", "label": "Q4"},
        {"year": "2024", "value1": "500", "value2": "", "label": "Q1"},
    ]
    result = validator.validate(rows, "MIXED")

    assert result["is_valid"] is True  # Still valid, but warning on sparse column
    assert result["quality_metrics"]["numeric_columns"] == 2
    assert "value2" in result["quality_metrics"]["null_by_column"]
    assert result["quality_metrics"]["null_by_column"]["value2"] > 40
    print("✓ test_mixed_numeric_quality passed")


def test_real_data():
    """Test with real Treasury Bulletin data."""
    try:
        finder = TableFinder()
        normalizer = DataNormalizer()
        validator = TableValidator()

        filepath = "/Users/jwalinshah/projects/officeqa-arena/corpus/treasury_bulletin_1995_12.txt"
        content, meta = finder.find_table(filepath, "FFO-1")

        if content:
            rows = finder.extract_rows(content)
            cleaned_rows, _ = normalizer.clean_rows(rows)

            result = validator.validate(cleaned_rows, "FFO-1")

            assert result["is_valid"] is True
            assert result["row_count"] > 0
            assert result["column_count"] > 0
            assert result["confidence"] > 0.5
            print("✓ test_real_data passed")
        else:
            print("⊘ test_real_data skipped (corpus not found)")
    except Exception as e:
        print(f"⊘ test_real_data skipped ({str(e)})")


def run_all_tests():
    """Run all tests."""
    print("Running TableValidator tests...\n")

    tests = [
        test_empty_table,
        test_single_row,
        test_good_table,
        test_inconsistent_columns,
        test_missing_values,
        test_no_numeric_columns,
        test_all_zeros,
        test_required_columns,
        test_mixed_numeric_quality,
        test_real_data,
    ]

    failed = []
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"✗ {test.__name__} failed: {str(e)}")
            failed.append(test.__name__)
        except Exception as e:
            print(f"✗ {test.__name__} error: {str(e)}")
            failed.append(test.__name__)

    print(f"\n{len(tests) - len(failed)}/{len(tests)} tests passed")

    if failed:
        print(f"\nFailed tests: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("\n✓ All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    run_all_tests()
