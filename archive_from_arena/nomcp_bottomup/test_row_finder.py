#!/usr/bin/env python3
"""Test suite for RowFinder component."""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from row_finder import RowFinder


def test_standard_treasury_format():
    """Test finding rows in standard Treasury format."""
    finder = RowFinder()
    rows = [
        {"Fiscal year or month": "1994", "Total receipts": "1350576"},
        {"Fiscal year or month": "1995", "Total receipts": "1350576"},
        {"Fiscal year or month": "1996", "Total receipts": "1413156"},
    ]
    row, idx, conf = finder.find_row_by_period(rows, year=1995, period_type="fiscal")
    assert row is not None
    assert idx == 1
    assert conf > 0.9


def test_fiscal_year_encoding():
    """Test fiscal year encoding (YYYYF format)."""
    finder = RowFinder()
    rows = [
        {"Fiscal year or month": "19941", "Total receipts": "100"},
        {"Fiscal year or month": "19951", "Total receipts": "200"},
        {"Fiscal year or month": "19961", "Total receipts": "300"},
    ]
    row, idx, conf = finder.find_row_by_period(rows, year=1995, period_type="fiscal")
    assert row is not None
    assert idx == 1


def test_monthly_data():
    """Test finding specific months in monthly data."""
    finder = RowFinder()
    rows = [
        {"Fiscal year or month": "Jan. 1995", "Total receipts": "100"},
        {"Fiscal year or month": "Feb. 1995", "Total receipts": "110"},
        {"Fiscal year or month": "Mar. 1995", "Total receipts": "120"},
    ]
    row, idx, conf = finder.find_row_by_period(rows, year=1995, month=2, period_type="calendar")
    assert row is not None
    assert idx == 1
    assert conf > 0.85


def test_mixed_formats():
    """Test finding rows with mixed period formats."""
    finder = RowFinder()
    rows = [
        {"Period": "FY1994", "Value": "100"},
        {"Period": "1995", "Value": "200"},
        {"Period": "Fiscal 1996", "Value": "300"},
    ]
    row, idx, conf = finder.find_row_by_period(rows, year=1995)
    assert row is not None
    assert idx == 1


def test_calendar_vs_fiscal():
    """Test distinguishing between calendar and fiscal years."""
    finder = RowFinder()
    rows = [
        {"date": "CY1995", "value": "100"},
        {"date": "1995", "value": "200"},
        {"date": "FY1995", "value": "300"},
    ]

    row_cy, idx_cy, conf_cy = finder.find_row_by_period(rows, year=1995, period_type="calendar")
    row_fy, idx_fy, conf_fy = finder.find_row_by_period(rows, year=1995, period_type="fiscal")

    assert idx_cy == 0 and conf_cy == 0.99
    assert idx_fy == 2 and conf_fy == 0.99


def test_all_candidates_ranking():
    """Test that candidates are ranked by confidence."""
    finder = RowFinder()
    rows = [
        {"period": "1995", "value": "1"},
        {"period": "FY1995", "value": "2"},
        {"period": "CY1995", "value": "3"},
        {"period": "1994", "value": "4"},
    ]
    candidates = finder.find_all_candidate_rows(rows, year=1995, period_type="fiscal")
    assert len(candidates) >= 2
    assert candidates[0][2] >= candidates[1][2]  # sorted by confidence


def test_parse_month_variants():
    """Test parsing various month formats."""
    finder = RowFinder()
    month_variants = [
        ("Jan. 1995", 1),
        ("January 1995", 1),
        ("january 1995", 1),
        ("1995 - Jan", 1),
        ("1995 - January", 1),
        ("Dec. 1995", 12),
        ("December 1995", 12),
    ]
    for text, expected_month in month_variants:
        parsed = finder.parse_period_string(text)
        assert parsed is not None, f"Failed to parse '{text}'"
        assert parsed["month"] == expected_month, f"Wrong month for '{text}'"


def test_no_match():
    """Test handling when year is not found."""
    finder = RowFinder()
    rows = [
        {"Fiscal year or month": "1994", "value": "100"},
        {"Fiscal year or month": "1996", "value": "200"},
    ]
    row, idx, conf = finder.find_row_by_period(rows, year=1995)
    assert row is None
    assert conf == 0.0


def test_missing_period_values():
    """Test skipping rows with missing or empty period values."""
    finder = RowFinder()
    rows = [
        {"Fiscal year or month": None, "value": "100"},
        {"Fiscal year or month": "", "value": "200"},
        {"Fiscal year or month": "1995", "value": "300"},
    ]
    row, idx, conf = finder.find_row_by_period(rows, year=1995)
    assert row is not None
    assert idx == 2


def test_empty_candidates():
    """Test when no candidates are found."""
    finder = RowFinder()
    rows = [
        {"period": "1994", "value": "1"},
        {"period": "1996", "value": "2"},
    ]
    candidates = finder.find_all_candidate_rows(rows, year=1995)
    assert len(candidates) == 0


def test_parse_fiscal_year_explicit():
    """Test parsing explicit fiscal year notations."""
    finder = RowFinder()
    cases = [
        ("FY1995", 1995, "fiscal", 0.99),
        ("FY 1995", 1995, "fiscal", 0.99),
        ("Fiscal 1995", 1995, "fiscal", 0.95),
        ("Fiscal-1995", 1995, "fiscal", 0.95),
    ]
    for text, exp_year, exp_period, min_conf in cases:
        parsed = finder.parse_period_string(text)
        assert parsed is not None
        assert parsed["year"] == exp_year
        assert parsed["period_type"] == exp_period
        assert parsed["confidence"] >= min_conf


def test_parse_calendar_year_explicit():
    """Test parsing explicit calendar year notations."""
    finder = RowFinder()
    cases = [
        ("CY1995", 1995, "calendar", 0.99),
        ("CY 1995", 1995, "calendar", 0.99),
        ("Calendar 1995", 1995, "calendar", 0.95),
        ("Calendar-1995", 1995, "calendar", 0.95),
    ]
    for text, exp_year, exp_period, min_conf in cases:
        parsed = finder.parse_period_string(text)
        assert parsed is not None
        assert parsed["year"] == exp_year
        assert parsed["period_type"] == exp_period
        assert parsed["confidence"] >= min_conf


def test_score_period_match():
    """Test period matching with different scores."""
    finder = RowFinder()

    # Exact match: fiscal year
    score = finder.score_period_match("19951", 1995, "fiscal")
    assert score >= 0.9

    # Year match but wrong month
    score = finder.score_period_match("Jan. 1995", 1995, "calendar", month=2)
    assert score == 0.0

    # Year match, month match
    score = finder.score_period_match("Feb. 1995", 1995, "calendar", month=2)
    assert score >= 0.85

    # No year match
    score = finder.score_period_match("1994", 1995, "fiscal")
    assert score == 0.0


def test_parse_ambiguous_year():
    """Test parsing ambiguous year formats."""
    finder = RowFinder()

    parsed = finder.parse_period_string("1995")
    assert parsed is not None
    assert parsed["year"] == 1995
    assert parsed["period_type"] == "unknown"
    assert 0.75 <= parsed["confidence"] <= 0.9


def test_period_label_extraction():
    """Test extracting period label from row dict."""
    finder = RowFinder()

    # Standard format
    row = {"Fiscal year or month": "1995", "Total receipts": "1000"}
    label = finder._get_period_label(row)
    assert label == "1995"

    # Alternative column name
    row = {"Period": "FY1995", "Value": "1000"}
    label = finder._get_period_label(row)
    assert label == "FY1995"

    # Fallback to first column
    row = {"Year": "1995", "Value": "1000"}
    label = finder._get_period_label(row)
    assert label == "1995"


def test_empty_rows():
    """Test handling empty rows list."""
    finder = RowFinder()
    row, idx, conf = finder.find_row_by_period([], year=1995)
    assert row is None
    assert idx is None
    assert conf == 0.0


def test_confidence_scores():
    """Test that confidence scores follow the expected order."""
    finder = RowFinder()

    # Explicit fiscal > ambiguous year
    score_explicit = finder.score_period_match("FY1995", 1995, "fiscal")
    score_ambiguous = finder.score_period_match("1995", 1995, "fiscal")
    assert score_explicit > score_ambiguous

    # Year + month > year only
    score_both = finder.score_period_match("Jan. 1995", 1995, "calendar", month=1)
    score_year_only = finder.score_period_match("1995", 1995, "calendar")
    # They're different, year_only gives 0 because month is required
    assert score_both > 0.0


def test_period_type_mismatch_heavy_penalty():
    """Test that fiscal/calendar mismatch is heavily penalized."""
    finder = RowFinder()

    # Searching for calendar year but row is fiscal -> 0.15
    score = finder.score_period_match("FY1995", 1995, "calendar")
    assert score == 0.15, f"Expected 0.15, got {score}"

    # Searching for fiscal year but row is calendar -> 0.15
    score = finder.score_period_match("CY1995", 1995, "fiscal")
    assert score == 0.15, f"Expected 0.15, got {score}"

    # Correct match should be much higher
    score_correct = finder.score_period_match("FY1995", 1995, "fiscal")
    assert score_correct == 0.99
    assert score_correct > score  # correct match >> mismatch


def test_fiscal_year_encoding_conservative_confidence():
    """Test that YYYYF fiscal encoding has conservative parse confidence."""
    finder = RowFinder()

    parsed = finder.parse_period_string("19951")
    assert parsed is not None
    # YYYYF encoding is ambiguous - could be FY or data artifact
    # Conservative: treat as unknown with lower confidence
    assert parsed["period_type"] in ("fiscal", "unknown")
    assert parsed["confidence"] <= 0.80, (
        f"Expected conservative confidence, got {parsed['confidence']}"
    )

    # Explicit FY prefix should still be high confidence
    parsed_fy = finder.parse_period_string("FY1995")
    assert parsed_fy is not None
    assert parsed_fy["period_type"] == "fiscal"
    assert parsed_fy["confidence"] >= 0.95


def test_month_only_label_with_target_month():
    """Test month-only rows match with low confidence when month matches."""
    finder = RowFinder()

    # Month-only label "February" when looking for month=2 in year 1995
    # Row has no year, but month matches -> 0.30
    score = finder.score_period_match("February", 1995, "calendar", month=2)
    assert score == 0.30, f"Expected 0.30, got {score}"

    # Month doesn't match -> 0.0
    score = finder.score_period_match("February", 1995, "calendar", month=3)
    assert score == 0.0

    # Looking for annual data (month=None) with month-only label -> 0.0
    score = finder.score_period_match("February", 1995, "calendar", month=None)
    assert score == 0.0


def test_month_only_rows_in_find():
    """Test finding month-only rows via find_row_by_period."""
    finder = RowFinder()
    rows = [
        {"date": "January", "value": "10"},
        {"date": "February", "value": "20"},
        {"date": "March", "value": "30"},
    ]

    # Should find February with confidence 0.30
    row, idx, conf = finder.find_row_by_period(rows, year=1995, period_type="calendar", month=2)
    assert row is not None
    assert idx == 1
    assert conf == 0.30


if __name__ == "__main__":
    # Run all tests
    import inspect

    test_functions = [
        (name, obj)
        for name, obj in inspect.getmembers(sys.modules[__name__])
        if inspect.isfunction(obj) and name.startswith("test_")
    ]

    print(f"Running {len(test_functions)} tests...")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, test_func in test_functions:
        try:
            test_func()
            print(f"✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {name}: {type(e).__name__}: {e}")
            failed += 1

    print("=" * 70)
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        sys.exit(1)
