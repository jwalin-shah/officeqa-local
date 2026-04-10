"""Tests for reward.py — fuzzy_match_answer() with numeric and text cases."""

import pytest

from reward import (
    extract_final_answer,
    extract_numbers_with_context,
    fuzzy_match_answer,
    has_significant_text,
    normalize_text,
    score_answer,
)

# ── Exact numeric match ─────────────────────────────────────────────────────


def test_exact_integer_match():
    ok, _ = fuzzy_match_answer("600", "600")
    assert ok


def test_exact_decimal_match():
    ok, _ = fuzzy_match_answer("3.14", "3.14")
    assert ok


def test_negative_number_match():
    ok, _ = fuzzy_match_answer("-523", "-523")
    assert ok


# ── Tolerance-based numeric match ────────────────────────────────────────────


def test_within_tolerance():
    ok, _ = fuzzy_match_answer("1000", "1005", tolerance=0.01)
    assert ok


def test_outside_tolerance():
    ok, _ = fuzzy_match_answer("1000", "1060", tolerance=0.05)
    assert not ok


def test_zero_tolerance_exact():
    ok, _ = fuzzy_match_answer("500", "500", tolerance=0.0)
    assert ok


def test_zero_tolerance_close():
    ok, _ = fuzzy_match_answer("500", "501", tolerance=0.0)
    assert not ok


# ── Text match ───────────────────────────────────────────────────────────────


def test_text_match_substring():
    ok, _ = fuzzy_match_answer("March 3 1977", "March 3 1977")
    assert ok


def test_text_match_case_insensitive():
    ok, _ = fuzzy_match_answer("defense", "The answer is Defense spending")
    assert ok


def test_text_no_match():
    ok, _ = fuzzy_match_answer("defense", "education spending")
    assert not ok


# ── Number with commas ───────────────────────────────────────────────────────


def test_comma_number():
    ok, _ = fuzzy_match_answer("1,580", "1580")
    assert ok


def test_comma_large_number():
    ok, _ = fuzzy_match_answer("1,234,567", "1234567")
    assert ok


# ── Unicode minus ────────────────────────────────────────────────────────────


def test_unicode_minus():
    ok, _ = fuzzy_match_answer("−523", "-523")
    assert ok


# ── Multi-number answers ─────────────────────────────────────────────────────


def test_multi_number_match():
    ok, _ = fuzzy_match_answer("[100, 200, 300]", "[100, 200, 300]")
    assert ok


def test_multi_number_partial():
    ok, _ = fuzzy_match_answer("[100, 200]", "[100, 999]", tolerance=0.05)
    assert not ok


# ── Edge cases ───────────────────────────────────────────────────────────────


def test_zero_ground_truth_and_pred():
    ok, _ = fuzzy_match_answer("0", "0")
    assert ok


def test_empty_gt_raises():
    with pytest.raises(ValueError):
        fuzzy_match_answer("", "42")


def test_empty_pred_raises():
    with pytest.raises(ValueError):
        fuzzy_match_answer("42", "")


# ── Helper functions ─────────────────────────────────────────────────────────


def test_normalize_text_unicode_minus():
    assert normalize_text("−5") == "-5"


def test_extract_numbers_with_context():
    nums = extract_numbers_with_context("The budget was 1,234 million")
    values = [n for n, _, _, _ in nums]
    assert 1234.0 in values


def test_extract_final_answer_tags():
    text = "blah blah <FINAL_ANSWER>42</FINAL_ANSWER> more text"
    assert extract_final_answer(text) == "42"


def test_extract_final_answer_no_tags():
    assert extract_final_answer("just some text") == "just some text"


def test_has_significant_text_numeric():
    has, _ = has_significant_text("1234")
    assert not has


def test_has_significant_text_words():
    has, _ = has_significant_text("defense spending 1234")
    assert has


# ── score_answer ─────────────────────────────────────────────────────────────


def test_score_answer_correct():
    assert score_answer("600", "600") == 1.0


def test_score_answer_wrong():
    assert score_answer("600", "999") == 0.0
