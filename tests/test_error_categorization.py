"""Tests for solve.py error categorization (categorize_error function).

Covers:
- Direct pipeline error strings classified correctly
- Unit scaling heuristic (answer off by ~1000x or ~1e6x)
- FY/CY confusion heuristic (question mentions FY/CY, answer moderately off)
- Wrong table heuristic (answer significantly off >50%)
- Verify missed (answer passed pipeline but is still wrong)
- Edge cases: non-numeric answers, years in answers, zero expected values
"""

from solve import categorize_error

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Direct pipeline error strings
# ═══════════════════════════════════════════════════════════════════════════════


class TestDirectErrors:
    """Explicit pipeline error strings are classified into their categories."""

    def test_decompose_failed(self):
        assert categorize_error("DECOMPOSE_FAILED", "100") == "decompose_failed"

    def test_decompose_failed_case_insensitive(self):
        assert categorize_error("decompose_failed", "100") == "decompose_failed"

    def test_retrieve_empty(self):
        assert categorize_error("RETRIEVE_EMPTY", "100") == "retrieve_empty"

    def test_extract_failed(self):
        assert categorize_error("EXTRACT_FAILED", "100") == "extract_failed"

    def test_no_values(self):
        assert categorize_error("NO_VALUES[v1]", "100") == "extract_failed"

    def test_compute_failed(self):
        assert categorize_error("COMPUTE_FAILED: division by zero", "100") == "compute_failed"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Unit scaling heuristic
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnitScaling:
    """When the answer is off by a typical unit conversion factor (~1000x or ~1e6x),
    categorize as unit_scaling."""

    def test_answer_1000x_too_large(self):
        """Answer is ~1000x the expected — likely unit mismatch (raw vs thousands)."""
        assert categorize_error("2,602,000", "2,602") == "unit_scaling"

    def test_answer_1000x_too_small(self):
        """Answer is ~0.001x the expected — likely unit mismatch (millions vs raw)."""
        assert categorize_error("2.602", "2,602") == "unit_scaling"

    def test_answer_1e6x_too_large(self):
        """Answer is ~1e6x the expected — likely millions vs raw."""
        assert categorize_error("2,602,000,000", "2,602") == "unit_scaling"

    def test_answer_1e6x_too_small(self):
        """Answer is ~1e-6x the expected — likely raw vs millions."""
        assert categorize_error("0.002602", "2,602") == "unit_scaling"

    def test_not_unit_scaling_when_close(self):
        """Answer within 50% is not unit_scaling."""
        cat = categorize_error("3,000", "2,602")
        assert cat != "unit_scaling"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. FY/CY confusion heuristic
# ═══════════════════════════════════════════════════════════════════════════════


class TestFYCYConfusion:
    """When the question mentions FY/CY and the answer is moderately off,
    categorize as fy_cy_confusion."""

    def test_fy_cy_moderate_off(self):
        """Answer is ~10% off and context mentions FY/CY."""
        # FY total may differ from CY total by a moderate amount
        assert categorize_error("fiscal year total: 28,000", "25,000") == "fy_cy_confusion"

    def test_cy_context_detected(self):
        """Calendar year context detected in expected answer."""
        assert categorize_error("CY total: 27,000", "25,000") == "fy_cy_confusion"

    def test_no_fy_cy_context_not_confusion(self):
        """Without FY/CY context, moderate off is not fy_cy_confusion."""
        cat = categorize_error("27,000", "25,000")
        assert cat != "fy_cy_confusion"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Wrong table heuristic
# ═══════════════════════════════════════════════════════════════════════════════


class TestWrongTable:
    """When the answer is significantly off (>50%), categorize as wrong_table."""

    def test_answer_2x_too_large(self):
        """Answer is 2x the expected — likely wrong table or row."""
        assert categorize_error("5,204", "2,602") == "wrong_table"

    def test_answer_half_expected(self):
        """Answer is ~0.5x the expected — likely a sub-category not total."""
        assert categorize_error("1,200", "2,602") == "wrong_table"

    def test_answer_3x_too_large(self):
        """Answer is 3x the expected — likely wrong table."""
        assert categorize_error("7,806", "2,602") == "wrong_table"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Verify missed (default)
# ═══════════════════════════════════════════════════════════════════════════════


class TestVerifyMissed:
    """When the pipeline produced a reasonable-looking answer that's still wrong,
    categorize as verify_missed."""

    def test_moderate_off_no_unit_or_fycy(self):
        """Answer is slightly off (<50%) and no unit/FYCY context — verify_missed."""
        cat = categorize_error("2,800", "2,602")
        assert cat == "verify_missed"

    def test_non_numeric_answer(self):
        """Non-numeric answer that doesn't match expected — verify_missed."""
        cat = categorize_error("some text answer", "2,602")
        assert cat == "verify_missed"

    def test_text_answer_mismatch(self):
        """Both are text but don't match — verify_missed."""
        cat = categorize_error("Treasury Department", "Defense Department")
        assert cat == "verify_missed"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Edge cases
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Edge cases for categorize_error."""

    def test_year_in_answer_not_confused_with_number(self):
        """A year like 1940 should not be the primary number extracted."""
        # If got="1940" and expected="2,602", 1940 would be a year and skipped
        # so the function would likely return verify_missed
        cat = categorize_error("1940", "2,602")
        # 1940 is filtered as a year, so no primary number found → verify_missed
        assert cat == "verify_missed"

    def test_zero_expected(self):
        """When expected is 0, avoid division by zero."""
        cat = categorize_error("5", "0")
        # No ratio can be computed → verify_missed or wrong_table
        assert cat in ("verify_missed", "wrong_table")

    def test_both_zero(self):
        """Both answers are 0 but still called as error — verify_missed."""
        cat = categorize_error("0", "0")
        # Both are 0, ratio is 0/0 — can't determine → verify_missed
        assert cat == "verify_missed"

    def test_negative_answer(self):
        """Negative answers are handled correctly."""
        cat = categorize_error("-2,602", "2,602")
        # Ratio is -1.0 — not unit_scaling, not wrong_table → verify_missed
        assert cat == "verify_missed"

    def test_percent_answer(self):
        """Percentage answers are handled."""
        cat = categorize_error("5.2%", "3.1%")
        # Both are small numbers, ratio ~1.67 → wrong_table
        assert cat in ("wrong_table", "verify_missed")

    def test_large_unit_scaling_millions(self):
        """Millions-to-billions unit confusion detected."""
        # expected = 2.6 billion, got = 2,600 million (= 2.6 billion in millions)
        # This is actually the same number, but if one is raw and other is in millions:
        assert categorize_error("2,600,000,000", "2,600") == "unit_scaling"
