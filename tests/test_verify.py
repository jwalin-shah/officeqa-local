"""Tests for verify.py — mentor/review prompt framing, auto-fix functions."""

from verify import auto_fix_fy_cy, auto_fix_units

# ═══════════════════════════════════════════════════════════════════════════════
# auto_fix_units
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoFixUnits:
    """Tests for auto_fix_units() — deterministic unit-scale mismatch detection."""

    def test_no_mismatch_when_units_match(self):
        """When source and target units match, no correction is needed."""
        result = auto_fix_units(
            answer="2,602",
            source_unit="millions",
            target_unit="millions",
        )
        assert result is None

    def test_no_mismatch_when_target_is_none(self):
        """When target_unit is None, we can't detect a mismatch."""
        result = auto_fix_units(
            answer="2,602",
            source_unit="millions",
            target_unit=None,
        )
        assert result is None

    def test_no_mismatch_when_source_is_none(self):
        """When source_unit is None, we can't detect a mismatch."""
        result = auto_fix_units(
            answer="2,602",
            source_unit=None,
            target_unit="millions",
        )
        assert result is None

    def test_thousands_to_millions(self):
        """Source is thousands but answer should be in millions → divide by 1000."""
        result = auto_fix_units(
            answer="2,602,000",
            source_unit="thousands",
            target_unit="millions",
        )
        assert result is not None
        assert result["corrected"] == "2,602"
        assert "thousands" in result["issue"]
        assert "millions" in result["issue"]

    def test_millions_to_thousands(self):
        """Source is millions but answer should be in thousands → multiply by 1000."""
        result = auto_fix_units(
            answer="2.6",
            source_unit="millions",
            target_unit="thousands",
        )
        assert result is not None
        assert result["corrected"] == "2,600"

    def test_millions_to_billions(self):
        """Source is millions but answer should be in billions → divide by 1000."""
        result = auto_fix_units(
            answer="2,600",
            source_unit="millions",
            target_unit="billions",
        )
        assert result is not None
        assert result["corrected"] == "2.6"

    def test_billions_to_millions(self):
        """Source is billions but answer should be in millions → multiply by 1000."""
        result = auto_fix_units(
            answer="2.6",
            source_unit="billions",
            target_unit="millions",
        )
        assert result is not None
        assert result["corrected"] == "2,600"

    def test_numeric_answer_with_commas(self):
        """Handles comma-formatted numeric answers."""
        result = auto_fix_units(
            answer="1,234,567",
            source_unit="thousands",
            target_unit="millions",
        )
        assert result is not None
        # 1,234,567 thousands = 1,234.567 millions
        assert "1,234.567" in result["corrected"]

    def test_non_numeric_answer_returns_none(self):
        """Non-numeric answers can't be unit-corrected."""
        result = auto_fix_units(
            answer="EXTRACT_FAILED",
            source_unit="thousands",
            target_unit="millions",
        )
        assert result is None

    def test_percent_unit_no_conversion(self):
        """Percent to percent — no mismatch."""
        result = auto_fix_units(
            answer="42.5",
            source_unit="percent",
            target_unit="percent",
        )
        assert result is None

    def test_unrecognized_unit_returns_none(self):
        """Unknown unit keys can't be auto-fixed."""
        result = auto_fix_units(
            answer="100",
            source_unit="gazillions",
            target_unit="millions",
        )
        assert result is None

    def test_returns_dict_with_issue_and_corrected(self):
        """Result dict always has 'issue' and 'corrected' keys."""
        result = auto_fix_units(
            answer="5,000",
            source_unit="thousands",
            target_unit="millions",
        )
        assert result is not None
        assert "issue" in result
        assert "corrected" in result
        assert isinstance(result["issue"], str)
        assert isinstance(result["corrected"], str)


# ═══════════════════════════════════════════════════════════════════════════════
# auto_fix_fy_cy
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoFixFyCy:
    """Tests for auto_fix_fy_cy() — deterministic FY/CY confusion detection."""

    def test_fy_used_for_cy_question(self):
        """When spec says calendar year but extraction used fiscal year, flag it."""
        result = auto_fix_fy_cy(
            spec={"period": "calendar"},
            extractions={
                "v1": {
                    "values": [1200],
                    "labels": ["Fiscal year 1940"],
                    "source_file": "bulletin_1940_06.json",
                }
            },
        )
        assert result is not None
        assert result["flagged"] is True
        assert "fiscal" in result["issue"].lower()
        assert "calendar" in result["issue"].lower()
        assert result["suggested_phase"] == "extract"

    def test_cy_used_for_fy_question(self):
        """When spec says fiscal year but extraction used calendar year, flag it."""
        result = auto_fix_fy_cy(
            spec={"period": "fiscal"},
            extractions={
                "v1": {
                    "values": [1100],
                    "labels": ["Calendar year 1940"],
                    "source_file": "bulletin_1940_12.json",
                }
            },
        )
        assert result is not None
        assert result["flagged"] is True
        assert "calendar" in result["issue"].lower()
        assert "fiscal" in result["issue"].lower()

    def test_no_confusion_when_periods_match(self):
        """When spec is fiscal and extraction is also fiscal, no flag."""
        result = auto_fix_fy_cy(
            spec={"period": "fiscal"},
            extractions={
                "v1": {
                    "values": [1200],
                    "labels": ["Fiscal year 1940"],
                    "source_file": "bulletin_1940_06.json",
                }
            },
        )
        assert result is None

    def test_no_confusion_when_no_period_in_spec(self):
        """When spec has no period info, can't detect FY/CY mismatch."""
        result = auto_fix_fy_cy(
            spec={},
            extractions={
                "v1": {
                    "values": [1200],
                    "labels": ["Fiscal year 1940"],
                    "source_file": "bulletin_1940_06.json",
                }
            },
        )
        assert result is None

    def test_no_confusion_when_no_labels(self):
        """When extraction has no labels, can't detect FY/CY from labels alone."""
        result = auto_fix_fy_cy(
            spec={"period": "calendar"},
            extractions={
                "v1": {
                    "values": [1200],
                    "labels": [],
                    "source_file": "bulletin_1940_06.json",
                }
            },
        )
        assert result is None

    def test_fy_detected_in_row_label(self):
        """FY/CY detection works even with 'FY YYYY' format in labels."""
        result = auto_fix_fy_cy(
            spec={"period": "calendar"},
            extractions={
                "v1": {
                    "values": [500],
                    "labels": ["FY 1940"],
                    "source_file": "bulletin_1940_06.json",
                }
            },
        )
        assert result is not None
        assert result["flagged"] is True

    def test_cy_detected_in_row_label(self):
        """FY/CY detection works with 'CY YYYY' format in labels."""
        result = auto_fix_fy_cy(
            spec={"period": "fiscal"},
            extractions={
                "v1": {
                    "values": [500],
                    "labels": ["CY 1940"],
                    "source_file": "bulletin_1940_12.json",
                }
            },
        )
        assert result is not None
        assert result["flagged"] is True

    def test_returns_dict_with_flagged_issue_suggested_phase(self):
        """Result dict has the expected keys."""
        result = auto_fix_fy_cy(
            spec={"period": "calendar"},
            extractions={
                "v1": {
                    "values": [1200],
                    "labels": ["Fiscal year 1940"],
                    "source_file": "bulletin_1940_06.json",
                }
            },
        )
        assert result is not None
        assert "flagged" in result
        assert "issue" in result
        assert "suggested_phase" in result


# ═══════════════════════════════════════════════════════════════════════════════
# verify_answer fail-open
# ═══════════════════════════════════════════════════════════════════════════════


class TestVerifyFailOpen:
    """Tests that verify_answer returns ok=True on internal errors."""

    def test_verify_returns_ok_true_on_llm_exception(self, monkeypatch):
        """When the LLM call raises an exception, verify_answer returns ok=True."""

        def mock_create(*args, **kwargs):
            raise RuntimeError("API error")

        from verify import client

        monkeypatch.setattr(client.chat.completions, "create", mock_create)

        from verify import verify_answer

        result = verify_answer(
            question="Test question",
            spec={"data_requests": [{"id": "v1"}]},
            extractions={"v1": {"values": [100]}},
            answer="100",
        )
        assert result["ok"] is True

    def test_verify_returns_ok_true_on_unparseable_json(self, monkeypatch):
        """When the LLM returns unparseable JSON, verify_answer returns ok=True."""

        class MockChoice:
            message = type("msg", (), {"content": "NOT JSON AT ALL"})()

        class MockResponse:
            choices = [MockChoice()]

        def mock_create(*args, **kwargs):
            return MockResponse()

        from verify import client

        monkeypatch.setattr(client.chat.completions, "create", mock_create)

        from verify import verify_answer

        result = verify_answer(
            question="Test question",
            spec={"data_requests": [{"id": "v1"}]},
            extractions={"v1": {"values": [100]}},
            answer="100",
        )
        assert result["ok"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt framing check
# ═══════════════════════════════════════════════════════════════════════════════


class TestPromptFraming:
    """Verify the mentor/intern framing is present in VERIFY_SYSTEM."""

    def test_verify_system_uses_mentor_framing(self):
        """VERIFY_SYSTEM contains senior analyst / intern / mentor language."""
        from verify import VERIFY_SYSTEM

        # Must contain some form of the mentor/intern framing
        lower = VERIFY_SYSTEM.lower()
        assert "senior" in lower or "mentor" in lower
        assert "intern" in lower

    def test_verify_system_preserves_checklist(self):
        """VERIFY_SYSTEM still contains the checklist items."""
        from verify import VERIFY_SYSTEM

        lower = VERIFY_SYSTEM.lower()
        assert "units" in lower
        assert "fiscal year" in lower or "fy" in lower
        assert "total" in lower

    def test_no_review_board_framing(self):
        """VERIFY_SYSTEM no longer uses 'review board' framing."""
        from verify import VERIFY_SYSTEM

        lower = VERIFY_SYSTEM.lower()
        assert "review board" not in lower
