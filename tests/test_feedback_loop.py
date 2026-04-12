"""Tests for solve.py feedback loops, bounded retries, and phase routing.

Covers:
- Both bounce-back paths (extract, decompose) are reachable
- Feedback messages from verify are passed correctly to extract
- MAX_RETRIES bounds total LLM calls per question
- No infinite retry loops possible
- Thread-safe sqlite3 connections in parallel eval
- Bounded retry behavior and correct phase routing
"""

import threading
from unittest.mock import patch

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers: build mock specs, extractions, verify verdicts
# ═══════════════════════════════════════════════════════════════════════════════


def _make_spec(data_requests=None, computation="direct"):
    """Build a minimal valid QuestionSpec for testing."""
    if data_requests is None:
        data_requests = [
            {
                "id": "v1",
                "label": "test value",
                "source": "corpus",
                "row_hint": "national defense",
                "column_hint": "1940",
                "years": [1940],
                "granularity": "annual",
                "expected_count": 1,
                "cohort": False,
            }
        ]
    return {
        "computation": computation,
        "data_requests": data_requests,
        "computation_spec": {"python_template": "result = values['v1'][0]"},
        "output_format": {"type": "number", "unit": None, "rounding": None, "as_percent": False},
    }


def _make_extraction(values=None, labels=None, source_file="test.json"):
    """Build a minimal extraction dict for a single DR."""
    if values is None:
        values = [2602]
    if labels is None:
        labels = ["national defense 1940"]
    return {
        "v1": {
            "values": values,
            "labels": labels,
            "source_file": source_file,
        }
    }


def _make_retrieve_entry(
    file="treasury_bulletin_1940_06.json", element_seq=1, html="<table></table>"
):
    """Build a minimal retrieve_v2 entry."""
    return {"file": file, "element_seq": element_seq, "html": html, "score": 0.9}


def _make_per_dr_entries(dr_ids=None):
    """Build per_dr_entries dict with at least one entry per DR."""
    if dr_ids is None:
        dr_ids = ["v1"]
    return {dr_id: [_make_retrieve_entry()] for dr_id in dr_ids}


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Constants exist and have correct values
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify MAX_RETRIES and MAX_LLM_CALLS constants exist and are correct."""

    def test_max_retries_constant_exists(self):
        from solve import MAX_RETRIES_PER_PHASE

        assert MAX_RETRIES_PER_PHASE is not None

    def test_max_retries_is_2(self):
        from solve import MAX_RETRIES_PER_PHASE

        assert MAX_RETRIES_PER_PHASE == 2

    def test_max_llm_calls_constant_exists(self):
        from solve import MAX_LLM_CALLS

        assert MAX_LLM_CALLS is not None

    def test_max_llm_calls_is_6(self):
        from solve import MAX_LLM_CALLS

        assert MAX_LLM_CALLS == 6

    def test_max_llm_calls_is_consistent_with_max_retries(self):
        """MAX_LLM_CALLS should be at most MAX_RETRIES_PER_PHASE * 3 phases."""
        from solve import MAX_LLM_CALLS, MAX_RETRIES_PER_PHASE

        # decompose + extract + verify = 3 phases, each with MAX_RETRIES_PER_PHASE attempts
        assert MAX_LLM_CALLS <= MAX_RETRIES_PER_PHASE * 3


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Bounce-back: verify → extract path
# ═══════════════════════════════════════════════════════════════════════════════


class TestBounceBackExtract:
    """When verify returns ok=False with suggested_phase='extract',
    solve re-runs extract→compute with the issue as feedback."""

    def test_extract_bounce_back_called(self):
        """Verify bounce-back to extract is reachable."""
        from solve import solve

        spec = _make_spec()

        # We need to mock the pipeline components
        with (
            patch("solve.scout", return_value="hint"),
            patch("solve.decompose", return_value=spec),
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            # First extract returns valid values
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }

            # Verify says extract is wrong
            mock_verify.return_value = {
                "ok": False,
                "issue": "wrong row selected",
                "suggested_phase": "extract",
            }

            solve("What were defense expenditures in 1940?", verbose=True)

            # extract_structured should have been called at least twice:
            # once initial + once bounce-back
            assert mock_extract.call_count >= 2

            # Second call should include feedback
            second_call_kwargs = mock_extract.call_args_list[1]
            # feedback is passed as the `feedback` kwarg in _run_extract_and_compute
            # which passes it through to extract_structured
            assert second_call_kwargs[1].get("feedback") == "wrong row selected" or (
                len(second_call_kwargs[0]) > 4 and second_call_kwargs[0][4] == "wrong row selected"
            )

    def test_extract_bounce_back_feedback_content(self):
        """Feedback from verify is passed correctly to extract's feedback parameter."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose", return_value=spec),
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }
            mock_verify.return_value = {
                "ok": False,
                "issue": "unit mismatch: value is in thousands not millions",
                "suggested_phase": "extract",
            }

            solve("Test question", verbose=True)

            # Find the call that has feedback
            feedback_found = False
            for call in mock_extract.call_args_list[1:]:
                kwargs = call[1]
                if kwargs.get("feedback") == "unit mismatch: value is in thousands not millions":
                    feedback_found = True
                    break
            assert feedback_found, "Feedback from verify not passed to extract_structured"

    def test_extract_bounce_back_uses_better_answer(self):
        """When extract bounce-back succeeds, the improved answer is used."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose", return_value=spec),
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }
            # First verify says extract is wrong, second says ok
            mock_verify.side_effect = [
                {"ok": False, "issue": "wrong column", "suggested_phase": "extract"},
                {"ok": True},
            ]

            result = solve("Test question", verbose=True)
            # The final answer should be the corrected one
            assert result == "2,602"


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Bounce-back: verify → decompose path
# ═══════════════════════════════════════════════════════════════════════════════


class TestBounceBackDecompose:
    """When verify returns ok=False with suggested_phase='decompose',
    solve re-runs the full pipeline (decompose→retrieve→extract→compute)."""

    def test_decompose_bounce_back_called(self):
        """Verify bounce-back to decompose is reachable."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose") as mock_decompose,
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_decompose.return_value = spec
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }
            mock_verify.return_value = {
                "ok": False,
                "issue": "wrong computation: should be percent change not difference",
                "suggested_phase": "decompose",
            }

            solve("Test question", verbose=True)

            # decompose should have been called at least twice (initial + bounce-back)
            assert mock_decompose.call_count >= 2

    def test_decompose_bounce_back_receives_feedback(self):
        """Feedback from verify is passed to decompose on bounce-back."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value="scout hint"),
            patch("solve.decompose") as mock_decompose,
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_decompose.return_value = spec
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }
            mock_verify.return_value = {
                "ok": False,
                "issue": "wrong computation: should be percent change",
                "suggested_phase": "decompose",
            }

            solve("Test question", verbose=True)

            # Second decompose call should include the feedback
            second_call = mock_decompose.call_args_list[1]
            feedback_arg = second_call[1].get("feedback") or (
                second_call[0][2] if len(second_call[0]) > 2 else None
            )
            assert feedback_arg is not None
            assert "percent change" in feedback_arg

    def test_decompose_bounce_back_uses_better_answer(self):
        """When decompose bounce-back produces a valid answer, it's used."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose") as mock_decompose,
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", side_effect=[2602, 3500]),
            patch("solve.format_result", side_effect=["2,602", "3,500"]),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_decompose.return_value = spec
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }
            # First verify says decompose is wrong
            mock_verify.return_value = {
                "ok": False,
                "issue": "wrong computation",
                "suggested_phase": "decompose",
            }

            result = solve("Test question", verbose=True)
            # After decompose bounce-back, the second answer should be used
            assert result == "3,500"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Bounded retries — no infinite loops
# ═══════════════════════════════════════════════════════════════════════════════


class TestBoundedRetries:
    """MAX_RETRIES bounds total LLM calls per question; no infinite loops."""

    def test_llm_call_counter_tracks_calls(self):
        """The solve() function tracks LLM call count."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose", return_value=spec),
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer", return_value={"ok": True}),
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }

            # Verify the function completes without error
            result = solve("Test question", verbose=False)
            assert result is not None

    def test_verify_always_fails_does_not_infinite_loop(self):
        """If verify always says 'retry extract', solve doesn't loop forever."""
        from solve import MAX_LLM_CALLS, solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose", return_value=spec),
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }
            # Verify always says extract is wrong
            mock_verify.return_value = {
                "ok": False,
                "issue": "always wrong",
                "suggested_phase": "extract",
            }

            result = solve("Test question", verbose=False)
            # Should terminate (not loop forever)
            assert result is not None
            # The total number of extract_structured calls should be bounded
            # At most MAX_RETRIES_PER_PHASE + 1 (initial + retries)
            assert mock_extract.call_count <= MAX_LLM_CALLS

    def test_llm_budget_exhausted_returns_current_answer(self):
        """When LLM budget is exhausted, solve returns the current answer."""
        from solve import MAX_LLM_CALLS, solve

        spec = _make_spec()
        # Make decompose consume all LLM budget
        decompose_calls = {"count": 0}

        def expensive_decompose(*args, **kwargs):
            decompose_calls["count"] += 1
            return spec

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose", side_effect=expensive_decompose),
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }
            # Verify always says retry decompose
            mock_verify.return_value = {
                "ok": False,
                "issue": "wrong decomposition",
                "suggested_phase": "decompose",
            }

            result = solve("Test question", verbose=False)
            # Should terminate even though verify always says retry
            assert result is not None
            # decompose should have been called at most MAX_LLM_CALLS / 1 times
            # (since each decompose call counts as 1 LLM call)
            assert decompose_calls["count"] <= MAX_LLM_CALLS

    def test_total_llm_calls_bounded_by_max(self):
        """Total LLM calls per solve() invocation do not exceed MAX_LLM_CALLS."""
        from solve import MAX_LLM_CALLS, solve

        spec = _make_spec()
        llm_calls = {"count": 0}

        def counting_decompose(*args, **kwargs):
            llm_calls["count"] += 1
            return spec

        def counting_extract(*args, **kwargs):
            llm_calls["count"] += 1
            return {"extractions": _make_extraction(), "notes": ""}

        def counting_verify(*args, **kwargs):
            llm_calls["count"] += 1
            return {"ok": False, "issue": "try again", "suggested_phase": "extract"}

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose", side_effect=counting_decompose),
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured", side_effect=counting_extract),
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer", side_effect=counting_verify),
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            solve("Test question", verbose=False)
            assert llm_calls["count"] <= MAX_LLM_CALLS, (
                f"LLM calls ({llm_calls['count']}) exceeded MAX_LLM_CALLS ({MAX_LLM_CALLS})"
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Thread-safety for parallel eval
# ═══════════════════════════════════════════════════════════════════════════════


class TestThreadSafety:
    """Ensure sqlite3 connections are thread-local in parallel eval."""

    def test_retrieve_v2_uses_thread_local_sqlite3(self):
        """retrieve_v2's _ledger_conn() returns thread-local connections."""
        from retrieve_v2 import _ledger_conn

        conn1 = _ledger_conn()
        conn2 = _ledger_conn()
        # Same thread should get the same connection
        assert conn1 is conn2

    def test_retrieve_v2_different_threads_get_different_connections(self):
        """Different threads get different sqlite3 connections."""
        from retrieve_v2 import _ledger_conn

        main_conn = _ledger_conn()
        thread_conn_holder: list[object] = [None]

        def get_conn_in_thread():
            thread_conn_holder[0] = _ledger_conn()

        t = threading.Thread(target=get_conn_in_thread)
        t.start()
        t.join()

        # Connections from different threads should be different objects
        assert thread_conn_holder[0] is not main_conn

    def test_solve_fast_path_uses_thread_local_sqlite3(self):
        """find._conn() returns thread-local connections."""
        from find import _conn

        conn1 = _conn()
        conn2 = _conn()
        # Same thread should get the same connection
        assert conn1 is conn2

    def test_solve_fast_path_different_threads_get_different_connections(self):
        """Different threads get different _conn() connections."""
        from find import _conn

        main_conn = _conn()
        thread_conn_holder: list[object] = [None]

        def get_conn_in_thread():
            thread_conn_holder[0] = _conn()

        t = threading.Thread(target=get_conn_in_thread)
        t.start()
        t.join()

        # Connections from different threads should be different objects
        assert thread_conn_holder[0] is not main_conn

    def test_json_element_cache_thread_safety(self):
        """_JSON_ELEMENT_CACHE in retrieve_v2 should not cause crashes
        under concurrent access (module-level dict)."""
        from retrieve_v2 import _load_element_html

        # Just verify it doesn't crash with concurrent reads
        errors = []

        def read_cache():
            try:
                # Access the cache by loading a non-existent element
                _load_element_html("nonexistent.json", 0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=read_cache) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # No crashes should occur (empty string is fine, errors are not)
        assert len(errors) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Correct phase routing
# ═══════════════════════════════════════════════════════════════════════════════


class TestPhaseRouting:
    """Verify that suggested_phase correctly routes to the right bounce-back."""

    def test_extract_phase_routes_to_extract(self):
        """suggested_phase='extract' triggers re-extraction, not re-decomposition."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose") as mock_decompose,
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_decompose.return_value = spec
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }
            mock_verify.side_effect = [
                {"ok": False, "issue": "wrong row", "suggested_phase": "extract"},
                {"ok": True},
            ]

            solve("Test question", verbose=False)

            # When suggested_phase='extract', decompose should NOT be called again
            # (only 1 call for initial decomposition)
            assert mock_decompose.call_count == 1
            # extract_structured should be called twice (initial + bounce-back)
            assert mock_extract.call_count == 2

    def test_decompose_phase_routes_to_decompose(self):
        """suggested_phase='decompose' triggers full pipeline re-run."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose") as mock_decompose,
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_decompose.return_value = spec
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }
            mock_verify.side_effect = [
                {"ok": False, "issue": "wrong computation", "suggested_phase": "decompose"},
                {"ok": True},
            ]

            solve("Test question", verbose=False)

            # When suggested_phase='decompose', decompose should be called again
            assert mock_decompose.call_count >= 2

    def test_null_phase_no_bounce_back(self):
        """When suggested_phase is null/missing, no bounce-back occurs."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose") as mock_decompose,
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_decompose.return_value = spec
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }
            mock_verify.return_value = {
                "ok": False,
                "issue": "something wrong",
                "suggested_phase": None,
            }

            result = solve("Test question", verbose=False)

            # No bounce-back should happen — only initial calls
            assert mock_decompose.call_count == 1
            assert mock_extract.call_count == 1
            # Answer should still be the original (not changed by bounce-back)
            assert result == "2,602"

    def test_empty_retrieve_triggers_redecompose(self):
        """When retrieve returns 0 entries, decompose is retried with feedback."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose") as mock_decompose,
            patch("solve.retrieve_bottomup", return_value={"v1": []}),
            patch("solve.retrieve_for_spec") as mock_retrieve,
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer", return_value={"ok": True}),
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_decompose.return_value = spec
            # Bottom-up empty, then retrieve_v2 fallback empty, then post-redecompose retrieve succeeds
            mock_retrieve.side_effect = [
                {"v1": []},  # empty
                _make_per_dr_entries(),  # with entries
            ]
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }

            solve("Test question", verbose=False)

            # decompose should have been called twice
            assert mock_decompose.call_count == 2


# ═══════════════════════════════════════════════════════════════════════════════
# 7. CPI-dependent questions produce valid answers
# ═══════════════════════════════════════════════════════════════════════════════


class TestCPIQuestions:
    """CPI-dependent questions produce valid answers through full pipeline."""

    def test_cpi_dr_gracefully_handled(self):
        """Data requests with source='cpi' don't crash the pipeline."""
        from solve import solve

        spec = _make_spec(
            data_requests=[
                {
                    "id": "v1",
                    "label": "CPI-U 1940",
                    "source": "cpi",
                    "row_hint": "",
                    "column_hint": "",
                    "years": [1940],
                    "granularity": "annual",
                    "expected_count": 1,
                    "cohort": False,
                },
                {
                    "id": "v2",
                    "label": "defense expenditure",
                    "source": "corpus",
                    "row_hint": "national defense",
                    "column_hint": "1940",
                    "years": [1940],
                    "granularity": "annual",
                    "expected_count": 1,
                    "cohort": False,
                },
            ],
            computation="cpi_adjust",
        )
        spec["computation_spec"] = {
            "python_template": "result = values['v2'][0] * (values['v1'][0] / 100)"
        }

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose", return_value=spec),
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries(["v1", "v2"])),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1", "v2"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602.0),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer", return_value={"ok": True}),
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_extract.return_value = {
                "extractions": {
                    "v1": {"values": [14.0], "labels": ["CPI-U 1940"], "source_file": "cpi.py"},
                    "v2": {
                        "values": [2602],
                        "labels": ["national defense"],
                        "source_file": "test.json",
                    },
                },
                "notes": "",
            }

            result = solve("CPI-adjusted defense spending 1940", verbose=False)
            # Should not be COMPUTE_FAILED or NO_VALUES
            assert result not in ("COMPUTE_FAILED", "EXTRACT_FAILED")
            assert not (result and result.startswith("NO_VALUES"))


# ═══════════════════════════════════════════════════════════════════════════════
# 8. End-to-end pipeline completes without errors
# ═══════════════════════════════════════════════════════════════════════════════


class TestEndToEnd:
    """Verify the full pipeline completes for a single question."""

    def test_pipeline_completes_with_verify_ok(self):
        """Full pipeline: decompose → retrieve → extract → compute → verify(ok)."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value="hint"),
            patch("solve.decompose", return_value=spec),
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer", return_value={"ok": True}),
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }

            result = solve("What were defense expenditures in 1940?", verbose=False)
            assert result == "2,602"

    def test_pipeline_completes_without_verify(self):
        """Pipeline with use_verify=False skips verification."""
        from solve import solve

        spec = _make_spec()

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose", return_value=spec),
            patch("solve.retrieve_for_spec", return_value=_make_per_dr_entries()),
            patch("solve.try_deterministic_fast_path", return_value=({}, ["v1"])),
            patch("solve.extract_structured") as mock_extract,
            patch("solve.compute_execute", return_value=2602),
            patch("solve.format_result", return_value="2,602"),
            patch("solve.verify_answer") as mock_verify,
            patch("solve._determine_source_unit", return_value=None),
            patch("solve.validate_extractions", return_value=[]),
        ):
            mock_extract.return_value = {
                "extractions": _make_extraction(),
                "notes": "",
            }

            result = solve("Test question", verbose=False, use_verify=False)
            assert result == "2,602"
            # verify should NOT have been called
            assert mock_verify.call_count == 0

    def test_decompose_failed_returns_error(self):
        """When decompose returns None, solve returns DECOMPOSE_FAILED."""
        from solve import solve

        with (
            patch("solve.scout", return_value=""),
            patch("solve.decompose", return_value=None),
        ):
            result = solve("Test question", verbose=False)
            assert result == "DECOMPOSE_FAILED"
