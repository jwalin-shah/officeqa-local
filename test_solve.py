#!/usr/bin/env python3
"""Smoke tests for the OfficeQA pipeline entry points.

Validates that core pipeline functions are importable and callable.
Does NOT make any LLM calls — all tests use deterministic code paths.
"""


def test_imports():
    """All pipeline modules are importable."""
    from compute import ComputeError, execute, format_result, validate_extractions  # noqa: F401
    from extract import build_context_from_entries, html_to_pipe_text, render_entry  # noqa: F401
    from retrieve_v2 import retrieve, retrieve_from_question  # noqa: F401
    from reward import fuzzy_match_answer, score_answer  # noqa: F401
    from scout import scout  # noqa: F401


def test_compute_simple():
    """A basic compute template executes correctly."""
    from compute import execute, format_result

    spec = {
        "data_requests": [{"id": "v1", "expected_count": 3}],
        "computation_spec": {"python_template": "result = sum(values['v1'])"},
        "output_format": {"type": "number"},
    }
    extractions = {"v1": {"values": [100, 200, 300]}}
    result = execute(spec, extractions)
    assert result == 600
    assert format_result(result, spec["output_format"]) == "600"


def test_format_result_rounding():
    """format_result applies rounding as specified."""
    from compute import format_result

    assert format_result(3.14159, {"type": "number", "rounding": "hundredths"}) == "3.14"
    assert format_result(3.14159, {"type": "number", "rounding": "integer"}) == "3.0"


def test_reward_basic():
    """fuzzy_match_answer works for exact and close numeric matches."""
    from reward import fuzzy_match_answer

    ok, _ = fuzzy_match_answer("600", "600")
    assert ok

    ok, _ = fuzzy_match_answer("600", "601", tolerance=0.01)
    assert ok


if __name__ == "__main__":
    test_imports()
    test_compute_simple()
    test_format_result_rounding()
    test_reward_basic()
    print("All test_solve.py tests passed.")
