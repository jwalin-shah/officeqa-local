"""Tests for compute.py — execute(), format_result(), sandbox security."""

import pytest

from compute import ComputeError, execute, format_result, validate_extractions

# ── execute() happy paths ────────────────────────────────────────────────────


def test_execute_sum():
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 3}],
        "computation_spec": {"python_template": "result = sum(values['v1'])"},
    }
    extractions = {"v1": {"values": [10, 20, 30]}}
    assert execute(spec, extractions) == 60


def test_execute_difference():
    spec = {
        "data_requests": [
            {"id": "v1", "expected_count": 1},
            {"id": "v2", "expected_count": 1},
        ],
        "computation_spec": {"python_template": "result = values['v2'][0] - values['v1'][0]"},
    }
    extractions = {
        "v1": {"values": [100]},
        "v2": {"values": [250]},
    }
    assert execute(spec, extractions) == 150


def test_execute_mean():
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 4}],
        "computation_spec": {"python_template": "result = mean(values['v1'])"},
    }
    extractions = {"v1": {"values": [10, 20, 30, 40]}}
    assert execute(spec, extractions) == 25.0


def test_execute_math_sqrt():
    """Templates using math module work because math is pre-injected."""
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 1}],
        "computation_spec": {"python_template": "result = math.sqrt(values['v1'][0])"},
    }
    extractions = {"v1": {"values": [144]}}
    assert execute(spec, extractions) == 12.0


def test_execute_statistics_stdev():
    """Templates using statistics module work via pre-injection."""
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 5}],
        "computation_spec": {"python_template": "result = statistics.stdev(values['v1'])"},
    }
    extractions = {"v1": {"values": [2, 4, 4, 4, 5, 5, 7, 9]}}
    result = execute(spec, extractions)
    assert abs(result - 2.138) < 0.01


def test_execute_re_module():
    """Templates can use the pre-injected re module."""
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 1}],
        "computation_spec": {
            "python_template": "result = len(re.findall(r'\\d+', 'abc123def456'))"
        },
    }
    extractions = {"v1": {"values": [0]}}
    assert execute(spec, extractions) == 2


def test_execute_numpy():
    """Templates can use the pre-injected numpy module."""
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 3}],
        "computation_spec": {"python_template": "result = float(np.mean(values['v1']))"},
    }
    extractions = {"v1": {"values": [10, 20, 30]}}
    assert execute(spec, extractions) == 20.0


def test_execute_strips_none_values():
    """None values in extractions are stripped before computation."""
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 3}],
        "computation_spec": {"python_template": "result = sum(values['v1'])"},
    }
    extractions = {"v1": {"values": [10, None, 30, None]}}
    assert execute(spec, extractions) == 40


def test_execute_list_result():
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 3}],
        "computation_spec": {"python_template": "result = [min(values['v1']), max(values['v1'])]"},
    }
    extractions = {"v1": {"values": [5, 15, 10]}}
    assert execute(spec, extractions) == [5, 15]


def test_execute_percent_change():
    spec = {
        "data_requests": [
            {"id": "v1", "expected_count": 1},
            {"id": "v2", "expected_count": 1},
        ],
        "computation_spec": {
            "python_template": (
                "result = (values['v2'][0] - values['v1'][0]) / values['v1'][0] * 100"
            )
        },
    }
    extractions = {
        "v1": {"values": [200]},
        "v2": {"values": [250]},
    }
    assert execute(spec, extractions) == 25.0


# ── Sandbox security ─────────────────────────────────────────────────────────


def test_import_blocked():
    """__import__ is NOT in safe_builtins — import os must fail."""
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 1}],
        "computation_spec": {"python_template": "import os; result = os.getcwd()"},
    }
    extractions = {"v1": {"values": [1]}}
    with pytest.raises(ComputeError, match="template execution failed"):
        execute(spec, extractions)


def test_dunder_import_blocked():
    """Direct __import__('os') is blocked."""
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 1}],
        "computation_spec": {"python_template": "result = __import__('os').getcwd()"},
    }
    extractions = {"v1": {"values": [1]}}
    with pytest.raises(ComputeError, match="template execution failed"):
        execute(spec, extractions)


def test_safe_builtins_no_import():
    """Verify __import__ is not in safe_builtins dict at all."""
    # We test this by inspecting the compute module directly
    import inspect

    from compute import execute as _execute

    source = inspect.getsource(_execute)
    # The safe_builtins dict should not include __import__
    assert '"__import__": __import__' not in source
    assert "'__import__': __import__" not in source


# ── Error handling ───────────────────────────────────────────────────────────


def test_execute_no_template():
    spec = {
        "data_requests": [{"id": "v1"}],
        "computation_spec": {},
    }
    extractions = {"v1": {"values": [1]}}
    with pytest.raises(ComputeError, match="no python_template"):
        execute(spec, extractions)


def test_execute_no_result_var():
    spec = {
        "data_requests": [{"id": "v1"}],
        "computation_spec": {"python_template": "x = 42"},
    }
    extractions = {"v1": {"values": [1]}}
    with pytest.raises(ComputeError, match="did not assign 'result'"):
        execute(spec, extractions)


def test_execute_template_error():
    spec = {
        "data_requests": [{"id": "v1"}],
        "computation_spec": {"python_template": "result = 1/0"},
    }
    extractions = {"v1": {"values": [1]}}
    with pytest.raises(ComputeError, match="template execution failed"):
        execute(spec, extractions)


# ── format_result() ──────────────────────────────────────────────────────────


def test_format_result_number():
    assert format_result(42, {"type": "number"}) == "42"
    assert format_result(3.14, {"type": "number"}) == "3.14"


def test_format_result_rounding():
    assert format_result(3.14159, {"type": "number", "rounding": "integer"}) == "3.0"
    assert format_result(3.14159, {"type": "number", "rounding": "tenths"}) == "3.1"
    assert format_result(3.14159, {"type": "number", "rounding": "hundredths"}) == "3.14"
    assert format_result(3.14159, {"type": "number", "rounding": "thousandths"}) == "3.142"


def test_format_result_list():
    assert format_result([1, 2, 3], {"type": "list"}) == "[1, 2, 3]"
    assert format_result([1.5, 2.3], {"type": "list", "rounding": "integer"}) == "[2.0, 2.0]"


def test_format_result_none():
    assert format_result(None, {"type": "number"}) == "None"


def test_format_result_string():
    assert format_result("hello", {"type": "string"}) == "hello"


# ── validate_extractions() ───────────────────────────────────────────────────


def test_validate_extractions_missing():
    spec = {"data_requests": [{"id": "v1", "expected_count": 3}]}
    # v1 is missing entirely
    warnings = validate_extractions(spec, {})
    assert len(warnings) == 1
    assert "missing extraction for v1" in warnings[0]


def test_validate_extractions_low_count():
    spec = {"data_requests": [{"id": "v1", "expected_count": 12}]}
    extractions = {"v1": {"values": [1, 2, 3]}}
    warnings = validate_extractions(spec, extractions)
    assert len(warnings) == 1
    assert "only 3/12" in warnings[0]


def test_validate_extractions_ok():
    spec = {"data_requests": [{"id": "v1", "expected_count": 3}]}
    extractions = {"v1": {"values": [1, 2, 3]}}
    warnings = validate_extractions(spec, extractions)
    assert warnings == []
