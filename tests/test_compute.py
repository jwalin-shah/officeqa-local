"""Tests for compute.py — execute(), format_result(), sandbox security, unit conversion."""

import pytest

from compute import (
    ComputeError,
    convert_unit,
    execute,
    format_result,
    parse_unit,
    validate_extractions,
)

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


# ── Enum dispatcher ─────────────────────────────────────────────────────────


def _dispatch_spec(op, n_drs=1, **extra):
    spec = {
        "computation": op,
        "data_requests": [{"id": f"v{i + 1}"} for i in range(n_drs)],
        "computation_spec": {"python_template": "result = None  # should not run"},
    }
    spec.update(extra)
    return spec


def test_dispatch_direct():
    spec = _dispatch_spec("direct")
    assert execute(spec, {"v1": {"values": [42]}}) == 42.0


def test_dispatch_sum():
    spec = _dispatch_spec("sum")
    assert execute(spec, {"v1": {"values": [1, 2, 3, 4]}}) == 10.0


def test_dispatch_difference():
    spec = _dispatch_spec("difference", n_drs=2)
    ex = {"v1": {"values": [100]}, "v2": {"values": [250]}}
    assert execute(spec, ex) == 150.0


def test_dispatch_percent_change():
    spec = _dispatch_spec("percent_change", n_drs=2)
    ex = {"v1": {"values": [200]}, "v2": {"values": [250]}}
    assert execute(spec, ex) == 25.0


def test_dispatch_percent_change_raises_on_zero_base():
    """percent_change uses v1 as base; zero base must not divide by zero."""
    spec = _dispatch_spec("percent_change", n_drs=2)
    with pytest.raises(ComputeError, match="percent_change: base is zero"):
        execute(spec, {"v1": {"values": [0]}, "v2": {"values": [100]}})


def test_dispatch_ratio():
    spec = _dispatch_spec("ratio", n_drs=2)
    assert execute(spec, {"v1": {"values": [10]}, "v2": {"values": [4]}}) == 2.5


def test_dispatch_ratio_raises_on_zero_denominator():
    """ratio is v1/v2; zero denominator must not divide by zero."""
    spec = _dispatch_spec("ratio", n_drs=2)
    with pytest.raises(ComputeError, match="ratio: denominator is zero"):
        execute(spec, {"v1": {"values": [10]}, "v2": {"values": [0]}})


def test_dispatch_average():
    assert execute(_dispatch_spec("average"), {"v1": {"values": [10, 20, 30]}}) == 20.0


def test_dispatch_max_min():
    assert execute(_dispatch_spec("max"), {"v1": {"values": [5, 12, 3]}}) == 12.0
    assert execute(_dispatch_spec("min"), {"v1": {"values": [5, 12, 3]}}) == 3.0


def test_dispatch_argmax_with_labels():
    spec = _dispatch_spec("argmax")
    ex = {"v1": {"values": [10, 50, 20], "labels": ["A", "B", "C"]}}
    assert execute(spec, ex) == "B"


def test_dispatch_linear_regression():
    spec = {
        "computation": "linear_regression",
        "data_requests": [{"id": "v1", "years": [2000, 2001, 2002, 2003]}],
        "computation_spec": {"python_template": "result = None"},
    }
    ex = {"v1": {"values": [10, 20, 30, 40]}}
    result = execute(spec, ex)
    assert isinstance(result, list) and len(result) == 2
    assert abs(result[0] - 10.0) < 1e-6  # slope
    assert abs(result[1] - (-19990.0)) < 1e-3  # intercept at x=0


def test_dispatch_cagr():
    # 100 → 200 over 10 years → ~7.1773%
    spec = {
        "computation": "cagr",
        "data_requests": [
            {"id": "v1", "years": [2000]},
            {"id": "v2", "years": [2010]},
        ],
        "computation_spec": {"python_template": "result = None"},
    }
    ex = {"v1": {"values": [100]}, "v2": {"values": [200]}}
    assert abs(execute(spec, ex) - 7.17734625) < 1e-4


def test_dispatch_gini():
    spec = _dispatch_spec("gini")
    # Equal distribution → 0
    assert abs(execute(spec, {"v1": {"values": [1, 1, 1, 1]}})) < 1e-9


def test_dispatch_unknown_op_falls_back_to_template():
    spec = {
        "computation": "custom",
        "data_requests": [{"id": "v1"}],
        "computation_spec": {"python_template": "result = values['v1'][0] * 7"},
    }
    assert execute(spec, {"v1": {"values": [6]}}) == 42


def test_dispatch_aliases():
    # "mean" aliases to "average"
    assert execute(_dispatch_spec("mean"), {"v1": {"values": [10, 20]}}) == 15.0
    # "correlation" aliases to pearson_correlation
    spec = _dispatch_spec("correlation", n_drs=2)
    ex = {"v1": {"values": [1, 2, 3]}, "v2": {"values": [2, 4, 6]}}
    assert abs(execute(spec, ex) - 1.0) < 1e-9


def test_dispatch_difference_raises_on_empty():
    spec = _dispatch_spec("difference", n_drs=2)
    with pytest.raises(ComputeError, match="need two non-empty"):
        execute(spec, {"v1": {"values": [5]}, "v2": {"values": []}})


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


# ── parse_unit() ─────────────────────────────────────────────────────────────


def test_parse_unit_millions():
    assert parse_unit("In millions of dollars") == "millions"
    assert parse_unit("In millions") == "millions"
    assert parse_unit("Millions of dollars") == "millions"
    assert parse_unit("millions") == "millions"


def test_parse_unit_thousands():
    assert parse_unit("In thousands of dollars") == "thousands"
    assert parse_unit("In thousands") == "thousands"
    assert parse_unit("Thousands") == "thousands"
    assert parse_unit("thousands") == "thousands"


def test_parse_unit_billions():
    assert parse_unit("In billions of dollars") == "billions"
    assert parse_unit("In billions") == "billions"
    assert parse_unit("Billions") == "billions"
    assert parse_unit("billions") == "billions"


def test_parse_unit_percent():
    assert parse_unit("Percent") == "percent"
    assert parse_unit("In percent") == "percent"
    assert parse_unit("percent") == "percent"


def test_parse_unit_none_or_empty():
    assert parse_unit("") is None
    assert parse_unit(None) is None


def test_parse_unit_unrecognized():
    assert parse_unit("Index numbers") is None
    assert parse_unit("some random text") is None


# ── convert_unit() ──────────────────────────────────────────────────────────


def test_convert_unit_thousands_to_millions():
    """Source=thousands (1e3), target=millions (1e6) → divide by 1000."""
    assert convert_unit(5000, "thousands", "millions") == 5.0


def test_convert_unit_millions_to_thousands():
    """Source=millions (1e6), target=thousands (1e3) → multiply by 1000."""
    assert convert_unit(5, "millions", "thousands") == 5000.0


def test_convert_unit_thousands_to_billions():
    """Source=thousands (1e3), target=billions (1e9) → divide by 1e6."""
    assert convert_unit(3_000_000, "thousands", "billions") == 3.0


def test_convert_unit_millions_to_billions():
    """Source=millions (1e6), target=billions (1e9) → divide by 1000."""
    assert convert_unit(2500, "millions", "billions") == 2.5


def test_convert_unit_same_units():
    """When source and target match, no conversion."""
    assert convert_unit(42, "millions", "millions") == 42.0
    assert convert_unit(100, "thousands", "thousands") == 100.0


def test_convert_unit_source_none():
    """No source unit → no conversion (return value unchanged)."""
    assert convert_unit(42, None, "millions") == 42.0


def test_convert_unit_target_none():
    """No target unit → no conversion (return value unchanged)."""
    assert convert_unit(42, "millions", None) == 42.0


def test_convert_unit_both_none():
    """Both None → no conversion."""
    assert convert_unit(42, None, None) == 42.0


def test_convert_unit_list():
    """convert_unit applies element-wise to lists."""
    assert convert_unit([1000, 2000, 3000], "thousands", "millions") == [1.0, 2.0, 3.0]


def test_convert_unit_percent_no_conversion():
    """Percent source + percent target → no conversion."""
    assert convert_unit(5.3, "percent", "percent") == 5.3


# ── format_result() with unit conversion ────────────────────────────────────


def test_format_result_unit_thousands_to_millions():
    """Source thousands, output millions → value / 1000."""
    fmt = {"type": "number", "unit": "millions"}
    assert format_result(5000, fmt, source_unit="thousands") == "5.0"


def test_format_result_unit_millions_to_billions():
    """Source millions, output billions → value / 1000."""
    fmt = {"type": "number", "unit": "billions"}
    assert format_result(2500, fmt, source_unit="millions") == "2.5"


def test_format_result_unit_same_no_conversion():
    """Same unit → no conversion applied."""
    fmt = {"type": "number", "unit": "millions"}
    assert format_result(42, fmt, source_unit="millions") == "42"


def test_format_result_unit_none_source():
    """No source unit → no conversion (backward compat)."""
    fmt = {"type": "number", "unit": "millions"}
    assert format_result(42, fmt, source_unit=None) == "42"


def test_format_result_unit_none_target():
    """No target unit in output_format → no conversion (backward compat)."""
    fmt = {"type": "number"}
    assert format_result(42, fmt, source_unit="millions") == "42"


def test_format_result_unit_both_none():
    """Both None → no conversion (backward compat)."""
    fmt = {"type": "number"}
    assert format_result(42, fmt, source_unit=None) == "42"


def test_format_result_unit_with_rounding():
    """Unit conversion + rounding should work together."""
    fmt = {"type": "number", "unit": "millions", "rounding": "integer"}
    # 4999 thousands → 4.999 millions → rounded to integer = 5.0
    assert format_result(4999, fmt, source_unit="thousands") == "5.0"


def test_format_result_unit_list_conversion():
    """Unit conversion on list results."""
    fmt = {"type": "list", "unit": "millions"}
    result = format_result([1000, 2000], fmt, source_unit="thousands")
    assert result == "[1.0, 2.0]"
