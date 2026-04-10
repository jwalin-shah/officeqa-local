#!/usr/bin/env python3
"""Deterministic computation harness for OfficeQA.

The LLM writes a python_template in the QuestionSpec (decompose step).
The LLM extracts raw values in the ExtractionResult (extract step).
This module runs the template against those values — no LLM arithmetic.
"""

import math
import re
import statistics

import numpy as np


class ComputeError(Exception):
    pass


# ── Unit conversion ──────────────────────────────────────────────────────────

UNIT_MULTIPLIERS: dict[str, float] = {
    "thousands": 1e3,
    "millions": 1e6,
    "billions": 1e9,
    "percent": 1.0,
}

# Canonical aliases that map to the same multiplier key
_UNIT_ALIASES: dict[str, str] = {
    "thousand": "thousands",
    "million": "millions",
    "billion": "billions",
    "%": "percent",
}


def parse_unit(raw: str | None) -> str | None:
    """Parse a table-header unit string into a canonical unit key.

    Handles strings like "In millions of dollars", "In thousands",
    "Percent", "billions", etc. Returns a key from UNIT_MULTIPLIERS
    or None if the unit can't be identified.
    """
    if not raw:
        return None
    text = raw.strip().lower()

    # Direct canonical match
    if text in UNIT_MULTIPLIERS:
        return text

    # Alias match
    if text in _UNIT_ALIASES:
        return _UNIT_ALIASES[text]

    # Scan for known scale words in the text
    for key in ("billions", "millions", "thousands", "percent"):
        if key in text:
            return key
    # Singular forms
    for singular, plural in _UNIT_ALIASES.items():
        if singular in text:
            return plural

    return None


def convert_unit(value, source_unit: str | None, target_unit: str | None):
    """Apply unit conversion when source and target units differ.

    Formula: converted = value * (source_multiplier / target_multiplier)
    E.g., source='thousands' (1e3) + target='millions' (1e6) → value / 1000.

    Returns the value unchanged if either unit is None or they match.
    Applies element-wise to lists/tuples.
    """
    if source_unit is None or target_unit is None:
        return value
    if source_unit == target_unit:
        return value

    src_mult = UNIT_MULTIPLIERS.get(source_unit)
    tgt_mult = UNIT_MULTIPLIERS.get(target_unit)
    if src_mult is None or tgt_mult is None:
        return value  # unrecognized unit — don't convert

    factor = src_mult / tgt_mult

    if isinstance(value, (list, tuple)):
        return [v * factor if isinstance(v, (int, float)) else v for v in value]
    if isinstance(value, (int, float)):
        return value * factor
    return value  # non-numeric — don't convert


def validate_extractions(spec: dict, extractions: dict) -> list[str]:
    """Return a list of warnings; empty if everything looks good."""
    warnings = []
    for dr in spec.get("data_requests", []):
        vid = dr["id"]
        if vid not in extractions:
            warnings.append(f"missing extraction for {vid}")
            continue
        vals = extractions[vid].get("values") or []
        nonnull = [v for v in vals if v is not None]
        expected = dr.get("expected_count") or 0
        if expected and len(nonnull) < max(1, int(expected * 0.8)):
            warnings.append(f"{vid}: only {len(nonnull)}/{expected} values extracted")
    return warnings


def execute(spec: dict, extractions: dict, verbose: bool = False):
    """Run the python_template against extracted values.

    Raises ComputeError on any failure.
    """
    values = {k: v.get("values", []) for k, v in extractions.items()}

    # Strip nulls so sum/mean/etc. don't blow up on None
    values = {k: [x for x in vs if x is not None] for k, vs in values.items()}

    template = spec.get("computation_spec", {}).get("python_template")
    if not template:
        raise ComputeError("no python_template in spec")

    # Collect year lists for regression templates that want actual years
    years_by_id = {}
    for dr in spec.get("data_requests", []):
        if dr.get("years"):
            years_by_id[dr["id"]] = list(dr["years"])

    # Lazy-import scipy so environments without it can still run sum/mean
    scipy_funcs = {}
    try:
        from scipy.stats import linregress, pearsonr

        scipy_funcs["linregress"] = linregress
        scipy_funcs["pearsonr"] = pearsonr
    except ImportError:
        pass

    # Expose common statistics helpers by bare name so templates can write
    # geometric_mean(xs) without qualifying statistics.geometric_mean.
    stats_helpers = {
        "mean": statistics.mean,
        "median": statistics.median,
        "stdev": statistics.stdev,
        "pstdev": statistics.pstdev,
        "variance": statistics.variance,
        "pvariance": statistics.pvariance,
        "geometric_mean": statistics.geometric_mean,
        "harmonic_mean": statistics.harmonic_mean,
        "fmean": statistics.fmean,
        "sqrt": math.sqrt,
        "log": math.log,
        "log10": math.log10,
        "exp": math.exp,
        "pow": math.pow,
    }

    local_vars = {
        "values": values,
        "years": years_by_id,
        "math": math,
        "statistics": statistics,
        "re": re,
        "np": np,
        "numpy": np,
        **stats_helpers,
        **scipy_funcs,
    }

    # CPI injection if needed
    if spec.get("cpi_needed"):
        try:
            from cpi import annual_cpi  # optional helper

            base = spec.get("cpi_base_year")
            target = spec.get("cpi_target_year")
            if base and target:
                local_vars["cpi_base"] = annual_cpi(base)
                local_vars["cpi_target"] = annual_cpi(target)
        except Exception as e:
            if verbose:
                print(f"  CPI injection failed: {e}")

    # Restricted builtins — allow common primitives needed by templates.
    # NOTE: __import__ is deliberately excluded to prevent arbitrary module
    # imports (security hardening). Pre-injected modules (math, statistics,
    # re, numpy) are available in the local namespace instead.
    safe_builtins = {
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "list": list,
        "tuple": tuple,
        "dict": dict,
        "float": float,
        "int": int,
        "round": round,
        "sorted": sorted,
        "zip": zip,
        "enumerate": enumerate,
        "map": map,
        "filter": filter,
        "print": print,
    }

    if verbose:
        print(f"  Executing template: {template}")

    try:
        exec(template, {"__builtins__": safe_builtins}, local_vars)  # nosec B102
    except Exception as e:
        raise ComputeError(f"template execution failed: {e}") from e

    if "result" not in local_vars:
        raise ComputeError("template did not assign 'result'")

    return local_vars["result"]


def format_result(result, output_format: dict, source_unit: str | None = None) -> str:
    """Format the computed result as a string for the reward function.

    If `source_unit` and output_format.unit differ, applies unit conversion
    before formatting. E.g., source='thousands', unit='millions' → divides
    by 1000.
    """
    if result is None:
        return "None"

    fmt = output_format or {}
    rtype = fmt.get("type", "number")
    rounding = fmt.get("rounding")
    target_unit = fmt.get("unit")

    # Apply unit conversion when source and target differ
    result = convert_unit(result, source_unit, target_unit)

    round_map = {
        "integer": 0,
        "tenths": 1,
        "hundredths": 2,
        "thousandths": 3,
        "ten_thousandths": 4,
    }
    digits = round_map.get(rounding)

    def _fmt_num(x):
        if digits is not None:
            x = round(float(x), digits)
        return str(x)

    if rtype == "list" or isinstance(result, (list, tuple)):
        items = [_fmt_num(x) if isinstance(x, (int, float)) else str(x) for x in result]
        return "[" + ", ".join(items) + "]"

    if isinstance(result, (int, float)):
        return _fmt_num(result)

    return str(result)


if __name__ == "__main__":
    # Quick self-test
    spec = {
        "data_requests": [{"id": "v1", "expected_count": 3, "years": [1940, 1941, 1942]}],
        "computation_spec": {"python_template": "result = sum(values['v1'])"},
        "output_format": {"type": "number"},
    }
    extractions = {"v1": {"values": [100, 200, 300]}}
    result = execute(spec, extractions, verbose=True)
    print(f"Result: {result}")
    print(f"Formatted: {format_result(result, spec['output_format'])}")
