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


_DOLLAR_SCALE_UNITS = frozenset({"thousands", "millions", "billions"})


def convert_unit(value, source_unit: str | None, target_unit: str | None):
    """Apply unit conversion when source and target units differ.

    Formula: converted = value * (source_multiplier / target_multiplier)
    E.g., source='thousands' (1e3) + target='millions' (1e6) → value / 1000.

    Special case: target_unit=None with a dollar-scale source (thousands/millions/billions)
    means the output format didn't specify a unit → treat target as raw (×1), so
    values in thousands get ×1000 to reach raw dollars. Non-dollar units (percent,
    index, count) are left unchanged when target is None.

    Applies element-wise to lists/tuples.
    """
    if source_unit is None:
        return value
    if target_unit is None:
        # Only expand dollar-scale units to raw when target is unspecified.
        if source_unit not in _DOLLAR_SCALE_UNITS:
            return value
        src_mult = UNIT_MULTIPLIERS.get(source_unit)
        if src_mult is None:
            return value
        if isinstance(value, (list, tuple)):
            return [v * src_mult if isinstance(v, (int, float)) else v for v in value]
        if isinstance(value, (int, float)):
            return value * src_mult
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


# ── Enum dispatcher ──────────────────────────────────────────────────────────

_DISPATCH_FALLBACK = object()  # sentinel: dispatcher declines, use template

# Operation name aliases — decompose LLM sometimes varies the enum string.
_OP_ALIASES: dict[str, str] = {
    "mean": "average",
    "arithmetic_mean": "average",
    "avg": "average",
    "standard_deviation": "stdev_sample",
    "std_dev": "stdev_sample",
    "stdev": "stdev_sample",
    "correlation": "pearson_correlation",
    "pearson": "pearson_correlation",
    "linreg": "linear_regression",
    "regression": "linear_regression",
    "ols": "linear_regression",
    "percent_difference": "percent_change",
    "pct_change": "percent_change",
    "cv": "coefficient_of_variation",
    "count": "count",
}


def _first_list(values: dict, dr_ids: list[str]) -> list:
    """Return the values list for the first DR that has data."""
    for did in dr_ids:
        vs = values.get(did) or []
        if vs:
            return list(vs)
    return []


def _nth_list(values: dict, dr_ids: list[str], n: int) -> list:
    """Return values for the n-th DR (0-indexed) or [] if missing."""
    if n >= len(dr_ids):
        return []
    return list(values.get(dr_ids[n]) or [])


def _scalar(lst: list, which: str = "first") -> float | None:
    if not lst:
        return None
    return float(lst[0] if which == "first" else lst[-1])


def _gini_compute(vals: list[float]) -> float:
    if not vals:
        return 0.0
    sorted_vals = sorted(float(v) for v in vals)
    n = len(sorted_vals)
    total = sum(sorted_vals)
    if total == 0 or n == 0:
        return 0.0
    cum = sum((i + 1) * v for i, v in enumerate(sorted_vals))
    return (2 * cum / (n * total)) - (n + 1) / n


def _theil_compute(vals: list[float]) -> float:
    pos = [float(v) for v in vals if v and v > 0]
    if not pos:
        return 0.0
    mean_val = sum(pos) / len(pos)
    if mean_val == 0:
        return 0.0
    return sum((v / mean_val) * math.log(v / mean_val) for v in pos) / len(pos)


def _kl_compute(p: list[float], q: list[float]) -> float:
    total = 0.0
    for pi, qi in zip(p, q, strict=False):
        if pi and qi and pi > 0 and qi > 0:
            total += pi * math.log(pi / qi)
    return total


def _dispatch_enum(
    op: str,
    spec: dict,
    values: dict,
    labels_by_id: dict,
    years_by_id: dict,
    dr_ids: list[str],
    verbose: bool,
):
    """Hand-written implementations for each computation enum value.

    Return `_DISPATCH_FALLBACK` to signal the caller should run the
    python_template fallback instead. Raise `ComputeError` for hard failures
    where fallback is unlikely to help (e.g., empty inputs to difference).
    """
    op = _OP_ALIASES.get(op, op)
    params = spec.get("computation_params") or {}

    if verbose:
        print(f"  dispatch: op={op} drs={dr_ids}")

    first = _first_list(values, dr_ids)

    # ── Single-value lookups ────────────────────────────────────────────────
    if op == "direct":
        if not first:
            raise ComputeError("direct: no values extracted")
        return float(first[0])

    # ── Aggregates over a list ──────────────────────────────────────────────
    if op == "sum":
        if not first:
            raise ComputeError("sum: no values extracted")
        return float(sum(first))
    if op == "average":
        if not first:
            raise ComputeError("average: no values extracted")
        return statistics.mean(first)
    if op == "median":
        if not first:
            raise ComputeError("median: no values extracted")
        return statistics.median(first)
    if op == "geometric_mean":
        pos = [float(v) for v in first if v and v > 0]
        if not pos:
            raise ComputeError("geometric_mean: no positive values")
        return statistics.geometric_mean(pos)
    if op == "harmonic_mean":
        pos = [float(v) for v in first if v and v > 0]
        if not pos:
            raise ComputeError("harmonic_mean: no positive values")
        return statistics.harmonic_mean(pos)
    if op == "max":
        if not first:
            raise ComputeError("max: no values extracted")
        return float(max(first))
    if op == "min":
        if not first:
            raise ComputeError("min: no values extracted")
        return float(min(first))
    if op == "count":
        return len(first)
    if op == "stdev_sample":
        if len(first) < 2:
            raise ComputeError("stdev_sample: need ≥2 values")
        return statistics.stdev(first)
    if op == "stdev_population":
        if not first:
            raise ComputeError("stdev_population: no values extracted")
        return statistics.pstdev(first)
    if op == "variance_sample":
        if len(first) < 2:
            raise ComputeError("variance_sample: need ≥2 values")
        return statistics.variance(first)
    if op == "variance_population":
        if not first:
            raise ComputeError("variance_population: no values extracted")
        return statistics.pvariance(first)
    if op == "coefficient_of_variation":
        if len(first) < 2:
            raise ComputeError("cv: need ≥2 values")
        m = statistics.mean(first)
        if m == 0:
            raise ComputeError("cv: mean is zero")
        return statistics.stdev(first) / m

    # ── Argmax / argmin (needs labels) ─────────────────────────────────────
    if op in ("argmax", "argmin"):
        if not first:
            raise ComputeError(f"{op}: no values extracted")
        first_id = dr_ids[0] if dr_ids else None
        labels = labels_by_id.get(first_id) if first_id else []
        idx = first.index(max(first)) if op == "argmax" else first.index(min(first))
        if labels and idx < len(labels):
            return str(labels[idx])
        return idx

    # ── Two-DR scalar ops ──────────────────────────────────────────────────
    if op in ("difference", "abs_difference", "percent_change", "abs_percent_change", "ratio"):
        # Decompose sometimes picks a binary enum but emits >2 DRs (e.g.
        # CPI- or FX-adjusted differences). The enum dispatch can only
        # consume the first two — if there are more, the formula must be
        # custom, so fall through to the python_template instead of
        # silently dropping inputs.
        if len(dr_ids) > 2 and spec.get("computation_spec", {}).get("python_template"):
            return _DISPATCH_FALLBACK
        v1 = _first_list(values, dr_ids[:1])
        v2 = _nth_list(values, dr_ids, 1)
        if not v1 or not v2:
            raise ComputeError(f"{op}: need two non-empty DRs (got v1={len(v1)}, v2={len(v2)})")
        # If either side has >1 value (e.g. monthly_all), the scalar path
        # would silently drop 11 of the 12 values. Decompose typically emits
        # a `sum(values['v1'])` template for this case — fall through to it
        # so we aggregate before comparing.
        if (len(v1) > 1 or len(v2) > 1) and spec.get("computation_spec", {}).get("python_template"):
            return _DISPATCH_FALLBACK
        a, b = float(v1[0]), float(v2[0])
        if op == "difference":
            return b - a
        if op == "abs_difference":
            return abs(b - a)
        if op == "percent_change":
            if a == 0:
                raise ComputeError("percent_change: base is zero")
            return (b - a) / a * 100.0
        if op == "abs_percent_change":
            if a == 0:
                raise ComputeError("abs_percent_change: base is zero")
            return abs((b - a) / a) * 100.0
        if op == "ratio":
            if b == 0:
                raise ComputeError("ratio: denominator is zero")
            return a / b

    # ── CAGR ────────────────────────────────────────────────────────────────
    if op == "cagr":
        # Two forms: (a) one list → start=first, end=last, years from DR years
        #            (b) two DRs → v1=start, v2=end, years from params or DRs
        start = end = None
        n_years = None
        if len(dr_ids) >= 2 and values.get(dr_ids[0]) and values.get(dr_ids[1]):
            start = float(values[dr_ids[0]][0])
            end = float(values[dr_ids[1]][0])
            yrs1 = years_by_id.get(dr_ids[0]) or []
            yrs2 = years_by_id.get(dr_ids[1]) or []
            if yrs1 and yrs2:
                n_years = yrs2[-1] - yrs1[0]
        elif first and len(first) >= 2:
            start = float(first[0])
            end = float(first[-1])
            yrs = years_by_id.get(dr_ids[0]) or []
            n_years = yrs[-1] - yrs[0] if len(yrs) >= 2 else len(first) - 1
        if n_years is None:
            n_years = params.get("years") or params.get("n")
        if start is None or end is None or not n_years or start <= 0:
            raise ComputeError(f"cagr: invalid inputs (start={start}, end={end}, n={n_years})")
        return ((end / start) ** (1.0 / n_years) - 1.0) * 100.0

    # ── Linear regression (returns [slope, intercept]) ─────────────────────
    if op == "linear_regression":
        ys = first
        if len(ys) < 2:
            raise ComputeError("linear_regression: need ≥2 points")
        first_id = dr_ids[0] if dr_ids else None
        xs_cfg = (params.get("x") or "years") if isinstance(params, dict) else "years"
        if xs_cfg == "years":
            xs = years_by_id.get(first_id) or list(range(len(ys)))
        elif isinstance(xs_cfg, list):
            xs = list(xs_cfg)
        else:
            xs = list(range(len(ys)))
        if len(xs) != len(ys):
            # Align — trim to shortest
            n = min(len(xs), len(ys))
            xs, ys = xs[:n], ys[:n]
        try:
            from scipy.stats import linregress

            lr = linregress(xs, ys)
            return [float(lr.slope), float(lr.intercept)]
        except ImportError:
            n = len(xs)
            mx = sum(xs) / n
            my = sum(ys) / n
            num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
            den = sum((xs[i] - mx) ** 2 for i in range(n))
            if den == 0:
                raise ComputeError("linear_regression: zero variance in x") from None
            slope = num / den
            intercept = my - slope * mx
            return [float(slope), float(intercept)]

    # ── Pearson correlation ────────────────────────────────────────────────
    if op == "pearson_correlation":
        v1 = _first_list(values, dr_ids[:1])
        v2 = _nth_list(values, dr_ids, 1)
        if len(v1) < 2 or len(v2) < 2:
            raise ComputeError("pearson_correlation: need two lists ≥2 values")
        n = min(len(v1), len(v2))
        v1, v2 = v1[:n], v2[:n]
        try:
            from scipy.stats import pearsonr

            r, _ = pearsonr(v1, v2)
            return float(r)
        except ImportError:
            m1 = sum(v1) / n
            m2 = sum(v2) / n
            num = sum((v1[i] - m1) * (v2[i] - m2) for i in range(n))
            d1 = math.sqrt(sum((v1[i] - m1) ** 2 for i in range(n)))
            d2 = math.sqrt(sum((v2[i] - m2) ** 2 for i in range(n)))
            if d1 == 0 or d2 == 0:
                raise ComputeError("pearson_correlation: zero variance") from None
            return num / (d1 * d2)

    # ── Shape stats ────────────────────────────────────────────────────────
    if op in ("skewness", "skew"):
        if len(first) < 3:
            raise ComputeError("skewness: need ≥3 values")
        try:
            from scipy.stats import skew

            return float(skew(first))
        except ImportError:
            raise ComputeError("skewness: scipy not installed") from None
    if op == "kurtosis":
        if len(first) < 4:
            raise ComputeError("kurtosis: need ≥4 values")
        try:
            from scipy.stats import kurtosis

            return float(kurtosis(first))
        except ImportError:
            raise ComputeError("kurtosis: scipy not installed") from None

    # ── Inequality / divergence ────────────────────────────────────────────
    if op == "gini":
        if not first:
            raise ComputeError("gini: no values extracted")
        return _gini_compute(first)
    if op == "theil_index":
        if not first:
            raise ComputeError("theil_index: no values extracted")
        return _theil_compute(first)
    if op == "kl_divergence":
        p = _first_list(values, dr_ids[:1])
        q = _nth_list(values, dr_ids, 1)
        if not p or not q:
            raise ComputeError("kl_divergence: need two non-empty DRs")
        return _kl_compute(p, q)

    # ── Transforms with params ─────────────────────────────────────────────
    if op == "box_cox":
        if not first:
            raise ComputeError("box_cox: no values extracted")
        lam = params.get("lambda") if isinstance(params, dict) else None
        if lam is None:
            raise ComputeError("box_cox: missing lambda param")
        x = float(first[0])
        return (x**lam - 1.0) / lam

    if op == "cpi_adjust":
        if not first:
            raise ComputeError("cpi_adjust: no values extracted")
        base = spec.get("cpi_base_year")
        target = spec.get("cpi_target_year")
        try:
            from cpi import annual_cpi  # type: ignore[import-not-found]

            cpi_base = annual_cpi(base) if base else None
            cpi_target = annual_cpi(target) if target else None
        except Exception as e:
            raise ComputeError(f"cpi_adjust: cpi lookup failed: {e}") from e
        if not cpi_base or not cpi_target:
            raise ComputeError("cpi_adjust: missing cpi_base_year / cpi_target_year")
        return float(first[0]) * (cpi_target / cpi_base)

    if op == "fx_convert":
        if not first:
            raise ComputeError("fx_convert: no values extracted")
        rate = (params or {}).get("fx_rate") or spec.get("fx_rate")
        if rate is None:
            # Decompose often puts the FX rate in the second DR rather
            # than as a static param. Use that value if present.
            second = _nth_list(values, dr_ids, 1)
            if second:
                rate = second[0]
        if rate is None:
            raise ComputeError("fx_convert: missing fx_rate param")
        return float(first[0]) * float(rate)

    if op == "var_parametric":
        if len(first) < 2:
            raise ComputeError("var_parametric: need ≥2 values")
        confidence = (params or {}).get("confidence", 0.95)
        mu = statistics.mean(first)
        sigma = statistics.stdev(first)
        try:
            from scipy.stats import norm

            z = norm.ppf(confidence)
        except ImportError:
            z = 1.645 if abs(confidence - 0.95) < 1e-6 else 2.326
        return float(mu + z * sigma)

    # Unknown op → let the python_template handle it
    return _DISPATCH_FALLBACK


def execute(spec: dict, extractions: dict, verbose: bool = False):
    """Run the computation defined in `spec` against extracted values.

    Dispatches on `spec['computation']` enum first (e.g. sum, difference,
    linear_regression) — each operation is a hand-written Python function,
    so the LLM never has to author arithmetic. Only `custom` / unknown ops
    fall through to `computation_spec.python_template`.
    """
    values = {k: (v.get("values") or []) for k, v in extractions.items()}
    labels_by_id = {k: v.get("labels") or [] for k, v in extractions.items()}

    # Strip nulls so sum/mean/etc. don't blow up on None
    values = {k: [x for x in (vs or []) if x is not None] for k, vs in values.items()}

    # Collect year lists for regression templates that want actual years
    years_by_id: dict[str, list[int]] = {}
    for dr in spec.get("data_requests", []):
        if dr.get("years"):
            years_by_id[dr["id"]] = list(dr["years"])

    computation = (spec.get("computation") or "").strip().lower()
    dr_ids = [dr.get("id") for dr in spec.get("data_requests", []) if dr.get("id")]
    template = spec.get("computation_spec", {}).get("python_template") or ""

    # When the python_template encodes a transformation the enum can't
    # express (box-cox, log, exp, custom def/lambda, etc.), the enum is a
    # lossy summary and running it would silently return the wrong answer.
    # Prefer the template in that case.
    template_is_richer = bool(template) and bool(
        re.search(
            r"\bbox_cox\b|\blog\b|\bexp\b|\bsqrt\b|\bdef\s|\blambda\b|\*\*|"
            r"\bmath\.|\bscipy\.|\btransform|\bnormalize|\bnp\.|\bpolyfit\b|"
            r"\bprojected\b|\bpredict(ed)?\b|\bforecast\b|\bextrapolat\w*",
            template,
        )
    )
    # Regressions/correlations with more than one DR almost always mean the
    # template is using the regression output (slope/intercept) to do a
    # further step — project a value, compare against another series, etc.
    # The enum dispatcher would just return [slope, intercept] and drop the
    # second DR, silently producing the wrong answer.
    if (
        computation in ("linear_regression", "pearson_correlation")
        and len([d for d in spec.get("data_requests", []) if d.get("id")]) > 1
        and template
    ):
        template_is_richer = True

    # ── Dispatch path: hand-written op for a known enum ─────────────────────
    if computation and computation not in ("custom", "") and not template_is_richer:
        try:
            dispatched = _dispatch_enum(
                computation, spec, values, labels_by_id, years_by_id, dr_ids, verbose
            )
            if dispatched is not _DISPATCH_FALLBACK:
                return dispatched
        except ComputeError:
            raise
        except Exception as e:
            if verbose:
                print(f"  dispatch failed for {computation!r}: {e} — falling back to template")

    template = spec.get("computation_spec", {}).get("python_template")
    if not template:
        raise ComputeError(f"no python_template in spec (computation={computation!r})")

    # Lazy-import scipy so environments without it can still run sum/mean
    scipy_funcs = {}
    try:
        from scipy.stats import kurtosis as _kurtosis
        from scipy.stats import linregress, pearsonr
        from scipy.stats import skew as _skew

        scipy_funcs["linregress"] = linregress
        scipy_funcs["pearsonr"] = pearsonr
        scipy_funcs["kurtosis"] = _kurtosis
        scipy_funcs["skew"] = _skew
        scipy_funcs["skewness"] = _skew  # alias used by decompose prompt
    except ImportError:
        pass

    # ── Common helper functions (available by bare name in templates) ───

    def _gini(values_list):
        """Compute Gini coefficient for a list of values."""
        if not values_list:
            return 0.0
        sorted_vals = sorted(values_list)
        n = len(sorted_vals)
        if n == 0:
            return 0.0
        mean_val = sum(sorted_vals) / n
        if mean_val == 0:
            return 0.0
        cum = 0
        for i, v in enumerate(sorted_vals):
            cum += (i + 1) * v
        return (2 * cum / (n * sum(sorted_vals))) - (n + 1) / n

    def _theil_index(values_list):
        """Compute Theil inequality index for a list of values."""
        if not values_list:
            return 0.0
        vals = [v for v in values_list if v and v > 0]
        if not vals:
            return 0.0
        import math as _m

        mean_val = sum(vals) / len(vals)
        if mean_val == 0:
            return 0.0
        n = len(vals)
        total = sum(v / mean_val * _m.log(v / mean_val) for v in vals if v > 0)
        return total / n

    def _kl_divergence(p_list, q_list):
        """Compute KL divergence KL(P||Q) between two probability lists."""
        if not p_list or not q_list:
            return 0.0
        import math as _m

        total = 0.0
        for p, q in zip(p_list, q_list, strict=False):
            if p > 0 and q > 0:
                total += p * _m.log(p / q)
        return total

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
        "gini": _gini,
        "theil_index": _theil_index,
        "kl_divergence": _kl_divergence,
        # ── Operation-name aliases ────────────────────────────────────
        # The decompose LLM sometimes uses the operation enum name from
        # the prompt (e.g., "stdev_population") instead of the Python
        # function name ("pstdev"). Map the enum names to their real
        # implementations so templates work regardless.
        "stdev_population": statistics.pstdev,
        "stdev_sample": statistics.stdev,
        "pearson_correlation": scipy_funcs.get("pearsonr", lambda x, y: (0.0, 0.0)),
        "cagr": lambda start, end, years: (
            (end / start) ** (1 / years) - 1 if start and start > 0 and years > 0 else 0.0
        ),
        # Probability normalization — templates sometimes use p_norm(xs) to
        # convert a list of counts/weights into probabilities summing to 1.
        "p_norm": lambda xs: [x / sum(xs) for x in xs] if xs and sum(xs) != 0 else xs,
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
    # Expose each DR id as a bare name — decompose sometimes writes
    # `v1[i]` instead of `values['v1'][i]`, and silently failing on the
    # name lookup would otherwise convert a valid extraction into a
    # compute error. `values` takes precedence for lookups that clash.
    # Also expose `{dr_id}_labels` so templates can do argmax by label.
    for _did, _vs in values.items():
        if _did and _did not in local_vars:
            local_vars[_did] = _vs
    for _did, _lbls in labels_by_id.items():
        _lbl_key = f"{_did}_labels"
        if _lbl_key not in local_vars:
            local_vars[_lbl_key] = _lbls

    # CPI injection if needed
    if spec.get("cpi_needed"):
        try:
            from cpi import annual_cpi  # type: ignore[import-unresolved]  # optional helper

            base = spec.get("cpi_base_year")
            target = spec.get("cpi_target_year")
            if base and target:
                local_vars["cpi_base"] = annual_cpi(base)
                local_vars["cpi_target"] = annual_cpi(target)
        except Exception as e:
            if verbose:
                print(f"  CPI injection failed: {e}")

    # Restricted builtins — allow common primitives needed by templates.
    # __import__ is replaced with _safe_import to prevent arbitrary module
    # imports (security hardening). Only pre-approved modules (math,
    # statistics, re, numpy, scipy) can be imported; `import os` etc. fail.
    _ALLOWED_MODULES = frozenset(
        {"math", "statistics", "re", "numpy", "np", "scipy", "scipy.stats"}
    )

    def _safe_import(name, *args, **kwargs):
        """Restricted import that only allows pre-approved modules."""
        if name in _ALLOWED_MODULES:
            return __import__(name, *args, **kwargs)  # noqa: PLC0414
        raise ImportError(f"import of '{name}' is not allowed in compute sandbox")

    safe_builtins = {
        "__import__": _safe_import,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "range": range,
        "all": all,
        "any": any,
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

    # Guard: detect empty DRs referenced in the template before exec so the
    # error message names the missing DRs instead of crashing with opaque
    # "list index out of range" or "KeyError" inside the sandbox.
    _empty_referenced: list[str] = []
    for _did, _vs in values.items():
        if (
            not _vs
            and _did
            and (
                _did in template
                or f"values['{_did}']" in template
                or f'values["{_did}"]' in template
            )
        ):
            _empty_referenced.append(_did)
    if _empty_referenced:
        raise ComputeError(f"template references DRs with no extracted values: {_empty_referenced}")

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
    if isinstance(rounding, str):
        digits = round_map.get(rounding)
    elif isinstance(rounding, (int, float)):
        digits = int(rounding)
    else:
        digits = None

    def _fmt_num(x):
        if digits is not None:
            x = round(float(x), digits)
        return str(x)

    if isinstance(result, (list, tuple)):
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
