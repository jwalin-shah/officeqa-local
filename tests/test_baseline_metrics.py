"""Tests for baseline metrics and decompose spec validation."""

import json
from pathlib import Path

BASELINE_PATH = Path("baseline_metrics.json")
DECOMPOSE_PATH = Path("decompose_eval.full.jsonl")

_BASELINE_METRICS_CACHE: dict | None = None


def load_baseline_metrics() -> dict:
    """Parse ``baseline_metrics.json`` once per process (shared by tests)."""
    global _BASELINE_METRICS_CACHE
    if _BASELINE_METRICS_CACHE is None:
        _BASELINE_METRICS_CACHE = json.loads(BASELINE_PATH.read_text())
    return _BASELINE_METRICS_CACHE


EXPECTED_TOP_LEVEL_KEYS = frozenset(
    {
        "retrieval",
        "retrieval_baseline_pre_improvement",
        "ledger",
        "oracle_extraction",
        "unit_conversion",
        "decompose_validation",
    }
)

RETRIEVAL_SNAPSHOT_KEYS = frozenset(
    {
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "recall_at_20",
        "recall_at_30",
        "file_level_recall_at_10",
        "total_questions",
        "elapsed_s",
    }
)

RECALL_AT_RANK_KEYS = (
    "recall_at_1",
    "recall_at_5",
    "recall_at_10",
    "recall_at_20",
    "recall_at_30",
)

LEDGER_KEYS = frozenset(
    {
        "reachability_file",
        "reachability_page",
        "reachability_page_any",
        "cell_present_page_numeric_pct",
        "total_questions",
    }
)

ORACLE_EXTRACTION_KEYS = frozenset({"accuracy", "n", "note"})

UNIT_CONVERSION_KEYS = frozenset(
    {
        "parse_unit_coverage",
        "format_result_source_unit",
        "test_coverage",
    }
)

DECOMPOSE_VALIDATION_KEYS = frozenset(
    {
        "total_lines",
        "parse_errors",
        "valid_specs",
        "missing_specs",
        "row_hint_present",
        "row_hint_pct",
        "row_hint_pct_of_total",
    }
)


def _assert_exact_keys(obj: dict, expected: frozenset[str], label: str) -> None:
    got = frozenset(obj)
    assert got == expected, f"{label}: keys {sorted(got)} != expected {sorted(expected)}"


def _assert_retrieval_snapshot_schema(obj: dict, label: str) -> None:
    _assert_exact_keys(obj, RETRIEVAL_SNAPSHOT_KEYS, label)
    for k in RECALL_AT_RANK_KEYS:
        assert isinstance(obj[k], (int, float)), f"{label}.{k} must be numeric"
    assert isinstance(obj["file_level_recall_at_10"], (int, float)), (
        f"{label}.file_level_recall_at_10 must be numeric"
    )
    assert isinstance(obj["total_questions"], int), f"{label}.total_questions must be int"
    assert isinstance(obj["elapsed_s"], (int, float)), f"{label}.elapsed_s must be numeric"


def _assert_retrieval_snapshot_values(obj: dict, label: str) -> None:
    for k in RECALL_AT_RANK_KEYS:
        v = obj[k]
        assert 0 <= v <= 100, f"{label}.{k}={v} out of range [0, 100]"
    fl = obj["file_level_recall_at_10"]
    assert 0 <= fl <= 100, f"{label}.file_level_recall_at_10={fl} out of range [0, 100]"
    ranks = [obj[k] for k in RECALL_AT_RANK_KEYS]
    for i in range(len(ranks) - 1):
        a, b = ranks[i], ranks[i + 1]
        assert a <= b, (
            f"{label}: recall@* not non-decreasing at "
            f"{RECALL_AT_RANK_KEYS[i]} <= {RECALL_AT_RANK_KEYS[i + 1]}: {ranks}"
        )
    assert obj["total_questions"] > 0, f"{label}.total_questions must be positive"
    assert obj["elapsed_s"] >= 0, f"{label}.elapsed_s must be non-negative"


def test_baseline_metrics_exists():
    assert BASELINE_PATH.exists(), "baseline_metrics.json must exist at project root"


def test_baseline_metrics_schema():
    data = load_baseline_metrics()
    _assert_exact_keys(data, EXPECTED_TOP_LEVEL_KEYS, "top-level")

    _assert_retrieval_snapshot_schema(data["retrieval"], "retrieval")
    _assert_retrieval_snapshot_schema(
        data["retrieval_baseline_pre_improvement"],
        "retrieval_baseline_pre_improvement",
    )

    led = data["ledger"]
    _assert_exact_keys(led, LEDGER_KEYS, "ledger")
    for k in (
        "reachability_file",
        "reachability_page",
        "reachability_page_any",
        "cell_present_page_numeric_pct",
    ):
        assert isinstance(led[k], (int, float)), f"ledger.{k} must be numeric"
    assert isinstance(led["total_questions"], int), "ledger.total_questions must be int"

    oe = data["oracle_extraction"]
    _assert_exact_keys(oe, ORACLE_EXTRACTION_KEYS, "oracle_extraction")
    assert isinstance(oe["accuracy"], (int, float)), "oracle_extraction.accuracy must be numeric"
    assert isinstance(oe["n"], int), "oracle_extraction.n must be int"
    assert isinstance(oe["note"], str), "oracle_extraction.note must be str"

    uc = data["unit_conversion"]
    _assert_exact_keys(uc, UNIT_CONVERSION_KEYS, "unit_conversion")
    assert isinstance(uc["parse_unit_coverage"], str), (
        "unit_conversion.parse_unit_coverage must be str"
    )
    assert isinstance(uc["format_result_source_unit"], bool), (
        "unit_conversion.format_result_source_unit must be bool"
    )
    assert isinstance(uc["test_coverage"], str), "unit_conversion.test_coverage must be str"

    dv = data["decompose_validation"]
    _assert_exact_keys(dv, DECOMPOSE_VALIDATION_KEYS, "decompose_validation")
    for k in (
        "total_lines",
        "parse_errors",
        "valid_specs",
        "missing_specs",
        "row_hint_present",
    ):
        assert isinstance(dv[k], int), f"decompose_validation.{k} must be int"
    for k in ("row_hint_pct", "row_hint_pct_of_total"):
        assert isinstance(dv[k], (int, float)), f"decompose_validation.{k} must be numeric"


def test_baseline_metrics_values_reasonable():
    data = load_baseline_metrics()

    for block_name in ("retrieval", "retrieval_baseline_pre_improvement"):
        _assert_retrieval_snapshot_values(data[block_name], block_name)

    led = data["ledger"]
    for k in (
        "reachability_file",
        "reachability_page",
        "reachability_page_any",
        "cell_present_page_numeric_pct",
    ):
        assert 0 <= led[k] <= 100, f"ledger.{k}={led[k]} out of range [0, 100]"
    assert led["total_questions"] > 0

    oe = data["oracle_extraction"]
    assert 0 <= oe["accuracy"] <= 100
    assert oe["n"] > 0

    dv = data["decompose_validation"]
    assert dv["parse_errors"] == 0
    assert dv["valid_specs"] + dv["missing_specs"] == dv["total_lines"]
    assert 0 <= dv["row_hint_pct"] <= 100
    assert 0 <= dv["row_hint_pct_of_total"] <= 100
    assert dv["row_hint_present"] <= dv["valid_specs"]


def test_decompose_specs_parse():
    """All 246 lines in decompose_eval.full.jsonl parse as valid JSON."""
    assert DECOMPOSE_PATH.exists(), "decompose_eval.full.jsonl must exist"

    total = 0
    parse_errors = 0
    with open(DECOMPOSE_PATH) as f:
        for _i, line in enumerate(f, 1):
            total += 1
            try:
                json.loads(line)
            except json.JSONDecodeError:
                parse_errors += 1

    assert total == 246, f"Expected 246 lines, got {total}"
    assert parse_errors == 0, f"{parse_errors} lines failed to parse"


def test_decompose_specs_row_hint_coverage():
    """≥95% of specs with data_requests have non-empty row_hint on first DR."""
    with_dr = 0
    has_row_hint = 0

    with open(DECOMPOSE_PATH) as f:
        for line in f:
            obj = json.loads(line)
            spec = obj.get("spec")
            if not spec:
                continue
            drs = spec.get("data_requests") or []
            if not drs:
                continue
            with_dr += 1
            rh = (drs[0].get("row_hint") or "").strip()
            if rh:
                has_row_hint += 1

    assert with_dr > 0, "No specs with data_requests found"
    pct = has_row_hint / with_dr * 100
    # Cached specs drift slightly with decompose changes; gate obvious regressions only.
    assert pct >= 92.0, f"row_hint coverage {pct:.1f}% < 92% ({has_row_hint}/{with_dr})"


def test_decompose_specs_have_required_fields():
    """Specs with data_requests have required structure."""
    with open(DECOMPOSE_PATH) as f:
        for line in f:
            obj = json.loads(line)
            assert "uid" in obj, "Missing uid"
            assert "question" in obj, "Missing question"

            spec = obj.get("spec")
            if not spec:
                continue  # Some specs may have errors

            drs = spec.get("data_requests") or []
            for dr in drs:
                assert "id" in dr, f"DR missing 'id' in {obj['uid']}"
                assert "label" in dr, f"DR missing 'label' in {obj['uid']}"
