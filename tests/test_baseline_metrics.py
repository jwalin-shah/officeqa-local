"""Tests for baseline metrics and decompose spec validation."""

import json
from pathlib import Path

BASELINE_PATH = Path("baseline_metrics.json")
DECOMPOSE_PATH = Path("decompose_eval.full.jsonl")


def test_baseline_metrics_exists():
    assert BASELINE_PATH.exists(), "baseline_metrics.json must exist at project root"


def test_baseline_metrics_schema():
    data = json.loads(BASELINE_PATH.read_text())

    # Required top-level keys
    assert "retrieval" in data
    assert "ledger" in data
    assert "oracle_extraction" in data

    # Retrieval fields
    r = data["retrieval"]
    assert "recall_at_1" in r
    assert "recall_at_5" in r
    assert "recall_at_10" in r
    assert isinstance(r["recall_at_1"], (int, float))
    assert isinstance(r["recall_at_5"], (int, float))
    assert isinstance(r["recall_at_10"], (int, float))

    # Ledger fields
    led = data["ledger"]
    assert "reachability_file" in led
    assert "reachability_page" in led
    assert isinstance(led["reachability_file"], (int, float))
    assert isinstance(led["reachability_page"], (int, float))

    # Oracle extraction fields
    oe = data["oracle_extraction"]
    assert "accuracy" in oe
    assert "n" in oe
    assert isinstance(oe["accuracy"], (int, float))
    assert isinstance(oe["n"], int)


def test_baseline_metrics_values_reasonable():
    data = json.loads(BASELINE_PATH.read_text())

    r = data["retrieval"]
    # Recall values should be between 0 and 100
    for key in ("recall_at_1", "recall_at_5", "recall_at_10"):
        assert 0 <= r[key] <= 100, f"{key}={r[key]} out of range"

    # recall@k should be monotonically increasing
    assert r["recall_at_1"] <= r["recall_at_5"] <= r["recall_at_10"]

    led = data["ledger"]
    assert 0 <= led["reachability_file"] <= 100
    assert 0 <= led["reachability_page"] <= 100

    oe = data["oracle_extraction"]
    assert 0 <= oe["accuracy"] <= 100
    assert oe["n"] > 0


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
    assert pct >= 95.0, f"row_hint coverage {pct:.1f}% < 95% ({has_row_hint}/{with_dr})"


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
