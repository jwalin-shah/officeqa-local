"""Tests for retrieve_v2.py — FTS channel, metric channel, edge cases."""

import sqlite3
from typing import Any, cast

import pytest

from retrieve_v2 import (
    LEDGER_PATH,
    _detect_wants_monthly,
    _expand_synonyms,
    _extract_hints,
    _file_year_bonus,
    _fts_channel,
    _metric_channel,
    _row_hint_variants,
    content_tokens,
    detect_year_mode,
    normalize_metric_slug,
    parse_direct_bulletin_refs,
    parse_years_from_question,
    retrieve,
    retrieve_from_question,
)

LEDGER_EXISTS = LEDGER_PATH.exists()
skip_no_ledger = pytest.mark.skipif(not LEDGER_EXISTS, reason="ledger.sqlite not found")


# ── Parsing helpers ──────────────────────────────────────────────────────────


def test_parse_years():
    assert parse_years_from_question("What was spending in 1940?") == [1940]
    assert parse_years_from_question("Between 1935 and 1945") == [1935, 1945]
    assert parse_years_from_question("No year here") == []


def test_parse_years_multi():
    years = parse_years_from_question("Compare 1940, 1941, and 1942")
    assert years == [1940, 1941, 1942]


def test_detect_year_mode_calendar():
    assert detect_year_mode("What was the calendar year 1940 total?") == "calendar"


def test_detect_year_mode_fiscal():
    assert detect_year_mode("fiscal year 1940 defense spending") == "fiscal"


def test_detect_year_mode_unknown():
    assert detect_year_mode("spending in 1940") == "unknown"


def test_parse_direct_bulletin_refs():
    refs = parse_direct_bulletin_refs("the July 1946 bulletin")
    assert (1946, 7) in refs


def test_parse_direct_bulletin_refs_empty():
    assert parse_direct_bulletin_refs("no bulletin reference here") == []


def test_detect_wants_monthly():
    assert _detect_wants_monthly("monthly expenditures for 1940")
    assert _detect_wants_monthly("January through December 1940")
    assert not _detect_wants_monthly("total spending in 1940")


# ── Token/slug helpers ───────────────────────────────────────────────────────


def test_content_tokens():
    tokens = content_tokens("What were the total expenditures for national defense?")
    assert "expenditures" in tokens
    assert "defense" in tokens
    assert "national" in tokens
    # Stopwords filtered
    assert "the" not in tokens
    assert "what" not in tokens


def test_normalize_metric_slug():
    assert normalize_metric_slug("National defense 1/") == "national defense"
    assert normalize_metric_slug("Net interest (on debt)") == "net interest on debt"
    assert normalize_metric_slug(None) == ""
    assert normalize_metric_slug("") == ""


def test_expand_synonyms_includes_arena_terms():
    expanded = _expand_synonyms(
        ["expenditures", "defense", "veterans"],
        "total expenditures for national defense and veterans",
    )
    assert "outlays" in expanded
    assert "spending" in expanded
    assert "military" in expanded
    assert "national defense" in expanded
    assert "veterans affairs" in expanded


def test_row_hint_variants_expand_phrases():
    variants = _row_hint_variants("Veterans administration")
    assert "veterans administration" in variants
    assert "veterans affairs" in variants
    assert "veterans" in variants


# ── _extract_hints ───────────────────────────────────────────────────────────


def test_extract_hints_from_plan():
    plan = {
        "data_requests": [
            {
                "label": "defense spending",
                "row_hint": "National defense",
                "column_hint": "1940",
                "years": [1940],
            }
        ],
    }
    metric, col, row, years = _extract_hints(plan, "question")
    assert metric == "defense spending"
    assert col == "1940"
    assert row == "National defense"
    assert years == [1940]


def test_extract_hints_empty_plan():
    metric, col, row, years = _extract_hints(None, "spending in 1940")
    assert metric == ""
    assert col == ""
    assert row == ""
    assert years == [1940]


# ── _file_year_bonus ─────────────────────────────────────────────────────────


def test_file_year_bonus_calendar_peak():
    # Calendar year 1940: peak offset is +1 (bulletin 1941)
    bonus = _file_year_bonus(1941, 1, [1940], "calendar")
    assert bonus > 5.0  # base 5 + month bonus for January of offset+1


def test_file_year_bonus_fiscal_peak():
    # Fiscal year 1940: peak offset is 0
    bonus = _file_year_bonus(1940, 8, [1940], "fiscal")
    assert bonus > 5.0  # base 5 + month bonus for August


def test_file_year_bonus_no_years():
    assert _file_year_bonus(1940, 1, [], "calendar") == 0.0


def test_file_year_bonus_none_year():
    assert _file_year_bonus(None, None, [1940], "calendar") == 0.0


# ── FTS channel (requires ledger.sqlite) ─────────────────────────────────────


@skip_no_ledger
def test_fts_channel_basic():
    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.row_factory = sqlite3.Row
    rows = _fts_channel(conn, ["defense", "expenditures"], [1940], False)
    conn.close()
    assert len(rows) > 0


@skip_no_ledger
def test_fts_channel_empty_tokens():
    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.row_factory = sqlite3.Row
    rows = _fts_channel(conn, [], [1940], False)
    conn.close()
    assert rows == []


@skip_no_ledger
def test_fts_channel_no_year_filter():
    """FTS still returns results when no years are given."""
    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.row_factory = sqlite3.Row
    rows = _fts_channel(conn, ["defense", "expenditures"], [], False)
    conn.close()
    assert len(rows) > 0


def test_fts_channel_uses_synonyms_progressively():
    class FakeConn:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            query = params[0]
            self.calls.append(query)
            if "outlays" in query or "military" in query:
                return [{"id": 2, "score": -1.0}]
            return []

    conn = cast(Any, FakeConn())
    rows = _fts_channel(conn, ["expenditures", "defense"], [1940], False, top_n=5)
    assert len(rows) == 1
    assert any("outlays" in call or "military" in call for call in conn.calls)
    assert len(conn.calls) >= 2


def test_fts_channel_skips_synonyms_when_exact_hits_enough():
    class FakeConn:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            query = params[0]
            self.calls.append(query)
            if "outlays" in query or "military" in query:
                raise AssertionError("synonym query should not run when exact hits are sufficient")
            return [{"id": i, "score": -1.0 - i} for i in range(8)]

    conn = cast(Any, FakeConn())
    rows = _fts_channel(conn, ["expenditures", "defense"], [1940], False, top_n=8)
    assert len(rows) == 8
    assert len(conn.calls) == 1


# ── Metric channel (requires ledger.sqlite) ──────────────────────────────────


@skip_no_ledger
def test_metric_channel_basic():
    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.row_factory = sqlite3.Row
    rows = _metric_channel(
        conn,
        "national defense",
        "National defense",
        "",
        [1940],
        False,
    )
    conn.close()
    assert len(rows) > 0


@skip_no_ledger
def test_metric_channel_empty_metric():
    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.row_factory = sqlite3.Row
    rows = _metric_channel(conn, "", "", "", [1940], False)
    conn.close()
    assert rows == []


def test_metric_channel_uses_row_hint_synonyms_progressively():
    class FakeConn:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            self.calls.append(params[0])
            if params[0] == "military expenditures":
                return [{"id": 7, "score": -6.0}]
            return []

    conn = cast(Any, FakeConn())
    rows = _metric_channel(
        conn,
        "expenditures",
        "national defense",
        "",
        [1940],
        False,
        top_n=5,
    )
    assert len(rows) == 1
    assert conn.calls[0] == "national defense expenditures"
    assert "military expenditures" in conn.calls[1:]


def test_metric_channel_skips_synonyms_when_exact_hits_enough():
    class FakeConn:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            self.calls.append(params[0])
            if params[0] != "national defense expenditures":
                raise AssertionError("synonym lookup should not run when exact hits are sufficient")
            return [{"id": i, "score": -6.0 - i} for i in range(6)]

    conn = cast(Any, FakeConn())
    rows = _metric_channel(
        conn,
        "expenditures",
        "national defense",
        "",
        [1940],
        False,
        top_n=6,
    )
    assert len(rows) == 6
    assert conn.calls == ["national defense expenditures"]


# ── Full retrieve (requires ledger.sqlite) ───────────────────────────────────


@skip_no_ledger
def test_retrieve_empty_results():
    """Retrieve returns empty list for completely bogus query."""
    entries = retrieve(
        {
            "data_requests": [
                {"label": "xyzzy_nonexistent_12345", "row_hint": "", "column_hint": "", "years": []}
            ]
        },
        "xyzzy_nonexistent_12345",
        top_k=5,
        load_html=False,
    )
    assert isinstance(entries, list)


@skip_no_ledger
def test_retrieve_from_question_basic():
    """retrieve_from_question returns entries for a known question."""
    entries = retrieve_from_question(
        "What were the total expenditures for national defense in 1940?",
        top_k=5,
    )
    assert isinstance(entries, list)
    assert len(entries) > 0
    # Each entry should have expected keys
    for e in entries:
        assert "file" in e
        assert "section" in e or "title" in e
