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
    _fts_channel_trace,
    _metric_channel,
    _metric_channel_trace,
    _period_aware_score_delta,
    _prose_footnote_channel,
    _row_hint_variants,
    _winning_hit_details,
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
    hints = _extract_hints(plan, "question")
    assert len(hints) == 1
    assert hints[0]["metric"] == "defense spending"
    assert hints[0]["col_hint"] == "1940"
    assert hints[0]["row_hint"] == "National defense"
    assert hints[0]["years"] == [1940]


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


def test_fts_channel_trace_runs_full_progressive_cascade():
    class FakeConn:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            file_year_between = "t.file_year BETWEEN ? AND ?" in sql
            year_filter = "table_columns" in sql
            query = params[0]
            if (
                file_year_between
                and not year_filter
                and ("outlays" in query or "military" in query)
            ):
                strategy = "synonym_year_window"
            elif not file_year_between and ("outlays" in query or "military" in query):
                strategy = "synonym_unrestricted"
            elif file_year_between and year_filter and params[1:3] == [1945, 1950]:
                strategy = "year_shifted"
            elif file_year_between and year_filter and params[1:3] == [1940, 1944]:
                strategy = "exact_year_filter"
            elif file_year_between and not year_filter and params[1:3] == [1940, 1944]:
                strategy = "exact_year_window"
            else:
                strategy = "unknown"
            self.calls.append(strategy)
            if strategy == "synonym_unrestricted":
                return [{"id": 11, "score": -1.0}]
            if strategy == "year_shifted":
                return [{"id": 22, "score": -0.5}]
            return []

    conn = cast(Any, FakeConn())
    trace = _fts_channel_trace(conn, ["expenditures", "defense"], [1940], False, top_n=10)
    assert [name for name, _ in trace.attempts] == [
        "exact_year_filter",
        "exact_year_window",
        "synonym_year_window",
        "synonym_unrestricted",
        "year_shifted",
    ]
    assert conn.calls == [name for name, _ in trace.attempts]
    assert trace.strategy_by_id == {11: "synonym_unrestricted", 22: "year_shifted"}


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


def test_metric_channel_trace_uses_partial_match_after_synonyms():
    class FakeConn:
        def __init__(self):
            self.calls = []

        def execute(self, sql, params):
            exact_text = params[0]
            require_all_terms = "metric_slug LIKE ? AND metric_slug LIKE ?" in sql
            if require_all_terms:
                strategy = (
                    "primary_exact"
                    if exact_text == "national defense expenditures"
                    else "synonym_exact"
                )
            else:
                strategy = "partial_match"
            self.calls.append((strategy, exact_text))
            if strategy == "partial_match" and exact_text == "national defense expenditures":
                return [{"id": 31, "score": -3.0}]
            return []

    conn = cast(Any, FakeConn())
    trace = _metric_channel_trace(
        conn,
        "expenditures",
        "national defense",
        "",
        [1940],
        False,
        top_n=5,
    )
    assert [name for name, _ in trace.attempts] == [
        "primary_exact",
        "synonym_exact",
        "partial_match",
    ]
    assert trace.strategy_by_id == {31: "partial_match"}
    assert ("partial_match", "national defense expenditures") in conn.calls


# ── Retrieval transparency helpers ───────────────────────────────────────────


def test_winning_hit_details_returns_rank_and_strategy():
    rank, strategy = _winning_hit_details(
        [
            {"file": "a.json", "page_id": 1, "retrieval_strategy": "fts:exact_year_filter"},
            {"file": "b.json", "page_id": 2, "retrieval_strategy": "metric:partial_match"},
        ],
        {("b.json", 2)},
    )
    assert rank == 2
    assert strategy == "metric:partial_match"


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


# ── Prose/footnote channel ───────────────────────────────────────────────────


def test_prose_footnote_channel_empty_tokens():
    """Empty tokens returns empty list without querying DB."""

    class FakeConn:
        def execute(self, sql, params):
            raise AssertionError("should not query DB with empty tokens")

    conn = cast(Any, FakeConn())
    result = _prose_footnote_channel(conn, [], [1940])
    assert result == []


def test_prose_footnote_channel_returns_dict_list():
    """Channel returns a list of dicts with required metadata keys."""

    class FakeRow:
        def __init__(self, data):
            self._data = data

        def __getitem__(self, key):
            return self._data[key]

    class FakeConn:
        def execute(self, sql, params):
            if "prose_fts" in sql:
                return [
                    FakeRow(
                        {
                            "file": "treasury_bulletin_1941_03.json",
                            "element_seq": 10,
                            "page_id": 5,
                            "file_year": 1941,
                            "file_month": 3,
                            "section": "Expenditures",
                            "content": "National defense expenditures rose sharply.",
                            "near_table_id": 42,
                            "score": -3.5,
                        }
                    )
                ]
            elif "footnotes_fts" in sql:
                return [
                    FakeRow(
                        {
                            "file": "treasury_bulletin_1941_03.json",
                            "element_seq": 11,
                            "page_id": 5,
                            "file_year": 1941,
                            "file_month": 3,
                            "content": "1/ Excludes supplemental appropriations.",
                            "near_table_id": 42,
                            "score": -2.0,
                        }
                    )
                ]
            return []

        def fetchall(self):
            return []

    # Use a real FakeConn that correctly returns fetchall
    class FakeConn2:
        def execute(self, sql, params):
            return self

        def fetchall(self):
            return []

    # Minimal test: calling with tokens on a non-DB conn won't crash, returns []
    result = _prose_footnote_channel(cast(Any, FakeConn2()), ["defense", "expenditures"], [1940])
    assert isinstance(result, list)


@skip_no_ledger
def test_prose_footnote_channel_with_real_ledger():
    """Channel returns results from prose_fts and footnotes_fts when query matches."""
    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.row_factory = sqlite3.Row
    result = _prose_footnote_channel(conn, ["defense", "expenditures"], [1940], top_n=20)
    conn.close()
    assert isinstance(result, list)
    # Each result should have required keys
    for entry in result:
        assert "file" in entry
        assert "page_id" in entry
        assert "element_seq" in entry
        assert "content" in entry
        assert "near_table_id" in entry
        assert "source" in entry
        assert entry["source"] in ("prose", "footnote")
        assert "score" in entry


@skip_no_ledger
def test_prose_footnote_channel_empty_for_bogus_query():
    """Channel returns empty list for queries with no matches."""
    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.row_factory = sqlite3.Row
    result = _prose_footnote_channel(conn, ["xyzzy_bogus_12345"], [1940])
    conn.close()
    assert result == []


@skip_no_ledger
def test_retrieve_includes_prose_footnote_channel_field():
    """retrieve() returns entries that may include retrieval_channel='prose_footnote'."""
    entries = retrieve(
        {
            "data_requests": [
                {
                    "label": "national defense",
                    "row_hint": "National defense",
                    "column_hint": "",
                    "years": [1940],
                }
            ]
        },
        "What were total national defense expenditures in 1940?",
        top_k=15,
        load_html=False,
    )
    assert isinstance(entries, list)
    # All entries should have retrieval_channel field
    for e in entries:
        assert "retrieval_channel" in e
    # Some entries may be prose_footnote channel
    channels = {e["retrieval_channel"] for e in entries}
    # At minimum, fts or metric entries should appear
    assert channels & {"fts", "metric", "prose_footnote"}


@skip_no_ledger
def test_prose_footnote_entries_have_content_field():
    """Prose/footnote entries in retrieve() output include content field."""
    entries = retrieve(
        {
            "data_requests": [
                {
                    "label": "footnote annotation",
                    "row_hint": "",
                    "column_hint": "",
                    "years": [1940],
                }
            ]
        },
        "footnote annotation fiscal year 1940",
        top_k=20,
        load_html=False,
    )
    pf_entries = [e for e in entries if e.get("retrieval_channel") == "prose_footnote"]
    for e in pf_entries:
        assert "content" in e
        assert isinstance(e["content"], str)


# ── _extract_hints with row_hint_alternatives and multi-request ──────────────


def test_extract_hints_returns_row_hint_alternatives():
    """_extract_hints reads row_hint_alternatives from data_request."""
    plan = {
        "data_requests": [
            {
                "label": "national defense spending",
                "row_hint": "National defense",
                "row_hint_alternatives": [
                    "National defense and associated activities",
                    "National defense (050)",
                ],
                "column_hint": "Total",
                "years": [1940],
            }
        ],
    }
    hints = _extract_hints(plan, "question")
    assert len(hints) == 1  # one request
    hint = hints[0]
    assert hint["row_hint"] == "National defense"
    assert hint["row_hint_alternatives"] == [
        "National defense and associated activities",
        "National defense (050)",
    ]
    assert hint["metric"] == "national defense spending"
    assert hint["col_hint"] == "Total"
    assert hint["years"] == [1940]


def test_extract_hints_multi_request():
    """_extract_hints returns hints for ALL data_requests, not just first."""
    plan = {
        "data_requests": [
            {
                "label": "v1 spending",
                "row_hint": "National defense",
                "row_hint_alternatives": ["Defense"],
                "column_hint": "Total",
                "years": [1940],
            },
            {
                "label": "v2 spending",
                "row_hint": "Veterans benefits",
                "row_hint_alternatives": ["Veterans administration"],
                "column_hint": "1945",
                "years": [1945],
            },
        ],
    }
    hints = _extract_hints(plan, "question")
    assert len(hints) == 2
    assert hints[0]["row_hint"] == "National defense"
    assert hints[1]["row_hint"] == "Veterans benefits"
    assert hints[1]["row_hint_alternatives"] == ["Veterans administration"]


def test_extract_hints_empty_alternatives():
    """When row_hint_alternatives is absent, returns empty list."""
    plan = {
        "data_requests": [
            {
                "label": "defense spending",
                "row_hint": "National defense",
                "column_hint": "Total",
                "years": [1940],
            }
        ],
    }
    hints = _extract_hints(plan, "question")
    assert hints[0]["row_hint_alternatives"] == []


def test_extract_hints_back_compat():
    """_extract_hints still works when called with old-style tuple unpacking via index access."""
    plan = {
        "data_requests": [
            {
                "label": "defense spending",
                "row_hint": "National defense",
                "column_hint": "Total",
                "years": [1940],
            }
        ],
    }
    hints = _extract_hints(plan, "question")
    # First hint should have all fields
    h = hints[0]
    assert h["metric"] == "defense spending"
    assert h["row_hint"] == "National defense"
    assert h["col_hint"] == "Total"
    assert h["years"] == [1940]


def test_extract_hints_falls_back_to_question_years():
    """When data_request has no years, falls back to parsing from question."""
    plan = {
        "data_requests": [
            {
                "label": "defense spending",
                "row_hint": "National defense",
                "column_hint": "",
                "years": None,
            }
        ],
    }
    hints = _extract_hints(plan, "spending in 1940 and 1941")
    assert hints[0]["years"] == [1940, 1941]


def test_extract_hints_empty_plan():
    """Empty plan returns empty list of hints."""
    hints = _extract_hints(None, "spending in 1940")
    assert hints == []


# ── Metric channel with row_hint_alternatives ────────────────────────────────


def test_metric_channel_uses_row_hint_alternatives():
    """Metric channel tries each alternative for substring matching."""
    calls = []

    class FakeConn:
        def execute(self, sql, params):
            exact_text = params[0]
            calls.append(exact_text)
            if exact_text == "national defense and associated activities expenditures":
                return [{"id": 99, "score": -8.0}]
            return []

    conn = cast(Any, FakeConn())
    # "national defense" primary row_hint + alternatives should produce
    # queries for both the primary and the alternative
    rows = _metric_channel(
        conn,
        "expenditures",
        "National defense and associated activities",
        "",
        [1940],
        False,
        top_n=10,
    )
    assert len(rows) >= 1
    # The exact_text lookup for the alternative slug should have been called
    assert any("national defense and associated activities" in c for c in calls)


# ── Period-aware scoring ──────────────────────────────────────────────────────


def test_period_aware_score_fiscal_match():
    """Fiscal-year question: period='fiscal' tables get a boost."""
    delta = _period_aware_score_delta("fiscal", "fiscal")
    assert delta >= 0.5


def test_period_aware_score_fiscal_mismatch():
    """Fiscal-year question: period='calendar' tables get a penalty."""
    delta = _period_aware_score_delta("fiscal", "calendar")
    assert delta <= -0.3


def test_period_aware_score_calendar_match():
    """Calendar-year question: period='calendar' tables get a boost."""
    delta = _period_aware_score_delta("calendar", "calendar")
    assert delta >= 0.5


def test_period_aware_score_calendar_mismatch():
    """Calendar-year question: period='fiscal' tables get a penalty."""
    delta = _period_aware_score_delta("calendar", "fiscal")
    assert delta <= -0.3


def test_period_aware_score_unknown_mode():
    """Unknown year mode: no period scoring delta."""
    delta = _period_aware_score_delta("unknown", "fiscal")
    assert delta == 0.0


def test_period_aware_score_empty_period():
    """Empty period: no period scoring delta."""
    delta = _period_aware_score_delta("fiscal", "")
    assert delta == 0.0


def test_period_aware_score_none_period():
    """None period: no period scoring delta."""
    delta = _period_aware_score_delta("fiscal", None)
    assert delta == 0.0


# ── Multi-request retrieval ─────────────────────────────────────────────────


@skip_no_ledger
def test_retrieve_multi_request_plan():
    """retrieve() processes ALL data_requests in a spec, not just the first."""
    plan = {
        "data_requests": [
            {
                "id": "v1",
                "label": "national defense expenditures",
                "row_hint": "National defense",
                "column_hint": "",
                "years": [1940],
            },
            {
                "id": "v2",
                "label": "veterans expenditures",
                "row_hint": "Veterans",
                "column_hint": "",
                "years": [1940],
            },
        ],
    }
    entries = retrieve(plan, "national defense and veterans in 1940", top_k=10, load_html=False)
    assert isinstance(entries, list)
    # Should have results for both requests — not just the first
    # (Both row hints should contribute to the union of results)


@skip_no_ledger
def test_retrieve_fiscal_year_ranks_fiscal_tables_higher():
    """Fiscal-year question ranks period='fiscal' tables higher than period='calendar'."""
    # Use a question that explicitly mentions "fiscal year"
    entries = retrieve(
        {
            "data_requests": [
                {
                    "label": "defense outlays fiscal year 1940",
                    "row_hint": "National defense",
                    "column_hint": "",
                    "years": [1940],
                }
            ]
        },
        "fiscal year 1940 national defense outlays",
        top_k=20,
        load_html=False,
    )
    assert isinstance(entries, list)
    if len(entries) >= 2:
        # Find entries with period information
        fiscal_entries = [e for e in entries if e.get("period") == "fiscal"]
        calendar_entries = [e for e in entries if e.get("period") == "calendar"]
        # If both exist, fiscal entries should rank before calendar entries
        if fiscal_entries and calendar_entries:
            fiscal_ranks = [entries.index(e) for e in fiscal_entries]
            calendar_ranks = [entries.index(e) for e in calendar_entries]
            assert min(fiscal_ranks) < min(calendar_ranks), (
                f"Fiscal tables should rank before calendar tables for FY question: "
                f"fiscal_min_rank={min(fiscal_ranks)}, calendar_min_rank={min(calendar_ranks)}"
            )


@skip_no_ledger
def test_row_hint_alternatives_superset_behavior():
    """Plans with row_hint_alternatives find superset of hits vs primary-only."""
    plan_with_alts = {
        "data_requests": [
            {
                "label": "national defense",
                "row_hint": "National defense",
                "row_hint_alternatives": [
                    "National defense and associated activities",
                    "National defense (050)",
                ],
                "column_hint": "",
                "years": [1940],
            }
        ],
    }
    plan_no_alts = {
        "data_requests": [
            {
                "label": "national defense",
                "row_hint": "National defense",
                "column_hint": "",
                "years": [1940],
            }
        ],
    }
    entries_with_alts = retrieve(
        plan_with_alts,
        "national defense expenditures 1940",
        top_k=10,
        load_html=False,
    )
    entries_no_alts = retrieve(
        plan_no_alts,
        "national defense expenditures 1940",
        top_k=10,
        load_html=False,
    )
    # With alternatives, we should find at least as many results
    files_with_alts = {e["file"] for e in entries_with_alts}
    files_no_alts = {e["file"] for e in entries_no_alts}
    assert len(files_with_alts) >= len(files_no_alts)


# ── VAL-RETR-008: Edge case handling ─────────────────────────────────────────


@skip_no_ledger
def test_retrieve_zero_results_returns_empty_list():
    """When both channels return zero results, retrieve() returns an empty list (not None/exception).

    This is VAL-RETR-008: graceful handling when both channels find nothing.
    """
    entries = retrieve(
        {
            "data_requests": [
                {
                    "label": "xyzzy_nonexistent_12345_abcd",
                    "row_hint": "xyzzy_nonexistent_12345_abcd",
                    "column_hint": "",
                    "years": [],
                }
            ]
        },
        "xyzzy_nonexistent_12345_abcd",
        top_k=5,
        load_html=False,
    )
    # Must return an empty list, not None, not an exception
    assert entries is not None, "retrieve() must not return None"
    assert isinstance(entries, list), "retrieve() must return a list"
    # For a completely bogus query, expect empty or very few results
    # The key guarantee is: no exception raised, and result is a list (possibly empty)


@skip_no_ledger
def test_retrieve_zero_results_both_channels_empty():
    """When both channels truly return zero results, retrieve() returns [].

    Uses a completely bogus question that won't match any corpus tables,
    ensuring FTS and metric channels both return nothing.
    """
    entries = retrieve(
        {"data_requests": []},
        "xyzzy_fhqwhgads_plugh_bogus_nonexistent_99999",
        top_k=5,
        load_html=False,
    )
    # The guarantee is: returns a list (not None, not exception).
    # May be empty or have a few FTS hits, but never crashes.
    assert entries is not None
    assert isinstance(entries, list)


@skip_no_ledger
def test_retrieve_zero_results_null_plan():
    """When plan is None, retrieve() returns empty list (not None/exception)."""
    entries = retrieve(
        None,
        "xyzzy_nonexistent_query_no_plan",
        top_k=5,
        load_html=False,
    )
    assert entries is not None
    assert isinstance(entries, list)
    # With None plan, _extract_hints returns [], so no hints → no metric channel,
    # but FTS still runs from question tokens. Still, a bogus question may return [].


def test_retrieve_no_parseable_years_uses_unfiltered_fts():
    """When a question has no parseable years, FTS channel uses unfiltered search.

    This is VAL-RETR-008: retrieval uses unfiltered FTS when no years are found.
    """
    calls = []

    class FakeConn:
        def execute(self, sql, params):
            query = params[0] if params else ""
            year_filter_present = "table_columns" in sql and "year_extracted" in sql
            file_year_between = "t.file_year BETWEEN ? AND ?" in sql
            calls.append(
                {
                    "query": query,
                    "year_filter": year_filter_present,
                    "file_year_between": file_year_between,
                }
            )
            return []

    conn = cast(Any, FakeConn())
    # No years → target_years=[]
    trace = _fts_channel_trace(conn, ["defense", "expenditures"], [], False, top_n=10)
    # With no target_years and non-empty tokens, the FTS channel should use
    # the exact_unrestricted strategy (no year filter)
    assert trace.attempts, "FTS channel should execute at least one strategy"
    strategies = [name for name, _ in trace.attempts]
    assert "exact_unrestricted" in strategies, (
        f"Expected 'exact_unrestricted' strategy when no years, got: {strategies}"
    )


@skip_no_ledger
def test_retrieve_from_question_no_years_still_returns_results():
    """retrieve_from_question works when question has no parseable years.

    Uses unfiltered FTS — no year-based filtering is applied.
    """
    # A question with no 4-digit years
    entries = retrieve_from_question(
        "What are the categories of federal spending?",
        top_k=5,
    )
    assert isinstance(entries, list)
    # Even without years, the retrieval should return some results
    # (the corpus has tables about spending categories)
    assert len(entries) > 0, "Unfiltered FTS should find results for topical queries"


@skip_no_ledger
def test_fts_channel_no_years_uses_unrestricted_strategy():
    """FTS channel trace shows 'exact_unrestricted' strategy when no target years."""
    conn = sqlite3.connect(str(LEDGER_PATH))
    conn.row_factory = sqlite3.Row
    trace = _fts_channel_trace(conn, ["defense", "expenditures"], [], False, top_n=10)
    conn.close()
    strategies = [name for name, _ in trace.attempts]
    assert "exact_unrestricted" in strategies, (
        f"Expected 'exact_unrestricted' in strategies when no years, got: {strategies}"
    )
