"""Tests for extract.py — html_to_pipe_text(), html_to_vertical_text(), rendering helpers, mocked extract_structured().

Also tests for the deterministic fast-path in solve.py (_try_deterministic_fast_path)."""

from unittest.mock import MagicMock, patch

from extract import (
    build_context_from_entries,
    clean_value,
    html_to_pipe_text,
    html_to_vertical_text,
    render_entry,
    to_num,
)

# ── html_to_pipe_text() ─────────────────────────────────────────────────────


def test_html_to_pipe_simple():
    html = (
        "<table>"
        "<tr><th>Category</th><th>1940</th><th>1941</th></tr>"
        "<tr><td>Defense</td><td>1580</td><td>2602</td></tr>"
        "</table>"
    )
    result = html_to_pipe_text(html)
    assert "| Category | 1940 | 1941 |" in result
    assert "| Defense | 1580 | 2602 |" in result


def test_html_to_pipe_empty():
    assert html_to_pipe_text("") == ""
    assert html_to_pipe_text(None) == ""  # type: ignore[arg-type]


def test_html_to_pipe_no_rows():
    html = "<table></table>"
    assert html_to_pipe_text(html) == ""


def test_html_to_pipe_max_rows():
    rows_html = "".join(f"<tr><td>Row {i}</td><td>{i}</td></tr>" for i in range(100))
    html = f"<table>{rows_html}</table>"
    result = html_to_pipe_text(html, max_rows=10)
    lines = result.strip().split("\n")
    # 10 data rows + 1 truncation message
    assert len(lines) == 11
    assert "truncated" in lines[-1].lower()


def test_html_to_pipe_nested_tags():
    html = "<table><tr><td><b>Bold</b> text</td><td>123</td></tr></table>"
    result = html_to_pipe_text(html)
    assert "Bold text" in result or "Bold  text" in result


# ── render_entry() ───────────────────────────────────────────────────────────


def test_render_entry_basic():
    entry = {
        "file": "treasury_bulletin_1941_01.json",
        "element_id": 5,
        "title": "Budget Receipts",
        "section": "Table 3",
        "caption": "",
        "html": (
            "<table>"
            "<tr><th>Item</th><th>1940</th></tr>"
            "<tr><td>Defense</td><td>1580</td></tr>"
            "</table>"
        ),
    }
    result = render_entry(entry)
    assert "treasury_bulletin_1941_01.json" in result
    assert "Budget Receipts" in result
    assert "| Defense | 1580 |" in result


def test_render_entry_no_html():
    entry = {"file": "test.json", "html": ""}
    assert render_entry(entry) == ""


# ── build_context_from_entries() ─────────────────────────────────────────────


def test_build_context_budget():
    entries = [
        {
            "file": "test.json",
            "element_id": 1,
            "title": "Table A",
            "section": "",
            "caption": "",
            "html": (
                "<table><tr><th>Row</th><th>Val</th></tr><tr><td>A</td><td>100</td></tr></table>"
            ),
        },
    ]
    ctx = build_context_from_entries(entries, char_budget=10000)
    assert "test.json" in ctx
    assert "| A | 100 |" in ctx


def test_build_context_budget_limit():
    entry = {
        "file": "test.json",
        "element_id": 1,
        "title": "T",
        "section": "",
        "caption": "",
        "html": (
            "<table>"
            "<tr><th>X</th></tr>"
            + "".join(f"<tr><td>{'A' * 200}</td></tr>" for _ in range(50))
            + "</table>"
        ),
    }
    ctx = build_context_from_entries([entry], char_budget=500)
    assert len(ctx) <= 600  # budget + some slack for truncation


# ── clean_value() / to_num() ────────────────────────────────────────────────


def test_clean_value_footnotes():
    assert clean_value("1,580 3/") == "1,580"
    assert clean_value("r/123") == "123"


def test_clean_value_negatives():
    assert clean_value("(523)") == "-523"


def test_to_num_basic():
    assert to_num("1,580") == 1580.0
    assert to_num("-523") == -523.0


def test_to_num_missing():
    assert to_num("nan") is None
    assert to_num("-") is None
    assert to_num("*") is None
    assert to_num("") is None


def test_to_num_dollar_percent():
    assert to_num("$1,234") == 1234.0
    assert to_num("5.5%") == 5.5


# ── extract_structured() with mocked LLM ────────────────────────────────────


def test_extract_structured_mock():
    """extract_structured returns parsed extractions from mocked LLM response."""
    from extract import extract_structured as _extract_structured

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content='{"extractions": {"v1": {"values": [100, 200], '
                '"labels": ["Jan", "Feb"], "source_file": "test.json", '
                '"confidence": "high"}}, "notes": "ok"}'
            )
        )
    ]

    spec = {
        "data_requests": [
            {
                "id": "v1",
                "label": "monthly values",
                "years": [1940],
                "granularity": "monthly_all",
            }
        ],
    }
    per_dr_entries = {
        "v1": [
            {
                "file": "test.json",
                "element_id": 1,
                "title": "T",
                "section": "",
                "caption": "",
                "html": (
                    "<table>"
                    "<tr><th>Month</th><th>Value</th></tr>"
                    "<tr><td>Jan</td><td>100</td></tr>"
                    "<tr><td>Feb</td><td>200</td></tr>"
                    "</table>"
                ),
            }
        ],
    }

    with patch("extract.client") as mock_client:
        mock_client.chat.completions.create.return_value = mock_response
        result = _extract_structured(spec, per_dr_entries, "test question")

    assert result is not None
    assert "extractions" in result
    assert result["extractions"]["v1"]["values"] == [100, 200]


def test_extract_structured_no_data_requests():
    """extract_structured returns None when no data_requests."""
    from extract import extract_structured as _extract_structured

    result = _extract_structured({}, {}, "test")
    assert result is None


# ── html_to_vertical_text() ─────────────────────────────────────────────────


def _make_12col_monthly_html() -> str:
    """Build a 12-column monthly table HTML (Jan–Dec 1940)."""
    months = [
        "Jan.",
        "Feb.",
        "Mar.",
        "Apr.",
        "May",
        "Jun.",
        "Jul.",
        "Aug.",
        "Sep.",
        "Oct.",
        "Nov.",
        "Dec.",
    ]
    header_cells = "<th>Category</th>" + "".join(f"<th>{m}</th>" for m in months)
    value_cells = "<td>National defense</td>" + "".join(
        f"<td>{v}</td>" for v in [132, 129, 143, 159, 154, 153, 177, 200, 219, 287, 376, 473]
    )
    return f"<table><tr>{header_cells}</tr><tr>{value_cells}</tr></table>"


def test_vertical_12_column_monthly_table():
    """Tables with ≥8 columns render in vertical format with month annotations."""
    html = _make_12col_monthly_html()
    result = html_to_vertical_text(html)
    assert "ROW: National defense" in result
    assert "Jan. (month 1): 132" in result
    assert "Dec. (month 12): 473" in result


def test_vertical_month_indices_chronological():
    """Month indices follow calendar order (Jan=1, Dec=12)."""
    html = _make_12col_monthly_html()
    result = html_to_vertical_text(html)
    # Verify all 12 month indices are present in order
    for i in range(1, 13):
        assert f"(month {i})" in result


def test_vertical_fiscal_year_column_order():
    """Fiscal-year-ordered columns (Oct→Sep) are annotated with month numbers."""
    # Fiscal year columns: Oct, Nov, Dec, Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep
    fy_months = [
        "Oct.",
        "Nov.",
        "Dec.",
        "Jan.",
        "Feb.",
        "Mar.",
        "Apr.",
        "May",
        "Jun.",
        "Jul.",
        "Aug.",
        "Sep.",
    ]
    header_cells = "<th>Category</th>" + "".join(f"<th>{m}</th>" for m in fy_months)
    value_cells = "<td>Defense</td>" + "".join(f"<td>{v}</td>" for v in range(10, 22))
    html = f"<table><tr>{header_cells}</tr><tr>{value_cells}</tr></table>"
    result = html_to_vertical_text(html)
    assert "ROW: Defense" in result
    # Oct is month 10 in FY ordering
    assert "Oct. (month 10): 10" in result
    # Sep is month 9
    assert "Sep. (month 9): 21" in result


def test_vertical_empty_html():
    """Empty/null HTML returns empty string."""
    assert html_to_vertical_text("") == ""
    assert html_to_vertical_text(None) == ""  # type: ignore[arg-type]


def test_vertical_narrow_table_not_used():
    """Narrow tables (<8 cols) should not be rendered vertically by render_entry."""
    # 3-column table — below default threshold
    html = (
        "<table>"
        "<tr><th>Category</th><th>1940</th><th>1941</th></tr>"
        "<tr><td>Defense</td><td>1580</td><td>2602</td></tr>"
        "</table>"
    )
    entry = {
        "file": "test.json",
        "element_id": 1,
        "title": "T",
        "section": "",
        "caption": "",
        "html": html,
    }
    result = render_entry(entry)
    # Should use pipe-delimited, not vertical format
    assert "| Defense | 1580 | 2602 |" in result


def test_vertical_wide_table_used_in_render_entry():
    """Wide tables (≥8 cols) use vertical format in render_entry."""
    html = _make_12col_monthly_html()
    entry = {
        "file": "test.json",
        "element_id": 1,
        "title": "Monthly Defense Spending",
        "section": "",
        "caption": "",
        "html": html,
    }
    result = render_entry(entry)
    # Should use vertical format, not pipe-delimited
    assert "ROW: National defense" in result
    assert "Jan. (month 1): 132" in result
    # Should NOT contain pipe-delimited row for this data
    assert "| National defense | 132 |" not in result


def test_vertical_configurable_threshold():
    """The column threshold is configurable via vertical_threshold param."""
    # 5-column table — below default threshold of 8
    header = "<tr><th>A</th><th>B</th><th>C</th><th>D</th><th>E</th></tr>"
    data = "<tr><td>Row1</td><td>1</td><td>2</td><td>3</td><td>4</td></tr>"
    html = f"<table>{header}{data}</table>"
    # Default threshold (8) → pipe format
    result_default = html_to_vertical_text(html)
    assert result_default == ""  # <8 cols → not rendered vertically

    # Lower threshold (5) → vertical format
    result_low = html_to_vertical_text(html, vertical_threshold=5)
    assert "ROW: Row1" in result_low


def test_vertical_skips_missing_values():
    """Vertical format skips nan/empty/missing cells."""
    header = "<tr><th>Category</th>" + "".join(f"<th>C{i}</th>" for i in range(1, 9)) + "</tr>"
    data = "<tr><td>Test</td><td>1</td><td>nan</td><td></td><td>-</td><td>5</td><td>*</td><td>7</td><td>8</td></tr>"
    html = f"<table>{header}{data}</table>"
    result = html_to_vertical_text(html)
    assert "ROW: Test" in result
    assert "C1: 1" in result
    assert "C5: 5" in result
    assert "C8: 8" in result
    # nan, empty, -, * should be skipped
    assert "C2: nan" not in result
    assert "C3:" not in result  # empty
    assert "C4: -" not in result


def test_vertical_multiple_rows():
    """Vertical format renders multiple rows, each with ROW: prefix."""
    header = "<tr><th>Category</th>" + "".join(f"<th>C{i}</th>" for i in range(1, 9)) + "</tr>"
    row1 = "<tr><td>Defense</td>" + "".join(f"<td>{i}</td>" for i in range(1, 9)) + "</tr>"
    row2 = "<tr><td>Veterans</td>" + "".join(f"<td>{i + 10}</td>" for i in range(1, 9)) + "</tr>"
    html = f"<table>{header}{row1}{row2}</table>"
    result = html_to_vertical_text(html)
    assert "ROW: Defense" in result
    assert "ROW: Veterans" in result
    assert "C1: 1" in result
    assert "C1: 11" in result


def test_vertical_max_rows():
    """Vertical format respects max_rows parameter."""
    header = "<tr><th>Category</th>" + "".join(f"<th>C{i}</th>" for i in range(1, 9)) + "</tr>"
    rows = "".join(
        f"<tr><td>Row {i}</td>" + "".join(f"<td>{i}</td>" for _ in range(8)) + "</tr>"
        for i in range(100)
    )
    html = f"<table>{header}{rows}</table>"
    result = html_to_vertical_text(html, max_rows=5)
    # Should have 5 ROW: labels + a truncation note
    row_count = result.count("ROW:")
    assert row_count == 5
    assert "truncated" in result.lower()


def test_vertical_non_month_columns_no_month_annotation():
    """Non-month column headers don't get month annotations."""
    header = (
        "<tr><th>Category</th>" + "".join(f"<th>Year {y}</th>" for y in range(1935, 1943)) + "</tr>"
    )
    data = "<tr><td>Defense</td>" + "".join(f"<td>{y}</td>" for y in range(1935, 1943)) + "</tr>"
    html = f"<table>{header}{data}</table>"
    result = html_to_vertical_text(html)
    assert "ROW: Defense" in result
    # No month annotations for year columns
    assert "(month" not in result
    # Still has column header: value
    assert "Year 1935:" in result


# ── Deterministic fast-path (_try_deterministic_fast_path in solve.py) ──────


def _make_annual_spec() -> dict:
    """Build a minimal spec with one annual data_request."""
    return {
        "computation": "direct",
        "data_requests": [
            {
                "id": "v1",
                "label": "National defense expenditures 1940",
                "source": "corpus",
                "row_hint": "National defense",
                "column_hint": "1940",
                "years": [1940],
                "granularity": "annual",
                "expected_count": 1,
                "cohort": False,
            }
        ],
        "computation_spec": {"python_template": "result = values['v1'][0]"},
        "output_format": {
            "type": "number",
            "unit": "millions",
            "rounding": None,
            "as_percent": False,
        },
    }


def _make_monthly_spec() -> dict:
    """Build a minimal spec with one monthly_all data_request."""
    return {
        "computation": "sum",
        "data_requests": [
            {
                "id": "v1",
                "label": "Monthly national defense expenditures CY 1940",
                "source": "corpus",
                "row_hint": "National defense",
                "column_hint": "",
                "years": [1940],
                "granularity": "monthly_all",
                "expected_count": 12,
                "cohort": False,
            }
        ],
        "computation_spec": {"python_template": "result = sum(values['v1'])"},
        "output_format": {
            "type": "number",
            "unit": "millions",
            "rounding": None,
            "as_percent": False,
        },
    }


def _make_multi_dr_spec() -> dict:
    """Build a spec with two annual data_requests."""
    return {
        "computation": "percent_change",
        "data_requests": [
            {
                "id": "v1",
                "label": "Defense expenditures 1938",
                "source": "corpus",
                "row_hint": "National defense",
                "column_hint": "1938",
                "years": [1938],
                "granularity": "annual",
                "expected_count": 1,
                "cohort": False,
            },
            {
                "id": "v2",
                "label": "Defense expenditures 1940",
                "source": "corpus",
                "row_hint": "National defense",
                "column_hint": "1940",
                "years": [1940],
                "granularity": "annual",
                "expected_count": 1,
                "cohort": False,
            },
        ],
        "computation_spec": {
            "python_template": "result = (values['v2'][0] - values['v1'][0]) / values['v1'][0] * 100"
        },
        "output_format": {
            "type": "number",
            "unit": "percent",
            "rounding": "hundredths",
            "as_percent": False,
        },
    }


def _make_table_entry(table_id: int = 42) -> dict:
    """Build a mock retrieve_v2 entry with a known file/element_seq."""
    return {
        "file": "treasury_bulletin_1941_01.json",
        "element_id": 5,
        "element_seq": 7,
        "page_id": 12,
        "file_year": 1941,
        "file_month": 1,
        "section": "Table 3",
        "title": "Budget Receipts and Outlays",
        "caption": "",
        "column_headers": ["Fiscal year", "Total", "National defense"],
        "row_labels": ["1938", "1939", "1940"],
        "years": [1938, 1939, 1940],
        "unit": "millions_usd",
        "period": "fiscal",
        "n_rows": 5,
        "n_cols": 3,
        "retrieval_strategy": "exact",
        "retrieval_channel": "metric",
        "html": "<table><tr><th>Year</th><th>Total</th><th>National defense</th></tr>"
        "<tr><td>1940</td><td>9468</td><td>1580</td></tr></table>",
    }


def test_fast_path_all_resolve_annual():
    """When all data_requests resolve deterministically, fast-path returns
    extractions dict without calling the LLM."""
    from solve import _try_deterministic_fast_path

    spec = _make_annual_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    # Mock: _get_table_id_from_entry returns 42, _build_cells_for_dr builds one cell,
    # resolve_cells returns a resolved value
    with (
        patch("solve._get_table_id_from_entry", return_value=42),
        patch(
            "solve._build_cells_for_dr",
            return_value=[{"row_leaf": "National defense", "col_leaf": "1940", "name": "v1"}],
        ),
        patch("solve.resolve_cells") as mock_resolve,
    ):
        mock_resolve.return_value = {
            "values": {"v1": 1580.0},
            "debug": {"v1": {"status": "ok", "raw": "1580", "num": 1580.0}},
        }
        result = _try_deterministic_fast_path(spec, per_dr, verbose=True)

    assert result is not None
    assert "extractions" in result
    assert result["extractions"]["v1"]["values"] == [1580.0]
    assert result["extractions"]["v1"]["confidence"] == "deterministic"
    # LLM was NOT called — resolve_cells was the only function invoked
    mock_resolve.assert_called_once()


def test_fast_path_partial_resolve_falls_back():
    """When resolve_cells returns None for any value, fast-path returns None
    and the caller should fall back to LLM extraction."""
    from solve import _try_deterministic_fast_path

    spec = _make_annual_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    with (
        patch("solve._get_table_id_from_entry", return_value=42),
        patch(
            "solve._build_cells_for_dr",
            return_value=[{"row_leaf": "National defense", "col_leaf": "1940", "name": "v1"}],
        ),
        patch("solve.resolve_cells") as mock_resolve,
    ):
        mock_resolve.return_value = {
            "values": {"v1": None},
            "debug": {"v1": {"status": "label_miss", "row_found": True, "col_found": False}},
        }
        result = _try_deterministic_fast_path(spec, per_dr, verbose=True)

    # Fast-path should fail (None), falling back to LLM
    assert result is None


def test_fast_path_external_source_graceful():
    """When a data_request has source='external', fast-path skips it
    and returns None (not a crash)."""
    from solve import _try_deterministic_fast_path

    spec = _make_annual_spec()
    # Override the source to 'external'
    spec["data_requests"][0]["source"] = "external"
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    result = _try_deterministic_fast_path(spec, per_dr, verbose=True)
    # Should return None gracefully, not raise
    assert result is None


def test_fast_path_cpi_source_skipped():
    """When a data_request has source='cpi', fast-path skips it."""
    from solve import _try_deterministic_fast_path

    spec = _make_annual_spec()
    spec["data_requests"][0]["source"] = "cpi"
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    result = _try_deterministic_fast_path(spec, per_dr, verbose=True)
    assert result is None


def test_fast_path_no_retrieved_entries():
    """When per_dr_entries has no entries for a DR, fast-path returns None."""
    from solve import _try_deterministic_fast_path

    spec = _make_annual_spec()
    per_dr = {"v1": []}  # empty entries

    result = _try_deterministic_fast_path(spec, per_dr, verbose=True)
    assert result is None


def test_fast_path_no_table_entries_only_pf():
    """When only prose/footnote entries exist (no table entries), fast-path returns None."""
    from solve import _try_deterministic_fast_path

    spec = _make_annual_spec()
    # Prose/footnote entries have 'content' but no 'html' with data
    pf_entry = {
        "file": "test.json",
        "element_id": None,
        "element_seq": 99,
        "page_id": 1,
        "section": "",
        "title": "",
        "caption": "",
        "content": "Some footnote text",
        "retrieval_channel": "prose_footnote",
        "html": "",
    }
    per_dr = {"v1": [pf_entry]}

    result = _try_deterministic_fast_path(spec, per_dr, verbose=True)
    assert result is None


def test_fast_path_table_id_not_found():
    """When the table_id can't be resolved from the entry, fast-path returns None."""
    from solve import _try_deterministic_fast_path

    spec = _make_annual_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    with patch("solve._get_table_id_from_entry", return_value=None):
        result = _try_deterministic_fast_path(spec, per_dr, verbose=True)

    assert result is None


def test_fast_path_cant_build_cells():
    """When _build_cells_for_dr returns None (unsupported granularity), fast-path returns None."""
    from solve import _try_deterministic_fast_path

    spec = _make_annual_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    with (
        patch("solve._get_table_id_from_entry", return_value=42),
        patch("solve._build_cells_for_dr", return_value=None),
    ):
        result = _try_deterministic_fast_path(spec, per_dr, verbose=True)

    assert result is None


def test_fast_path_multi_dr_all_resolve():
    """When multiple data_requests all resolve, fast-path returns extractions
    for all DRs."""
    from solve import _try_deterministic_fast_path

    spec = _make_multi_dr_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries, "v2": entries}

    with (
        patch("solve._get_table_id_from_entry", return_value=42),
        patch("solve._build_cells_for_dr") as mock_build,
        patch("solve.resolve_cells") as mock_resolve,
    ):
        # First call for v1, second for v2
        mock_build.side_effect = [
            [{"row_leaf": "National defense", "col_leaf": "1938", "name": "v1"}],
            [{"row_leaf": "National defense", "col_leaf": "1940", "name": "v2"}],
        ]
        mock_resolve.side_effect = [
            {"values": {"v1": 1200.0}, "debug": {}},
            {"values": {"v2": 1580.0}, "debug": {}},
        ]
        result = _try_deterministic_fast_path(spec, per_dr, verbose=True)

    assert result is not None
    assert result["extractions"]["v1"]["values"] == [1200.0]
    assert result["extractions"]["v2"]["values"] == [1580.0]


def test_fast_path_multi_dr_partial_fails():
    """When one of two data_requests fails to resolve, fast-path returns None
    (falls back to full LLM extraction for all DRs)."""
    from solve import _try_deterministic_fast_path

    spec = _make_multi_dr_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries, "v2": entries}

    with (
        patch("solve._get_table_id_from_entry", return_value=42),
        patch("solve._build_cells_for_dr") as mock_build,
        patch("solve.resolve_cells") as mock_resolve,
    ):
        mock_build.side_effect = [
            [{"row_leaf": "National defense", "col_leaf": "1938", "name": "v1"}],
            [{"row_leaf": "National defense", "col_leaf": "1940", "name": "v2"}],
        ]
        # v1 resolves, v2 doesn't
        mock_resolve.side_effect = [
            {"values": {"v1": 1200.0}, "debug": {}},
            {"values": {"v2": None}, "debug": {"v2": {"status": "label_miss"}}},
        ]
        result = _try_deterministic_fast_path(spec, per_dr, verbose=True)

    # Fast-path should fail because v2 didn't resolve
    assert result is None


def test_fast_path_monthly_all_resolve():
    """When monthly_all DR resolves all 12 values, fast-path returns
    extractions with 12 values in order."""
    from solve import _try_deterministic_fast_path

    spec = _make_monthly_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    monthly_cells = [
        {"row_leaf": "National defense", "col_leaf": f"month_{m}", "name": f"m{m:02d}"}
        for m in range(1, 13)
    ]
    monthly_values = {f"m{m:02d}": float(m * 100) for m in range(1, 13)}

    with (
        patch("solve._get_table_id_from_entry", return_value=42),
        patch("solve._build_cells_for_dr", return_value=monthly_cells),
        patch("solve.resolve_cells") as mock_resolve,
    ):
        mock_resolve.return_value = {"values": monthly_values, "debug": {}}
        result = _try_deterministic_fast_path(spec, per_dr, verbose=True)

    assert result is not None
    assert len(result["extractions"]["v1"]["values"]) == 12
    assert result["extractions"]["v1"]["values"][0] == 100.0  # Jan
    assert result["extractions"]["v1"]["values"][11] == 1200.0  # Dec


def test_fast_path_monthly_incomplete_fails():
    """When monthly_all DR resolves fewer than 12 values, fast-path returns None."""
    from solve import _try_deterministic_fast_path

    spec = _make_monthly_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    # Only 10 months resolved (e.g., missing Nov/Dec)
    incomplete_cells = [
        {"row_leaf": "National defense", "col_leaf": f"month_{m}", "name": f"m{m:02d}"}
        for m in range(1, 11)  # Only 10 months
    ]
    incomplete_values = {f"m{m:02d}": float(m * 100) for m in range(1, 11)}

    with (
        patch("solve._get_table_id_from_entry", return_value=42),
        patch("solve._build_cells_for_dr", return_value=incomplete_cells),
        patch("solve.resolve_cells") as mock_resolve,
    ):
        mock_resolve.return_value = {"values": incomplete_values, "debug": {}}
        result = _try_deterministic_fast_path(spec, per_dr, verbose=True)

    # Should fail because we don't have 12 values
    assert result is None


def test_fast_path_values_format_matches_llm_extraction():
    """The extractions dict from fast-path has the same structure as LLM extraction,
    so compute can process it identically."""
    from compute import execute as compute_execute

    spec = _make_annual_spec()
    # Simulate what fast-path would return
    fast_path_extractions = {
        "v1": {
            "values": [1580.0],
            "labels": ["National defense"],
            "source_file": "treasury_bulletin_1941_01.json",
            "confidence": "deterministic",
        }
    }
    # Compute should work with this format just like LLM extraction
    result = compute_execute(spec, fast_path_extractions, verbose=False)
    assert result == 1580.0


def test_fast_path_monthly_values_format_matches_compute():
    """Monthly values from fast-path work correctly with compute's sum template."""
    from compute import execute as compute_execute

    spec = _make_monthly_spec()
    monthly_vals = [
        100.0,
        110.0,
        120.0,
        130.0,
        140.0,
        150.0,
        160.0,
        170.0,
        180.0,
        190.0,
        200.0,
        210.0,
    ]
    fast_path_extractions = {
        "v1": {
            "values": monthly_vals,
            "labels": [f"month {m}" for m in range(1, 13)],
            "source_file": "test.json",
            "confidence": "deterministic",
        }
    }
    result = compute_execute(spec, fast_path_extractions, verbose=False)
    assert result == sum(monthly_vals)


def test_fast_path_empty_spec():
    """Fast-path returns None when spec has no data_requests."""
    from solve import _try_deterministic_fast_path

    result = _try_deterministic_fast_path({}, {}, verbose=True)
    assert result is None


def test_fast_path_logging_on_success(capsys):
    """Fast-path logs when all DRs resolve deterministically."""
    from solve import _try_deterministic_fast_path

    spec = _make_annual_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    with (
        patch("solve._get_table_id_from_entry", return_value=42),
        patch(
            "solve._build_cells_for_dr",
            return_value=[{"row_leaf": "National defense", "col_leaf": "1940", "name": "v1"}],
        ),
        patch("solve.resolve_cells", return_value={"values": {"v1": 1580.0}, "debug": {}}),
    ):
        _try_deterministic_fast_path(spec, per_dr, verbose=True)

    captured = capsys.readouterr()
    assert "Fast-path" in captured.out or "deterministic" in captured.out.lower()


def test_fast_path_logging_on_failure(capsys):
    """Fast-path logs when it falls back (unresolved cells)."""
    from solve import _try_deterministic_fast_path

    spec = _make_annual_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    with (
        patch("solve._get_table_id_from_entry", return_value=42),
        patch(
            "solve._build_cells_for_dr",
            return_value=[{"row_leaf": "National defense", "col_leaf": "1940", "name": "v1"}],
        ),
        patch(
            "solve.resolve_cells",
            return_value={"values": {"v1": None}, "debug": {"v1": {"status": "label_miss"}}},
        ),
    ):
        _try_deterministic_fast_path(spec, per_dr, verbose=True)

    captured = capsys.readouterr()
    assert "Fast-path" in captured.out or "fast-path" in captured.out.lower()


def test_fast_path_with_run_extract_and_compute():
    """_run_extract_and_compute tries the fast-path first, and skips LLM
    when all values resolve deterministically."""
    from solve import _run_extract_and_compute

    spec = _make_annual_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    with (
        patch("solve._try_deterministic_fast_path") as mock_fp,
        patch("solve.extract_structured") as mock_llm,
        patch("solve.compute_execute", return_value=1580.0) as mock_compute,
        patch("solve.format_result", return_value="1580"),
    ):
        mock_fp.return_value = {
            "extractions": {
                "v1": {
                    "values": [1580.0],
                    "labels": ["National defense"],
                    "source_file": "test.json",
                    "confidence": "deterministic",
                }
            },
            "notes": "v1: resolved deterministically",
        }
        answer, extraction = _run_extract_and_compute(spec, per_dr, "test question", verbose=True)

    # Fast-path was attempted
    mock_fp.assert_called_once()
    # LLM extract was NOT called
    mock_llm.assert_not_called()
    # Compute was called with the fast-path extractions
    mock_compute.assert_called_once()
    assert answer == "1580"


def test_fast_path_fallback_to_llm():
    """_run_extract_and_compute falls back to LLM when fast-path returns None."""
    from solve import _run_extract_and_compute

    spec = _make_annual_spec()
    entries = [_make_table_entry(table_id=42)]
    per_dr = {"v1": entries}

    with (
        patch("solve._try_deterministic_fast_path", return_value=None) as mock_fp,
        patch("solve.extract_structured") as mock_llm,
        patch("solve.validate_extractions", return_value=[]),
        patch("solve.compute_execute", return_value=1580.0),
        patch("solve.format_result", return_value="1580"),
    ):
        mock_llm.return_value = {
            "extractions": {
                "v1": {
                    "values": [1580.0],
                    "labels": ["National defense"],
                    "source_file": "test.json",
                    "confidence": "high",
                }
            },
            "notes": "LLM extraction",
        }
        answer, extraction = _run_extract_and_compute(spec, per_dr, "test question", verbose=True)

    # Fast-path was attempted
    mock_fp.assert_called_once()
    # LLM extract WAS called (fallback)
    mock_llm.assert_called_once()
    assert answer == "1580"
