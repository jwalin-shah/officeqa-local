"""Tests for extract.py — html_to_pipe_text(), html_to_vertical_text(), rendering helpers, mocked extract_structured()."""

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
