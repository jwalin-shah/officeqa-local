"""Tests for extract.py — html_to_pipe_text(), rendering helpers, mocked extract_structured()."""

from unittest.mock import MagicMock, patch

from extract import build_context_from_entries, clean_value, html_to_pipe_text, render_entry, to_num

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
