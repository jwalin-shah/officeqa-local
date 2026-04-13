"""Tests for column-0 rowspan label propagation in parse_table_lxml."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from build_ledger import parse_table_lxml


def test_rowspan_label_propagates():
    html = """<table>
    <tr><th>Category</th><th>Jan</th><th>Feb</th></tr>
    <tr><td rowspan="3">National defense</td><td>100</td><td>110</td></tr>
    <tr><td>120</td><td>130</td></tr>
    <tr><td>140</td><td>150</td></tr>
    </table>"""
    result = parse_table_lxml(html)
    # Column 0 of data rows should all be "National defense"
    data_rows = [r for i, r in enumerate(result["grid"]) if not result["is_header"][i]]
    assert all(row[0] == "National defense" for row in data_rows), (
        f"Expected all rows to have label 'National defense', got: {[r[0] for r in data_rows]}"
    )


def test_rowspan_label_stops_at_new_origin():
    html = """<table>
    <tr><th>Category</th><th>Value</th></tr>
    <tr><td rowspan="2">Receipts</td><td>100</td></tr>
    <tr><td>200</td></tr>
    <tr><td>Expenditures</td><td>300</td></tr>
    </table>"""
    result = parse_table_lxml(html)
    data_rows = [r for i, r in enumerate(result["grid"]) if not result["is_header"][i]]
    labels = [r[0] for r in data_rows]
    assert labels == ["Receipts", "Receipts", "Expenditures"], f"Got: {labels}"


def test_numeric_rowspan_not_propagated():
    """Numeric cells with rowspan should NOT have their value copied to extension rows."""
    html = """<table>
    <tr><th>Category</th><th>Value</th></tr>
    <tr><td>Item A</td><td rowspan="2">999</td></tr>
    <tr><td>Item B</td></tr>
    </table>"""
    result = parse_table_lxml(html)
    data_rows = [r for i, r in enumerate(result["grid"]) if not result["is_header"][i]]
    # Column 1 of row 1 (extension) should be empty, not "999"
    assert data_rows[1][1] == "", f"Expected empty extension in col 1, got: {data_rows[1][1]}"


def test_empty_label_not_propagated():
    """An empty origin cell in column 0 should not forward-fill subsequent rows."""
    html = """<table>
    <tr><th>Category</th><th>Value</th></tr>
    <tr><td></td><td>100</td></tr>
    <tr><td>Real label</td><td>200</td></tr>
    </table>"""
    result = parse_table_lxml(html)
    data_rows = [r for i, r in enumerate(result["grid"]) if not result["is_header"][i]]
    labels = [r[0] for r in data_rows]
    # Empty origin should remain empty; "Real label" should not be affected
    assert labels[0] == "", f"Expected empty first label, got: {labels[0]!r}"
    assert labels[1] == "Real label", f"Expected 'Real label', got: {labels[1]!r}"
