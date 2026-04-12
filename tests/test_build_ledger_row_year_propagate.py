"""Unit tests for propagate_row_years_long_format (no ledger.sqlite)."""

from __future__ import annotations

import pytest

from build_ledger import propagate_row_years_long_format


def _row(
    row_index: int,
    *,
    year: int | None = None,
    month: int | None = None,
    section: bool = False,
) -> dict:
    return {
        "row_index": row_index,
        "row_leaf": "",
        "row_path": "",
        "indent_level": 0,
        "year_extracted": year,
        "month_extracted": month,
        "is_section_header": 1 if section else 0,
    }


def test_propagates_month_rows_after_anchor() -> None:
    rows = [
        _row(0, year=1940, month=1),
        _row(1, month=2),
        _row(2, month=3),
        _row(3, month=12),
    ]
    propagate_row_years_long_format(rows)
    assert [r["year_extracted"] for r in rows] == [1940, 1940, 1940, 1940]


def test_section_header_resets_current_year() -> None:
    rows = [
        _row(0, year=1940, month=1),
        _row(1, month=2),
        _row(2, section=True),
        _row(3, month=3),
        _row(4, year=1950, month=1),
        _row(5, month=2),
    ]
    propagate_row_years_long_format(rows)
    assert [r["year_extracted"] for r in rows] == [
        1940,
        1940,
        None,
        None,
        1950,
        1950,
    ]


def test_no_anchor_no_propagation() -> None:
    rows = [_row(0, month=3), _row(1, month=4)]
    propagate_row_years_long_format(rows)
    assert rows[0]["year_extracted"] is None
    assert rows[1]["year_extracted"] is None


def test_explicit_year_row_not_overwritten() -> None:
    rows = [
        _row(0, year=1940, month=1),
        _row(1, year=1939, month=12),
    ]
    propagate_row_years_long_format(rows)
    assert rows[1]["year_extracted"] == 1939


def test_section_header_year_not_applied_before_reset_in_walk_order() -> None:
    """walk_table clears current_year on section rows before reading their year."""
    rows = [
        _row(0, year=1940, month=1),
        _row(1, month=2),
        _row(2, year=2000, month=None, section=True),
        _row(3, month=5),
    ]
    propagate_row_years_long_format(rows)
    assert rows[2]["year_extracted"] == 2000
    assert rows[3]["year_extracted"] is None


@pytest.mark.parametrize("section_flag", [1, True])
def test_truthy_section_header_triggers_reset(section_flag: bool | int) -> None:
    rows = [
        _row(0, year=1940, month=1),
        {"row_index": 1, "is_section_header": section_flag},
        _row(2, month=6),
    ]
    propagate_row_years_long_format(rows)
    assert rows[2]["year_extracted"] is None
