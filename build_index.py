#!/usr/bin/env python3
"""Build a table-level index from parsed bulletin JSONs.

Each table element becomes one index row with:
  - source file + element id (back-pointer)
  - nearest preceding title / section_header / caption (what the table is *about*)
  - column headers (first <tr>)
  - row labels (first <td> of each subsequent <tr>)
  - years mentioned (union of year tokens in headers + labels + body)
  - raw html (so retrieval can hand the exact table to the extractor)

Output: corpus_index.pkl — a list[dict], one entry per table.
"""

import json
import pickle
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

CORPUS_JSON = Path("corpus_json")
INDEX_PATH = Path("corpus_index.pkl")

YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b")
FILENAME_YEAR_MONTH_RE = re.compile(r"treasury_bulletin_(\d{4})_(\d{2})")


class TableParser(HTMLParser):
    """Minimal <table> parser: extracts rows as lists of cell texts."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._current_row = []
        elif tag in ("td", "th") and self._current_row is not None:
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "tr" and self._current_row is not None:
            if self._current_row:
                self.rows.append(self._current_row)
            self._current_row = None
        elif tag in ("td", "th") and self._current_cell is not None:
            text = " ".join("".join(self._current_cell).split()).strip()
            self._current_row.append(text) if self._current_row is not None else None
            self._current_cell = None

    def handle_data(self, data: str) -> None:
        if self._current_cell is not None:
            self._current_cell.append(data)


def parse_table_html(html: str) -> tuple[list[str], list[str], list[list[str]]]:
    """Return (column_headers, row_labels, all_rows)."""
    if not html:
        return [], [], []
    p = TableParser()
    try:
        p.feed(html)
    except Exception:
        return [], [], []
    rows = p.rows
    if not rows:
        return [], [], []
    header = rows[0]
    row_labels = [r[0] for r in rows[1:] if r and r[0]]
    return header, row_labels, rows


def extract_years(texts: list[str]) -> list[int]:
    years: set[int] = set()
    for t in texts:
        for m in YEAR_RE.findall(t or ""):
            years.add(int(m))
    return sorted(years)


def _content_text(element: dict) -> str:
    c = element.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, dict):
        return " ".join(str(v) for v in c.values() if isinstance(v, (str, int, float)))
    if isinstance(c, list):
        return " ".join(str(x) for x in c if isinstance(x, (str, int, float)))
    return ""


def index_file(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        print(f"  SKIP {path.name}: {e}", file=sys.stderr)
        return []

    doc = data.get("document") or {}
    elements = doc.get("elements") or []
    if not elements:
        return []

    m = FILENAME_YEAR_MONTH_RE.search(path.name)
    file_year = int(m.group(1)) if m else None
    file_month = int(m.group(2)) if m else None

    entries: list[dict] = []
    current_title = ""
    current_section = ""
    current_caption = ""

    for el in elements:
        etype = el.get("type")
        text = _content_text(el)

        if etype == "title":
            current_title = text or current_title
            continue
        if etype == "section_header":
            current_section = text or current_section
            continue
        if etype == "caption":
            current_caption = text or current_caption
            continue

        if etype != "table":
            continue

        headers, row_labels, all_rows = parse_table_html(text)
        if not all_rows:
            continue

        flat_cells: list[str] = []
        for r in all_rows:
            flat_cells.extend(r)
        body_text = " ".join(flat_cells)

        years = extract_years(
            [
                current_title,
                current_section,
                current_caption,
                " ".join(headers),
                " ".join(row_labels),
                body_text,
            ]
        )

        entries.append(
            {
                "file": path.name,
                "element_id": el.get("id"),
                "file_year": file_year,
                "file_month": file_month,
                "title": current_title,
                "section": current_section,
                "caption": current_caption,
                "column_headers": headers,
                "row_labels": row_labels,
                "years": years,
                "n_rows": len(all_rows),
                "html": text,
            }
        )

        current_caption = ""

    return entries


def build() -> None:
    files = sorted(CORPUS_JSON.glob("treasury_bulletin_*.json"))
    if not files:
        print(f"No files in {CORPUS_JSON}", file=sys.stderr)
        sys.exit(1)

    index: list[dict] = []
    for i, f in enumerate(files, 1):
        entries = index_file(f)
        index.extend(entries)
        if i % 50 == 0 or i == len(files):
            print(f"  {i}/{len(files)} files, {len(index)} tables so far")

    INDEX_PATH.write_bytes(pickle.dumps(index))
    print(f"\nWrote {INDEX_PATH} — {len(index)} table entries from {len(files)} files")

    files_with_tables = len({e["file"] for e in index})
    avg_tables = len(index) / files_with_tables if files_with_tables else 0
    print(f"  files with at least one table: {files_with_tables}")
    print(f"  avg tables per file: {avg_tables:.1f}")

    with_headers = sum(1 for e in index if e["column_headers"])
    with_rows = sum(1 for e in index if e["row_labels"])
    with_section = sum(1 for e in index if e["section"])
    with_years = sum(1 for e in index if e["years"])
    print(f"  tables with column headers: {with_headers} ({with_headers / len(index) * 100:.1f}%)")
    print(f"  tables with row labels:     {with_rows} ({with_rows / len(index) * 100:.1f}%)")
    print(f"  tables with a section:      {with_section} ({with_section / len(index) * 100:.1f}%)")
    print(f"  tables with year tokens:    {with_years} ({with_years / len(index) * 100:.1f}%)")


if __name__ == "__main__":
    build()
