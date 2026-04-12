#!/usr/bin/env python3
"""Extract answers from Treasury Bulletin files.

Strategy:
1. Parse files into table blocks, split blocks into sub-sections
2. Use difflib fuzzy matching to find the best sub-section for the question
3. Send focused context (header + matched sub-section) to LLM
"""

import contextlib
import os
import re
import subprocess
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.getenv("OFFICEQA_MODEL", "deepseek-chat")

client = OpenAI(
    api_key=os.getenv("DEDALUS_API_KEY"),
    base_url=os.getenv("DEDALUS_API_BASE"),
)

SYNONYMS = {
    "expenditures": ["outlays", "spending", "disbursements"],
    "receipts": ["revenue", "income", "collections"],
    "defense": ["national defense", "military"],
    "debt": ["public debt", "obligations"],
    "veterans": ["veterans' administration", "veterans administration", "VA"],
    "international": ["foreign", "claims", "liabilities"],
    "currency": ["coin", "circulation", "money"],
    "savings": ["savings bonds", "series E", "series H"],
}


def llm(system: str, user: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content or ""


# ── HTML table rendering (for retrieve_v2 entries) ──────────────────────────


class _TableHTMLParser(HTMLParser):
    """Parse a single <table> into rows of cells. Handles th/td, flattens
    nested tags, preserves cell text."""

    def __init__(self):
        super().__init__()
        self.rows: list[list[str]] = []
        self._current_row: list[str] | None = None
        self._current_cell: list[str] | None = None

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "tr":
            self._current_row = []
        elif t in ("td", "th") and self._current_row is not None:
            self._current_cell = []

    def handle_endtag(self, tag):
        t = tag.lower()
        if t in ("td", "th") and self._current_cell is not None and self._current_row is not None:
            text = " ".join("".join(self._current_cell).split()).strip()
            self._current_row.append(text)
            self._current_cell = None
        elif t == "tr" and self._current_row is not None:
            if any(c for c in self._current_row):
                self.rows.append(self._current_row)
            self._current_row = None

    def handle_data(self, data):
        if self._current_cell is not None:
            self._current_cell.append(data)

    def handle_entityref(self, name):
        if self._current_cell is not None:
            self._current_cell.append(f"&{name};")


def _disambiguate_row_labels(rows: list[list[str]]) -> list[list[str]]:
    """Prefix bare month-name row labels with the inferred year.

    Treasury tables often have rows like:
        1939-December | 125
        January       | 132
        February      | 129
        ...
        December      | 473
        1941          | 2602

    The bare months after "1939-December" belong to 1940 (the next calendar
    year). This function detects the pattern and rewrites bare months to
    "1940-January", "1940-February", etc. so the LLM can unambiguously
    identify each row.

    Only modifies the first cell (row label) of data rows (skips header row 0).
    """
    if len(rows) < 2:
        return rows

    current_year: int | None = None
    month_names = {
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "jan",
        "feb",
        "mar",
        "apr",
        "jun",
        "jul",
        "aug",
        "sep",
        "sept",
        "oct",
        "nov",
        "dec",
        "jan.",
        "feb.",
        "mar.",
        "apr.",
        "may.",
        "jun.",
        "jul.",
        "aug.",
        "sep.",
        "sept.",
        "oct.",
        "nov.",
        "dec.",
    }

    result = [rows[0]]  # keep header as-is
    for row in rows[1:]:
        if not row:
            result.append(row)
            continue
        label = row[0].strip()
        label_lower = label.lower()

        # Detect "YYYY-MonthName" or "YYYY MonthName" pattern
        year_month = re.match(r"^(\d{4})[\s\-](\w+)", label)
        if year_month:
            y = int(year_month.group(1))
            m = year_month.group(2).lower().rstrip(".")
            if m in month_names or m + "." in month_names:
                month_idx = _detect_month_index(year_month.group(2))
                current_year = y + 1 if month_idx == 12 else y
            result.append(row)
            continue

        # Bare "YYYY" row — reset year tracking
        if re.match(r"^\d{4}$", label):
            current_year = None
            result.append(row)
            continue

        # Bare month name — prefix with inferred year
        if current_year is not None and label_lower.rstrip(".") in month_names:
            new_label = f"{current_year}-{label}"
            result.append([new_label] + row[1:])
            continue

        result.append(row)

    return result


def html_to_pipe_text(html: str, max_rows: int = 80) -> str:
    """Render a <table> HTML blob into pipe-delimited lines the LLM can read.

    Returns '' on parse failure. Caps rows to avoid runaway context bloat on
    huge tables (some bulletins have 200+ row tables)."""
    if not html:
        return ""
    try:
        parser = _TableHTMLParser()
        parser.feed(html)
        parser.close()
    except Exception:
        return ""
    if not parser.rows:
        return ""
    rows = _disambiguate_row_labels(parser.rows)
    lines = []
    for i, row in enumerate(rows):
        if i >= max_rows:
            lines.append(f"... ({len(rows) - max_rows} more rows truncated)")
            break
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ── Month detection helpers for vertical serialization ───────────────────────

# Calendar month order (1-indexed)
_CALENDAR_MONTHS: dict[str, int] = {
    "jan": 1,
    "jan.": 1,
    "january": 1,
    "feb": 2,
    "feb.": 2,
    "february": 2,
    "mar": 3,
    "mar.": 3,
    "march": 3,
    "apr": 4,
    "apr.": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "jun.": 6,
    "june": 6,
    "jul": 7,
    "jul.": 7,
    "july": 7,
    "aug": 8,
    "aug.": 8,
    "august": 8,
    "sep": 9,
    "sep.": 9,
    "sept": 9,
    "sept.": 9,
    "september": 9,
    "oct": 10,
    "oct.": 10,
    "october": 10,
    "nov": 11,
    "nov.": 11,
    "november": 11,
    "dec": 12,
    "dec.": 12,
    "december": 12,
}

# Fiscal year starts in Oct: Oct=1, Nov=2, ..., Sep=12
_FY_MONTH_ORDER: list[str] = [
    "oct",
    "nov",
    "dec",
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
]


def _detect_month_index(col_header: str) -> int | None:
    """Detect the calendar month number from a column header.

    Returns 1–12 if the header is a month name, None otherwise.
    Handles variants like 'Jan.', 'Jan', 'January'.
    Rejects false positives like 'Marketable securities' (starts with 'mar').
    """
    h = col_header.strip().lower()
    # Direct month name match
    if h in _CALENDAR_MONTHS:
        return _CALENDAR_MONTHS[h]
    # Check if header starts with a month name (e.g., "Jan. > 1940")
    for month_name, idx in _CALENDAR_MONTHS.items():
        if h.startswith(month_name):
            # Verify the next character is NOT alphabetic (prevents
            # 'Marketable securities' matching 'mar', 'Octane' matching 'oct', etc.)
            next_pos = len(month_name)
            if next_pos >= len(h) or not h[next_pos].isalpha():
                return idx
    return None


def _detect_fy_column_order(headers: list[str]) -> bool:
    """Detect if columns follow fiscal-year ordering (Oct→Sep).

    Returns True if the first few data columns match the FY month sequence.
    """
    month_indices = []
    for h in headers[1:]:  # skip row-label column
        idx = _detect_month_index(h)
        if idx is not None:
            month_indices.append(idx)
        else:
            break  # stop at first non-month column

    if len(month_indices) < 3:
        return False

    # Check if they follow FY order: 10, 11, 12, 1, 2, 3, ...
    fy_expected = [10, 11, 12, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    for start in range(len(fy_expected)):
        pattern = fy_expected[start:] + fy_expected[:start]
        if month_indices == pattern[: len(month_indices)]:
            return True
    return False


def _annotate_month(col_header: str, fy_order: bool) -> str:
    """Add month index annotation to a column header if it's a month name.

    For calendar order: Jan. (month 1), Feb. (month 2), etc.
    For fiscal year order: Oct. (month 10), Nov. (month 11), etc.
    Non-month columns are returned unchanged.
    """
    month_idx = _detect_month_index(col_header)
    if month_idx is None:
        return col_header
    return f"{col_header.strip()} (month {month_idx})"


def html_to_vertical_text(html: str, max_rows: int = 80, vertical_threshold: int = 8) -> str:
    """Render a <table> HTML blob into vertical key-value format.

    For tables with ≥vertical_threshold columns, produces:
        ROW: <row_label>
          <col_1> (month 1): <value>
          <col_2> (month 2): <value>
          ...

    Month names in column headers get annotated with their month index.
    Fiscal-year column order (Oct→Sep) is detected and annotated correctly.

    Returns '' if table has < vertical_threshold columns, on parse failure,
    or for empty input."""
    if not html:
        return ""
    try:
        parser = _TableHTMLParser()
        parser.feed(html)
        parser.close()
    except Exception:
        return ""
    if not parser.rows:
        return ""

    # Need at least a header row + one data row
    if len(parser.rows) < 2:
        return ""

    disambiguated = _disambiguate_row_labels(parser.rows)
    header_row = disambiguated[0]
    data_rows = disambiguated[1:]

    # Check column count (header columns include row label)
    if len(header_row) < vertical_threshold:
        return ""

    # Detect fiscal-year column ordering
    fy_order = _detect_fy_column_order(header_row)

    # Build annotated column labels
    col_labels = [_annotate_month(h, fy_order) for h in header_row]

    lines: list[str] = []
    for i, row in enumerate(data_rows):
        if i >= max_rows:
            lines.append(f"... ({len(data_rows) - max_rows} more rows truncated)")
            break

        # First cell is the row label
        row_label = row[0].strip() if row else ""
        lines.append(f"ROW: {row_label}")

        # Remaining cells are values
        for j in range(1, len(col_labels)):
            if j >= len(row):
                break
            raw_val = row[j].strip()
            # Skip missing/empty values
            if not raw_val or raw_val.lower() in ("nan", "-", "*", "..."):
                continue
            # Clean and convert numeric values for clarity
            num = to_num(raw_val)
            if num is not None:
                # Format: drop trailing .0 for integers
                val_str = str(int(num)) if num == int(num) else str(num)
            else:
                val_str = clean_value(raw_val)
            lines.append(f"  {col_labels[j]}: {val_str}")

    return "\n".join(lines)


def render_entry(entry: dict, vertical_threshold: int = 8, max_rows: int = 80) -> str:
    """Turn a retrieve_v2 table entry into a labelled context block.

    Wide tables (≥vertical_threshold columns) are rendered in vertical
    key-value format; narrow tables use the traditional pipe-delimited format."""
    file = entry.get("file") or "?"
    section = (entry.get("section") or "").strip()
    title = (entry.get("title") or "").strip()
    caption = (entry.get("caption") or "").strip()
    unit = (entry.get("unit") or "").strip()
    near_content = entry.get("near_content") or []
    element_id = entry.get("element_id")
    page_id = entry.get("page_id")

    header_bits = [f"# {file}"]
    if element_id is not None:
        header_bits[0] += f" (table #{element_id})"
    if page_id is not None:
        header_bits[0] += f" [page {page_id}]"
    if title:
        header_bits.append(f"Title: {title}")
    if section and section != title:
        header_bits.append(f"Section: {section}")
    if caption:
        header_bits.append(f"Caption: {caption}")
    if unit:
        header_bits.append(f"Unit Metadata: {unit}")
    for content in near_content:
        if content:
            header_bits.append(f"Adjacent Context: {content.strip()}")

    # Prose/footnote entries carry text in "content", not HTML tables
    content = (entry.get("content") or "").strip()
    html = entry.get("html") or ""

    if content and not html:
        body = content
    else:
        # Try vertical format first for wide tables
        vertical = html_to_vertical_text(
            html, max_rows=max_rows, vertical_threshold=vertical_threshold
        )
        if vertical:
            body = vertical
        else:
            body = html_to_pipe_text(html, max_rows=max_rows)
            if not body:
                return ""

    return "\n".join(header_bits) + "\n" + body


def build_context_from_entries(
    entries: list[dict],
    char_budget: int = 10000,
    vertical_threshold: int = 8,
    max_rows: int = 80,
) -> str:
    """Render a list of retrieve_v2 entries into one context blob, stopping at budget."""
    parts: list[str] = []
    total = 0
    for e in entries:
        block = render_entry(e, vertical_threshold=vertical_threshold, max_rows=max_rows)
        if not block:
            continue
        if total + len(block) > char_budget:
            remaining = char_budget - total
            if remaining > 500:
                parts.append(block[:remaining] + "\n... (truncated)")
            break
        parts.append(block)
        total += len(block) + 2
    return "\n\n".join(parts)


# ── Value helpers ────────────────────────────────────────────────────────────


def clean_value(v: str) -> str:
    """Clean Treasury table cell values: footnotes, negatives, symbols."""
    v = re.sub(r"\s*\d+/\s*$", "", v)
    v = re.sub(r"^[rpe]/\s*", "", v)
    v = re.sub(r"[rpe]\s*$", "", v)
    m = re.match(r"^\(([0-9,.]+)\)$", v)
    if m:
        v = "-" + m.group(1)
    return v.strip()


def to_num(v: str) -> float | None:
    """Parse a cleaned string value to a number."""
    v = clean_value(v).replace(",", "").replace("$", "").replace("%", "").lstrip("+")
    if not v or v in ("nan", "-", "*", "...", ""):
        return None
    try:
        return float(v)
    except ValueError:
        return None


# ── Table structure parsing ──────────────────────────────────────────────────


def find_table_blocks(lines: list[str]) -> list[tuple[int, int]]:
    """Find contiguous blocks of pipe-delimited lines."""
    blocks = []
    in_block = False
    start = 0
    for i, line in enumerate(lines):
        has_pipe = "|" in line and line.strip() != "|"
        if has_pipe and not in_block:
            start = i
            in_block = True
        elif not has_pipe and in_block:
            if i - start >= 3:
                blocks.append((start, i))
            in_block = False
    if in_block and len(lines) - start >= 3:
        blocks.append((start, len(lines)))
    return blocks


def parse_header(raw: str) -> str:
    """Clean multi-level header: 'Unnamed: 0_level_0 > Budget > 1940' → 'Budget > 1940'."""
    parts = [p.strip() for p in raw.split(">")]
    meaningful = [
        p
        for p in parts
        if not re.match(r"Unnamed:\s*\d+_level_\d+", p) and p.lower() not in ("nan", "")
    ]
    return " > ".join(meaningful) if meaningful else raw.strip()


def split_row(line: str) -> list[str]:
    """Split a pipe-delimited line into cells."""
    cells = [c.strip() for c in line.split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def is_section_header(cells: list[str]) -> bool:
    """Check if a row is a section header (label + all nan/empty)."""
    if len(cells) < 2:
        return False
    label = cells[0].strip()
    if not label or label.lower() == "nan":
        return False
    # A section header has a text label and all other cells are nan/empty
    data_cells = cells[1:]
    nan_count = sum(1 for c in data_cells if c.lower() in ("nan", "", "-"))
    return nan_count >= len(data_cells) * 0.8


def is_separator(line: str) -> bool:
    """Check if line is a separator like | --- | --- |."""
    cells = [c.strip() for c in line.split("|") if c.strip()]
    return all(re.match(r"^-+$", c) for c in cells) if cells else False


class TableSection:
    """A section within a table block: section label + data rows."""

    def __init__(
        self,
        label: str,
        header_line: str,
        data_lines: list[str],
        file_line_start: int,
        context_above: list[str] = None,
    ):
        self.label = label
        self.header_line = header_line
        self.data_lines = data_lines
        self.file_line_start = file_line_start
        self.context_above = context_above or []
        # Parse header columns
        self.headers = [parse_header(h) for h in split_row(header_line)]

    def raw_text(self) -> str:
        """Return context + header + data as raw text for the LLM."""
        prefix = ""
        if self.context_above:
            prefix = "Context: " + " | ".join(self.context_above) + "\n"
        return prefix + self.header_line + "\n" + "\n".join(self.data_lines)


def parse_table_sections(
    lines: list[str], block_start: int, context_above: list[str] = None
) -> list[TableSection]:
    """Parse a table block into sections.

    A section starts with a 'section header' row (label + nan cells)
    followed by data rows, until the next section header or end of block.
    """
    if len(lines) < 3:
        return []

    # First line is the header
    header_line = lines[0]
    # Skip separator
    data_start = 1
    if data_start < len(lines) and is_separator(lines[data_start]):
        data_start = 2

    sections = []
    current_label = ""
    current_data: list[str] = []
    current_start = block_start + data_start

    for i in range(data_start, len(lines)):
        line = lines[i]
        if "|" not in line:
            continue
        cells = split_row(line)
        if not cells:
            continue

        if is_section_header(cells):
            # Save previous section if it has data
            if current_data:
                sections.append(
                    TableSection(
                        current_label, header_line, current_data, current_start, context_above
                    )
                )
            current_label = cells[0].strip()
            current_data = []
            current_start = block_start + i
        else:
            current_data.append(line)

    # Save last section
    if current_data:
        sections.append(
            TableSection(current_label, header_line, current_data, current_start, context_above)
        )

    # If no section headers found, treat entire block as one section
    if not sections and data_start < len(lines):
        all_data = [ln for ln in lines[data_start:] if "|" in ln and not is_separator(ln)]
        if all_data:
            sections.append(
                TableSection("", header_line, all_data, block_start + data_start, context_above)
            )

    return sections


# ── Fuzzy matching ───────────────────────────────────────────────────────────


def expand_terms(terms: list[str]) -> list[str]:
    """Add domain synonyms."""
    expanded = list(terms)
    for t in terms:
        for key, syns in SYNONYMS.items():
            if key in t.lower():
                expanded.extend(syns)
    return list(dict.fromkeys(expanded))


def score_section(section: TableSection, query_terms: list[str], years: list[int]) -> float:
    """Score how well a table section matches the query.

    Uses SequenceMatcher for fuzzy matching against:
    - Section label (strongest signal)
    - Column headers
    - Row labels (first column of data rows)
    - Year presence in data
    """
    score = 0.0
    expanded = expand_terms(query_terms)

    # 1. Match section label against query terms
    if section.label:
        label_lower = section.label.lower()
        for term in expanded:
            ratio = SequenceMatcher(None, term.lower(), label_lower).ratio()
            if term.lower() in label_lower or label_lower in term.lower():
                ratio = max(ratio, 0.9)
            if ratio >= 0.5:
                score += ratio * 40  # section label is strongest signal

    # 2. Match column headers
    for header in section.headers:
        h_clean = parse_header(header).lower()
        for term in expanded:
            ratio = SequenceMatcher(None, term.lower(), h_clean).ratio()
            if term.lower() in h_clean:
                ratio = max(ratio, 0.85)
            if ratio >= 0.5:
                score += ratio * 15

    # 3. Match row labels (first column)
    for line in section.data_lines[:20]:
        cells = split_row(line)
        if cells:
            row_label = cells[0].lower()
            for term in expanded:
                if term.lower() in row_label:
                    score += 5
                    break

    # 4. Year presence in data rows + monthly completeness
    year_strs = [str(y) for y in years]
    data_text = "\n".join(section.data_lines)
    for ys in year_strs:
        if ys in data_text:
            score += 20

    # Count monthly rows for each year by scanning sequentially:
    # Pattern: "1940-January" row, then "February", "March", ... "December"
    months_full = [
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ]
    months_abbr = [
        "jan.",
        "feb.",
        "mar.",
        "apr.",
        "may.",
        "jun.",
        "jul.",
        "aug.",
        "sep.",
        "oct.",
        "nov.",
        "dec.",
    ]
    for y in years:
        in_year = False
        month_count = 0
        for line in section.data_lines:
            cells = split_row(line)
            if not cells:
                continue
            label = cells[0].lower().strip()
            # Start of year's monthly data
            if f"{y}-" in label:
                in_year = True
                month_count += 1
            elif in_year:
                # Bare month names following the year-prefixed row
                if any(label.startswith(m) for m in months_full) or any(
                    label.startswith(m) for m in months_abbr
                ):
                    month_count += 1
                elif re.match(r"^\d{4}", label):
                    break  # next year's data started
        if month_count >= 10:
            score += 40

    # 5. Data richness: prefer sections with actual numeric data
    num_count = 0
    for line in section.data_lines[:10]:
        for cell in split_row(line)[1:]:
            val = to_num(cell)
            if val is not None:
                n = abs(val)
                if n >= 10:
                    num_count += 1
    if num_count >= 5:
        score += 15
    elif num_count == 0:
        score -= 20  # likely a TOC or metadata section

    # 6. Co-occurrence bonus: sections matching MULTIPLE distinct terms are much more likely correct
    # Check which query terms appear in label + headers combined
    all_text = (section.label + " " + " ".join(section.headers)).lower()
    all_text += " " + " ".join(section.data_lines[:5]).lower()
    matched_terms = set()
    for term in expanded:
        if term.lower() in all_text:
            matched_terms.add(term.lower())
    if len(matched_terms) >= 3:
        score += 50
    elif len(matched_terms) >= 2:
        score += 25

    return score


# ── Context building ─────────────────────────────────────────────────────────


def _extract_query_terms(question: str, keywords: list[str]) -> list[str]:
    """Extract meaningful terms from question + keywords for matching."""
    stop = {
        "the",
        "for",
        "and",
        "was",
        "were",
        "what",
        "total",
        "using",
        "only",
        "reported",
        "values",
        "individual",
        "calendar",
        "months",
        "fiscal",
        "year",
        "these",
        "this",
        "that",
        "with",
        "from",
        "millions",
        "nominal",
        "dollars",
        "specifically",
        "corresponding",
        "rounded",
        "nearest",
        "percent",
        "change",
        "absolute",
        "place",
        "value",
        "sum",
        "all",
        "not",
        "how",
        "much",
        "many",
        "which",
        "each",
        "every",
        "should",
        "report",
        "federal",
        "government",
        "table",
        "taking",
        "cost",
        "amount",
        "monthly",
        "annual",
        "exclude",
        "offsets",
        "adjustments",
        "figure",
        "include",
        "contain",
        "number",
        "page",
        "billion",
        "thousand",
        "use",
        "determine",
        "find",
        "according",
        "data",
        "states",
        "united",
        "treasury",
        "bulletin",
    }
    q_words = re.findall(r"[A-Za-z][\w'-]+", question)
    terms = [w for w in q_words if w.lower() not in stop and len(w) > 2]
    # Also filter keywords through stop list
    terms.extend(k for k in keywords if k not in terms and k.lower() not in stop and len(k) > 2)
    return terms


def build_context(
    files: list[str],
    years: list[int],
    keywords: list[str],
    char_budget: int = 8000,
    max_sections: int = 5,
) -> str:
    """Build focused context by finding the best table sections for a query.

    1. Parse all table blocks in all files into sections
    2. Score each section by fuzzy match against keywords + year presence
    3. Return up to max_sections top sections, capped at char_budget
    """
    # Collect and score all sections from all files
    scored: list[tuple[float, str, TableSection]] = []

    for fpath in files:
        fname = Path(fpath).name
        try:
            with open(fpath) as f:
                file_lines = f.readlines()
        except Exception:
            continue

        blocks = find_table_blocks(file_lines)
        for block_start, block_end in blocks:
            # Grab up to 5 lines of text above the table block for context (e.g. units)
            ctx_start = max(0, block_start - 5)
            context_above = [ln.strip() for ln in file_lines[ctx_start:block_start] if ln.strip()]

            block_lines = file_lines[block_start:block_end]
            sections = parse_table_sections(block_lines, block_start, context_above)
            for section in sections:
                s = score_section(section, keywords, years)
                if s > 0:
                    scored.append((s, fname, section))

    if not scored:
        return ""

    scored.sort(key=lambda x: -x[0])

    context_parts = []
    total_chars = 0
    seen_headers = set()

    for _score, fname, section in scored[:max_sections]:
        header_key = section.header_line[:200]
        include_header = header_key not in seen_headers
        seen_headers.add(header_key)

        part = f"# {fname} (line {section.file_line_start + 1})"
        if section.label:
            part += f" — section: {section.label}"
        part += "\n"
        if section.context_above:
            part += "Context: " + " | ".join(section.context_above) + "\n"
        if include_header:
            part += section.header_line + "\n"
        part += "\n".join(section.data_lines)

        if total_chars + len(part) > char_budget:
            remaining = char_budget - total_chars
            if remaining > 300:
                part = part[:remaining]
            else:
                break
        context_parts.append(part)
        total_chars += len(part)

    return "\n\n".join(context_parts)


# ── CPI handling ─────────────────────────────────────────────────────────────


def detect_cpi_need(question: str) -> bool:
    q = question.lower()
    return any(
        kw in q
        for kw in [
            "inflation",
            "real dollar",
            "constant dollar",
            "adjusted dollar",
            "cpi",
            "consumer price",
            "real term",
            "nominal to real",
        ]
    )


def get_cpi_context(question: str) -> str:
    if not detect_cpi_need(question):
        return ""
    years = [int(y) for y in re.findall(r"\b(1[89]\d{2}|20[0-2]\d)\b", question)]
    parts = ["\n--- CPI DATA ---"]
    for y in years:
        try:
            r = subprocess.run(
                ["python3", "cpi.py", "annual", str(y)], capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                parts.append(f"CPI annual average for {y}: {r.stdout.strip()}")
        except Exception:
            pass
    if len(years) >= 2:
        try:
            r = subprocess.run(
                ["python3", "cpi.py", "annual", str(years[0]), str(years[-1])],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                parts.append(f"CPI multiplier {years[0]} -> {years[-1]}: {r.stdout.strip()}")
        except Exception:
            pass
    return "\n".join(parts) if len(parts) > 1 else ""


# ── LLM extraction ──────────────────────────────────────────────────────────

EXTRACT_SYSTEM = """You extract answers from U.S. Treasury Bulletin data tables.

You receive pre-selected table sections that are most likely to contain the answer.
Each section has a header row (column names), data rows, and sometimes a "Context:"
line containing text found immediately above the table.

UNITS AND SCALE (CRITICAL):
- Check the "Context:" line and table headers for scale indicators like
  "In millions of dollars", "(In thousands of dollars)", "In billions", or "percent".
- If the question asks for a value in a specific unit (e.g. "in millions") but
  the table is in a different unit, you MUST perform the conversion.
- If no unit is explicitly stated in the context or headers, assume nominal
  dollars as printed.

TABLE FORMAT:
- Pipe-delimited: | row_label | value1 | value2 | ...
- Headers may use ">" for multi-level columns
- Values like "1,580 3/" mean 1580 (ignore footnote markers)
- (123) means negative 123
- "nan" or "-" means no data

KEY RULES:
- "Calendar year YYYY" = SUM the 12 monthly rows (Jan-Dec). Do NOT use the annual summary row.
- "Fiscal year YYYY" = use the annual summary row for that year.
- Count column positions carefully by counting pipes from left to right.
- Show your work step by step for any computation.

Put the final answer on the LAST line by itself.
Numbers: no units, no %, no $, no commas (e.g., 2602, 1608.80, -523)
Text answers: return the text (e.g., March 3 1977)"""


def extract_answer(
    question: str, files: list[str], keywords: list[str], verbose: bool = False
) -> str:
    """Build context from files and extract answer via single LLM call."""
    years = [int(y) for y in re.findall(r"\b(1[89]\d{2}|20[0-2]\d)\b", question)]
    q_terms = _extract_query_terms(question, keywords)
    context = build_context(files, years, q_terms)
    if not context:
        return "Data not found"

    cpi = get_cpi_context(question)
    user_msg = f"Question: {question}\n\n{context}{cpi}\n\nExtract the answer."

    if verbose:
        print(f"  Extract: {len(context)} chars context, {len(cpi)} chars CPI")

    raw = llm(EXTRACT_SYSTEM, user_msg, max_tokens=1200)
    if verbose:
        print(f"  LLM raw:\n{raw[:500]}")

    return raw.strip()


# ── Structured extraction (v2) ──────────────────────────────────────────────

EXTRACT_STRUCTURED_SYSTEM = """You are a collaborator on a research effort to
answer U.S. Treasury Bulletin questions. Your job is to extract raw numbers
faithfully from the tables you are given — the pipeline downstream trusts what
you return and cannot recover from a fabricated value. Work carefully; a
correct null is worth more than a confident guess.

It is OK to return null for a value you cannot find. It is not OK to invent
one. When a cell is missing, unreadable, or ambiguous, return null and explain
what you saw in `notes`. The pipeline can retry on a flagged null; it cannot
undo a plausible-looking fake number.

You extract raw numeric values from U.S. Treasury Bulletin tables to fill
slots defined in a QuestionSpec.

CRITICAL RULE: You DO NOT compute anything. You only extract raw numbers from
tables and cite where they came from. Python will run the computation later.

EXTRACTION CHECKLIST (these are the failure modes we paid to find):
  - UNITS AND SCALE (CRITICAL): Scan "Unit Metadata", "Adjacent Context",
    table title, and captions for unit markers (e.g., "In millions of dollars",
    "(In thousands of dollars)", "In billions", or "percent").
    "Adjacent Context" contains prose/footnotes from the same page and
    frequently carries these critical scale markers.
    A correct `source_unit` is mandatory for downstream math to work.
    Return the RAW number as printed in the table — the formatter handles
    unit conversion. Do not pre-scale.
  - FISCAL YEAR BOUNDARIES. The spec's granularity tells you whether the
    answer is an annual/FY row or a sum of 12 monthly rows, but when you're
    confirming you grabbed the right row remember:
      - pre-1977  FY = Jul 1 (YYYY-1) through Jun 30 YYYY
      - post-1976 FY = Oct 1 (YYYY-1) through Sep 30 YYYY
      - CY        = Jan 1 through Dec 31 (sum of 12 monthly rows)
    A bare "YYYY" row in a post-1976 table is the fiscal year, not the
    calendar year. Never return an FY row when the spec asked for CY.
  - TOTAL vs SUB-CATEGORY: if the spec asks for a specific child line, never
    return the "Total X" row value. Total rows already include all children.
  - ACTUAL vs ESTIMATED: when both are present, prefer the actual row.
  - PERCENTAGE COLUMNS: when the table already contains a "% increase",
    "% change", or "Percent change" column, extract the TABLE'S printed
    percentage directly. Do NOT compute percentages from raw amounts —
    the table's pre-rounded value is what downstream expects.
  - ANNUAL vs MONTHLY: a monthly row is not an annual total. For
    granularity="annual", return the single year-labeled row; for
    granularity="monthly_all", return all 12 monthly rows.
  - COLUMN POSITION: wide tables with multi-level headers are easy to misread.
    If multiple columns could plausibly match, return the one whose header
    text contains the `column_hint` substring. Confirm by header text, never
    guess by position.
  - VINTAGE: the spec carries `vintage: "latest"` or `"as_reported"`. When
    "latest" (the default), prefer revised/canonical values — a later
    bulletin's restated figure beats the original. When "as_reported", use
    the value as it was originally published and ignore later revisions.

TABLE FORMAT:
- Pipe-delimited: | row_label | val1 | val2 | ...
- Multi-level headers use ">" (e.g. "Budget > 1940")
- (123) means -123
- "nan" or "-" means missing
- "1,580 3/" means 1580 (ignore footnote markers)

FOR EACH data_request in the QuestionSpec:
- granularity="monthly_all": extract all 12 monthly rows Jan→Dec for the year
  (rows like "1940-January", "February", "March", ..., "December"). Return 12 numbers.
- granularity="annual": extract the SINGLE annual row | YYYY | value
  (the single year-labeled row, NOT the monthly rows). Return 1 number.
- granularity="multi_year_annual": one annual row per year in the years list,
  in the same order as years. Return len(years) numbers.
- granularity="monthly_range": rows from month_start to month_end, in order.
- granularity="specific_month": one specific month row.

values[] must contain exactly expected_count numbers (or null for missing).
Return the raw extracted number — no scaling, no unit conversion, no math.

SOURCE_UNIT (REQUIRED): For every extraction, set `source_unit` to the unit
the RAW values are printed in. Look at the table caption, section header,
column header, or row label for phrases like "in millions", "(thousands of
dollars)", "$ billions", "percent", "Index", etc. Use exactly one of:
  - "thousands_usd" — table values are in thousands of dollars
  - "millions_usd"  — table values are in millions of dollars
  - "billions_usd"  — table values are in billions of dollars
  - "usd"           — table values are in raw dollars
  - "percent"       — table values are percentages
  - "index"         — index points / basis points
  - "count"         — counts / units of something other than money
  - null            — only if you genuinely cannot tell
This field is load-bearing: downstream math uses it to scale into the unit
the question asks for. A wrong source_unit silently corrupts every answer
that involves a non-linear op (box-cox, log, geometric mean, ratios across
mixed-unit tables). When in doubt, look for "(in millions of dollars)" near
the title — that is the most common form.

GROUNDING (REQUIRED): For EACH value you extract, you MUST cite the exact
cell coordinates where you found it. This is how we verify your work:
  - "row_label": the EXACT text of the first column in that row, as printed
  - "col_label": the EXACT column header text above the cell you read
These coordinates let the pipeline cross-check your extraction against the
parsed table. If you cannot identify the exact row/column for a value,
return null for that value — a verifiable null is better than an ungrounded
number.

For PROSE / text entries (non-table context), set row_label and col_label to
null and put the source passage in notes.

Output ONLY valid JSON matching this schema:
{
  "extractions": {
    "v1": {
      "values": [132, 129, 143, 159, 154, 153, 177, 200, 219, 287, 376, 473],
      "labels": ["1940-January", "February", "March", ...],
      "row_labels": ["1940-January", "February", "March", ...],
      "col_labels": ["National defense", "National defense", ...],
      "source_file": "treasury_bulletin_1941_01.txt",
      "source_unit": "millions_usd",
      "confidence": "high"
    }
  },
  "notes": "<anything unusual, e.g. had to skip annual row>"
}"""


# ── CY Row Filtering (VAL-EXTR-005) ─────────────────────────────────────────


def _is_annual_or_fy_label(label: str) -> bool:
    """Check if a row label represents an annual or fiscal-year summary row.

    Matches:
    - Bare 4-digit year: "1940"
    - "Fiscal year YYYY" / "FY YYYY" / "Fiscal YYYY"
    """
    label = label.strip()
    return bool(
        re.match(r"^\d{4}$", label)
        or re.match(r"^(Fiscal\s+year|FY|Fiscal)\s+\d{4}", label, re.IGNORECASE)
    )


def filter_cy_rows(text: str, granularity: str) -> str:
    """Filter annual/FY summary rows from rendered context for CY questions.

    When granularity='monthly_all', removes rows where the label is a bare
    year (e.g., "1940") or a fiscal-year designation (e.g., "Fiscal year 1940").
    Only monthly data rows are kept, preventing the LLM from picking the wrong
    total.

    Handles both pipe-delimited and vertical serialization formats.
    For other granularities, returns text unchanged.
    """
    if granularity != "monthly_all" or not text:
        return text

    lines = text.split("\n")
    filtered: list[str] = []
    in_filtered_row = False  # tracking vertical ROW entries to skip

    for line in lines:
        stripped = line.strip()

        # ── Vertical format: check ROW entries ──────────────────────────
        if stripped.startswith("ROW:"):
            row_label = stripped[4:].strip()
            if _is_annual_or_fy_label(row_label):
                in_filtered_row = True
                continue  # skip this ROW line
            else:
                in_filtered_row = False  # new ROW is not filtered

        # Skip value lines under a filtered vertical ROW
        if in_filtered_row:
            if line.startswith("  "):  # indented value line
                continue
            elif stripped == "":
                # Blank line may or may not be inside filtered ROW —
                # treat as boundary to stop skipping
                in_filtered_row = False
            else:
                # Non-indented, non-blank line — end of filtered ROW values
                in_filtered_row = False

        # ── Pipe format: check first cell ──────────────────────────────
        if "|" in stripped and not in_filtered_row:
            cells = _split_pipe_line(stripped)
            if cells and _is_annual_or_fy_label(cells[0]):
                continue

        filtered.append(line)

    return "\n".join(filtered)


# ── Monthly Pre-Extraction (VAL-EXTR-004) ──────────────────────────────────


def _split_pipe_line(line: str) -> list[str]:
    """Split a pipe-delimited line into cells, removing empty edge cells."""
    cells = [c.strip() for c in line.split("|")]
    if cells and cells[0] == "":
        cells = cells[1:]
    if cells and cells[-1] == "":
        cells = cells[:-1]
    return cells


def _detect_month_from_label(label: str, target_year: int | None) -> int | None:
    """Detect calendar month index from a row label.

    Patterns handled:
    - "1940-January" → 1 (when target_year matches)
    - "1940 January" → 1
    - "January" / "Jan." → 1
    Returns None if label doesn't look like a month row.
    """
    label = label.strip().lower()

    # Year-month pattern: "1940-January", "1940 January"
    if target_year:
        for month_name, idx in _CALENDAR_MONTHS.items():
            prefix_dash = f"{target_year}-{month_name}"
            prefix_space = f"{target_year} {month_name}"
            if label.startswith(prefix_dash) or label.startswith(prefix_space):
                return idx

    # Bare month name: "January", "Feb.", "mar"
    # Avoid matching year-prefixed labels for wrong years
    if not re.match(r"^\d{4}", label):
        for month_name, idx in _CALENDAR_MONTHS.items():
            if label == month_name or label.startswith(month_name):
                return idx

    return None


def _parse_vertical_monthly_values(text: str, row_hint: str = "") -> list[float] | None:
    """Parse vertical-format rendered text for 12 monthly values with month
    annotations.

    Looks for patterns like "Jan. (month 1): 132" under ROW entries
    matching the optional row_hint. Returns list of 12 floats in month
    order, or None if all 12 can't be found.
    """
    month_values: dict[int, float] = {}
    in_target_row = True  # by default, accept any ROW

    for line in text.split("\n"):
        stripped = line.strip()

        # Track which ROW we're in
        if stripped.startswith("ROW:"):
            row_label = stripped[4:].strip().lower()
            if row_hint:
                hint_lower = row_hint.lower()
                in_target_row = hint_lower in row_label or row_label in hint_lower
            else:
                in_target_row = True

        if in_target_row:
            # Match "(month N): value" pattern
            m = re.match(r".*\(month\s+(\d+)\):\s*([0-9,.\-]+)", stripped)
            if m:
                month_idx = int(m.group(1))
                val_str = m.group(2)
                val = to_num(val_str)
                if val is not None and 1 <= month_idx <= 12:
                    month_values[month_idx] = val

    if len(month_values) == 12:
        return [month_values[i] for i in range(1, 13)]
    return None


def _parse_pipe_monthly_row_values(text: str, dr: dict) -> list[float] | None:
    """Parse pipe-delimited text where months are ROWS (not columns).

    Looks for rows like:
      | 1940-January | 132 | ...
      | February | 129 | ...
    Identifies the target column by matching the DR's row_hint against column
    headers, then extracts values from monthly rows.
    """
    row_hint = (dr.get("row_hint") or "").lower()
    years = dr.get("years") or []
    target_year = years[0] if years else None

    lines = text.split("\n")

    # Find the target column index by matching row_hint in header cells
    target_col_idx: int | None = None

    for line in lines:
        if "|" not in line:
            continue
        cells = _split_pipe_line(line)
        if not cells:
            continue

        for j, cell in enumerate(cells):
            cell_lower = cell.lower().strip()
            if row_hint and (row_hint in cell_lower or cell_lower in row_hint):
                target_col_idx = j
                break

        if target_col_idx is not None:
            break

    if target_col_idx is None:
        return None

    # Extract values from monthly rows
    month_values: dict[int, float] = {}

    for line in lines:
        if "|" not in line:
            continue
        cells = _split_pipe_line(line)
        if not cells:
            continue

        first_cell = cells[0].strip()
        month_idx = _detect_month_from_label(first_cell, target_year)
        if month_idx is not None and target_col_idx < len(cells):
            val = to_num(cells[target_col_idx])
            if val is not None:
                month_values[month_idx] = val

    if len(month_values) == 12:
        return [month_values[i] for i in range(1, 13)]
    return None


def _format_value_list(values: list[float]) -> str:
    """Format a list of numeric values for the pre-extraction annotation."""
    parts: list[str] = []
    for v in values:
        if v == int(v):
            parts.append(str(int(v)))
        else:
            parts.append(str(v))
    return "[" + ", ".join(parts) + "]"


def pre_extract_monthly_values(text: str, dr: dict, spec: dict) -> str | None:
    """Pre-extract monthly values from rendered context for CY sum questions.

    When granularity='monthly_all' and computation='sum', parses the rendered
    table text for 12 monthly values and returns an annotation:

        PRE-EXTRACTED MONTHLY VALUES for CY YYYY: [v1, v2, ..., v12]
        (Count: 12 — sum for calendar year total)

    Works by parsing the rendered text — no database access required.
    Returns None if values can't be extracted or conditions aren't met.
    """
    granularity = dr.get("granularity", "")
    computation = spec.get("computation", "")
    years = dr.get("years") or []

    if granularity != "monthly_all" or computation != "sum" or not years:
        return None

    target_year = years[0]
    row_hint = dr.get("row_hint") or ""

    # Strategy 1: Parse vertical format with month annotations
    values = _parse_vertical_monthly_values(text, row_hint=row_hint)

    # Strategy 2: Parse pipe format with months as rows
    if not values:
        values = _parse_pipe_monthly_row_values(text, dr)

    if values and len(values) == 12:
        return (
            f"PRE-EXTRACTED MONTHLY VALUES for CY {target_year}: "
            f"{_format_value_list(values)} "
            f"(Count: 12 — sum for calendar year total)"
        )

    return None


def _resolve_external_dr(dr: dict) -> list[float] | None:
    """Resolve a non-corpus data_request directly from a known source.

    Returns a list of values to inject into extractions, or None if the
    source is unknown / unsupported. Skips the LLM extractor entirely
    for these — corpus tables don't contain CPI/FX/external lookups,
    so asking the LLM to find them there always returns nulls.
    """
    src = (dr.get("source") or "corpus").lower()
    if src == "corpus":
        return None

    years = [int(y) for y in (dr.get("years") or []) if y is not None]

    if src == "cpi":
        try:
            from cpi import A as _CPI_A  # BLS official annual averages
        except Exception:
            return None
        out: list[float] = []
        for y in years:
            val = _CPI_A.get(y)
            if val is not None:
                out.append(val)
        return out or None

    if src in ("fx", "external"):
        try:
            from external_data import resolve_fx_dr
        except Exception:
            return None
        return resolve_fx_dr(dr)

    return None


# ── Multi-year entry coverage ──────────────────────────────────────────────


def _ensure_year_coverage(entries: list[dict], year_set: set[int]) -> list[dict]:
    """Reorder entries so at least one entry per required year appears early.

    For multi-year DRs (continuous_monthly, multi_year_annual), the first
    file's tables can consume the entire char budget before entries covering
    later years are rendered.  This promotes one entry per uncovered year
    to the front, then appends the rest in their original order.
    """
    if len(year_set) <= 1:
        return entries
    covered: set[int] = set()
    priority: list[dict] = []
    rest: list[dict] = []
    for e in entries:
        e_years = set(e.get("years") or [])
        uncovered = (e_years & year_set) - covered
        if uncovered:
            covered.update(uncovered)
            priority.append(e)
        else:
            rest.append(e)
    return priority + rest


def extract_structured(
    spec: dict,
    per_dr_entries: dict,
    question: str,
    verbose: bool = False,
    feedback: str = "",
    _alt_render: bool = False,
) -> dict | None:
    """Extract structured values per the QuestionSpec's data_requests.

    `per_dr_entries` is {dr_id: [retrieve_v2 entries]}. Each entry carries the
    table HTML inline — we render it straight to pipe-delimited text. No
    .txt re-scanning, no per-DR fuzzy section scoring. The specific tables
    retrieve_v2 identified are exactly what the LLM sees."""
    data_requests = spec.get("data_requests") or []
    if not data_requests:
        return None

    # ── Pre-resolve non-corpus DRs (CPI, FX, external) ──────────────────
    # The LLM extractor only sees corpus tables, so source=cpi/fx/external
    # DRs would always come back null. Resolve them deterministically here
    # and inject the values into the final extractions dict.
    preinjected: dict[str, dict] = {}
    corpus_drs: list[dict] = []
    for dr in data_requests:
        src = (dr.get("source") or "corpus").lower()
        if src == "corpus":
            corpus_drs.append(dr)
            continue
        vals = _resolve_external_dr(dr)
        if vals:
            preinjected[dr.get("id", "?")] = {
                "values": vals,
                "label": dr.get("label", ""),
                "source": src,
            }
        else:
            # Unresolvable external DR — still let the LLM try in case
            # the value happens to live in a corpus table.
            corpus_drs.append(dr)

    per_request_context: list[str] = []
    # Top-k retrieval returns 20 entries per DR. With dedupe_by_file we care
    # about seeing rank 19-20 as often as rank 1, so the budget has to be wide
    # enough to fit the full list even when each table has a lot of rows.
    per_request_budget = max(6000, 48000 // max(1, len(data_requests)))

    any_hit = False
    for dr in corpus_drs:
        dr_id = dr.get("id", "?")
        dr_label = dr.get("label", "")
        dr_years = [y for y in (dr.get("years") or [])]
        granularity = dr.get("granularity", "?")
        entries = per_dr_entries.get(dr_id) or []
        # Reorder entries so tables relevant to this DR's years and section come first.
        # Matters when multiple DRs share an entry pool (e.g. oracle eval or
        # multi-year questions) — the wrong table otherwise wins the char
        # budget race and the right one gets truncated, leaving the LLM to
        # return nulls because "the 1953 data wasn't in the context".
        section_hint = (dr.get("section_hint") or "").lower()
        if (dr_years or section_hint) and entries:
            dr_year_set = {int(y) for y in dr_years}

            def _entry_match_score(sec: str, years: set[int], e: dict) -> tuple[int, int, int]:
                # 1. Section/Title match (highest priority)
                section_score = 0
                if sec:
                    e_title = (e.get("title") or "").lower()
                    e_section = (e.get("section") or "").lower()
                    if sec in e_title or sec in e_section:
                        section_score = -1

                # 2. Year overlap
                file_year = e.get("file_year")
                if file_year is None:
                    file_year = 0
                e_years = set(e.get("years") or [])
                overlap = len(years & e_years)
                overlap_score = -overlap if overlap else 0

                # 3. Proximity to DR years
                proximity = min(
                    (abs(int(file_year) - y) for y in years),
                    default=9999,
                )
                return (
                    section_score,
                    0 if overlap else 1,
                    proximity if not overlap else overlap_score,
                )

            entries = sorted(
                entries,
                key=lambda e, _sh=section_hint, _ys=dr_year_set: _entry_match_score(_sh, _ys, e),
            )
            # For multi-year DRs, promote one entry per uncovered year to
            # the front so the char budget is spread across all years.
            if len(dr_year_set) > 1:
                entries = _ensure_year_coverage(entries, dr_year_set)
        _vt = 999 if _alt_render else 8
        _mr = 150 if _alt_render else 80
        ctx = (
            build_context_from_entries(
                entries,
                char_budget=per_request_budget,
                vertical_threshold=_vt,
                max_rows=_mr,
            )
            if entries
            else ""
        )

        # Apply CY row filtering for monthly_all questions — suppress
        # annual/FY summary rows so the LLM doesn't pick the wrong total
        if ctx and granularity == "monthly_all":
            ctx = filter_cy_rows(ctx, granularity)

        header = (
            f"=== Context for {dr_id} ({dr_label}) — years={dr_years} granularity={granularity} ==="
        )
        if ctx:
            # Try pre-extraction for monthly_all + sum DRs
            annotation = pre_extract_monthly_values(ctx, dr, spec)
            if annotation:
                ctx = f"{annotation}\n\n{ctx}"
            per_request_context.append(f"{header}\n{ctx}")
            any_hit = True
        else:
            per_request_context.append(f"{header}\n(no tables retrieved for this data_request)")

    if not any_hit and not preinjected:
        return None
    context = "\n\n".join(per_request_context)

    import json as _json

    # Only show the LLM the corpus DRs it's responsible for. Pre-resolved
    # ones are merged back in after the call.
    spec_json = _json.dumps(
        {
            "computation": spec.get("computation"),
            "data_requests": corpus_drs or data_requests,
        },
        indent=2,
    )

    # If every DR was pre-resolved, skip the LLM call entirely.
    if not corpus_drs:
        return {"extractions": preinjected}

    feedback_block = ""
    if feedback:
        feedback_block = (
            f"\nREVIEW FEEDBACK from previous attempt — fix this specifically:\n{feedback}\n"
        )

    user_msg = (
        f"QuestionSpec:\n{spec_json}\n\n"
        f"Original question (for context): {question}\n"
        f"{feedback_block}\n"
        f"Table sections (scoped per data_request):\n{context}\n\n"
        f"Fill the extractions JSON. Return ONLY JSON."
    )

    if verbose:
        print(
            f"  Extract(v2): {len(context)} chars context, "
            f"{len(data_requests)} data_requests (per-request budget {per_request_budget})"
        )

    raw = llm(EXTRACT_STRUCTURED_SYSTEM, user_msg, max_tokens=2000)
    if verbose:
        print(f"  LLM raw:\n{raw[:500]}")

    import re as _re

    cleaned = _re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        result = _json.loads(cleaned)
    except _json.JSONDecodeError:
        try:
            result = _json.loads(cleaned[: cleaned.rfind("}") + 1])
        except Exception:
            return None

    if "extractions" not in result:
        return None

    # Merge in pre-resolved external DRs (CPI etc.) so the cohesion check
    # below sees a complete picture.
    if preinjected:
        merged = dict(result.get("extractions") or {})
        for k, v in preinjected.items():
            merged[k] = v
        result["extractions"] = merged

    # ── Cohesion: ensure every spec DR id was filled ─────────────────────
    # The LLM occasionally returns keys that don't match spec ids (e.g.
    # "value_1" instead of "v1") or silently drops a DR — both manifest as
    # downstream COMPUTE_FAIL "v1=1, v2=0". Try to remap by position/label
    # first, then retry once with explicit feedback if anything is still
    # missing.
    spec_dr_ids = [dr.get("id") for dr in data_requests if dr.get("id")]
    extractions = result.get("extractions") or {}

    def _missing_ids(ex: dict) -> list[str]:
        # A DR is "missing" if it's absent, has no values list, OR has a
        # values list where every entry is null. The all-null case is the
        # dominant failure mode: the LLM fills in labels but gives up on
        # numbers, which looks filled to a naive emptiness check but is
        # indistinguishable from missing to the compute phase.
        out = []
        for did in spec_dr_ids:
            v = ex.get(did) or {}
            vals = v.get("values") or []
            if not vals or not any(x is not None for x in vals):
                out.append(did)
        return out

    if isinstance(extractions, dict) and spec_dr_ids:
        # Position-based fallback: if extractions has the right number of
        # entries but mismatched keys, remap by spec order.
        missing = _missing_ids(extractions)
        if missing and len(extractions) == len(spec_dr_ids):
            ex_keys = list(extractions.keys())
            if any(k not in spec_dr_ids for k in ex_keys):
                remapped = {
                    spec_dr_ids[i]: extractions[ex_keys[i]] for i in range(len(spec_dr_ids))
                }
                if not _missing_ids(remapped):
                    result["extractions"] = remapped
                    extractions = remapped

        # ── Combined quality retry: missing + incomplete + labels-but-null ──
        # One retry addresses all extraction quality issues at once so we
        # stay within the LLM-call budget (max 2 calls per extract phase).
        missing = _missing_ids(extractions)

        def _incomplete_drs() -> list[tuple[str, int, int]]:
            out: list[tuple[str, int, int]] = []
            for dr in data_requests:
                did = dr.get("id")
                exp = dr.get("expected_count")
                if not did or not exp or not isinstance(exp, int):
                    continue
                if did in (missing or []):
                    continue  # already counted as missing
                v = extractions.get(did) or {}
                vals = [x for x in (v.get("values") or []) if x is not None]
                if 0 < len(vals) < exp:
                    out.append((did, len(vals), exp))
            return out

        incomplete = _incomplete_drs()

        if (missing or incomplete) and not feedback:
            feedback_parts: list[str] = []

            # Detect labels-but-null pattern → use alt rendering on retry
            has_labels_null = (
                any((extractions.get(did) or {}).get("labels") for did in missing)
                if missing
                else False
            )

            if missing:
                missing_details = []
                for did in missing:
                    v = extractions.get(did) or {}
                    dr_meta = next((d for d in data_requests if d.get("id") == did), {}) or {}
                    hint = (
                        f"{did}: label={dr_meta.get('label')!r} "
                        f"years={dr_meta.get('years')} "
                        f"row_hint={dr_meta.get('row_hint')!r} "
                        f"granularity={dr_meta.get('granularity')!r}"
                    )
                    prev_labels = v.get("labels")
                    prev_notes = v.get("notes")
                    if prev_labels:
                        hint += f" — you returned labels={prev_labels} but all values were null"
                    if prev_notes:
                        hint += f" (note: {prev_notes!r})"
                    missing_details.append(hint)
                feedback_parts.append(
                    "These data_requests had NO usable values:\n  - "
                    + "\n  - ".join(missing_details)
                )

            if incomplete:
                incomplete_details = []
                for did, got, exp in incomplete:
                    dr_meta = next((d for d in data_requests if d.get("id") == did), {}) or {}
                    labels = (extractions.get(did) or {}).get("labels") or []
                    incomplete_details.append(
                        f"{did}: got {got}/{exp} values "
                        f"(labels={labels}, years={dr_meta.get('years')}, "
                        f"granularity={dr_meta.get('granularity')!r}). "
                        f"Look for the missing entries in other tables."
                    )
                feedback_parts.append(
                    "These data_requests had FEWER values than expected:\n  - "
                    + "\n  - ".join(incomplete_details)
                )

            retry_feedback = (
                "\n\n".join(feedback_parts)
                + "\n\nScan ALL provided tables for each problematic id. "
                "Different DRs may need different tables. If a DR asks for "
                "a different year than another, look for the table whose "
                "file_year or row labels match that DR's year. Only return "
                "null for an individual cell if the row exists but the cell "
                "is truly blank/-/nan."
            )
            if verbose:
                labels = []
                if missing:
                    labels.append(f"missing={missing}")
                if incomplete:
                    labels.append(f"incomplete={[(d, g, e) for d, g, e in incomplete]}")
                print(f"  Quality retry: {', '.join(labels)}")

            retry = extract_structured(
                spec,
                per_dr_entries,
                question,
                verbose=verbose,
                feedback=retry_feedback,
                _alt_render=has_labels_null,
            )
            if retry and retry.get("extractions"):
                retry_ex = retry["extractions"]
                merged = dict(extractions)
                # For missing DRs: take retry if any non-null values
                for did in missing:
                    retry_v = retry_ex.get(did) or {}
                    retry_vals = retry_v.get("values") or []
                    if any(x is not None for x in retry_vals):
                        merged[did] = retry_v
                # For incomplete DRs: take retry if more non-null values
                for did, got, _exp in incomplete:
                    retry_v = retry_ex.get(did) or {}
                    retry_nonnull = [x for x in (retry_v.get("values") or []) if x is not None]
                    if len(retry_nonnull) > got:
                        merged[did] = retry_v
                result["extractions"] = merged
                extractions = merged

    _normalize_extraction_units(
        result.get("extractions") or {}, data_requests, spec, verbose=verbose
    )
    _verify_against_ledger(
        result.get("extractions") or {}, data_requests, per_dr_entries, verbose=verbose
    )
    _filter_cohort_aggregates(result.get("extractions") or {}, data_requests, verbose=verbose)
    return result


# ── Ledger cross-check (post-extraction) ─────────────────────────────────


def _verify_against_ledger(
    extractions: dict,
    data_requests: list[dict],
    per_dr_entries: dict,
    verbose: bool = False,
) -> None:
    """In-place: cross-check extracted values against the HTML table cells.

    For each extraction that cites row_labels and col_labels, parses the
    source table HTML and verifies the LLM's value matches the actual cell.
    Flags mismatches in the extraction's 'verification' field. On mismatch,
    replaces the value with the ground-truth cell value if unambiguous.
    """
    for dr in data_requests:
        dr_id = dr.get("id")
        if not dr_id:
            continue
        ex = extractions.get(dr_id)
        if not isinstance(ex, dict):
            continue
        values = ex.get("values") or []
        row_labels = ex.get("row_labels") or []
        col_labels = ex.get("col_labels") or []
        if not row_labels or not col_labels:
            continue
        if len(row_labels) != len(values) or len(col_labels) != len(values):
            continue

        entries = per_dr_entries.get(dr_id) or []
        if not entries:
            continue

        # Build a lookup from the HTML tables: (row_label_lower, col_label_lower) → cell_value
        cell_map: dict[tuple[str, str], float | None] = {}
        for entry in entries:
            html = entry.get("html") or ""
            if not html:
                continue
            try:
                parser = _TableHTMLParser()
                parser.feed(html)
                parser.close()
            except Exception:
                continue
            if len(parser.rows) < 2:
                continue
            headers = parser.rows[0]
            for row in parser.rows[1:]:
                if not row:
                    continue
                rl = row[0].strip().lower()
                for ci in range(1, min(len(row), len(headers))):
                    cl = headers[ci].strip().lower()
                    val = to_num(row[ci])
                    cell_map[(rl, cl)] = val

        if not cell_map:
            continue

        corrections = 0
        for i, (rl, cl, val) in enumerate(zip(row_labels, col_labels, values, strict=False)):
            if val is None or rl is None or cl is None:
                continue
            key = (str(rl).strip().lower(), str(cl).strip().lower())
            if key not in cell_map:
                continue
            ground_truth = cell_map[key]
            if ground_truth is None:
                continue
            # Allow small float tolerance (< 0.01% relative)
            if val != 0 and abs(val - ground_truth) / abs(val) < 0.0001:
                continue
            if val != ground_truth:
                corrections += 1
                values[i] = ground_truth

        if corrections > 0:
            ex["values"] = values
            ex["verification"] = f"corrected {corrections} values via ledger cross-check"
            if verbose:
                print(f"  Ledger cross-check [{dr_id}]: corrected {corrections} values")


# ── Cohort aggregate filtering (post-extraction) ──────────────────────────

_COHORT_AGGREGATE_RE = re.compile(
    r"\btotal\b|\ball\s+other\b|\bgrand\s+total\b|\bsubtotal\b",
    re.IGNORECASE,
)

_COHORT_OTHER_PREFIX_RE = re.compile(r"^other\s+", re.IGNORECASE)


def _is_aggregate_label(label: str) -> bool:
    """Check if a row label looks like an aggregate/total/regional summary."""
    s = label.strip()
    if _COHORT_AGGREGATE_RE.search(s):
        return True
    return bool(_COHORT_OTHER_PREFIX_RE.match(s))


def _filter_cohort_aggregates(
    extractions: dict,
    data_requests: list[dict],
    verbose: bool = False,
) -> None:
    """In-place: remove aggregate/total rows from cohort DR extractions.

    For data_requests with cohort=True, filters out entries whose labels
    match aggregate patterns (Total, All other, Other <region>, etc.).
    Prevents max/min/argmax from picking summary rows instead of
    individual entity rows. No-op when filtering would remove all rows.
    """
    for dr in data_requests:
        if not dr.get("cohort"):
            continue
        dr_id = dr.get("id")
        if not dr_id:
            continue
        ex = extractions.get(dr_id)
        if not isinstance(ex, dict):
            continue
        values = ex.get("values") or []
        labels = ex.get("labels") or []
        if not labels or len(labels) != len(values):
            continue

        keep: list[int] = []
        removed: list[str] = []
        for i, label in enumerate(labels):
            if _is_aggregate_label(str(label)):
                removed.append(str(label))
            else:
                keep.append(i)

        if not removed or not keep:
            continue

        ex["values"] = [values[i] for i in keep]
        ex["labels"] = [labels[i] for i in keep]
        if verbose:
            print(f"  Cohort filter [{dr_id}]: removed {len(removed)} aggregate rows: {removed}")


# ── Unit normalization (post-extraction) ───────────────────────────────────

# Map the LLM's source_unit enum (and a few common variants) onto compute's
# canonical unit keys ("thousands"/"millions"/"billions"/"percent"/"dollars").
_LLM_UNIT_MAP: dict[str, str] = {
    "thousands_usd": "thousands",
    "thousand_usd": "thousands",
    "thousands": "thousands",
    "thousand": "thousands",
    "millions_usd": "millions",
    "million_usd": "millions",
    "millions": "millions",
    "million": "millions",
    "billions_usd": "billions",
    "billion_usd": "billions",
    "billions": "billions",
    "billion": "billions",
    "usd": "dollars",
    "dollars": "dollars",
    "percent": "percent",
    "%": "percent",
}


def _canonical_unit(raw: str | None) -> str | None:
    if not raw:
        return None
    return _LLM_UNIT_MAP.get(raw.strip().lower())


# Phrases like "(in millions)", "nominal billions", "thousands of dollars"
# inside a DR label tell us the unit the question wants the value in.
_TARGET_UNIT_RE = re.compile(
    r"\b(thousand[s]?|million[s]?|billion[s]?|trillion[s]?|percent|%)\b",
    re.IGNORECASE,
)


def _infer_target_unit(dr: dict) -> str | None:
    for field in ("label", "unit_hint", "description"):
        text = dr.get(field) or ""
        m = _TARGET_UNIT_RE.search(text)
        if m:
            return _canonical_unit(m.group(1))
    return None


def _normalize_extraction_units(
    extractions: dict,
    data_requests: list[dict],
    spec: dict | None = None,
    verbose: bool = False,
) -> None:
    """In-place: scale each DR's values from source_unit → target unit.

    Box-cox, log, geometric mean, and any other non-scale-invariant op needs
    its inputs in the unit the question is asking about. We can't fix this
    after compute runs, so we fix it here, before compute sees the values.

    No-op when source_unit is missing, target unit can't be inferred from the
    DR label, or the units already match. Records `unit_normalized_from`
    /`unit_normalized_to` on the extraction so downstream can audit.
    """
    from compute import convert_unit

    # Spec-level fallback target unit (output_format.unit). Used when the DR
    # label doesn't carry a unit phrase — common when decompose stores the
    # target unit at the spec level instead of repeating it per DR.
    spec_target = None
    if spec:
        of_unit = (spec.get("output_format") or {}).get("unit")
        if of_unit:
            spec_target = _canonical_unit(of_unit)

    for dr in data_requests:
        dr_id = dr.get("id")
        if not dr_id:
            continue
        ex = extractions.get(dr_id)
        if not isinstance(ex, dict):
            continue
        src = _canonical_unit(ex.get("source_unit"))
        tgt = _infer_target_unit(dr) or spec_target
        if not src or not tgt or src == tgt:
            continue
        if "dollars" in (src, tgt) and {src, tgt} != {"dollars"}:
            # Treat raw dollars as a real scale: 1.0 vs 1e3/1e6/1e9.
            from compute import UNIT_MULTIPLIERS

            UNIT_MULTIPLIERS.setdefault("dollars", 1.0)
        vals = ex.get("values") or []
        new_vals = convert_unit(vals, src, tgt)
        if new_vals is vals:
            continue
        ex["values"] = new_vals
        ex["unit_normalized_from"] = src
        ex["unit_normalized_to"] = tgt
        if verbose:
            print(f"  Unit-normalize {dr_id}: {src} → {tgt}")


# ── CLI + Testing ────────────────────────────────────────────────────────────


def _load_benchmark():
    import csv

    rows = []
    with open("officeqa_full.csv") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def _test_oracle(n: int = 20):
    """Test extraction with gold source files."""
    rows = _load_benchmark()
    if n:
        rows = rows[:n]

    from reward import fuzzy_match_answer

    correct, total = 0, 0
    for row in rows:
        gold_files = [f"corpus/{f.strip()}" for f in row["source_files"].split("\n") if f.strip()]
        question = row["question"]
        expected = row["answer"].strip()

        q_words = re.findall(r"[A-Za-z][\w'-]+", question)
        keywords = expand_terms([w for w in q_words if len(w) > 3][:5])

        got = extract_answer(question, gold_files, keywords)
        match, rationale = fuzzy_match_answer(expected, got, tolerance=0.01)

        total += 1
        if match:
            correct += 1
            print(f"  OK {row['uid']}: {got}")
        else:
            print(f"  MISS {row['uid']}: expected={expected!r} got={got!r} ({rationale})")

    print(f"\nOracle extraction: {correct}/{total} = {correct / total * 100:.1f}%")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]

    if "--test-oracle" in args:
        n = 20
        for i, a in enumerate(args):
            if a == "--n" and i + 1 < len(args):
                with contextlib.suppress(ValueError):
                    n = int(args[i + 1])
        _test_oracle(n)
    elif args:
        print("Usage: uv run python extract.py --test-oracle [--n N]")
