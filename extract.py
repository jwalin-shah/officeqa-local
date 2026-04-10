#!/usr/bin/env python3
"""Extract answers from Treasury Bulletin files.

Strategy:
1. Parse files into table blocks, split blocks into sub-sections
2. Use difflib fuzzy matching to find the best sub-section for the question
3. Send focused context (header + matched sub-section) to LLM
"""

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
    lines = []
    for i, row in enumerate(parser.rows):
        if i >= max_rows:
            lines.append(f"... ({len(parser.rows) - max_rows} more rows truncated)")
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
    """
    h = col_header.strip().lower()
    # Direct month name match
    if h in _CALENDAR_MONTHS:
        return _CALENDAR_MONTHS[h]
    # Check if header starts with a month name (e.g., "Jan. > 1940")
    for month_name, idx in _CALENDAR_MONTHS.items():
        if h.startswith(month_name):
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

    header_row = parser.rows[0]
    data_rows = parser.rows[1:]

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


def render_entry(entry: dict, vertical_threshold: int = 8) -> str:
    """Turn a retrieve_v2 table entry into a labelled context block.

    Wide tables (≥vertical_threshold columns) are rendered in vertical
    key-value format; narrow tables use the traditional pipe-delimited format."""
    file = entry.get("file") or "?"
    section = (entry.get("section") or "").strip()
    title = (entry.get("title") or "").strip()
    caption = (entry.get("caption") or "").strip()
    element_id = entry.get("element_id")

    header_bits = [f"# {file}"]
    if element_id is not None:
        header_bits[0] += f" (table #{element_id})"
    if title:
        header_bits.append(f"Title: {title}")
    if section and section != title:
        header_bits.append(f"Section: {section}")
    if caption:
        header_bits.append(f"Caption: {caption}")

    html = entry.get("html") or ""

    # Try vertical format first for wide tables
    vertical = html_to_vertical_text(html, vertical_threshold=vertical_threshold)
    if vertical:
        body = vertical
    else:
        body = html_to_pipe_text(html)
        if not body:
            return ""

    return "\n".join(header_bits) + "\n" + body


def build_context_from_entries(entries: list[dict], char_budget: int = 10000) -> str:
    """Render a list of retrieve_v2 entries into one context blob, stopping at budget."""
    parts: list[str] = []
    total = 0
    for e in entries:
        block = render_entry(e)
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

    def __init__(self, label: str, header_line: str, data_lines: list[str], file_line_start: int):
        self.label = label
        self.header_line = header_line
        self.data_lines = data_lines
        self.file_line_start = file_line_start
        # Parse header columns
        self.headers = [parse_header(h) for h in split_row(header_line)]

    def raw_text(self) -> str:
        """Return header + data as raw text for the LLM."""
        return self.header_line + "\n" + "\n".join(self.data_lines)

    def get_cell(self, row_idx: int, col_name: str) -> str | None:
        """Get a cell value by row index and column name."""
        if row_idx >= len(self.data_lines):
            return None
        cells = split_row(self.data_lines[row_idx])
        for i, h in enumerate(self.headers):
            if h == col_name and i < len(cells):
                return cells[i]
        return None


def parse_table_sections(lines: list[str], block_start: int) -> list[TableSection]:
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
                    TableSection(current_label, header_line, current_data, current_start)
                )
            current_label = cells[0].strip()
            current_data = []
            current_start = block_start + i
        else:
            current_data.append(line)

    # Save last section
    if current_data:
        sections.append(TableSection(current_label, header_line, current_data, current_start))

    # If no section headers found, treat entire block as one section
    if not sections and data_start < len(lines):
        all_data = [ln for ln in lines[data_start:] if "|" in ln and not is_separator(ln)]
        if all_data:
            sections.append(TableSection("", header_line, all_data, block_start + data_start))

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
            if to_num(cell) is not None:
                n = abs(to_num(cell))
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
            block_lines = file_lines[block_start:block_end]
            sections = parse_table_sections(block_lines, block_start)
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
Each section has a header row (column names) and data rows.

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

EXTRACT_STRUCTURED_SYSTEM = """You extract raw numeric values from U.S. Treasury
Bulletin tables to fill slots defined in a QuestionSpec.

CRITICAL RULE: You DO NOT compute anything. You only extract raw numbers from
tables and cite where they came from. Python will run the computation later.

EXTRACTION CHECKLIST (these are the failure modes we paid to find):
  - UNITS: note the table's scale ("in millions", "in thousands") in `notes`
    if it isn't the obvious default. Return the RAW number as printed — the
    formatter handles unit conversion. Do not pre-scale.
  - TOTAL vs SUB-CATEGORY: if the spec asks for a specific child line, never
    return the "Total X" row value. Total rows already include all children.
  - ACTUAL vs ESTIMATED: when both are present, prefer the actual row.
  - ANNUAL vs MONTHLY: a monthly row is not an annual total. For
    granularity="annual", return the single year-labeled row; for
    granularity="monthly_all", return all 12 monthly rows.
  - COLUMN POSITION: wide tables with multi-level headers are easy to misread.
    Confirm the column by its header text, not its position.

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

Output ONLY valid JSON matching this schema:
{
  "extractions": {
    "v1": {
      "values": [132, 129, 143, 159, 154, 153, 177, 200, 219, 287, 376, 473],
      "labels": ["1940-January", "February", "March", ...],
      "source_file": "treasury_bulletin_1941_01.txt",
      "confidence": "high"
    }
  },
  "notes": "<anything unusual, e.g. had to skip annual row>"
}"""


def extract_structured(
    spec: dict, per_dr_entries: dict, question: str, verbose: bool = False, feedback: str = ""
) -> dict | None:
    """Extract structured values per the QuestionSpec's data_requests.

    `per_dr_entries` is {dr_id: [retrieve_v2 entries]}. Each entry carries the
    table HTML inline — we render it straight to pipe-delimited text. No
    .txt re-scanning, no per-DR fuzzy section scoring. The specific tables
    retrieve_v2 identified are exactly what the LLM sees."""
    data_requests = spec.get("data_requests") or []
    if not data_requests:
        return None

    per_request_context: list[str] = []
    # Top-k retrieval returns 10 entries per DR. With dedupe_by_file we care
    # about seeing rank 9-10 as often as rank 1, so the budget has to be wide
    # enough to fit the full list even when each table has a lot of rows.
    per_request_budget = max(6000, 24000 // max(1, len(data_requests)))

    any_hit = False
    for dr in data_requests:
        dr_id = dr.get("id", "?")
        dr_label = dr.get("label", "")
        dr_years = [y for y in (dr.get("years") or [])]
        entries = per_dr_entries.get(dr_id) or []
        ctx = build_context_from_entries(entries, char_budget=per_request_budget) if entries else ""

        header = (
            f"=== Context for {dr_id} ({dr_label}) — "
            f"years={dr_years} granularity={dr.get('granularity', '?')} ==="
        )
        if ctx:
            per_request_context.append(f"{header}\n{ctx}")
            any_hit = True
        else:
            per_request_context.append(f"{header}\n(no tables retrieved for this data_request)")

    if not any_hit:
        return None
    context = "\n\n".join(per_request_context)

    import json as _json

    spec_json = _json.dumps(
        {
            "computation": spec.get("computation"),
            "data_requests": data_requests,
        },
        indent=2,
    )

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
    return result


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
                try:
                    n = int(args[i + 1])
                except ValueError:
                    pass
        _test_oracle(n)
    elif args:
        print("Usage: uv run python extract.py --test-oracle [--n N]")
