#!/usr/bin/env python3
"""intern.py — Python "research intern" for OfficeQA.

Ported from arena v14/solve_v14.py (the 184.5/246 = 75% winning approach).
Python does all the hard work: parse tables, find matching cells, precompute.
LLM gets a structured briefing and acts as a reviewer, not an extractor.

Usage:
    from intern import build_briefing
    briefing_text = build_briefing(question, corpus_json_files)
"""

import json
import math
import os
import re
import statistics
from collections import defaultdict
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path

# ── CPI-U 1913–2024 ──────────────────────────────────────────────────────────
CPI = {
    1913: 9.9,
    1914: 10.0,
    1915: 10.1,
    1916: 10.9,
    1917: 12.8,
    1918: 15.1,
    1919: 17.3,
    1920: 20.0,
    1921: 17.9,
    1922: 16.8,
    1923: 17.1,
    1924: 17.1,
    1925: 17.5,
    1926: 17.7,
    1927: 17.4,
    1928: 17.2,
    1929: 17.2,
    1930: 16.7,
    1931: 15.2,
    1932: 13.6,
    1933: 12.9,
    1934: 13.4,
    1935: 13.7,
    1936: 13.9,
    1937: 14.4,
    1938: 14.1,
    1939: 13.9,
    1940: 14.0,
    1941: 14.7,
    1942: 16.3,
    1943: 17.3,
    1944: 17.6,
    1945: 18.0,
    1946: 19.5,
    1947: 22.3,
    1948: 24.1,
    1949: 23.8,
    1950: 24.1,
    1951: 26.0,
    1952: 26.5,
    1953: 26.7,
    1954: 26.9,
    1955: 26.8,
    1956: 27.2,
    1957: 28.1,
    1958: 28.9,
    1959: 29.1,
    1960: 29.6,
    1961: 29.9,
    1962: 30.2,
    1963: 30.6,
    1964: 31.0,
    1965: 31.5,
    1966: 32.4,
    1967: 33.4,
    1968: 34.8,
    1969: 36.7,
    1970: 38.8,
    1971: 40.5,
    1972: 41.8,
    1973: 44.4,
    1974: 49.3,
    1975: 53.8,
    1976: 56.9,
    1977: 60.6,
    1978: 65.2,
    1979: 72.6,
    1980: 82.4,
    1981: 90.9,
    1982: 96.5,
    1983: 99.6,
    1984: 103.9,
    1985: 107.6,
    1986: 109.6,
    1987: 113.6,
    1988: 118.3,
    1989: 124.0,
    1990: 130.7,
    1991: 136.2,
    1992: 140.3,
    1993: 144.5,
    1994: 148.2,
    1995: 152.4,
    1996: 156.9,
    1997: 160.5,
    1998: 163.0,
    1999: 166.6,
    2000: 172.2,
    2001: 177.1,
    2002: 179.9,
    2003: 184.0,
    2004: 188.9,
    2005: 195.3,
    2006: 201.6,
    2007: 207.3,
    2008: 215.3,
    2009: 214.5,
    2010: 218.1,
    2011: 224.9,
    2012: 229.6,
    2013: 233.0,
    2014: 236.7,
    2015: 237.0,
    2016: 240.0,
    2017: 245.1,
    2018: 251.1,
    2019: 255.7,
    2020: 258.8,
    2021: 271.0,
    2022: 292.7,
    2023: 304.7,
    2024: 314.2,
}

# Fiscal year convention: pre-1977 = Jul–Jun, post-1976 = Oct–Sep
FY_RULES = (
    "FY RULES: Pre-1977 FY = Jul(Y-1)–Jun(Y). Post-1976 FY = Oct(Y-1)–Sep(Y). "
    "e.g. FY 1940 = Jul 1939–Jun 1940. FY 1980 = Oct 1979–Sep 1980."
)

MONTHS = [
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
MON3 = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

STOP = {
    "what",
    "were",
    "was",
    "the",
    "of",
    "in",
    "for",
    "a",
    "an",
    "to",
    "and",
    "or",
    "is",
    "how",
    "much",
    "many",
    "total",
    "amount",
    "during",
    "fiscal",
    "year",
    "calendar",
    "fy",
    "cy",
    "from",
    "by",
    "on",
    "at",
    "as",
    "it",
    "its",
    "be",
    "are",
    "this",
    "that",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "may",
    "might",
    "shall",
    "can",
    "per",
    "than",
    "into",
    "over",
    "about",
    "between",
    "through",
    "with",
    "not",
    "no",
    "all",
    "each",
    "every",
    "some",
    "any",
    "been",
    "being",
    "using",
    "specifically",
    "only",
    "reported",
    "values",
    "individual",
    "months",
    "these",
    "those",
    "which",
    "sum",
    "given",
    "following",
    "based",
    "according",
    "report",
    "value",
    "number",
    "figure",
    "data",
    "information",
    "monthly",
    "annual",
}

SYNONYMS = {
    "expenditures": ["outlays", "spending", "disbursements"],
    "outlays": ["expenditures", "spending"],
    "receipts": ["revenue", "income", "collections"],
    "revenue": ["receipts", "income", "collections"],
    "defense": ["national defense", "military"],
    "debt": ["public debt", "obligations"],
    "interest": ["interest cost", "net interest"],
    "tax": ["taxation", "taxes"],
    "imports": ["merchandise imports"],
    "exports": ["merchandise exports"],
    "surplus": ["excess"],
    "deficit": ["shortfall"],
}


# ── Question parsing ──────────────────────────────────────────────────────────


def extract_years(text: str) -> list[int]:
    years: set[int] = set()
    for m in re.finditer(r"\b(1[89]\d{2}|20[0-2]\d)s\b", text):
        years.update(range(int(m.group(1)), int(m.group(1)) + 10))
    for m in re.finditer(
        r"\b(1[89]\d{2}|20[0-2]\d)\s*(?:to|through|-|–)\s*(1[89]\d{2}|20[0-2]\d)\b", text
    ):
        y1, y2 = int(m.group(1)), int(m.group(2))
        if 0 < y2 - y1 <= 30:
            years.update(range(y1, y2 + 1))
    for m in re.finditer(
        r"\bbetween\s+(1[89]\d{2}|20[0-2]\d)\s+and\s+(1[89]\d{2}|20[0-2]\d)\b", text
    ):
        y1, y2 = int(m.group(1)), int(m.group(2))
        if 0 < y2 - y1 <= 30:
            years.update(range(y1, y2 + 1))
    for m in re.finditer(r"\b(1[89]\d{2}|20[0-2]\d)\b", text):
        years.add(int(m.group(1)))
    return sorted(years)


def extract_keywords(question: str, extra: str | None = None) -> list[str]:
    words = re.findall(r"[a-zA-Z]+", question.lower())
    kw = [w for w in words if w not in STOP and len(w) > 2]
    if extra:
        kw.extend(re.findall(r"[a-zA-Z]+", extra.lower()))
    expanded = list(kw)
    seen = set(k.lower() for k in expanded)
    for k in kw:
        for key, syns in SYNONYMS.items():
            if key == k or key in k:
                for s in syns:
                    if s.lower() not in seen:
                        expanded.append(s)
                        seen.add(s.lower())
    return list(dict.fromkeys(expanded))


def detect_period(question: str) -> str:
    q = question.lower()
    if re.search(r"\bfiscal\s+year\b|\bfy\s*\d|\bfy\b", q):
        return "fiscal"
    if re.search(r"\bcalendar\s+year\b|\bcy\s*\d|\bcy\b", q):
        return "calendar"
    return "calendar"  # Treasury default


def detect_operation(question: str) -> str:
    q = question.lower()
    if re.search(
        r"percent(age)?\s+(change|increase|decrease|growth|decline)|% change|growth rate|grew by", q
    ):
        return "pct_change"
    if re.search(
        r"\bdifference\b|\bhow much (more|less|greater|larger|smaller)\b|\bchange in\b|\bnet change\b",
        q,
    ):
        return "difference"
    if re.search(r"\bratio\b|\btimes\b|\bfold\b|\bproportion\b", q):
        return "ratio"
    if re.search(r"\baverage\b|\bmean\b", q):
        return "mean"
    if re.search(r"\bstandard deviation\b|\bstd dev\b|\bstdev\b|\bvolatility\b|\bvariance\b", q):
        return "stdev"
    if re.search(r"\bspread\b|\bdifference between.{0,40}and\b|\byield.{0,20}minus\b", q):
        return "spread"
    if re.search(r"\bregression\b|\bslope\b|\bintercept\b|\bforecast\b|\btrend\b", q):
        return "regression"
    if re.search(r"\bhighest\b|\blargest\b|\bmaximum\b|\bgreatest\b|\bpeak\b", q):
        return "max"
    if re.search(r"\blowest\b|\bsmallest\b|\bminimum\b|\bleast\b", q):
        return "min"
    if re.search(r"\bgeometric\s+mean\b", q):
        return "geometric_mean"
    if re.search(r"\bsum\b|\btotal\b|\bcombined\b|\baggregate\b", q):
        return "sum"
    return "lookup"


def detect_units(text: str) -> str:
    t = text.lower()
    if "in billions" in t or "billions of dollars" in t:
        return "billions of dollars"
    if "in millions" in t or "millions of dollars" in t:
        return "millions of dollars"
    if "in thousands" in t or "thousands of dollars" in t:
        return "thousands of dollars"
    if "percent" in t or "%" in t:
        return "percent"
    return "dollars (units unclear)"


# ── Number helpers ────────────────────────────────────────────────────────────


def parse_number(s: str) -> float | None:
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    s = re.sub(r"\s*[a-z0-9]+/\s*$", "", s)  # footnote refs
    s = re.sub(r"[*]+$", "", s)
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    if s in ("", "-", "...", "---", "\u2014", "\u2013", "n.a.", "N/A", "(X)", "X"):
        return None
    m = re.match(r"^[rpe]\s+(.+)$", s)
    if m:
        s = m.group(1).replace(",", "")
    try:
        val = float(s)
        return -val if neg else val
    except ValueError:
        return None


# ── HTML Table Parser ─────────────────────────────────────────────────────────


class _HTMLParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tables: list[list[list[dict]]] = []
        self._cur_table: list | None = None
        self._cur_row: list | None = None
        self._in_cell = False
        self._cell_tag = None
        self._cell_text = ""
        self._colspan = 1
        self._rowspan = 1

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "table":
            self._cur_table = []
        elif tag == "tr" and self._cur_table is not None:
            self._cur_row = []
        elif tag in ("th", "td") and self._cur_row is not None:
            self._in_cell = True
            self._cell_tag = tag
            self._cell_text = ""
            self._colspan = int(a.get("colspan", 1))
            self._rowspan = int(a.get("rowspan", 1))

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._in_cell:
            self._cur_row.append(
                {
                    "text": self._cell_text.strip(),
                    "tag": self._cell_tag,
                    "colspan": self._colspan,
                }
            )
            self._in_cell = False
        elif tag == "tr" and self._cur_row is not None:
            if self._cur_table is not None:
                self._cur_table.append(self._cur_row)
            self._cur_row = None
        elif tag == "table" and self._cur_table is not None:
            self.tables.append(self._cur_table)
            self._cur_table = None

    def handle_data(self, data):
        if self._in_cell:
            self._cell_text += data


def _clean_cell(s: str) -> str:
    s = re.sub(r"\s*[a-z0-9]+/\s*$", "", s)
    s = re.sub(r"^[rpe]/\s*", "", s)
    s = re.sub(r"[rpe]\s*$", "", s)
    return s.strip()


def parse_html_to_entries(html: str, file_stem: str = "") -> list[dict]:
    """Parse HTML table into vertical entries: [{row_label, column, value, numeric, file}]."""
    p = _HTMLParser()
    try:
        p.feed(html)
    except Exception:
        return []

    entries = []
    for table_rows in p.tables:
        if not table_rows:
            continue
        headers: list[str] = []
        data_start = 0
        for ri, row in enumerate(table_rows):
            if any(c["tag"] == "th" for c in row):
                hdr_cells: list[str] = []
                for c in row:
                    text = _clean_cell(c["text"])
                    for _ in range(c["colspan"]):
                        hdr_cells.append(text)
                if not headers:
                    headers = hdr_cells
                else:
                    for j in range(min(len(headers), len(hdr_cells))):
                        if hdr_cells[j] and headers[j]:
                            headers[j] = f"{headers[j]} > {hdr_cells[j]}"
                        elif hdr_cells[j]:
                            headers[j] = hdr_cells[j]
                data_start = ri + 1
            else:
                break
        if not headers and table_rows:
            headers = [_clean_cell(c["text"]) for c in table_rows[0]]
            data_start = 1

        for row in table_rows[data_start:]:
            cells = [_clean_cell(c["text"]) for c in row]
            if not cells:
                continue
            label = re.sub(r"\.{2,}\s*$", "", cells[0]).strip()
            label = re.sub(r"\s*\.{3,}\s*", " ", label).strip()
            if not label or re.match(r"^[\s\-=_.]+$", label):
                continue
            for ci in range(1, len(cells)):
                raw = cells[ci]
                if not raw or raw in ("-", "nan", "...", "(X)"):
                    continue
                hdr = headers[ci] if ci < len(headers) else f"col_{ci}"
                entries.append(
                    {
                        "row_label": label,
                        "column": hdr,
                        "value": raw,
                        "numeric": parse_number(raw),
                        "file": file_stem,
                    }
                )
    return entries


# ── Monthly aggregation ───────────────────────────────────────────────────────


def collect_monthly(entries: list[dict], year: int, keywords: list[str]) -> list[dict]:
    """Collect 12 monthly entries for a year, both row-oriented and col-oriented tables."""
    y_s = str(year)
    kw_top = keywords[:8]

    # Strategy 0: ledger entries have explicit year/month fields — use them directly.
    # Group by row_label, find rows with all 12 months of the target year.
    by_row: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("year") == year and e.get("month") is not None and e["numeric"] is not None:
            by_row[e["row_label"]].append(e)

    # Strategy 0b: ledger year labels can be off by 1 (file year used when no explicit
    # year prefix in col_path). Find the best keyword-matching row that has >= 10 monthly
    # entries regardless of year — the gold file context implies the data is for ~year.
    by_row_any: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e.get("month") is not None and e["numeric"] is not None:
            by_row_any[e["row_label"]].append(e)

    # Pick the best across both: run both strategies and take the higher-scoring result.
    # This prevents generic-keyword rows ("Expenditures 1/ 3/ > 1940") from beating
    # domain-specific rows ("Budget: > National defense") just because the row label
    # contains the target year.
    best_s0_sc, best_s0_entries = -1, []
    for row_label, row_entries in by_row.items():
        if len(row_entries) < 10:
            continue
        sc = sum(2 for k in kw_top if k in row_label.lower())
        if sc > best_s0_sc:
            best_s0_sc, best_s0_entries = sc, row_entries

    best_s0b_sc, best_s0b_entries = -1, []
    for rl, es in by_row_any.items():
        if len(es) < 10:
            continue
        sc = sum(2 for k in kw_top if k in rl.lower())
        if sc > best_s0b_sc:
            best_s0b_sc, best_s0b_entries = sc, es

    if best_s0_entries or best_s0b_entries:
        # Prefer the higher-scoring candidate; 0b wins ties (may include off-by-1-year data)
        chosen = None
        if best_s0b_sc > best_s0_sc and best_s0b_entries:
            chosen = best_s0b_entries
        elif best_s0_entries:
            chosen = best_s0_entries
        elif best_s0b_entries:
            chosen = best_s0b_entries
        if chosen is not None:
            # Filter out entries with year clearly before the target (e.g., Dec of prior year
            # that appears in the same table as the target year's monthly series).
            filtered = [e for e in chosen if e.get("year") is None or e["year"] >= year]
            if len(filtered) >= 10:
                return filtered
            return chosen  # fall back to unfiltered if filtering removed too many

    # Strategy 1: row-oriented — year+month in row_label, category in column
    by_col: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e["numeric"] is None:
            continue
        ll = e["row_label"].lower()
        if any(m in ll for m in MONTHS + MON3):
            by_col[e["column"].lower()].append(e)

    for col_key, col_entries in by_col.items():
        if not any(k in col_key for k in kw_top):
            continue
        in_year = False
        year_monthly: list[dict] = []
        seen_months: set[str] = set()
        for e in col_entries:
            ll = e["row_label"].lower().strip()
            ll_clean = re.sub(r"[.\s]+$", "", ll)
            month_key = None
            for i, full in enumerate(MONTHS):
                if full in ll_clean or MON3[i] in ll_clean.split("-")[-1].split()[0:1]:
                    month_key = full
                    break
            if y_s in ll and month_key:
                in_year = True
                if month_key not in seen_months:
                    seen_months.add(month_key)
                    year_monthly.append(e)
                continue
            if in_year and month_key:
                if re.match(r"^\d{4}", ll_clean):
                    if y_s not in ll_clean:
                        break
                    continue
                if month_key not in seen_months:
                    seen_months.add(month_key)
                    year_monthly.append(e)
                continue
            if in_year and not month_key:
                if re.match(r"^\d{4}", ll_clean) and y_s not in ll_clean:
                    break
        if len(year_monthly) >= 10:
            return year_monthly

    # Strategy 2: col-oriented — year in column header, month in row label
    monthly = [
        e
        for e in entries
        if e["numeric"] is not None
        and y_s in e["column"].lower()
        and any(m in e["row_label"].lower() for m in MON3 + MONTHS)
        and any(k in e["row_label"].lower() for k in kw_top)
    ]
    if len(monthly) >= 10:
        return monthly

    # Strategy 3: ledger column-oriented — year AND month both in col_path
    # (e.g. col_path='1940 > Apr.' — month isn't in row_label but IS in column)
    # Group by row_label, score by keyword, pick the best series.
    by_row_col: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        if e["numeric"] is None:
            continue
        col_l = e["column"].lower()
        if y_s in col_l and any(m in col_l for m in MON3 + MONTHS):
            by_row_col[e["row_label"]].append(e)
    if by_row_col:
        best_rl, best_sc3, best_rl_entries = None, -1, []
        for row_label, rl_entries in by_row_col.items():
            if len(rl_entries) < 10:
                continue
            sc = sum(2 for k in kw_top if k in row_label.lower())
            if sc > best_sc3:
                best_sc3, best_rl, best_rl_entries = sc, row_label, rl_entries
        if best_rl_entries:
            return best_rl_entries

    return []


# ── Best-value lookup ─────────────────────────────────────────────────────────


def best_value(entries: list[dict], keywords: list[str], year: int) -> dict | None:
    yr_s = str(year)
    kw_phrase = " ".join(keywords[:4])
    best, best_sc = None, -1
    for e in entries:
        if e["numeric"] is None:
            continue
        sc = 0
        ll, cl = e["row_label"].lower(), e["column"].lower()
        if yr_s in cl:
            sc += 5
        if re.match(rf"^{yr_s}$", ll.split("-")[0].strip()):
            sc += 5
        if yr_s in e["value"]:
            sc += 2
        sc += sum(2 for k in keywords if k in ll or k in cl)
        combined = f"{ll} {cl}"
        if kw_phrase and SequenceMatcher(None, kw_phrase, combined).ratio() >= 0.5:
            sc += 1
        if "total" in ll:
            sc += 1
        if any(m in ll for m in MONTHS + MON3):
            sc -= 1
        if sc > best_sc:
            best_sc, best = sc, e
    return best


# ── Evidence builder ──────────────────────────────────────────────────────────


def build_evidence_from_ledger(
    file_stem: str,
    keywords: list[str],
    years: list[int],
    db_conn,
) -> dict | None:
    """Build evidence dict from ledger.sqlite — faster and more reliable than HTML parsing.

    Queries the metrics view for all cells in the given file, then scores and
    selects candidates exactly like the HTML-parsing path.
    """
    try:
        rows = db_conn.execute(
            """
            SELECT row_path, col_path, year, month, value_raw, value, unit
            FROM metrics
            WHERE file = ?
            ORDER BY row_path, col_path, year, month
            """,
            (f"{file_stem}.json",),
        ).fetchall()
    except Exception:
        return None

    if not rows:
        return None

    entries: list[dict] = []
    units_seen: list[str] = []
    for row_path, col_path, year, month, value_raw, value_num, unit in rows:
        raw = (
            str(value_raw)
            if value_raw is not None
            else (str(value_num) if value_num is not None else "")
        )
        if not raw:
            continue
        # Build column label: if month col, include year prefix
        col_label = col_path or ""
        entries.append(
            {
                "row_label": row_path or "",
                "column": col_label,
                "value": raw,
                "numeric": float(value_num) if value_num is not None else None,
                "file": file_stem,
                "year": year,
                "month": month,
            }
        )
        if unit and unit not in units_seen:
            units_seen.append(unit)

    extracted: dict[int, dict] = {}
    for y in years:
        v = best_value(entries, keywords, y)
        if v:
            extracted[y] = v

    text_lower = " ".join(f"{e['row_label']} {e['column']}" for e in entries).lower()
    relevance = sum(1 for k in keywords if k in text_lower) * 2 + sum(
        1 for y in years if str(y) in text_lower
    )

    units_str = units_seen[0] if units_seen else "dollars (units unclear)"

    return {
        "file": file_stem,
        "units": units_str,
        "period": _detect_evidence_period(entries),
        "entries": entries,
        "extracted_values": extracted,
        "relevance": relevance,
    }


def build_evidence(
    corpus_json_path: str, keywords: list[str], years: list[int], db_conn=None
) -> dict:
    """Load a corpus_json file and build evidence dict.

    Prefers ledger.sqlite when a db_conn is provided (faster, no HTML parsing).
    Falls back to HTML parsing from corpus_json if ledger has no data.
    """
    stem = Path(corpus_json_path).stem

    # Try ledger first
    if db_conn is not None:
        ev = build_evidence_from_ledger(stem, keywords, years, db_conn)
        if ev:
            return ev

    # Fall back to HTML parsing
    entries: list[dict] = []
    units = ""

    try:
        with open(corpus_json_path) as f:
            doc = json.load(f)
        elements = doc.get("document", {}).get("elements", [])
        section_context = ""
        for elem in elements:
            etype = elem.get("type", "")
            if etype in ("section_header", "title"):
                section_context = (elem.get("content") or "").strip()
            if etype != "table":
                continue
            html = elem.get("content") or ""
            if not html:
                continue
            if not units:
                u = detect_units(html)
                if u != "dollars (units unclear)":
                    units = u
            parsed = parse_html_to_entries(html, stem)
            entries.extend(parsed)
    except Exception:
        pass

    extracted: dict[int, dict] = {}
    for y in years:
        v = best_value(entries, keywords, y)
        if v:
            extracted[y] = v

    text_lower = " ".join(f"{e['row_label']} {e['column']}" for e in entries).lower()
    relevance = sum(1 for k in keywords if k in text_lower) * 2 + sum(
        1 for y in years if str(y) in text_lower
    )

    return {
        "file": stem,
        "units": units or "dollars (units unclear)",
        "period": _detect_evidence_period(entries),
        "entries": entries,
        "extracted_values": extracted,
        "relevance": relevance,
    }


def _detect_evidence_period(entries: list[dict]) -> str:
    text = " ".join(e["row_label"].lower() for e in entries[:200])
    fy = sum(1 for s in ("fiscal year", "fiscal years") if s in text)
    cy = sum(1 for s in ("calendar year", "calendar years") if s in text)
    mo = sum(1 for s in MONTHS + MON3 if re.search(r"\b" + s + r"\b", text))
    if mo >= 3:
        return "calendar"
    return "calendar" if cy > fy else "fiscal" if fy > cy else "unknown"


# ── Precomputation ────────────────────────────────────────────────────────────


def _get_vals(evidence: list[dict], years: list[int]) -> dict[int, float]:
    result: dict[int, float] = {}
    for y in years:
        for ev in evidence:
            v = ev.get("extracted_values", {}).get(y)
            if v and v["numeric"] is not None:
                result[y] = v["numeric"]
                break
    return result


def precompute(
    op: str, evidence: list[dict], keywords: list[str], years: list[int]
) -> tuple[float | None, str | None]:
    """Try to compute the answer deterministically. Returns (value, trace) or (None, None)."""
    if not evidence or not years:
        return None, None
    all_entries = [e for ev in evidence for e in ev.get("entries", [])]

    if op == "lookup" and years:
        for ev in evidence:
            v = ev.get("extracted_values", {}).get(years[0])
            if v and v["numeric"] is not None:
                return v["numeric"], f"lookup: {v['row_label']}, {v['column']} = {v['value']}"

    if op == "sum":
        y0 = years[0]
        for evi in evidence:
            monthly = collect_monthly(evi.get("entries", []), y0, keywords)
            if len(monthly) >= 10:
                t = sum(e["numeric"] for e in monthly)
                labels = ", ".join(e["row_label"] for e in monthly[:3])
                return t, f"SUM of {len(monthly)} monthly values ({labels}...) = {t:,.2f}"
        if len(years) > 2:
            vals = _get_vals(evidence, years)
            if len(vals) == len(years):
                t = sum(vals.values())
                detail = " + ".join(f"{vals[y]:,.2f} ({y})" for y in years)
                return t, f"SUM across {len(years)} years: {detail} = {t:,.2f}"

    if op == "pct_change" and len(years) >= 2:
        y_old, y_new = years[0], years[-1]
        vo_m = vn_m = None
        for evi in evidence:
            ents = evi.get("entries", [])
            if vo_m is None:
                m = collect_monthly(ents, y_old, keywords)
                if len(m) >= 10:
                    vo_m = sum(e["numeric"] for e in m)
            if vn_m is None:
                m = collect_monthly(ents, y_new, keywords)
                if len(m) >= 10:
                    vn_m = sum(e["numeric"] for e in m)
        if vo_m is not None and vn_m is not None and vo_m != 0:
            p = round((vn_m - vo_m) / abs(vo_m) * 100, 2)
            return p, f"pct_change(monthly_sum {y_old}={vo_m:,.2f}, {y_new}={vn_m:,.2f}) = {p:.2f}%"
        vals = _get_vals(evidence, [y_old, y_new])
        vo, vn = vals.get(y_old), vals.get(y_new)
        if vo is not None and vn is not None and vo != 0:
            p = round((vn - vo) / abs(vo) * 100, 2)
            return p, f"pct_change({vo:,.2f} → {vn:,.2f}) = {p:.2f}%"

    if op == "difference" and len(years) >= 2:
        y_old, y_new = years[0], years[-1]
        vo_m = vn_m = None
        for evi in evidence:
            ents = evi.get("entries", [])
            if vo_m is None:
                m = collect_monthly(ents, y_old, keywords)
                if len(m) >= 10:
                    vo_m = sum(e["numeric"] for e in m)
            if vn_m is None:
                m = collect_monthly(ents, y_new, keywords)
                if len(m) >= 10:
                    vn_m = sum(e["numeric"] for e in m)
        if vo_m is not None and vn_m is not None:
            d = vn_m - vo_m
            return d, f"difference monthly_sum({y_old}={vo_m:,.2f}, {y_new}={vn_m:,.2f}) = {d:,.2f}"
        vals = _get_vals(evidence, [y_old, y_new])
        vo, vn = vals.get(y_old), vals.get(y_new)
        if vo is not None and vn is not None:
            d = vn - vo
            return d, f"difference: {vn:,.2f} - {vo:,.2f} = {d:,.2f}"

    if op == "ratio" and len(years) >= 2:
        y_old, y_new = years[0], years[-1]
        vals = _get_vals(evidence, [y_old, y_new])
        vo, vn = vals.get(y_old), vals.get(y_new)
        if vo is not None and vn is not None and vo != 0:
            r = round(vn / vo, 4)
            return r, f"ratio: {vn:,.2f} / {vo:,.2f} = {r:.4f}"

    if op == "mean":
        year_sums: dict[int, float] = {}
        for y in years:
            for evi in evidence:
                m = collect_monthly(evi.get("entries", []), y, keywords)
                if len(m) >= 10:
                    year_sums[y] = sum(e["numeric"] for e in m)
                    break
        if len(year_sums) >= 2:
            a = round(sum(year_sums.values()) / len(year_sums), 2)
            detail = ", ".join(f"{y}={v:,.2f}" for y, v in sorted(year_sums.items()))
            return a, f"mean({detail}) = {a:,.2f}"
        vals = _get_vals(evidence, years)
        if len(vals) >= 2:
            a = round(sum(vals.values()) / len(vals), 2)
            return a, f"mean of {len(vals)} values = {a:,.2f}"

    if op == "stdev" and len(years) >= 3:
        vals_list: list[tuple[int, float]] = []
        for y in years:
            for evi in evidence:
                m = collect_monthly(evi.get("entries", []), y, keywords)
                if len(m) >= 10:
                    vals_list.append((y, sum(e["numeric"] for e in m)))
                    break
        if len(vals_list) < 3:
            vals = _get_vals(evidence, years)
            vals_list = [(y, vals[y]) for y in sorted(vals)]
        if len(vals_list) >= 3:
            numbers = [v for _, v in vals_list]
            sd = round(statistics.pstdev(numbers), 2)
            detail = ", ".join(f"{y}={v:,.2f}" for y, v in vals_list)
            return sd, f"stdev({detail}) = {sd:,.2f}"

    if op in ("max", "min") and years:
        vals = _get_vals(evidence, years)
        if len(vals) >= 2:
            pick = max if op == "max" else min
            best_yr = pick(vals, key=vals.get)  # type: ignore[arg-type]
            return vals[best_yr], f"{op.upper()} in {best_yr}: {vals[best_yr]:,.2f}"

    if op == "geometric_mean" and years:
        vals = _get_vals(evidence, years)
        numbers = [v for v in vals.values() if v > 0]
        if len(numbers) >= 2:
            gm = round(math.exp(sum(math.log(v) for v in numbers) / len(numbers)), 2)
            return gm, f"geometric_mean of {len(numbers)} values = {gm:,.2f}"

    if op == "regression" and len(years) >= 3:
        vals = _get_vals(evidence, years)
        if len(vals) >= 3:
            xs = sorted(vals.keys())
            ys_v = [vals[x] for x in xs]
            n = len(xs)
            x_mean = sum(xs) / n
            y_mean = sum(ys_v) / n
            ss_xy = sum((xs[i] - x_mean) * (ys_v[i] - y_mean) for i in range(n))
            ss_xx = sum((xs[i] - x_mean) ** 2 for i in range(n))
            if ss_xx != 0:
                slope = round(ss_xy / ss_xx, 4)
                intercept = round(y_mean - slope * x_mean, 2)
                return slope, f"regression slope={slope}, intercept={intercept}"

    return None, None


# ── CPI adjustment ────────────────────────────────────────────────────────────


def maybe_cpi_adjust(question: str, answer: float, years: list[int]) -> tuple[float, str | None]:
    q = question.lower()
    if not any(t in q for t in ("constant dollar", "real dollar", "adjusted for inflation")):
        return answer, None
    m = re.search(r"in\s+((?:19|20)\d{2})\s+dollars", q)
    if m and years:
        tgt, src = int(m.group(1)), years[0]
        if src in CPI and tgt in CPI:
            adj = round(answer * CPI[tgt] / CPI[src], 2)
            return adj, f"CPI adjust: {answer:,.2f} × {CPI[tgt]}/{CPI[src]} = {adj:,.2f}"
    return answer, None


# ── Conflict detection ────────────────────────────────────────────────────────


def detect_conflicts(evidence: list[dict], years: list[int]) -> list[str]:
    conflicts = []
    for y in years:
        srcs = [
            (ev["file"], ev["extracted_values"][y]["numeric"])
            for ev in evidence
            if y in ev.get("extracted_values", {})
            and ev["extracted_values"][y]["numeric"] is not None
        ]
        if len(srcs) < 2:
            continue
        base = abs(srcs[0][1]) or 1
        for i in range(1, len(srcs)):
            pct = abs(srcs[i][1] - srcs[0][1]) / base * 100
            if pct > 2:
                conflicts.append(
                    f"Year {y}: {srcs[0][0]}={srcs[0][1]:,.2f} vs "
                    f"{srcs[i][0]}={srcs[i][1]:,.2f} (diff={pct:.1f}%)"
                )
    return conflicts


# ── Briefing ──────────────────────────────────────────────────────────────────


def format_briefing(
    question: str,
    keywords: list[str],
    years: list[int],
    period: str,
    op: str,
    evidence: list[dict],
    answer: float | None,
    trace: str | None,
    conflicts: list[str],
    caveats: list[str],
) -> str:
    fmt_ans = (
        (str(int(answer)) if isinstance(answer, float) and answer == int(answer) else str(answer))
        if answer is not None
        else "NONE (manual review required)"
    )

    lines = [
        "=" * 60,
        f"ANSWER: {fmt_ans}",
        f"COMPUTATION: {trace}" if trace else "COMPUTATION: (none — LLM review needed)",
    ]
    if conflicts:
        lines.append(f"⚠ CONFLICT: {'; '.join(conflicts)}")
    if caveats:
        lines.append(f"CAVEATS: {'; '.join(caveats)}")
    lines += ["=" * 60, ""]
    lines.append(
        f"Question type: {op} | Period: {period} | Years: {', '.join(str(y) for y in years)}"
    )
    lines.append(FY_RULES)
    lines.append("")

    for i, ev in enumerate(evidence[:3]):
        ptag = ev["period"].upper() if ev["period"] != "unknown" else "?"
        mismatch = period != "unknown" and ev["period"] != "unknown" and ev["period"] != period
        suf = (
            " ⚠ PERIOD MISMATCH"
            if mismatch
            else (" ✓" if not mismatch and ev["period"] != "unknown" else "")
        )
        lines.append(f"Evidence {i + 1}: {ev['file']} ({ev['units']}, {ptag}{suf})")
        for y in years:
            v = ev.get("extracted_values", {}).get(y)
            if v:
                lines.append(f"  Year {y}: {v['row_label']}, {v['column']} = {v['value']}")
        # Top-15 scored entries
        entries = ev.get("entries", [])
        if entries:
            scored = sorted(
                entries,
                key=lambda e: (
                    sum(3 for y in years if str(y) in e["row_label"] or str(y) in e["column"])
                    + sum(
                        2
                        for k in keywords[:4]
                        if k in e["row_label"].lower() or k in e["column"].lower()
                    )
                    + (1 if e["numeric"] is not None else 0)
                ),
                reverse=True,
            )[:15]
            for e in scored:
                lines.append(f"  {e['row_label']}, {e['column']}: {e['value']}")
        lines.append("")

    if not evidence:
        lines += ["EVIDENCE: (none found)", ""]

    lines.append("---")
    return "\n".join(lines)


# ── Main entry point ──────────────────────────────────────────────────────────


def build_briefing(
    question: str,
    corpus_json_paths: list[str],
    extra_keywords: str | None = None,
    db_conn=None,
) -> str:
    """Build a structured research briefing from gold corpus_json files.

    Args:
        question: The OfficeQA question.
        corpus_json_paths: Absolute paths to corpus_json/*.json files.
        extra_keywords: Optional space-separated extra search terms.
        db_conn: Optional sqlite3 connection to ledger.sqlite for fast cell lookup.
                 When provided, uses the ledger instead of re-parsing HTML.

    Returns:
        Briefing string for LLM reviewer.
    """
    keywords = extract_keywords(question, extra_keywords)
    years = extract_years(question)
    period = detect_period(question)
    op = detect_operation(question)
    caveats: list[str] = []

    # Build evidence from each gold file (ledger if available, HTML fallback)
    evidence = [
        build_evidence(p, keywords, years, db_conn=db_conn)
        for p in corpus_json_paths
        if os.path.exists(p)
    ]
    evidence.sort(key=lambda e: e["relevance"], reverse=True)

    # Precompute
    answer, trace = precompute(op, evidence, keywords, years)

    # CPI adjust if needed
    if answer is not None:
        answer, cpi_note = maybe_cpi_adjust(question, answer, years)
        if cpi_note:
            caveats.append(cpi_note)

    conflicts = detect_conflicts(evidence, years)

    return format_briefing(
        question, keywords, years, period, op, evidence, answer, trace, conflicts, caveats
    )
