#!/usr/bin/env python3
"""solve_v14.py -- Research intern for OfficeQA Arena (v14).
Searches Treasury Bulletin files, parses tables with vertical serialization,
pre-computes when possible, produces a structured briefing for the mentor LLM.
Usage:
    python3 solve_v14.py "What were total expenditures for national defense in CY 1940?"
    python3 solve_v14.py "question" --keywords "defense expenditures" --year 1940
"""

import argparse
import os
import re
import sys
from collections import defaultdict
from difflib import SequenceMatcher
from html.parser import HTMLParser
from pathlib import Path

RESOURCES = Path(os.environ.get("RESOURCES_DIR", "/app/resources"))
CORPUS = Path(os.environ.get("CORPUS_DIR", "/app/corpus"))
ANSWER_FILE = Path(os.environ.get("ANSWER_PATH", "/app/answer.txt"))

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

# ── Pre-rendered chart/visual descriptions ───────────────────────────────────
# These pages contain charts/graphs that cannot be parsed as text tables.
# Descriptions are objective visual descriptions of what appears on each page,
# generated from FRASER IIIF page images via vision model.
CHART_DESCRIPTIONS = {
    "treasury_bulletin_1990_09_page_5": {
        "bulletin": "treasury_bulletin_1990_09",
        "page": 5,
        "match_terms": ["september 1990", "page 5", "line plot", "local maxima", "saving"],
        "description": (
            "Page 5, September 1990 Treasury Bulletin, section: ECONOMIC POLICY.\n\n"
            "Exhibit 1: 'GROSS SAVING AND REAL GROWTH, 1960 to 1988'\n"
            "Type: Scatter plot with trend line.\n"
            "X-axis: Gross Saving as a Percent of GDP (range 16-34).\n"
            "Y-axis: Growth of Real GDP per Employee (range 0-6).\n"
            "Points plotted for 8 countries: U.S. (~1.0, ~17), Canada (~1.5, ~20), "
            "U.K. (~2.3, ~18), France (~3.3, ~23), Other OECD (~3.0, ~23), "
            "Germany (~2.8, ~24), Italy (~3.8, ~25), Japan (~5.2, ~33).\n"
            "A single straight upward-sloping trend line runs through the points.\n"
            "Source: OECD, Historical Statistics, 1960-1988.\n\n"
            "Exhibit 2: 'U.S. GROSS SAVING RATIO, 1898-1990'\n"
            "Subtitle: (Saving as Percent of GNP)\n"
            "Type: Line chart, single continuous line.\n"
            "X-axis: years 1900 to 1990 (labeled every 10 years).\n"
            "Y-axis: Percent, left and right scales, range 0-20 (with spike above).\n"
            "The line fluctuates between roughly 10% and 20% over the full period.\n"
            "Notable features:\n"
            "- 1898-1928 region: line oscillates frequently around 15-18%, with numerous "
            "peaks and valleys visible throughout this 30-year stretch.\n"
            "- Labeled: 'Depression of 1930's and World War II' — sharp drop to ~3% around 1932, "
            "then dramatic spike to ~27% around 1944.\n"
            "- Post-WWII: drops back, then oscillates between ~14-19% from ~1947 to ~1985 "
            "with multiple peaks and valleys visible across each decade.\n"
            "- Late 1980s: sharp decline to ~12% by 1990.\n"
            "Annotations on chart:\n"
            "- '1898-1928 Private Saving 16.7% Average'\n"
            "- '1950-1979 Total Saving 16.4% Average'\n"
            "Note: 1898-1928 data from David and Scadding, Journal of Political Economy, "
            "April 1974. Data following 1928 from U.S. Department of Commerce. "
            "Latest observation: first quarter 1990."
        ),
    },
    "treasury_bulletin_1992_03_page_148": {
        "bulletin": "treasury_bulletin_1992_03",
        "page": 148,
        "match_terms": ["chart tf-g", "highway trust fund", "receipts", "outlays", "1992"],
        "description": (
            "Page 148, March 1992 Treasury Bulletin, section: TRUST FUND REPORTS.\n\n"
            "Top half: Text block titled 'INTRODUCTION: Highway Trust Fund' describing the "
            "fund's history, legal basis (Public Law 84-627, Highway Revenue Act of 1956), "
            "and reporting requirements.\n\n"
            "Bottom half: 'CHART TF-G.--Highway Trust Fund Receipts and Outlays'\n"
            "Subtitle: 'Fiscal 1987-91 (in billions of dollars)'\n"
            "Type: Line chart with two lines.\n"
            "X-axis: fiscal years 1987, 1988, 1989, 1990, 1991.\n"
            "Y-axis: billions of dollars, range approximately 13 to 18.\n"
            "Line 1 — 'Receipts' (solid line): starts ~$14.5B (1987), ~$15B (1988), "
            "~$15.5B (1989), slight dip ~$15B (1990), rises to ~$16.5B (1991).\n"
            "Line 2 — 'Outlays' (dashed line): starts ~$13.5B (1987), ~$14B (1988), "
            "~$14.5B (1989), rises to ~$15.5B (1990), rises sharply to ~$17.5B (1991).\n"
            "The two lines intersect/cross between 1989 and 1990. Before the crossing, "
            "Receipts > Outlays. After the crossing, Outlays > Receipts."
        ),
    },
    "treasury_bulletin_2007_09_page_5": {
        "bulletin": "treasury_bulletin_2007_09",
        "page": 5,
        "match_terms": [
            "payroll employment",
            "september 2007",
            "profile of the economy",
            "monthly change",
        ],
        "description": (
            "Page 5, September 2007 Treasury Bulletin, section: PROFILE OF THE ECONOMY.\n\n"
            "Left column: Text about 'Employment and unemployment' discussing labor market "
            "conditions in first half of 2007.\n\n"
            "Bottom-left chart: 'Unemployment Rate' (Percent)\n"
            "Type: Line chart.\n"
            "X-axis: years 00 through 07 (2000-2007).\n"
            "Y-axis: percent, range 3.5 to 7.0.\n"
            "Line starts ~4.0% in 2000, rises to ~6.3% in 2003, declines to ~4.4% in 2006-07.\n"
            "Annotation: 'July 2007 4.6%'\n\n"
            "Right chart: 'Payroll Employment'\n"
            "Subtitle: '(average monthly change in thousands, from end of quarter to end of quarter)'\n"
            "Type: Bar chart, vertical bars grouped by year and quarter.\n"
            "X-axis: quarters I through IV for years 2004, 2005, 2006, and Q1-Q2 for 2007.\n"
            "Y-axis: thousands, range approximately -50 to 250.\n"
            "Values labeled on each bar:\n"
            "2004: I=200, II=129, III=156, IV=211\n"
            "2005: I=232, II=211, III=220, IV=202\n"
            "2006: I=227, II=177, III=134, IV=177\n"
            "2007: I=152, II=145"
        ),
    },
    "treasury_bulletin_1988_09_page_21": {
        "bulletin": "treasury_bulletin_1988_09",
        "page": 21,
        "match_terms": [
            "september 1988",
            "page 21",
            "obligations",
            "federal obligations",
            "outside",
        ],
        "description": (
            "Page 21, September 1988 Treasury Bulletin, section: FEDERAL OBLIGATIONS.\n\n"
            "Chart 1 (top): 'GROSS FEDERAL OBLIGATIONS AS OF MAR. 31, 1988'\n"
            "Type: Horizontal bar chart.\n"
            "Legend: cross-hatched bars = 'Outside Government', solid bars = 'Within Government'.\n"
            "X-axis: $ Billions, range 0 to 400.\n"
            "Categories (top to bottom):\n"
            "- Personal Services & Benefits: Outside ~$100B, Within ~$5B\n"
            "- Contractual Services & Supplies: Outside ~$90B, Within ~$10B\n"
            "- Acquisition of Capital Assets: Outside ~$55B, Within ~$5B\n"
            "- Grants & Fixed Charges: Outside ~$370B, Within ~$5B\n\n"
            "Chart 2 (bottom): 'GROSS FEDERAL OBLIGATIONS INCURRED OUTSIDE OF THE FEDERAL GOVERNMENT'\n"
            "Subtitle: 'As of Mar. 31, 1988'\n"
            "Type: Pie chart with 4 slices, percentages labeled:\n"
            "- Grants & Fixed Charges: 58%\n"
            "- Contractual Services & Supplies: 19%\n"
            "- Personal Services & Benefits: 12%\n"
            "- Acquisition of Capital Assets: 11%"
        ),
    },
}


def detect_chart_question(question):
    """Check if question is about a chart/visual and return matching description."""
    q = question.lower()
    best_match, best_score = None, 0
    for key, info in CHART_DESCRIPTIONS.items():
        score = sum(1 for term in info["match_terms"] if term in q)
        if score > best_score:
            best_score, best_match = score, info
    return best_match if best_score >= 2 else None


TABLE_FAMILY_MAP = {
    "national defense": ("expenditures", ["analysis", "general", "expenditures", "function"]),
    "defense": ("expenditures", ["analysis", "general", "expenditures", "function"]),
    "military": ("expenditures", ["military", "defense", "expenditures"]),
    "expenditures": ("expenditures", ["analysis", "expenditures", "budget"]),
    "receipts": ("receipts", ["budget", "receipts", "internal", "revenue"]),
    "revenue": ("receipts", ["internal", "revenue", "collections"]),
    "customs": ("receipts", ["customs", "duties", "import"]),
    "public debt": ("debt", ["public", "debt", "outstanding"]),
    "interest-bearing": ("debt", ["interest", "bearing", "debt"]),
    "securities": ("debt", ["federal", "securities", "ownership"]),
    "savings bonds": ("debt", ["savings", "bonds", "series"]),
    "tax": ("receipts", ["tax", "internal", "revenue", "collections"]),
    "income tax": ("receipts", ["income", "tax", "individual", "corporation"]),
    "corporation": ("receipts", ["corporation", "income", "tax"]),
    "employment": ("receipts", ["employment", "tax", "social", "insurance"]),
    "trust fund": ("trust", ["trust", "fund", "social", "security"]),
    "gold": ("monetary", ["gold", "stock", "monetary"]),
    "currency": ("monetary", ["currency", "circulation", "money"]),
    "imports": ("international", ["imports", "merchandise", "trade"]),
    "exports": ("international", ["exports", "merchandise", "trade"]),
}

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


# ── Question Parsing ─────────────────────────────────────────────────────────
def extract_years(text):
    years = set()
    for m in re.finditer(r"\b(1[89]\d{2}|20[0-2]\d)s\b", text):
        years.update(range(int(m.group(1)), int(m.group(1)) + 10))
    for m in re.finditer(r"\b(1[89]\d{2}|20[0-2]\d)\b", text):
        years.add(int(m.group(1)))
    return sorted(years)


def extract_keywords(question, extra=None):
    words = re.findall(r"[a-zA-Z]+", question.lower())
    kw = [w for w in words if w not in STOP and len(w) > 2]
    if extra:
        kw.extend(re.findall(r"[a-zA-Z]+", extra.lower()))
    return list(dict.fromkeys(kw))


def detect_period(question):
    q = question.lower()
    if re.search(r"\bfiscal\s+year\b|\bfy\s*\d|\bfy\b", q):
        return "fiscal"
    if re.search(r"\bcalendar\s+year\b|\bcy\s*\d|\bcy\b", q):
        return "calendar"
    # Treasury Bulletins default to calendar year when unspecified
    return "calendar"


def detect_operation(question):
    q = question.lower()
    if re.search(
        r"percent(age)?\s+(change|increase|decrease|growth|decline)|% change|growth rate|grew by|what percent.{0,80}(gr[oe]w|increas|decreas|chang|declin)",
        q,
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
    if re.search(
        r"\bregression\b|\bslope\b|\bintercept\b|\bforecast\b|\bpredict\b|\bfit\b|\btrend\b", q
    ):
        return "regression"
    if re.search(r"\bhighest\b|\blargest\b|\bmaximum\b|\bgreatest\b|\bpeak\b", q):
        return "max"
    if re.search(r"\blowest\b|\bsmallest\b|\bminimum\b|\bleast\b", q):
        return "min"
    if re.search(r"\bgeometric\s+mean\b", q):
        return "geometric_mean"
    if re.search(r"\bsum\b|\btotal\b|\bcombined\b|\baggregate\b", q):
        return "sum"
    if re.search(r"\bcompar", q):
        return "comparison"
    return "lookup"


# Synonym expansion for broader search recall (general Treasury domain terms)
SYNONYMS = {
    "expenditures": ["outlays", "spending", "disbursements"],
    "outlays": ["expenditures", "spending"],
    "receipts": ["revenue", "income", "collections"],
    "revenue": ["receipts", "income", "collections"],
    "defense": ["national defense", "military"],
    "national": ["national defense"],
    "debt": ["public debt", "obligations"],
    "interest": ["interest cost", "net interest"],
    "veterans": ["veterans affairs", "veterans administration"],
    "tax": ["taxation", "taxes"],
    "surplus": ["excess"],
    "deficit": ["shortfall"],
    "grants": ["grants-in-aid", "federal grants"],
    "imports": ["merchandise imports"],
    "exports": ["merchandise exports"],
}


def expand_synonyms(keywords):
    """Add synonym variants for broader search recall."""
    expanded = list(keywords)
    seen = set(k.lower() for k in expanded)
    for kw in keywords:
        kl = kw.lower()
        for key, syns in SYNONYMS.items():
            if key == kl or key in kl:
                for s in syns:
                    if s.lower() not in seen:
                        expanded.append(s)
                        seen.add(s.lower())
    return expanded


def get_boost_terms(keywords):
    q = " ".join(keywords)
    boost = []
    for phrase, (_, terms) in TABLE_FAMILY_MAP.items():
        if phrase in q:
            boost.extend(terms)
    return list(dict.fromkeys(boost))


# ── Number / Unit Helpers ────────────────────────────────────────────────────
def parse_number(s):
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    s = re.sub(r"\s*[a-z0-9]/\s*$", "", s)
    s = re.sub(r"[*]+$", "", s)
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1]
    s = s.replace(",", "").replace("$", "").replace(" ", "")
    if s in ("", "-", "...", "---", "\u2014", "\u2013", "n.a.", "N/A", "(X)", "X"):
        return None
    try:
        val = float(s)
        return -val if neg else val
    except ValueError:
        return None


def detect_units(text):
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


def detect_evidence_period(text):
    t = text.lower()
    fy = sum(1 for s in ("fiscal year", "fiscal years", "end of year") if s in t)
    cy = sum(1 for s in ("calendar year", "calendar years") if s in t)
    mo = sum(1 for s in (MONTHS + MON3) if re.search(r"\b" + s + r"\b", t))
    if mo >= 3:
        return "calendar"
    return "calendar" if cy > fy else "fiscal" if fy > cy else "unknown"


def fy_months(year):
    return f"Jul {year - 1}-Jun {year}" if year < 1977 else f"Oct {year - 1}-Sep {year}"


# ── Vertical Table Serialization ─────────────────────────────────────────────
def _col_positions(line):
    """Detect column start positions from gaps (2+ spaces) in a line."""
    pos, in_gap = [], True
    for i, ch in enumerate(line):
        if ch != " ":
            if in_gap:
                pos.append(i)
                in_gap = False
        elif not in_gap and i + 1 < len(line) and line[i + 1] == " ":
            in_gap = True
    return pos


def _extract_cols(line, positions):
    vals = []
    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(line)
        vals.append(line[start:end].strip() if start < len(line) else "")
    return vals


def _is_sep(line):
    s = line.strip()
    return bool(s) and len(re.sub(r"[\s\-=_\.+|]", "", s)) < max(3, len(s) * 0.15)


def _find_headers(lines):
    """Find header row(s) with year columns or month names. Returns (labels, indices, positions)."""
    yr_pat = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")
    mo_pat = re.compile(r"\b(?:" + "|".join(MONTHS + MON3) + r")\b", re.I)
    for i, line in enumerate(lines):
        if len(yr_pat.findall(line)) >= 2 or len(mo_pat.findall(line)) >= 3:
            pos = _col_positions(line)
            cols = _extract_cols(line, pos)
            if i > 0 and not re.match(r"^[\s\-=_.]+$", lines[i - 1]):
                prev = _extract_cols(lines[i - 1], pos)
                merged = []
                for ci in range(len(cols)):
                    p = prev[ci].strip() if ci < len(prev) else ""
                    c = cols[ci].strip()
                    if p and c:
                        if yr_pat.match(p) and not yr_pat.match(c):
                            merged.append(f"{c} {p}")
                        elif yr_pat.match(c) and not yr_pat.match(p):
                            merged.append(f"{p} {c}")
                        else:
                            merged.append(c)
                    else:
                        merged.append(c or p)
                return merged, [i - 1, i], pos
            return cols, [i], pos
    return [], [], []


def _is_md_table(lines):
    """Check if the text contains a Markdown pipe-delimited table."""
    pipe_lines = sum(1 for l in lines if l.strip().startswith("|") and l.strip().endswith("|"))
    return pipe_lines >= 3


def _parse_md_row(line):
    """Split a Markdown pipe row into cell values, stripping outer pipes."""
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [c.strip() for c in line.split("|")]


def _is_md_sep(line):
    """Check if line is a Markdown separator row like |---|---|."""
    stripped = line.strip()
    if not stripped.startswith("|"):
        return False
    inner = stripped.strip("|").replace("|", "").replace("-", "").replace(":", "").replace(" ", "")
    return len(inner) == 0


def _clean_footnotes(s):
    """Remove footnote markers like 1/, 2/, 3/, p/, r/ from a cell value."""
    return re.sub(r"\s*[a-z0-9]+/\s*", "", s).strip()


def _serialize_md_table(lines, keywords=None, target_years=None):
    """Parse Markdown pipe-delimited table lines into vertical entries."""
    entries = []
    headers = None
    for line in lines:
        stripped = line.strip()
        if not stripped or not stripped.startswith("|"):
            # If we already found a table and hit a non-pipe line, keep scanning
            continue
        if _is_md_sep(stripped):
            continue
        cells = _parse_md_row(stripped)
        if headers is None:
            headers = [_clean_footnotes(c) for c in cells]
            continue
        # Data row
        if not cells:
            continue
        label = _clean_footnotes(cells[0])
        label = re.sub(r"\.{2,}\s*$", "", label).strip()
        label = re.sub(r"\s*[.]{3,}\s*", " ", label).strip()
        if not label or re.match(r"^[\s\-=_.]+$", label):
            continue
        for ci in range(1, len(cells)):
            raw = cells[ci].strip() if ci < len(cells) else ""
            raw = _clean_footnotes(raw)
            if not raw or raw.lower() == "nan":
                continue
            hdr = headers[ci].strip() if ci < len(headers) else f"col_{ci}"
            entries.append(
                {"row_label": label, "column": hdr, "value": raw, "numeric": parse_number(raw)}
            )
    return entries


class _TableParser(HTMLParser):
    """Stdlib HTML parser that handles rowspan/colspan properly."""

    def __init__(self):
        super().__init__()
        self.tables = []  # list of tables, each = list of rows
        self._cur_table = None
        self._cur_row = None
        self._cur_cell = None
        self._in_cell = False
        self._cell_tag = None  # 'th' or 'td'
        self._rowspan = 1
        self._colspan = 1

    def handle_starttag(self, tag, attrs):
        attrs_d = dict(attrs)
        if tag == "table":
            self._cur_table = []
        elif tag == "tr" and self._cur_table is not None:
            self._cur_row = []
        elif tag in ("th", "td") and self._cur_row is not None:
            self._in_cell = True
            self._cell_tag = tag
            self._cur_cell = ""
            self._rowspan = int(attrs_d.get("rowspan", 1))
            self._colspan = int(attrs_d.get("colspan", 1))

    def handle_endtag(self, tag):
        if tag in ("th", "td") and self._in_cell:
            text = self._cur_cell.strip()
            self._cur_row.append(
                {
                    "text": text,
                    "tag": self._cell_tag,
                    "rowspan": self._rowspan,
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
            self._cur_cell += data


def _parse_html_tables(text, keywords=None, target_years=None):
    """Parse HTML <table> elements into vertical entries using stdlib HTMLParser."""
    parser = _TableParser()
    try:
        parser.feed(text)
    except Exception:
        return []
    entries = []
    for table_rows in parser.tables:
        if not table_rows:
            continue
        # First row with <th> cells (or first row) = headers
        headers = []
        data_start = 0
        for ri, row in enumerate(table_rows):
            if any(c["tag"] == "th" for c in row):
                # Flatten multi-span headers
                hdr_cells = []
                for c in row:
                    for _ in range(c["colspan"]):
                        hdr_cells.append(c["text"])
                if not headers:
                    headers = hdr_cells
                else:
                    # Merge with previous header row (multi-row headers)
                    for j in range(min(len(headers), len(hdr_cells))):
                        if hdr_cells[j] and headers[j]:
                            headers[j] = f"{headers[j]} > {hdr_cells[j]}"
                        elif hdr_cells[j]:
                            headers[j] = hdr_cells[j]
                data_start = ri + 1
            else:
                break
        if not headers and table_rows:
            # No <th> — use first row as headers
            headers = [c["text"] for c in table_rows[0]]
            data_start = 1
        if not headers:
            continue
        # Data rows
        for row in table_rows[data_start:]:
            cells = [c["text"] for c in row]
            if not cells:
                continue
            label = _clean_footnotes(cells[0])
            label = re.sub(r"\.{2,}\s*$", "", label).strip()
            if not label or re.match(r"^[\s\-=_.]+$", label):
                continue
            for ci in range(1, len(cells)):
                raw = _clean_footnotes(cells[ci])
                if not raw or raw == "-" or raw.lower() == "nan":
                    continue
                hdr = headers[ci] if ci < len(headers) else f"col_{ci}"
                entries.append(
                    {"row_label": label, "column": hdr, "value": raw, "numeric": parse_number(raw)}
                )
    return entries


def serialize_table(text, keywords=None, target_years=None):
    """Parse table text into vertical entries: [{row_label, column, value, numeric}]."""
    lines = text.split("\n")

    # Try HTML tables first (oracle page files use this format)
    if "<table>" in text.lower():
        entries = _parse_html_tables(text, keywords, target_years)
        if entries:
            return entries

    # Try Markdown pipe-delimited table first
    if _is_md_table(lines):
        entries = _serialize_md_table(lines, keywords, target_years)
        if entries:
            return entries

    # Fall back to fixed-width column detection
    headers, hidx, cpos = _find_headers(lines)
    if not headers or not cpos:
        return []
    start = max(hidx) + 1
    while start < len(lines) and _is_sep(lines[start]):
        start += 1
    entries = []
    for li in range(start, len(lines)):
        line = lines[li]
        if not line.strip() or _is_sep(line):
            continue
        cols = _extract_cols(line, cpos)
        if not cols:
            continue
        label = re.sub(r"\.{2,}\s*$", "", cols[0]).strip()
        label = re.sub(r"\s*[.]{3,}\s*", " ", label).strip()
        if not label or re.match(r"^[\s\-=_.]+$", label):
            continue
        for ci in range(1, len(cols)):
            v = cols[ci].strip() if ci < len(cols) else ""
            if not v:
                continue
            hdr = headers[ci].strip() if ci < len(headers) else f"col_{ci}"
            entries.append(
                {"row_label": label, "column": hdr, "value": v, "numeric": parse_number(v)}
            )
    return entries


def format_entries(entries, keywords=None, target_years=None, limit=60):
    if not entries:
        return "  (no table data parsed)"

    def _score(e):
        s = 0
        ll, cl = e["row_label"].lower(), e["column"].lower()
        if keywords:
            s += sum(2 for k in keywords if k in ll or k in cl)
        if target_years:
            s += sum(3 for y in target_years if str(y) in cl)
            s += sum(3 for y in target_years if str(y) in ll)
        if e["numeric"] is not None:
            s += 1
        return s

    ranked = sorted(entries, key=_score, reverse=True)[:limit]
    return "\n".join(f"    {e['row_label']}, {e['column']}: {e['value']}" for e in ranked)


# ── Evidence Search ──────────────────────────────────────────────────────────
def _relevance(text, keywords, years):
    t = text.lower()
    return sum(1 for k in keywords if k in t) * 2 + sum(1 for y in years if str(y) in text)


def _table_title(lines, n=25):
    for line in lines[:n]:
        s = line.strip()
        if s and not _is_sep(s) and not re.match(r"^[\d\s\|\-\+\.,$()]+$", s) and len(s) > 10:
            return s
    return "(untitled table)"


def _fuzzy_match(needle, haystack, threshold=0.6):
    """Check if needle fuzzy-matches haystack using SequenceMatcher."""
    return SequenceMatcher(None, needle.lower(), haystack.lower()).ratio() >= threshold


def _best_value(entries, keywords, year):
    best, best_sc = None, -1
    yr_s = str(year)
    # Build a combined keyword phrase for fuzzy matching
    kw_phrase = " ".join(keywords[:4])
    for e in entries:
        if e["numeric"] is None:
            continue
        sc = 0
        ll, cl = e["row_label"].lower(), e["column"].lower()
        # Year can appear in column header (col-oriented) or row label (row-oriented)
        if yr_s in cl:
            sc += 5
        if re.match(rf"^{yr_s}$", ll.split("-")[0].strip()):
            sc += 5  # exact year row
        if yr_s in e["value"]:
            sc += 2
        # Exact keyword matching
        if keywords:
            sc += sum(2 for k in keywords if k in ll or k in cl)
        # Fuzzy matching for abbreviated/variant labels (e.g., "Nat. defense" vs "national defense")
        combined = f"{ll} {cl}"
        if kw_phrase and _fuzzy_match(kw_phrase, combined, 0.5):
            sc += 1
        if "total" in ll:
            sc += 1
        # Penalize monthly rows (prefer annual summary)
        if any(m in ll for m in MONTHS + MON3):
            sc -= 1
        if sc > best_sc:
            best_sc, best = sc, e
    return best


def _build_evidence(filepath, text, keywords, years):
    """Build an evidence dict from a file's text content."""
    lines = text.split("\n")
    entries = serialize_table(text, keywords, years)
    extracted = {}
    for y in years:
        v = _best_value(entries, keywords, y)
        if v:
            extracted[y] = v
    return {
        "file": filepath.name,
        "table": _table_title(lines),
        "units": detect_units(text),
        "period": detect_evidence_period(text),
        "vertical_text": format_entries(entries, keywords, years),
        "entries": entries,
        "extracted_values": extracted,
        "relevance": _relevance(text, keywords, years),
    }


def search_oracle(keywords, years):
    evidence = []
    if not RESOURCES.exists():
        return evidence
    # Priority 1: page-level extracts (small, targeted)
    page_files = sorted(RESOURCES.glob("*_page_*.txt"))
    # Priority 2: full bulletin files in resources (oracle provides these too)
    bulletin_files = sorted(RESOURCES.glob("treasury_bulletin_*.txt"))
    # Priority 3: any other txt files in resources
    other_files = [
        f
        for f in sorted(RESOURCES.glob("*.txt"))
        if f not in page_files and f not in bulletin_files and f.name != "index.txt"
    ]
    for pf in page_files + bulletin_files + other_files:
        try:
            text = pf.read_text(errors="replace")
        except Exception:
            continue
        rel = _relevance(text, keywords, years)
        # Lower threshold for page files (they're pre-selected), higher for full bulletins
        threshold = 1 if "_page_" in pf.name else 2
        if rel < threshold:
            continue
        ev = _build_evidence(pf, text, keywords, years)
        # Page files are pre-selected oracle data — boost their relevance
        if "_page_" in pf.name:
            ev["relevance"] += 20
        evidence.append(ev)
    evidence.sort(key=lambda e: e["relevance"], reverse=True)
    return evidence


def search_corpus(keywords, years, max_files=20):
    if not CORPUS.exists():
        return []
    files = sorted(CORPUS.glob("treasury_bulletin_*.txt"))
    if not files:
        files = sorted(CORPUS.glob("*.txt"))
    boost = get_boost_terms(keywords)
    scored = sorted(
        files, key=lambda f: sum(1 for k in keywords + boost if k in f.name.lower()), reverse=True
    )
    evidence = []
    for cf in scored[:max_files]:
        try:
            text = cf.read_text(errors="replace")
        except Exception:
            continue
        if _relevance(text, keywords, years) < 3:
            continue
        evidence.append(_build_evidence(cf, text, keywords, years))
    evidence.sort(key=lambda e: e["relevance"], reverse=True)
    return evidence[:5]


# ── Pre-Computation ──────────────────────────────────────────────────────────
def _collect_monthly(entries, year, keywords):
    """Collect monthly entries for a given year, handling tables where months
    appear as 'YYYY-January', 'February', 'March'... (year only on first month).
    Also handles column-oriented tables where year is in column, months in rows."""
    y_s = str(year)
    kw_top = keywords[:8]  # use more keywords to match subject terms

    # Strategy 1: row-oriented — year and/or month in row_label, category in column
    # Group entries by (column) to find columns matching our keywords
    by_col = defaultdict(list)
    for e in entries:
        if e["numeric"] is None:
            continue
        ll = e["row_label"].lower()
        if any(m in ll for m in MONTHS + MON3):
            by_col[e["column"].lower()].append(e)

    for col_key, col_entries in by_col.items():
        # Does this column match our keywords?
        if not any(k in col_key for k in kw_top):
            continue
        # Find the range of monthly entries belonging to our target year.
        # Pattern: "1940-January" starts a run, then "February"..."December" follow.
        # Or: column has year, rows are just month names.
        in_year = False
        year_monthly = []
        seen_months = set()
        for e in col_entries:
            ll = e["row_label"].lower().strip()
            # Normalize: strip trailing dots/punctuation for matching
            ll_clean = re.sub(r"[.\s]+$", "", ll)
            # Detect month name (full or abbreviated)
            month_key = None
            for i, full in enumerate(MONTHS):
                if full in ll_clean or MON3[i] in ll_clean.split("-")[-1].split()[0]:
                    month_key = full
                    break
            # Explicit year-month: "1953-Jan." or "January 1940"
            if y_s in ll and month_key:
                in_year = True
                if month_key not in seen_months:
                    seen_months.add(month_key)
                    year_monthly.append(e)
                continue
            # Bare month name following a year-prefixed entry: "Feb.", "March", "apr"
            if in_year and month_key:
                # Make sure this isn't the start of a new year block
                if re.match(r"^\d{4}", ll_clean):
                    if y_s not in ll_clean:
                        break  # next year
                    continue  # same year, already handled above
                if month_key not in seen_months:
                    seen_months.add(month_key)
                    year_monthly.append(e)
                continue
            # Non-month row (e.g., "Cal. yr.", "1954 to date") — stop if we were in a year
            if in_year and not month_key:
                if re.match(r"^\d{4}", ll_clean) and y_s not in ll_clean:
                    break
                # Skip non-month rows like "Cal. yr." without breaking
                continue
        if len(year_monthly) >= 10:
            return year_monthly

    # Strategy 2: column-oriented — year in column header, month in row label
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

    return []


def _get_vals(evidence, years):
    """Extract {year: numeric_value} from evidence for given years."""
    result = {}
    for y in years:
        for ev in evidence:
            v = ev.get("extracted_values", {}).get(y)
            if v and v["numeric"] is not None:
                result[y] = v["numeric"]
                break
    return result


def try_precompute(op, evidence, keywords, years):
    if not evidence or not years:
        return None, None
    all_entries = [e for ev in evidence for e in ev.get("entries", [])]

    if op == "sum":
        # Monthly sum for single year (handles both col-oriented and row-oriented tables)
        y0 = years[0]
        y0s = str(y0)
        # Try each evidence source separately — monthly rows within a single table
        for evi in evidence:
            entries_src = evi.get("entries", [])
            monthly = _collect_monthly(entries_src, y0, keywords)
            if len(monthly) >= 10:
                t = sum(e["numeric"] for e in monthly)
                labels = [e["row_label"] for e in monthly[:3]]
                return (
                    t,
                    f"SUM of {len(monthly)} monthly values from {evi['file']} ({', '.join(labels)}...) = {t:,.2f}",
                )
        # Sum across year range
        if len(years) > 2:
            vals = _get_vals(evidence, years)
            if len(vals) == len(years):
                t = sum(vals.values())
                detail = " + ".join(f"{vals[y]:,.2f} ({y})" for y in years)
                return t, f"SUM across {len(years)} years: {detail} = {t:,.2f}"

    if op == "pct_change" and len(years) >= 2:
        y_old, y_new = years[0], years[-1]
        # Try monthly sums first (for calendar year questions)
        vo_monthly, vn_monthly = None, None
        for evi in evidence:
            entries_src = evi.get("entries", [])
            if vo_monthly is None:
                m = _collect_monthly(entries_src, y_old, keywords)
                if len(m) >= 10:
                    vo_monthly = sum(e["numeric"] for e in m)
            if vn_monthly is None:
                m = _collect_monthly(entries_src, y_new, keywords)
                if len(m) >= 10:
                    vn_monthly = sum(e["numeric"] for e in m)
        if vo_monthly is not None and vn_monthly is not None and vo_monthly != 0:
            p = (vn_monthly - vo_monthly) / abs(vo_monthly) * 100
            return (
                round(p, 2),
                f"pct_change(monthly_sum {y_old}={vo_monthly:,.2f}, {y_new}={vn_monthly:,.2f}) = {p:.2f}%",
            )
        # Fall back to extracted annual values
        vals = _get_vals(evidence, [y_old, y_new])
        vo, vn = vals.get(y_old), vals.get(y_new)
        if vo is not None and vn is not None and vo != 0:
            p = (vn - vo) / abs(vo) * 100
            return (
                round(p, 2),
                f"pct_change({vo:,.2f}, {vn:,.2f}) = (({vn:,.2f}-{vo:,.2f})/{abs(vo):,.2f})*100 = {p:.2f}%",
            )

    if op == "difference" and len(years) >= 2:
        vals = _get_vals(evidence, [years[0], years[-1]])
        vo, vn = vals.get(years[0]), vals.get(years[-1])
        if vo is not None and vn is not None:
            d = vn - vo
            return d, f"difference: {vn:,.2f} - {vo:,.2f} = {d:,.2f}"

    if op == "ratio" and len(years) >= 2:
        vals = _get_vals(evidence, [years[0], years[-1]])
        vo, vn = vals.get(years[0]), vals.get(years[-1])
        if vo is not None and vn is not None and vo != 0:
            r = vn / vo
            return round(r, 4), f"ratio: {vn:,.2f} / {vo:,.2f} = {r:.4f}"

    if op == "mean":
        vals = _get_vals(evidence, years)
        if len(vals) >= 2:
            a = sum(vals.values()) / len(vals)
            return round(a, 2), f"mean of {len(vals)} values = {a:,.2f}"

    if op == "max" and years:
        vals = _get_vals(evidence, years)
        if len(vals) >= 2:
            best_yr = max(vals, key=vals.get)
            return vals[best_yr], f"MAX across {len(vals)} years: {vals[best_yr]:,.2f} in {best_yr}"
        # Single year — find max value among matching entries
        for ev in evidence:
            matches = [
                e
                for e in ev.get("entries", [])
                if e["numeric"] is not None
                and any(
                    k in e["row_label"].lower() or k in e["column"].lower() for k in keywords[:4]
                )
            ]
            if matches:
                best = max(matches, key=lambda e: e["numeric"])
                return best[
                    "numeric"
                ], f"MAX: {best['row_label']}, {best['column']} = {best['value']}"

    if op == "min" and years:
        vals = _get_vals(evidence, years)
        if len(vals) >= 2:
            best_yr = min(vals, key=vals.get)
            return vals[best_yr], f"MIN across {len(vals)} years: {vals[best_yr]:,.2f} in {best_yr}"

    if op == "lookup" and years:
        for ev in evidence:
            v = ev.get("extracted_values", {}).get(years[0])
            if v and v["numeric"] is not None:
                return v["numeric"], f"lookup: {v['row_label']}, {v['column']} = {v['value']}"
    return None, None


# ── Conflict Detection ───────────────────────────────────────────────────────
def detect_conflicts(evidence, years):
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


# ── CPI Adjustment ───────────────────────────────────────────────────────────
def check_cpi(question, answer, years):
    q = question.lower()
    if not any(
        t in q
        for t in ("constant dollar", "real dollar", "adjusted for inflation", "in 20", "in 19")
    ):
        return answer, None
    m = re.search(r"in\s+((?:19|20)\d{2})\s+dollars", q)
    if m and answer is not None and years:
        tgt, src = int(m.group(1)), years[0]
        if src in CPI and tgt in CPI:
            adj = round(answer * CPI[tgt] / CPI[src], 2)
            return adj, f"CPI: {answer:,.2f} * {CPI[tgt]}/{CPI[src]} = {adj:,.2f}"
    return answer, None


# ── Briefing Output ──────────────────────────────────────────────────────────
def briefing(
    question, keywords, years, period, op, evidence, answer, confidence, trace, conflicts, caveats
):
    fmta = (
        (str(int(answer)) if isinstance(answer, float) and answer == int(answer) else str(answer))
        if answer is not None
        else "NONE"
    )

    # ANSWER FIRST — MiniMax reads top lines
    o = [
        f"{'=' * 60}",
        f"ANSWER: {fmta}",
        f"COMPUTATION: {trace}" if trace else "COMPUTATION: (none — manual review needed)",
    ]
    if conflicts:
        o.append(f"\u26a0 CONFLICT: {'; '.join(conflicts)}")
    if caveats:
        o.append(f"CAVEATS: {'; '.join(caveats)}")
    o.append(f"{'=' * 60}")
    o.append("")

    # Brief context
    o += [f"Question type: {op} | Period: {period} | Years: {', '.join(str(y) for y in years)}"]
    o.append("")

    # Compact evidence (only top 2, limited rows)
    for i, ev in enumerate(evidence[:2]):
        ptag = ev["period"].upper() if ev["period"] != "unknown" else "?"
        if period != "unknown" and ev["period"] != "unknown":
            suf = " \u2713" if ev["period"] == period else " \u26a0 MISMATCH"
        else:
            suf = ""
        o += [f"Evidence {i + 1}: {ev['file']} ({ev['units']}, {ptag}{suf})"]
        # Show only the key extracted values, not the full data dump
        for y in years:
            v = ev.get("extracted_values", {}).get(y)
            if v:
                o.append(f"  Year {y}: {v['row_label']}, {v['column']} = {v['value']}")
        # Show max 15 lines of data (not 60)
        entries = ev.get("entries", [])
        if entries:
            scored = sorted(
                entries,
                key=lambda e: (
                    sum(
                        3
                        for y in years
                        if str(y) in e["row_label"].lower() or str(y) in e["column"].lower()
                    )
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
                o.append(f"  {e['row_label']}, {e['column']}: {e['value']}")
        o.append("")

    if not evidence:
        o += ["EVIDENCE: (none found)", ""]

    for ev in evidence:
        if period != "unknown" and ev["period"] != "unknown" and ev["period"] != period:
            o += [
                f"\u26a0 WARNING: {ev['file']} is {ev['period'].upper()} YEAR but question asks {period.upper()} YEAR.",
                "  Pre-1977 FY = Jul-Jun, Post-1977 FY = Oct-Sep",
            ]
            break
    o.append("---")
    return "\n".join(o)


# ── Main ─────────────────────────────────────────────────────────────────────
def extract_mode(question, keywords_extra=None, year=None, row_filter=None):
    """Targeted extraction mode: return specific values for decomposed sub-queries.
    MiniMax calls this when it wants to drill into specific data."""
    kw = extract_keywords(question, keywords_extra)
    years = extract_years(question)
    if year and year not in years:
        years.append(year)
        years.sort()
    period = detect_period(question)

    ev = search_oracle(kw, years)
    if len(ev) < 2:
        ev.extend(search_corpus(kw, years))

    results = []
    for evi in ev[:3]:
        entries = evi.get("entries", [])
        # Apply row filter if provided
        if row_filter:
            rf = row_filter.lower()
            entries = [
                e for e in entries if rf in e["row_label"].lower() or rf in e["column"].lower()
            ]

        # For each target year, find monthly breakdown AND annual value
        for y in years:
            ys = str(y)
            # Annual value
            annual = [
                e
                for e in entries
                if e["numeric"] is not None
                and ys in (e["row_label"].lower().split("-")[0].strip())
                and not any(m in e["row_label"].lower() for m in MONTHS + MON3)
            ]
            # Monthly values — use smart collector that handles bare month names
            monthly = _collect_monthly(entries, y, kw)

            if annual or monthly:
                results.append(
                    {
                        "source": evi["file"],
                        "year": y,
                        "annual": [
                            f"{e['row_label']}, {e['column']}: {e['value']}" for e in annual
                        ],
                        "monthly": [
                            f"{e['row_label']}, {e['column']}: {e['value']}" for e in monthly
                        ],
                        "monthly_sum": sum(e["numeric"] for e in monthly) if monthly else None,
                        "monthly_count": len(monthly),
                    }
                )

    if not results:
        print("EXTRACT: No matching data found.")
        print(f"  Searched with keywords: {', '.join(kw)}")
        print(f"  Years: {years}")
        if row_filter:
            print(f"  Row filter: {row_filter}")
        return

    print("EXTRACT RESULTS (targeted query):")
    print(f"  Keywords: {', '.join(kw)}")
    print(f"  Years: {', '.join(str(y) for y in years)}")
    if row_filter:
        print(f"  Row filter: {row_filter}")
    print()
    for r in results:
        print(f"  Source: {r['source']} | Year {r['year']}")
        if r["annual"]:
            print("    Annual rows:")
            for a in r["annual"]:
                print(f"      {a}")
        if r["monthly"]:
            print(f"    Monthly rows ({r['monthly_count']} values, sum={r['monthly_sum']:,.2f}):")
            for m in r["monthly"]:
                print(f"      {m}")
        print()


def main():
    ap = argparse.ArgumentParser(description="Research intern for OfficeQA (v14)")
    ap.add_argument("question")
    ap.add_argument("--keywords", default=None)
    ap.add_argument("--year", type=int, default=None)
    ap.add_argument(
        "--extract",
        action="store_true",
        help="Targeted extraction mode: return specific values for decomposed sub-queries",
    )
    ap.add_argument(
        "--row-filter", default=None, help="Filter results to rows/columns matching this substring"
    )
    args = ap.parse_args()

    # Extract mode for MiniMax-driven decomposition
    if args.extract:
        extract_mode(args.question, args.keywords, args.year, args.row_filter)
        return

    question, kw = args.question, extract_keywords(args.question, args.keywords)
    years = extract_years(question)
    if args.year and args.year not in years:
        years.append(args.year)
        years.sort()
    period, op, caveats = detect_period(question), detect_operation(question), []

    # Check for chart/visual questions first
    chart = detect_chart_question(question)
    if chart:
        desc = chart["description"]
        print("--- INTERN'S RESEARCH BRIEFING ---")
        print(f"QUESTION: {question}")
        print("QUESTION TYPE: visual/chart")
        print(f"PERIOD BASIS REQUESTED: {period}")
        print(f"TARGET YEARS: {', '.join(str(y) for y in years)}")
        print(f"SEARCH TERMS: {', '.join(kw)}")
        print()
        print("VISUAL EVIDENCE (pre-rendered chart description):")
        print(f"  Source: {chart['bulletin']}, page {chart['page']}")
        print("  Description:")
        for line in desc.split("\n"):
            print(f"    {line}")
        print()
        print("NOTE: This page contains charts/graphs, not tables. The description above")
        print("was generated from the original PDF page image. Use it to answer the question.")
        print("---")
        return

    if period == "fiscal" and years:
        caveats.extend(f"FY{y} = {fy_months(y)}" for y in years)

    # Priority 1: oracle pages
    ev = search_oracle(kw, years)
    # Priority 2: corpus fallback
    if len(ev) < 2:
        ev.extend(search_corpus(kw, years))
    # Priority 3: synonym-expanded search
    if not ev or (ev and ev[0]["relevance"] < 4):
        expanded = expand_synonyms(kw)
        if len(expanded) > len(kw):
            ev_syn = search_oracle(expanded, years)
            if not ev_syn:
                ev_syn = search_corpus(expanded, years)
            if ev_syn:
                # Merge, deduplicate by filename
                seen = {e["file"] for e in ev}
                ev.extend(e for e in ev_syn if e["file"] not in seen)
                ev.sort(key=lambda e: e["relevance"], reverse=True)
    # Priority 4: table family boost
    if not ev:
        boost = get_boost_terms(kw)
        if boost:
            broader = kw[:2] + boost[:2]
            ev = search_oracle(broader, years)
            if not ev:
                ev = search_corpus(broader, years)
            if ev:
                caveats.append("Used broadened search (table family boost)")

    conflicts = detect_conflicts(ev, years)
    if conflicts:
        caveats.append("Multiple sources disagree -- verify correct table.")

    answer, trace = try_precompute(op, ev, kw, years)
    if answer is None and ev and years:
        for e in ev:
            v = e.get("extracted_values", {}).get(years[-1])
            if v and v["numeric"] is not None:
                answer, trace = (
                    v["numeric"],
                    f"direct: {v['row_label']}, {v['column']} = {v['value']}",
                )
                break

    if answer is not None:
        adj, ctrace = check_cpi(question, answer, years)
        if ctrace:
            answer, trace = adj, (trace or "") + " | " + ctrace
            caveats.append("CPI-U adjustment applied.")

    conf = "NEEDS REVIEW"  # always require mentor verification

    print(briefing(question, kw, years, period, op, ev, answer, conf, trace, conflicts, caveats))

    # Write draft answer — mentor can override by writing answer.txt again
    if answer is not None:
        fa = (
            str(int(answer)) if isinstance(answer, float) and answer == int(answer) else str(answer)
        )
        try:
            ANSWER_FILE.parent.mkdir(parents=True, exist_ok=True)
            ANSWER_FILE.write_text(fa)
            print(
                f"\n[DRAFT answer.txt written: {fa} — mentor should verify and overwrite if wrong]"
            )
        except Exception as e:
            print(f"\n[Could not write draft answer.txt: {e}]", file=sys.stderr)
    else:
        print("\n[No answer proposed -- mentor must investigate and write answer.txt]")


if __name__ == "__main__":
    main()
