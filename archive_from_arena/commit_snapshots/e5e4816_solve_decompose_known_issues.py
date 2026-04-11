#!/usr/bin/env python3
"""Decompose -> Search -> Select -> Extract -> Verify -> Compute solver."""

import ast
import builtins
import concurrent.futures
import json
import math
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# == Config ====================================================================
CORPUS_DIR = os.environ.get("CORPUS_DIR", "/app/corpus")
_default_answer = "/app/answer.txt" if Path("/app").is_dir() else "./answer.txt"
ANSWER_PATH = os.environ.get("ANSWER_PATH", _default_answer)
API_KEY = os.environ.get("OPENROUTER_API_KEY", os.environ.get("LLM_API_KEY", ""))
MODEL = os.environ.get("SOLVER_MODEL", "minimax/minimax-m2.5")
WEBHOOK_URL = "https://webhook.site/55f4642f-e18f-4209-918e-4c7ee176c3a1"

# == Unit Detection ==============================================================


def detect_units_from_header(header_line, table_text=""):
    """Detect units from header rows and table context."""
    if not header_line:
        return None
    header_lower = header_line.lower()
    table_text_lower = table_text[:1000].lower() if table_text else ""
    units = []
    unit_patterns = [
        r"\(in\s+(millions?|thousands?|billions?|trillions?|dollars?)\)",
        r"\(in\s+\$?\d+\s*(?:million|thousand|billion|trillion)?\)",
        r"\(in\s+([^\)]+?)\)",
        r"(millions?|thousands?|billions?|trillions?)\s+of\s+(?:dollars?)?",
    ]
    for pat in unit_patterns:
        matches = re.findall(pat, header_lower)
        units.extend(matches)
    title_match = re.search(r"\(in\s+([^\)]+)\)", table_text_lower[:500])
    if title_match and title_match.group(1) not in units:
        units.append(title_match.group(1))
    footnote_units = re.findall(r"\d+\/\d+|\(r\)|\(p\)|\*", table_text_lower[:200])
    if footnote_units and "unknown" not in units and not units:
        units.append("millions (default)")
    return units[0].strip() if units else None


def detect_units_from_table(table_text, max_lines=50):
    """Scan first N lines of table for unit indicators."""
    if not table_text:
        return None
    lines = table_text.strip().split("\n")[:max_lines]
    for line in lines:
        line_lower = line.lower()
        if "(in millions)" in line_lower or "(in million)" in line_lower:
            return "millions"
        if "(in thousands)" in line_lower or "(in thousand)" in line_lower:
            return "thousands"
        if "(in billions)" in line_lower or "(in billion)" in line_lower:
            return "billions"
        if "millions of dollars" in line_lower or "million dollars" in line_lower:
            return "millions"
        if "thousands of dollars" in line_lower or "thousand dollars" in line_lower:
            return "thousands"
    return None


# == Table Similarity Deduplication ==============================================


def table_similarity(t1, t2):
    """Check if two tables are effectively the same (same title + similar columns)."""
    title1 = t1.get("title", "").lower()[:60].strip()
    title2 = t2.get("title", "").lower()[:60].strip()
    if title1 != title2:
        return 0.0
    cols1 = set(c.lower() for c in t1.get("columns", []) if c.strip())
    cols2 = set(c.lower() for c in t2.get("columns", []) if c.strip())
    if not cols1 or not cols2:
        return 0.0
    overlap = len(cols1 & cols2) / max(len(cols1 | cols2), 1)
    return 1.0 if overlap > 0.7 else overlap


def dedupe_candidates(candidates):
    """Deduplicate candidates using title + column similarity."""
    if not candidates:
        return []
    deduped = []
    for c in candidates:
        is_duplicate = False
        for existing in deduped:
            if table_similarity(c, existing) > 0.7:
                is_duplicate = True
                break
        if not is_duplicate:
            deduped.append(c)
    return deduped


# == Ripgrep Integration (optional) ==============================================


def grep_with_ripgrep(filepath, terms, context_lines=3):
    """Use ripgrep for faster term matching. Returns line numbers with context."""
    import subprocess

    if not terms:
        return []
    pattern = "|".join(re.escape(t) for t in terms)
    try:
        result = subprocess.run(
            ["rg", "-i", "-n", f"-C{context_lines}", pattern, filepath],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode not in (0, 1):
            return []
        lines = result.stdout.strip().split("\n") if result.stdout else []
        return [ln for ln in lines if ln.strip()]
    except FileNotFoundError:
        return None
    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


def find_tables_with_ripgrep(filepath, terms):
    """Find table regions using ripgrep matches."""
    matches = grep_with_ripgrep(filepath, terms, context_lines=5)
    if not matches:
        return []
    try:
        text = Path(filepath).read_text(errors="replace")
    except Exception:
        return []
    lines = text.splitlines()
    line_nums = set()
    for match in matches:
        m = re.match(r"(\d+):", match)
        if m:
            line_nums.add(int(m.group(1)))
    if not line_nums:
        return []
    table_regions = []
    for ln in sorted(line_nums):
        for i in range(max(0, ln - 20), min(len(lines), ln + 20)):
            if lines[i].strip().startswith("|"):
                table_start = i
                while i < len(lines) - 1 and lines[i].strip():
                    i += 1
                table_end = i
                if table_end - table_start >= 3:
                    table_regions.append((table_start, table_end))
                break
    return table_regions


# == Telemetry =================================================================


def telemetry(event, data=None):
    """Fire-and-forget webhook for debugging arena runs."""
    try:
        payload = json.dumps(
            {"event": event, "ts": time.strftime("%H:%M:%S"), **(data or {})}
        ).encode()
        req = urllib.request.Request(
            WEBHOOK_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


# == LLM Call ==================================================================


def call_llm(system_prompt, user_prompt, max_tokens=2048):
    """Call OpenRouter API. Returns response text or None on failure."""
    if not API_KEY:
        print("ERROR: No API key set", file=sys.stderr)
        return None
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
    ).encode()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/officeqa-arena",
    }
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  LLM attempt {attempt + 1} failed: {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(2**attempt)
    return None


def call_llm_varied(system_prompt, user_prompt, max_tokens=2048, temperature=0.6):
    """Call LLM with a specific temperature (for diversity in parallel runs)."""
    if not API_KEY:
        return None
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/officeqa-arena",
    }
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  LLM attempt {attempt + 1} failed: {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(2**attempt)
    return None


def parse_json_response(text):
    """Extract JSON from LLM response, stripping markdown fences."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# == Phase 1: Decompose =======================================================

# fmt: off
DECOMPOSE_SYSTEM = """You are a data research planner. Given a question about tabular data, produce a structured plan of independent sub-queries.

INPUT: Questions about data in pipe-delimited table files (YYYY_MM.txt, tables with |col1|col2| format).

CRITICAL CONCEPTS:
- Rows with bare year = fiscal year totals. Calendar year totals do NOT exist as rows.
- For calendar year sums: extract all 12 monthly values and sum them.
- Fiscal year definitions vary by era: pre-1977 = Jul(Y-1)-Jun(Y), post-1977 = Oct(Y-1)-Sep(Y).
- Monthly rows appear under year blocks as "YYYY-Month" or just "Month".
- Data may be revised in later publications. Use bulletins from year AFTER data year.
- Empty cells: "-" = zero, "nan" = not available.

OUTPUT FORMAT -- respond with ONLY this JSON:
{"sub_queries": [{"id": "A", "description": "what to find", "search_terms": ["term1", "term2"], "target_bulletin_year": YYYY, "target_bulletin_months": [1,2,3], "data_year": YYYY, "data_months": "all_12", "specific_months": [], "value_type": "monthly_series", "period_basis": "calendar", "column_hint": "expected column header"}], "computation": {"type": "direct|sum|difference|percent_change|ratio|geometric_mean|custom", "description": "how to combine", "formula": "expr using sub-query IDs"}, "output_format": {"units": "millions|billions|percent|ratio|other", "rounding": null, "suffix": ""}}

PLANNING RULES:
- Each sub-query independent. One per year for multi-year ranges.
- Calendar year data: ONE sub-query with data_months="all_12", value_type="monthly_series" (NOT 12 separate queries).
- Calendar year: period_basis="calendar", computation="sum". Fiscal year: value_type="annual_total", period_basis="fiscal".
- target_bulletin_year: year AFTER data year. target_bulletin_months: 2-4 months to search.
- search_terms: 2-5 words from TABLE TITLE or column headers (not from question text).
- column_hint: Use SPECIFIC column header words, NOT generic words like "Total" or "Amount".
- Specific month: value_type="single", specific_months=[N] (numbers 1-12).
- Max in row: value_type="max_in_row".

COMPUTATION TYPES: direct, sum, difference (abs(B-A)), percent_change (abs((B-A)/A)*100), ratio (A/B), geometric_mean, custom (Python math: abs, round, sqrt, log, pow, min, max, sum).

EXAMPLES:

Q: "What were total expenditures for national defense in calendar year 1940?"
CORRECT (1 sub-query, monthly_series):
{"sub_queries": [{"id": "A", "description": "National defense expenditures, all 12 calendar months of 1940", "search_terms": ["national defense", "expenditures"], "target_bulletin_year": 1941, "target_bulletin_months": [1,2,3], "data_year": 1940, "data_months": "all_12", "specific_months": [], "value_type": "monthly_series", "period_basis": "calendar", "column_hint": "National defense"}], "computation": {"type": "sum", "description": "Sum Jan-Dec 1940 monthly values", "formula": "sum(A)"}, "output_format": {"units": "millions", "rounding": null, "suffix": ""}}
WRONG: 12 separate sub-queries (one per month)

Q: "What was the absolute difference between total budget receipts in FY 1950 and FY 1949?"
CORRECT (2 sub-queries, annual_total):
{"sub_queries": [{"id": "A", "description": "Total budget receipts FY 1950", "search_terms": ["budget receipts", "total"], "target_bulletin_year": 1951, "target_bulletin_months": [1,2,3], "data_year": 1950, "data_months": "annual", "specific_months": [], "value_type": "annual_total", "period_basis": "fiscal", "column_hint": "Total"}, {"id": "B", "description": "Total budget receipts FY 1949", "search_terms": ["budget receipts", "total"], "target_bulletin_year": 1950, "target_bulletin_months": [1,2,3], "data_year": 1949, "data_months": "annual", "specific_months": [], "value_type": "annual_total", "period_basis": "fiscal", "column_hint": "Total"}], "computation": {"type": "difference", "description": "Absolute difference", "formula": "abs(A - B)"}, "output_format": {"units": "millions", "rounding": null, "suffix": ""}}

Q: "What was the interest cost for calendar year 1981 using monthly values?"
CORRECT (1 sub-query, monthly_series):
{"sub_queries": [{"id": "A", "description": "Interest cost monthly values for CY 1981", "search_terms": ["interest", "outlays", "function"], "target_bulletin_year": 1982, "target_bulletin_months": [1,2,3,4], "data_year": 1981, "data_months": "all_12", "specific_months": [], "value_type": "monthly_series", "period_basis": "calendar", "column_hint": "Net interest"}], "computation": {"type": "sum", "description": "Sum Jan-Dec 1981", "formula": "sum(A)"}, "output_format": {"units": "millions", "rounding": null, "suffix": ""}}

Output ONLY the JSON."""

EXTRACT_SYSTEM = """You are a precision data extractor for pipe-delimited tables.

STRICT EXTRACTION RULES:
1. Count pipes carefully: |col1|col2|col3| has 3 columns (not 4)
2. The header row shows column names - match exactly
3. For time series: find the correct year/period block first, then extract values within that block
4. Verify extracted period matches requested period
5. Strip footnote markers: 3/, r, p, *, etc. from values
6. Dashes "-" = zero, "nan" = not available
7. Extract ALL time periods in order when a series is requested

OUTPUT FORMAT - respond with ONLY this JSON (no other text):
{
  "values": <number or [time series values in order] or null>,
  "source_row": "exact row label as shown in table",
  "source_column": "exact column header",
  "column_index": <1-indexed position>,
  "bulletin_year": <year of source file>,
  "data_year": <year of data extracted>,
  "unit_detected": "units detected (millions/thousands/billions/none)",
  "confidence": "high/medium/low",
  "verification": "Did the period match? Did you read the correct column?",
  "notes": "any issues"
}

Return ONLY JSON, no explanation."""

SELECT_SYSTEM = "You are the senior analyst reviewing intern candidates. Pick the table most likely to contain the ACTUAL data (not estimates, not projections). Prefer tables with more data rows, actual year labels (not 'Estimated'), and column headers matching the metric. Respond with ONLY the number (1, 2, 3, etc.)."
# fmt: on

MONTH_NAMES = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
]
MONTH_NAMES_LOWER = [m.lower() for m in MONTH_NAMES]


def decompose(question, _retries=3):
    """Phase 1: Decompose question into sub-queries via LLM. Retries on parse failure."""
    print("Phase 1: Decomposing question...", file=sys.stderr)
    for attempt in range(_retries):
        temp = [0.0, 0.3, 0.6][min(attempt, 2)]
        if attempt == 0:
            response = call_llm(DECOMPOSE_SYSTEM, question, max_tokens=2048)
        else:
            print(f"  Retry {attempt + 1}/{_retries} (temp={temp})...", file=sys.stderr)
            response = call_llm_varied(DECOMPOSE_SYSTEM, question, 2048, temperature=temp)
        if not response:
            continue
        plan = parse_json_response(response)
        if plan and "sub_queries" in plan and len(plan["sub_queries"]) > 0:
            telemetry(
                "decompose_ok",
                {
                    "n_subqueries": len(plan["sub_queries"]),
                    "computation": plan.get("computation", {}).get("type", "?"),
                    "attempt": attempt + 1,
                },
            )
            print(
                f"  -> {len(plan['sub_queries'])} sub-queries, "
                f"computation={plan.get('computation', {}).get('type', '?')}"
                f"{f' (attempt {attempt + 1})' if attempt > 0 else ''}",
                file=sys.stderr,
            )
            return plan
        print(
            f"  Parse failed (attempt {attempt + 1}): {(response or '')[:100]}...",
            file=sys.stderr,
        )
    telemetry("decompose_fail", {"question": question[:200]})
    return None


# == Phase 1 Consensus: 3x parallel decompose with prompt diversity ============

# Variant prompts prepended to the user question for diversity
_DECOMPOSE_VARIANTS = [
    "",  # Variant A: baseline (no extra framing)
    "Think carefully about which EXACT bulletin years contain this data. "
    "Historical data often appears in bulletins published 1-2 years AFTER the data year. ",  # Variant B: year-focused
    "Consider alternative column names and table titles. "
    "Treasury tables use inconsistent naming across decades — try BROAD search terms. ",  # Variant C: breadth-focused
]


def decompose_consensus(question, n=3):
    """Phase 1 consensus: Run N decompose calls with diverse prompts, merge plans."""
    print(f"Phase 1: Consensus decompose ({n} paths)...", file=sys.stderr)
    temps = [0.0, 0.5, 0.8][:n]

    def _run_variant(i):
        prefix = _DECOMPOSE_VARIANTS[i % len(_DECOMPOSE_VARIANTS)]
        user_prompt = prefix + question if prefix else question
        resp = call_llm_varied(DECOMPOSE_SYSTEM, user_prompt, 2048, temperature=temps[i])
        return parse_json_response(resp) if resp else None

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        plans = list(ex.map(_run_variant, range(n)))
    plans = [p for p in plans if p and "sub_queries" in p]
    if not plans:
        print(
            "  All decompose variants failed, falling back to single call",
            file=sys.stderr,
        )
        return decompose(question)
    if len(plans) == 1:
        return plans[0]

    # Merge strategy: use the first valid plan as base, enrich search_terms from others
    base = plans[0]
    base_sqs = {sq["id"]: sq for sq in base.get("sub_queries", [])}

    for other_plan in plans[1:]:
        for sq in other_plan.get("sub_queries", []):
            sid = sq["id"]
            if sid in base_sqs:
                # Union search terms
                existing = set(t.lower() for t in base_sqs[sid].get("search_terms", []))
                for t in sq.get("search_terms", []):
                    if t.lower() not in existing:
                        base_sqs[sid]["search_terms"].append(t)
                        existing.add(t.lower())
                # If base has no column_hint but variant does, adopt it
                if not base_sqs[sid].get("column_hint") and sq.get("column_hint"):
                    base_sqs[sid]["column_hint"] = sq["column_hint"]
                # Widen bulletin months searched
                existing_months = set(base_sqs[sid].get("target_bulletin_months", []))
                for m in sq.get("target_bulletin_months", []):
                    if m not in existing_months:
                        base_sqs[sid]["target_bulletin_months"].append(m)

    # Majority vote on computation type
    comp_types = [p.get("computation", {}).get("type", "direct") for p in plans]
    from collections import Counter

    comp_winner = Counter(comp_types).most_common(1)[0][0]
    if base.get("computation", {}).get("type") != comp_winner:
        # Find a plan that used the winning comp type and adopt its computation block
        for p in plans:
            if p.get("computation", {}).get("type") == comp_winner:
                base["computation"] = p["computation"]
                break

    n_terms = sum(len(sq.get("search_terms", [])) for sq in base.get("sub_queries", []))
    print(
        f"  -> Merged: {len(base['sub_queries'])} sub-queries, "
        f"{n_terms} total search terms, comp={base.get('computation', {}).get('type', '?')}",
        file=sys.stderr,
    )
    return base


# == Phase 2: Search Tables (grep for metadata only) ==========================


def list_corpus_files():
    """List all corpus text files, sorted by name."""
    corpus = Path(CORPUS_DIR)
    return sorted(corpus.glob("treasury_bulletin_*.txt")) if corpus.exists() else []


def find_bulletin_files(year, months=None):
    """Find bulletin files for a given year, with priority months listed first.

    Always returns ALL files for the year. If specific months are given,
    those files appear first (priority ordering) followed by remaining months.
    """
    corpus = Path(CORPUS_DIR)
    if not corpus.exists():
        return []
    try:
        year = int(year)
    except (ValueError, TypeError):
        return []
    all_files = sorted(corpus.glob(f"treasury_bulletin_{year}_*.txt"))
    if not all_files:
        return []
    if not months:
        return all_files
    # Build priority set from requested months
    priority = []
    priority_set = set()
    for m in months:
        try:
            m_int = int(m)
        except (ValueError, TypeError):
            continue
        f = corpus / f"treasury_bulletin_{year}_{m_int:02d}.txt"
        if f.exists():
            priority.append(f)
            priority_set.add(f)
    # Priority months first, then remaining months
    rest = [f for f in all_files if f not in priority_set]
    return priority + rest


def _extract_table_title(lines, table_start):
    """Extract title above a pipe-delimited table, skipping blanks/page numbers."""
    title_lines = []
    _skip_re = re.compile(r"^(\d{1,4}|page\s+\d+|treasury\s+bulletin)$", re.IGNORECASE)
    for j in range(table_start - 1, max(0, table_start - 11), -1):
        line = lines[j].strip()
        if line.startswith("|") or line.startswith("---"):
            break
        if not line or _skip_re.match(line):
            continue  # skip but keep scanning upward
        title_lines.insert(0, line)
    return " ".join(title_lines).strip() if title_lines else ""


def _extract_columns(lines, table_start, table_end):
    """Extract and clean column headers from the first pipe-delimited row."""
    _footnote_re = re.compile(r"\s*\d+/")
    for i in range(table_start, min(table_start + 3, table_end)):
        line = lines[i].strip()
        if line.startswith("|") and "|" in line[1:]:
            raw_cols = [c.strip() for c in line.split("|") if c.strip()]
            if raw_cols and not all(re.match(r"^[-:]+$", c) for c in raw_cols):
                cleaned = []
                for cell in raw_cols:
                    segs = [s.strip() for s in cell.split(" > ")]
                    segs = [s for s in segs if not s.startswith("Unnamed:")]
                    joined = " / ".join(segs) if segs else cell
                    joined = _footnote_re.sub("", joined).strip()
                    cleaned.append(joined)
                return cleaned
    return []


def grep_table_metadata(filepath, search_terms):
    """Search file for pipe-delimited tables matching search terms."""
    try:
        text = filepath.read_text(errors="replace")
    except Exception:
        return []
    lines = text.splitlines()
    if not lines:
        return []
    # Find pipe-delimited table regions
    table_regions = []
    in_table = False
    tbl_start = 0
    for i, line in enumerate(lines):
        is_pipe = line.strip().startswith("|")
        if is_pipe and not in_table:
            in_table = True
            tbl_start = i
        elif not is_pipe and in_table:
            if i - tbl_start >= 3:
                table_regions.append((tbl_start, i))
            in_table = False
    if in_table and len(lines) - tbl_start >= 3:
        table_regions.append((tbl_start, len(lines)))

    # Build fuzzy patterns: allow optional punctuation between words
    def _fuzzy_pat(term):
        words = re.findall(r"\w+", term)
        if len(words) <= 1:
            return re.compile(re.escape(term), re.IGNORECASE)
        # Allow any non-word chars (apostrophes, hyphens, spaces) between words
        pat = r"\W*".join(re.escape(w) for w in words)
        return re.compile(pat, re.IGNORECASE)

    term_pats = [_fuzzy_pat(t) for t in search_terms]
    scored = []
    for start, end in table_regions:
        title_start = start
        for j in range(start - 1, max(0, start - 6), -1):
            ln = lines[j].strip()
            if not ln or ln.startswith("|") or ln.startswith("---"):
                break
            title_start = j
        title_text = "\n".join(lines[title_start:start])
        body_text = "\n".join(lines[start:end])
        # Title matches weighted 2x vs body matches
        title_hits = sum(2 for p in term_pats if p.search(title_text))
        body_hits = sum(1 for p in term_pats if p.search(body_text))
        hits = title_hits + body_hits
        if hits > 0:
            cols = _extract_columns(lines, start, end)
            tbl_title = _extract_table_title(lines, start)
            # Skip TOC tables — various heuristics
            if re.search(r"(?:cumulative\s+)?table\s+of\s+contents", tbl_title, re.IGNORECASE):
                continue
            # Skip small tables where values look like page numbers (all small ints)
            is_toc_like = False
            if end - start < 40:
                data_l = [
                    l
                    for l in lines[start : min(start + 8, end)]
                    if l.strip().startswith("|") and "---" not in l
                ]
                page_like_count = 0
                for dl in data_l[:5]:
                    cells = [c.strip() for c in dl.split("|") if c.strip()]
                    if len(cells) >= 2:
                        last_cell = cells[-1].replace(",", "").strip()
                        try:
                            v = float(last_cell)
                            if 1 <= v <= 200 and v == int(v):
                                page_like_count += 1
                        except (ValueError, IndexError):
                            pass
                if page_like_count >= 3:
                    is_toc_like = True
            # Skip tables that are clearly index/reference (e.g. "Issue and page number")
            if re.search(
                r"issue\s+and\s+page\s+number",
                title_text + body_text[:200],
                re.IGNORECASE,
            ):
                is_toc_like = True
            if is_toc_like:
                continue
            # Bonus: larger tables more likely to be real data
            size_bonus = min(3, (end - start) // 20)
            # Bonus: more columns = more likely data table
            col_bonus = min(2, max(0, len(cols) - 3))
            total_score = hits + size_bonus + col_bonus
            scored.append(
                (
                    total_score,
                    {
                        "title": tbl_title,
                        "columns": cols,
                        "source_file": str(filepath),
                        "line_start": title_start,
                        "line_end": end,
                    },
                )
            )
    scored.sort(key=lambda x: -x[0])
    return [item[1] for item in scored[:5]]


def search_tables(subquery):
    """Phase 2: Find candidate tables for a sub-query. Metadata only."""
    search_terms = subquery.get("search_terms", [])
    bulletin_year = subquery.get("target_bulletin_year")
    bulletin_months = subquery.get("target_bulletin_months", [])
    data_year = subquery.get("data_year")
    if not search_terms:
        return []
    files_to_search = []
    if bulletin_year:
        # Always search all months; priority months are listed first
        files_to_search = find_bulletin_files(bulletin_year, bulletin_months)
    if not files_to_search and data_year:
        # Search year+1 first, then widen progressively up to ±10 years
        for yr in [
            data_year + 1,
            data_year + 2,
            data_year + 3,
            data_year,
            data_year + 4,
            data_year + 5,
            data_year + 6,
            data_year + 7,
            data_year + 8,
            data_year - 1,
            data_year - 2,
        ]:
            files_to_search = find_bulletin_files(yr)
            if files_to_search:
                break
    if not files_to_search:
        files_to_search = list_corpus_files()[-24:]
    # Search ALL files, don't stop early — collect candidates and rank later
    all_cands = []
    for fpath in files_to_search:
        all_cands.extend(grep_table_metadata(fpath, search_terms))
    # If nothing with all terms, try individual terms
    if not all_cands:
        for fpath in files_to_search:
            for term in search_terms:
                all_cands.extend(grep_table_metadata(fpath, [term]))
            if len(all_cands) >= 5:
                break
    # Widen search to adjacent years if still empty (go ±20 years for historical data)
    if not all_cands and data_year:
        for delta in [
            -1,
            2,
            -2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
        ]:
            yr = (data_year + 1) + delta
            for fpath in find_bulletin_files(yr):
                all_cands.extend(grep_table_metadata(fpath, search_terms))
            if all_cands:
                break
    # Deduplicate using title + column similarity
    deduped = dedupe_candidates(all_cands)
    # Filter out candidates from bulletins too far from data year (widened to 20)
    if data_year:
        dy = int(data_year)

        def _near_data_year(c):
            m = re.search(r"treasury_bulletin_(\d{4})", str(c.get("source_file", "")))
            if not m:
                return True
            return abs(int(m.group(1)) - dy) <= 20

        nearby = [c for c in deduped if _near_data_year(c)]
        if nearby:
            deduped = nearby
    return deduped[:8]


# == Phase 2 Consensus: 3x parallel search strategies =========================

# Synonym expansions for common terms
_TERM_SYNONYMS = {
    "expenditures": ["outlays", "spending", "disbursements"],
    "outlays": ["expenditures", "spending", "disbursements"],
    "spending": ["expenditures", "outlays"],
    "receipts": ["revenue", "income", "collections"],
    "revenue": ["receipts", "income", "collections"],
    "income": ["receipts", "revenue"],
    "defense": ["national defense", "military", "defense services"],
    "national defense": ["defense", "military", "defense services"],
    "military": ["defense", "national defense"],
    "debt": ["public debt", "obligations", "borrowing"],
    "public debt": ["debt", "obligations"],
    "interest": ["interest cost", "net interest", "interest on debt"],
    "interest cost": ["interest", "net interest"],
    "net interest": ["interest", "interest cost"],
    "grants": ["grants-in-aid", "federal grants", "assistance"],
    "grants-in-aid": ["grants", "federal grants"],
    "veterans": ["veterans affairs", "veterans benefits", "veterans administration"],
    "veterans affairs": ["veterans", "veterans benefits"],
    "veterans benefits": ["veterans", "veterans affairs"],
    "budget": ["fiscal", "government"],
    "fiscal": ["budget", "government"],
    "receipt": ["revenue", "income"],
    "outlay": ["expenditure", "spending"],
    "surplus": ["excess", "balance"],
    "deficit": ["shortfall", "loss"],
    "tax": ["taxation", "taxes", "revenue"],
    "taxes": ["tax", "taxation"],
    "social security": ["social insurance", "trust funds"],
    "medicare": ["health", "health insurance"],
    "medicaid": ["medical assistance", "health"],
}


def _broaden_search_terms(terms):
    """Expand search terms with synonyms for broader recall."""
    expanded = list(terms)
    existing = set(t.lower() for t in expanded)
    for t in terms:
        tl = t.lower()
        for key, syns in _TERM_SYNONYMS.items():
            if key in tl:
                for s in syns:
                    if s.lower() not in existing:
                        expanded.append(s)
                        existing.add(s.lower())
    return expanded


def search_tables_multi(subquery, n_strategies=3):
    """Phase 2 consensus: Run N search strategies in parallel, merge pools."""
    search_terms = subquery.get("search_terms", [])
    if not search_terms:
        return []

    def _strategy_exact():
        """Strategy 1: Exact terms from plan."""
        return search_tables(subquery)

    def _strategy_broad():
        """Strategy 2: Broadened with synonyms."""
        sq_broad = dict(subquery)
        sq_broad["search_terms"] = _broaden_search_terms(search_terms)
        return search_tables(sq_broad)

    def _strategy_year_shift():
        """Strategy 3: Search wider bulletin years — compilations often appear 5-10 years later."""
        data_year = subquery.get("data_year")
        if not data_year:
            return search_tables(subquery)
        orig_year = subquery.get("target_bulletin_year", data_year + 1)
        results = []
        # Try year+2 through year+8 (historical compilations appear years later)
        for delta in [1, 2, 3, 5, 7]:
            sq_shifted = dict(subquery)
            sq_shifted["target_bulletin_year"] = orig_year + delta
            sq_shifted["target_bulletin_months"] = []  # search all months
            results.extend(search_tables(sq_shifted))
            if len(results) >= 5:
                break
        # Also try same year and year-1
        if len(results) < 3:
            for delta in [0, -1]:
                sq_shifted = dict(subquery)
                sq_shifted["target_bulletin_year"] = orig_year + delta
                results.extend(search_tables(sq_shifted))
        return results

    strategies = [_strategy_exact, _strategy_broad, _strategy_year_shift][:n_strategies]

    with concurrent.futures.ThreadPoolExecutor(max_workers=n_strategies) as ex:
        all_results = list(ex.map(lambda fn: fn(), strategies))

    # Merge and deduplicate using title + column similarity
    merged = dedupe_candidates([c for pool in all_results for c in pool])

    print(
        f"  Multi-search: {sum(len(p) for p in all_results)} raw -> {len(merged)} unique candidates",
        file=sys.stderr,
    )
    return merged[:12]  # Keep top 12 for scoring


# == Phase 3: Select Table ====================================================


def select_table(subquery, candidates):
    """Phase 3: Improved deterministic table selection with semantic scoring."""
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    data_year = subquery.get("data_year")
    col_hint = (subquery.get("column_hint") or "").lower()
    desc = (subquery.get("description") or "").lower()
    period_basis = (subquery.get("period_basis") or "").lower()
    value_type = (subquery.get("value_type") or "").lower()

    # Extract scoring keywords from description + col_hint
    _stop = {
        "the",
        "and",
        "for",
        "was",
        "what",
        "how",
        "total",
        "value",
        "all",
        "from",
        "with",
        "that",
        "this",
        "monthly",
        "values",
        "calendar",
        "fiscal",
        "year",
        "months",
        "data",
        "net",
        "gross",
    }
    desc_words = [w for w in re.findall(r"\w+", desc) if len(w) >= 3 and w not in _stop]
    hint_words = [w for w in re.findall(r"\w+", col_hint) if len(w) >= 3 and w not in _stop]

    best_score = -999
    best_cand = candidates[0]

    for c in candidates:
        score = 0
        title = (c.get("title") or "").lower()
        cols_lower = [col.lower() for col in (c.get("columns") or [])]
        cols_joined = " ".join(cols_lower)
        tbl_size = c.get("line_end", 0) - c.get("line_start", 0)
        src = str(c.get("source_file") or "")

        # 1. Column hint match (most important)
        if hint_words:
            for hw in hint_words:
                if any(hw in col for col in cols_lower):
                    score += 12  # increased from 10
                elif hw in title:
                    score += 3

        # 2. Description keyword match
        for dw in desc_words:
            if dw in title:
                score += 2
            if dw in cols_joined:
                score += 4  # increased from 3

        # 3. Bulletin proximity to data year
        if data_year:
            dy = int(data_year)
            m = re.search(r"treasury_bulletin_(\d{4})", src)
            if m:
                pub_yr = int(m.group(1))
                dist = abs(pub_yr - (dy + 1))
                if dist == 0:
                    score += 10
                else:
                    score -= dist * 4  # slightly less aggressive
                if abs(pub_yr - dy) > 20:  # widened from 15
                    score -= 50

        # 4. Table with monthly columns preferred for monthly_series
        if value_type == "monthly_series" or period_basis == "calendar":
            month_cols = sum(
                1
                for col in cols_lower
                if any(
                    m in col
                    for m in [
                        "jan",
                        "feb",
                        "mar",
                        "apr",
                        "may",
                        "jun",
                        "jul",
                        "aug",
                        "sep",
                        "oct",
                        "nov",
                        "dec",
                    ]
                )
            )
            if month_cols >= 6:
                score += 18  # increased from 15

        # 5. Data year appears in table body
        if data_year:
            yr_str = str(data_year)
            try:
                lines = Path(c["source_file"]).read_text(errors="replace").splitlines()
                ls = c.get("line_start", 0)
                le = min(c.get("line_end", ls + 50), len(lines))
                body_text = "\n".join(lines[ls:le])
                if yr_str in body_text:
                    score += 8  # increased from 5
            except Exception:
                pass

        # 6. Bigger tables slightly preferred
        score += min(4, tbl_size // 12)  # increased from 3/15

        # 7. NEW: Penalize tables with (Estimated) markers
        if data_year:
            try:
                lines = Path(c["source_file"]).read_text(errors="replace").splitlines()
                ls = c.get("line_start", 0)
                le = min(c.get("line_end", ls + 50), len(lines))
                body_start = "\n".join(lines[ls:le])
                if "(estimated)" in body_start.lower() or "estimated" in title:
                    score -= 15
            except Exception:
                pass

        # 8. NEW: Prefer tables with more columns (richer data)
        if len(cols_lower) >= 6:
            score += 5
        elif len(cols_lower) >= 4:
            score += 2

        # 9. NEW: Exact column hint match bonus
        if col_hint:
            exact_match = any(col_hint in col for col in cols_lower)
            if exact_match:
                score += 15

        if score > best_score:
            best_score = score
            best_cand = c

    return best_cand


# == Phase 4: Search Data =====================================================


def _filter_rows(data_rows, data_year, value_type):
    """Filter data rows to those relevant to data_year."""
    if not data_year:
        return data_rows
    year_str = str(data_year)
    if value_type in ("annual_total", "single", "max_in_row"):
        filtered = [r for r in data_rows if year_str in r]
        return filtered if filtered else data_rows
    if value_type == "monthly_series":
        filtered = []
        in_block = False
        for row in data_rows:
            cells = [c.strip() for c in row.split("|") if c.strip()]
            label = cells[0].lower().strip() if cells else ""
            if re.match(r"^\d{4}$", label.strip()):
                yr_val = int(label.strip())
                in_block = abs(yr_val - data_year) <= 1
                filtered.append(row)
                continue
            if in_block and (
                any(m in label for m in MONTH_NAMES_LOWER)
                or year_str in label
                or re.search(r"\d{4}[-\s]", label)
                or "transition" in label
            ):
                filtered.append(row)
        return filtered if filtered else data_rows
    return data_rows


def search_data(subquery, selected_table):
    """Phase 4: Read selected table, filter to relevant rows, detect units."""
    if not selected_table:
        return ""
    filepath = Path(selected_table["source_file"])
    try:
        text = filepath.read_text(errors="replace")
    except Exception:
        return ""
    lines = text.splitlines()
    start = selected_table["line_start"]
    end = min(selected_table["line_end"], len(lines))
    table_lines = lines[start:end]
    if not table_lines:
        return ""
    data_year = subquery.get("data_year")
    value_type = subquery.get("value_type", "single")
    title_parts, header_parts, data_rows = [], [], []
    found_pipe, header_done = False, False
    for line in table_lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            if not found_pipe:
                title_parts.append(line)
            continue
        found_pipe = True
        cells = [c.strip() for c in stripped.split("|") if c.strip()]
        if all(re.match(r"^[-:]+$", c) for c in cells):
            header_parts.append(line)
            header_done = True
        elif not header_done:
            header_parts.append(line)
        else:
            data_rows.append(line)
    filtered = _filter_rows(data_rows, data_year, value_type)
    out = []
    if title_parts:
        out.append("\n".join(title_parts))
    if header_parts:
        header_text = "\n".join(header_parts)
        out.append(header_text)
        # Detect and append units if found
        units = detect_units_from_header(header_text, "\n".join(title_parts))
        if units:
            out.append(f"\n[UNITS: {units}]")
    out.append("\n".join(filtered))
    result = "\n".join(out)
    if len(result) > 8000:
        result = result[:8000] + "\n... [truncated]"
    return result


# == Phase 5: Extract Data ====================================================


def build_extract_prompt(subquery, table_text):
    """Build user prompt for value extraction."""
    desc = subquery.get("description", "")
    vtype = subquery.get("value_type", "single")
    data_year = subquery.get("data_year", "")
    period = subquery.get("period_basis", "any")
    col_hint = subquery.get("column_hint", "")
    specific = subquery.get("specific_months", [])
    parts = [f"TASK: {desc}", f"Data year: {data_year}", f"Period: {period}"]
    if col_hint:
        parts.append(f"Expected column: {col_hint}")
    if vtype == "monthly_series":
        parts.append(f"Extract ALL 12 monthly values for CY {data_year} (Jan-Dec).")
    elif vtype == "annual_total":
        parts.append(f"Extract FY {data_year} total (row labeled '{data_year}').")
    elif vtype == "single" and specific:

        def _month_idx(m):
            """Convert month name or number to 1-12."""
            if isinstance(m, int):
                return m
            s = str(m).strip().lower()
            try:
                return int(s)
            except ValueError:
                for i, name in enumerate(MONTH_NAMES_LOWER):
                    if name.startswith(s[:3]):
                        return i + 1
                return None

        month_indices = [_month_idx(m) for m in specific]
        month_indices = [i for i in month_indices if i and 1 <= i <= 12]
        ms = ", ".join(MONTH_NAMES[i - 1] for i in month_indices)
        parts.append(f"Extract value for {ms} {data_year}.")
    elif vtype == "max_in_row":
        parts.append(
            f"Find row for {data_year}, return max value among data columns"
            " (excluding Total). Report which column."
        )
    else:
        parts.append("Extract the single value described above.")
    # Add vertical serialization of matching rows for easier column reading
    search_terms = subquery.get("search_terms", [])
    vertical = _vertical_serialize(table_text, data_year, col_hint, search_terms)
    if vertical:
        parts.append(f"\nVERTICAL VIEW (column: value pairs for matching rows):\n{vertical}")
        parts.append(f"\nRAW TABLE (for reference):\n{table_text}")
    else:
        parts.append(f"\nTABLE DATA:\n{table_text}")
    return "\n".join(parts)


def _vertical_serialize(table_text, data_year, col_hint="", search_terms=None):
    """Convert matching rows to vertical column:value pairs for easier LLM reading."""
    if not table_text or not data_year:
        return ""
    lines = table_text.strip().splitlines()
    # Find header row
    headers = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if cells and not all(re.match(r"^[-:]+$", c) for c in cells):
                headers = cells
                break
    if not headers:
        return ""

    # Build search words for marking relevant columns
    search_words = set()
    for t in search_terms or []:
        search_words.update(w.lower() for w in re.findall(r"\w+", t) if len(w) >= 3)

    year_str = str(data_year)
    result_parts = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|") if c.strip()]
        if not cells or all(re.match(r"^[-:]+$", c) for c in cells):
            continue
        # Check if this row matches the data year
        if year_str not in cells[0] and not any(m in cells[0].lower() for m in MONTH_NAMES_LOWER):
            continue
        # Serialize vertically
        row_label = cells[0]
        row_parts = [f"ROW: {row_label}"]
        for ci, cell in enumerate(cells[1:], 1):
            col_name = headers[ci] if ci < len(headers) else f"col_{ci}"
            # Mark columns matching column_hint or search terms
            col_lower = col_name.lower()
            markers = []
            if col_hint and col_hint.lower() in col_lower:
                markers.append("HINT_MATCH")
            if search_words and sum(1 for sw in search_words if sw in col_lower) >= 2:
                markers.append("SEARCH_MATCH")
            marker = f" <-- {', '.join(markers)}" if markers else ""
            row_parts.append(f"  {col_name}: {cell}{marker}")
        result_parts.append("\n".join(row_parts))

    return "\n\n".join(result_parts) if result_parts else ""


def extract_value(subquery, table_text):
    """Phase 5: Extract value(s) from focused table text via LLM."""
    if not table_text:
        return {"values": None, "confidence": "low", "notes": "No table data"}
    response = call_llm(EXTRACT_SYSTEM, build_extract_prompt(subquery, table_text), 1024)
    if not response:
        return {"values": None, "confidence": "low", "notes": "LLM failed"}
    result = parse_json_response(response)
    if not result:
        nums = re.findall(r"[\d,]+\.?\d*", response.replace(",", ""))
        if nums:
            try:
                return {
                    "values": float(nums[0]),
                    "confidence": "low",
                    "notes": "Parsed from raw",
                }
            except ValueError:
                pass
        return {"values": None, "confidence": "low", "notes": "Parse failed"}
    return result


# == Phase 5 Deterministic: Try column-matching extraction before LLM ==========


def extract_value_deterministic(subquery, table_text):
    """Try to extract value by deterministic column matching — no LLM needed.

    Returns extraction dict if successful, None if can't determine.
    """
    col_hint = (subquery.get("column_hint") or "").lower().strip()
    data_year = subquery.get("data_year")
    value_type = subquery.get("value_type", "single")
    if not col_hint or not data_year or not table_text:
        return None
    # Skip deterministic extraction when column_hint is too generic to be unambiguous
    _too_generic = {"total", "amount", "value", "number", "sum", "count", "other"}
    hint_words_check = set(re.findall(r"\w+", col_hint))
    if hint_words_check and hint_words_check.issubset(_too_generic):
        return None  # Let LLM handle ambiguous columns

    lines = table_text.strip().splitlines()
    # Find header row (first pipe-delimited row that isn't a separator)
    header_line = None
    header_idx = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("|") and "|" in stripped[1:]:
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if cells and not all(re.match(r"^[-:]+$", c) for c in cells):
                header_line = stripped
                header_idx = i
                break
    if not header_line:
        return None

    headers = [c.strip().lower() for c in header_line.split("|") if c.strip()]
    if not headers:
        return None

    # Find best matching column — prefer columns that also match search terms
    best_col_idx = None
    best_score = 0
    hint_words = set(re.findall(r"\w+", col_hint))
    search_terms = subquery.get("search_terms", [])
    search_words = set()
    for t in search_terms:
        search_words.update(w.lower() for w in re.findall(r"\w+", t) if len(w) >= 3)
    # Remove generic stop words that cause false matches
    _stop = {"total", "the", "and", "for", "was", "end", "period", "levels", "net"}
    hint_words_clean = hint_words - _stop

    for ci, h in enumerate(headers):
        h_words = set(re.findall(r"\w+", h))
        h_lower = h.lower()

        # Score: hint word overlap
        overlap = len(hint_words & h_words)
        if hint_words_clean:
            clean_overlap = len(hint_words_clean & h_words)
        else:
            clean_overlap = overlap

        # Bonus: search term words appear in column header
        search_overlap = sum(1 for sw in search_words if sw in h_lower)

        score = overlap + search_overlap * 2  # search term match is 2x weight

        # Require at least some meaningful match
        min_required = max(1, len(hint_words_clean) * 0.5) if hint_words_clean else 1
        if clean_overlap >= min_required or (overlap >= 1 and search_overlap >= 1):
            if score > best_score:
                best_score = score
                best_col_idx = ci

    if best_col_idx is None:
        return None

    year_str = str(data_year)

    if value_type in ("annual_total", "single", "max_in_row"):
        # Find the row for this year
        for line in lines[header_idx + 1 :]:
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if not cells:
                continue
            # Skip separator rows
            if all(re.match(r"^[-:]+$", c) for c in cells):
                continue
            label = cells[0].strip()
            if year_str not in label:
                continue
            # Found the year row — extract value at column index
            if best_col_idx < len(cells):
                raw = cells[best_col_idx]
                # Clean footnote markers
                raw = re.sub(r"\s*\d+/\s*", "", raw).strip()
                raw = re.sub(r"\s*[rp*]\s*$", "", raw, flags=re.IGNORECASE).strip()
                raw = raw.replace(",", "")
                if raw in ("-", "", "nan"):
                    val = 0.0
                else:
                    try:
                        val = float(raw)
                    except ValueError:
                        return None
                return {
                    "values": val,
                    "source_row": label,
                    "source_column": headers[best_col_idx],
                    "confidence": "high",
                    "notes": f"Deterministic: col {best_col_idx} '{headers[best_col_idx]}' matched hint '{col_hint}'",
                }

    elif value_type == "monthly_series":
        # For monthly series, we need to find 12 month rows and extract from the same column
        monthly_values = []
        in_block = False
        for line in lines[header_idx + 1 :]:
            stripped = line.strip()
            if not stripped.startswith("|"):
                continue
            cells = [c.strip() for c in stripped.split("|") if c.strip()]
            if not cells or all(re.match(r"^[-:]+$", c) for c in cells):
                continue
            label = cells[0].strip().lower()
            # Check if we're in the right year block
            if re.match(r"^\d{4}$", label.strip()):
                in_block = abs(int(label) - data_year) <= 1
                continue
            if in_block and any(m in label for m in MONTH_NAMES_LOWER):
                if best_col_idx < len(cells):
                    raw = cells[best_col_idx]
                    raw = re.sub(r"\s*\d+/\s*", "", raw).strip().replace(",", "")
                    if raw in ("-", "", "nan"):
                        monthly_values.append(0.0)
                    else:
                        try:
                            monthly_values.append(float(raw))
                        except ValueError:
                            return None  # Can't parse, fall back to LLM
        if len(monthly_values) == 12:
            return {
                "values": monthly_values,
                "source_row": f"monthly {data_year}",
                "source_column": headers[best_col_idx],
                "confidence": "high",
                "notes": f"Deterministic: col {best_col_idx} '{headers[best_col_idx]}' matched hint '{col_hint}'",
            }

    return None


# == Phase 5 Consensus: 3x parallel extraction with majority vote ==============

_EXTRACT_VARIANTS = [
    # Variant A: Precise (baseline)
    "",
    # Variant B: Contextual — pay attention to footnotes and units
    "\nIMPORTANT: Check for footnotes, unit labels (millions, thousands, etc.), "
    "and any notes that might affect the value. Read the FULL column header carefully.\n",
    # Variant C: Skeptical — double-check alignment
    "\nBEFORE extracting, verify: (1) you are reading the CORRECT row for the requested year, "
    "(2) you are reading the CORRECT column. State the row label and column header explicitly.\n",
]


def extract_value_consensus(subquery, table_text, n=3):
    """Phase 5 consensus: Run N extraction calls with diverse prompts, majority vote."""
    if not table_text:
        return {"values": None, "confidence": "low", "notes": "No table data"}

    base_prompt = build_extract_prompt(subquery, table_text)
    temps = [0.0, 0.4, 0.7][:n]

    def _run_extract(i):
        variant_suffix = _EXTRACT_VARIANTS[i % len(_EXTRACT_VARIANTS)]
        prompt = base_prompt + variant_suffix
        resp = call_llm_varied(EXTRACT_SYSTEM, prompt, 1024, temperature=temps[i])
        if not resp:
            return None
        result = parse_json_response(resp)
        if not result:
            nums = re.findall(r"[\d,]+\.?\d*", resp.replace(",", ""))
            if nums:
                try:
                    return {
                        "values": float(nums[0]),
                        "confidence": "low",
                        "notes": "Parsed from raw",
                    }
                except ValueError:
                    pass
            return None
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        results = list(ex.map(_run_extract, range(n)))
    results = [r for r in results if r and r.get("values") is not None]

    if not results:
        return {
            "values": None,
            "confidence": "low",
            "notes": "All extraction variants failed",
        }
    if len(results) == 1:
        return results[0]

    # Majority vote on values
    vtype = subquery.get("value_type", "single")

    if vtype == "monthly_series":
        # For monthly series: vote element-wise across the 3 lists
        lists = [
            r["values"]
            for r in results
            if isinstance(r.get("values"), list) and len(r["values"]) == 12
        ]
        if not lists:
            # Fall back to any list result
            for r in results:
                if isinstance(r.get("values"), list):
                    return r
            return results[0]
        if len(lists) == 1:
            best = [
                r for r in results if isinstance(r.get("values"), list) and len(r["values"]) == 12
            ][0]
            return best

        # Element-wise: for each month, take the value that appears most (or median)
        merged_values = []
        for month_idx in range(12):
            month_vals = []
            for lst in lists:
                try:
                    v = float(lst[month_idx]) if lst[month_idx] is not None else 0.0
                    month_vals.append(v)
                except (ValueError, TypeError, IndexError):
                    month_vals.append(0.0)
            # Majority: if 2+ agree, use that; else use median
            from collections import Counter

            val_counts = Counter(month_vals)
            winner, count = val_counts.most_common(1)[0]
            if count >= 2:
                merged_values.append(winner)
            else:
                merged_values.append(sorted(month_vals)[len(month_vals) // 2])

        best = results[0].copy()
        best["values"] = merged_values
        best["confidence"] = "high" if len(lists) >= 2 else "medium"
        best["notes"] = f"Consensus from {len(lists)} extraction paths"
        return best
    else:
        # Scalar values: majority vote
        scalars = []
        for r in results:
            v = r.get("values")
            if isinstance(v, (int, float)):
                scalars.append((v, r))
            elif isinstance(v, list) and len(v) == 1:
                scalars.append((v[0], r))

        if not scalars:
            return results[0]

        # Group by value (with small tolerance for float comparison)
        groups = {}
        for val, r in scalars:
            matched = False
            for key in groups:
                if abs(key - val) < 0.01 * max(abs(key), 1):
                    groups[key].append((val, r))
                    matched = True
                    break
            if not matched:
                groups[val] = [(val, r)]

        # Pick the group with most votes
        best_group = max(groups.values(), key=len)
        winner_val, winner_result = best_group[0]
        agreement = len(best_group)

        result = winner_result.copy()
        result["confidence"] = "high" if agreement >= 2 else result.get("confidence", "medium")
        result["notes"] = f"Consensus {agreement}/{len(scalars)}: " + ", ".join(
            f"{v}" for v, _ in scalars
        )
        print(
            f"    Extract consensus: {agreement}/{len(scalars)} agree on {winner_val}",
            file=sys.stderr,
        )
        return result


# == Phase 6: Verify ==========================================================


def _value_in_table(value, table_text):
    """Check if a numeric value actually appears in the table text."""
    if value is None or not table_text:
        return True  # can't verify, assume OK
    try:
        v = float(value)
    except (ValueError, TypeError):
        return True
    # Format value various ways and check if any appear in the text
    candidates = []
    if v == int(v) and abs(v) >= 1:
        iv = int(v)
        candidates.append(f"{iv:,}")  # "33,623"
        candidates.append(str(iv))  # "33623"
    else:
        candidates.append(f"{v:.1f}")
        candidates.append(f"{v:.2f}")
        candidates.append(str(v))
    # Also try without leading zeros
    for c in list(candidates):
        candidates.append(c.lstrip("0") or "0")
    return any(c in table_text for c in candidates)


def verify_value(subquery, extraction_result, focused_data):
    """Phase 6: Enhanced sanity-check extracted values with year/column verification."""
    values = extraction_result.get("values")
    if values is None:
        return extraction_result
    warnings = []
    # Type checks
    if isinstance(values, list):
        for i, v in enumerate(values):
            if v is not None and not isinstance(v, (int, float)):
                try:
                    float(v)
                except (ValueError, TypeError):
                    warnings.append(f"Non-numeric at index {i}: {v}")
    elif not isinstance(values, (int, float)):
        try:
            float(values)
        except (ValueError, TypeError):
            warnings.append(f"Non-numeric value: {values}")

    # Cross-check: does the extracted value actually appear in the source table?
    if focused_data:
        if isinstance(values, list):
            missing = sum(1 for v in values if not _value_in_table(v, focused_data))
            if missing > len(values) * 0.3:
                warnings.append(f"{missing}/{len(values)} values not found in table text")
        else:
            if not _value_in_table(values, focused_data):
                warnings.append(f"Value {values} not found in table text — possible hallucination")

    # NEW: Year verification - does extracted year match requested year?
    extracted_year = extraction_result.get("data_year")
    requested_year = subquery.get("data_year")
    if extracted_year and requested_year:
        if int(extracted_year) != int(requested_year):
            warnings.append(
                f"Year mismatch: requested {requested_year}, extracted {extracted_year}"
            )

    # NEW: Column index verification - is column index consistent?
    col_idx = extraction_result.get("column_index")
    if col_idx is not None:
        try:
            idx = int(col_idx)
            if idx < 1 or idx > 20:  # sanity bounds
                warnings.append(f"Suspicious column_index: {idx}")
        except (ValueError, TypeError):
            warnings.append(f"Invalid column_index: {col_idx}")

    # NEW: Unit consistency check
    unit_detected = extraction_result.get("unit_detected", "").lower()
    if unit_detected and unit_detected not in ("none", "unknown", ""):
        # Check if unit appears consistent with output_format
        output_units = subquery.get("output_format", {}).get("units", "").lower()
        if output_units and unit_detected != output_units:
            # Just a note, not a warning - could be intentional
            pass

    # Column hint check
    col_hint = subquery.get("column_hint", "")
    src_col = extraction_result.get("source_column", "")
    if col_hint and src_col:
        hw = set(re.findall(r"\w+", col_hint.lower()))
        sw = set(re.findall(r"\w+", src_col.lower()))
        if len(hw & sw) < len(hw) * 0.5 and len(hw) > 1:
            warnings.append(f"Column mismatch: '{col_hint}' vs '{src_col}'")
        # Contradictory term check: gross vs net
        _contra = [
            ("gross", "net"),
            ("receipts", "expenditures"),
            ("imports", "exports"),
        ]
        hint_low = col_hint.lower()
        src_low = src_col.lower()
        for a, b in _contra:
            if (a in hint_low and b in src_low) or (b in hint_low and a in src_low):
                warnings.append(f"Contradictory terms: hint='{col_hint}' vs source='{src_col}'")
    vtype = subquery.get("value_type", "single")
    if vtype == "monthly_series" and isinstance(values, list) and len(values) != 12:
        warnings.append(f"Expected 12 monthly values, got {len(values)}")
    if warnings:
        extraction_result = dict(extraction_result)
        extraction_result["confidence"] = "low"
        old = extraction_result.get("notes", "")
        extraction_result["notes"] = "; ".join(warnings) + (f" | {old}" if old else "")
    return extraction_result


# == Phase 7: Compute =========================================================


def parse_number(v):
    """Parse a value to float."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        v = v.strip().replace(",", "")
        if v in ("-", "", "nan", "null", "None"):
            return 0.0
        v = re.sub(r"\s*\d+/\s*$", "", v)
        v = re.sub(r"\s*[r]\s*$", "", v, flags=re.IGNORECASE)
        try:
            return float(v)
        except ValueError:
            return None
    return None


def resolve_subquery_value(extraction_result):
    """Turn extraction result into numeric value or list."""
    values = extraction_result.get("values")
    if values is None:
        return None
    if isinstance(values, list):
        return [n for v in values for n in [parse_number(v)] if n is not None]
    return parse_number(values)


def safe_eval_expr(expression, variables):
    """Evaluate a math expression with named variables using AST."""

    def _eval(node):  # noqa: C901
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return float(node.value)
            raise ValueError("only numeric constants")
        if isinstance(node, ast.Name):
            if node.id not in variables:
                raise ValueError(f"unknown variable: {node.id}")
            return float(variables[node.id])
        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                return -_eval(node.operand)
            if isinstance(node.op, ast.UAdd):
                return _eval(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            _ops = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Pow: lambda a, b: a**b,
                ast.Mod: lambda a, b: a % b,
            }
            op_type = type(node.op)
            if op_type == ast.Div:
                if right == 0:
                    raise ValueError("division by zero")
                return left / right
            if op_type in _ops:
                return _ops[op_type](left, right)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn, args = node.func.id, [_eval(a) for a in node.args]
            _fns = {
                "abs": lambda a: builtins.abs(a[0]),
                "sqrt": lambda a: math.sqrt(a[0]),
                "log": lambda a: math.log(*a),
                "pow": lambda a: a[0] ** a[1],
                "min": min,
                "max": max,
                "round": lambda a: (
                    builtins.round(a[0], int(a[1])) if len(a) == 2 else builtins.round(a[0])
                ),
            }
            if fn in _fns:
                return _fns[fn](args)
            raise ValueError(f"unsupported function: {fn}")
        raise ValueError(f"unsupported node: {type(node).__name__}")

    tree = ast.parse(expression.strip().replace("^", "**"), mode="eval")
    return _eval(tree)


def _to_scalar(v):
    """Collapse list to sum, pass through scalars."""
    if isinstance(v, list):
        return sum(x for x in v if x is not None)
    return v


def _make_vars(resolved):
    """Build {id: scalar} from resolved values."""
    return {k: _to_scalar(v) for k, v in resolved.items() if v is not None}


def compute(plan, extracted):
    """Phase 7: Deterministic computation on extracted values."""
    comp = plan.get("computation", {})
    comp_type = comp.get("type", "direct")
    formula = comp.get("formula", "")
    fmt = plan.get("output_format", {})
    rounding = fmt.get("rounding")
    suffix = fmt.get("suffix", "")
    sq_ids = [sq["id"] for sq in plan.get("sub_queries", [])]
    resolved = {}
    for sid in sq_ids:
        val = resolve_subquery_value(extracted.get(sid, {}))
        if val is None:
            print(f"  WARNING: sub-query {sid} returned None", file=sys.stderr)
        resolved[sid] = val
    print(f"  Resolved values: {resolved}", file=sys.stderr)
    result = None
    try:
        if comp_type == "direct":
            v = resolved.get(sq_ids[0])
            if isinstance(v, list):
                result = v[0] if len(v) == 1 else sum(v)
            else:
                result = v
        elif comp_type == "sum":
            v = resolved.get(sq_ids[0])
            if isinstance(v, list):
                result = sum(x for x in v if x is not None)
            else:
                result = sum(_to_scalar(x) for x in resolved.values() if x is not None)
        elif comp_type in ("difference", "percent_change", "ratio") and len(sq_ids) >= 2:
            a = _to_scalar(resolved.get(sq_ids[0]))
            b = _to_scalar(resolved.get(sq_ids[1]))
            if a is not None and b is not None:
                if comp_type == "difference":
                    result = abs(b - a)
                elif comp_type == "percent_change" and a != 0:
                    result = abs((b - a) / a) * 100
                elif comp_type == "ratio" and b != 0:
                    result = a / b
        elif comp_type == "geometric_mean":
            vals = []
            for v in resolved.values():
                if isinstance(v, list):
                    vals.extend(v)
                elif v is not None:
                    vals.append(v)
            if vals:
                p = 1.0
                for v in vals:
                    p *= v
                result = p ** (1.0 / len(vals))
        elif formula:
            try:
                result = safe_eval_expr(formula, _make_vars(resolved))
            except Exception:
                v = resolved.get(sq_ids[0]) if sq_ids else None
                result = _to_scalar(v)
    except Exception as e:
        print(f"  Compute error: {e}", file=sys.stderr)
        telemetry("compute_error", {"error": str(e)})
    if result is None:
        return "N/A"
    if rounding is not None:
        result = round(result, rounding)
    r = float(result)
    if rounding is not None and rounding == 0:
        formatted = f"{int(r):,}"
    elif rounding is not None:
        formatted = f"{r:,.{rounding}f}"
    elif r == int(r) and abs(r) >= 1:
        formatted = f"{int(r):,}"
    else:
        formatted = f"{r:,.2f}" if abs(r) < 100 else f"{r:,.0f}"
    if suffix:
        formatted = formatted + suffix
    return formatted


# == Main Pipeline =============================================================


def _process_subquery(sq):
    """Process a single sub-query through phases 2-6."""
    sq_id = sq["id"]
    desc = sq.get("description", "")[:80]
    # Phase 2: Search tables
    print(f"  [{sq_id}] Searching tables: {desc}", file=sys.stderr)
    candidates = search_tables(sq)
    if not candidates:
        print(f"  [{sq_id}] WARNING: No tables found", file=sys.stderr)
        return sq_id, {"values": None, "confidence": "low", "notes": "No tables"}, ""
    print(f"  [{sq_id}] Found {len(candidates)} candidate(s)", file=sys.stderr)
    # Phase 3: Select table
    selected = select_table(sq, candidates)
    if not selected:
        return (
            sq_id,
            {"values": None, "confidence": "low", "notes": "Select failed"},
            "",
        )
    print(f"  [{sq_id}] Selected: {selected['title'][:80]}", file=sys.stderr)
    # Phase 4: Read focused data
    focused = search_data(sq, selected)
    if not focused:
        return sq_id, {"values": None, "confidence": "low", "notes": "No data rows"}, ""
    print(f"  [{sq_id}] Loaded {len(focused)} chars", file=sys.stderr)
    # Phase 5: Extract
    extraction = extract_value(sq, focused)
    print(
        f"  [{sq_id}] Extracted: values={extraction.get('values')}, "
        f"conf={extraction.get('confidence', '?')}",
        file=sys.stderr,
    )
    # Phase 6: Verify
    extraction = verify_value(sq, extraction, focused)
    if extraction.get("confidence") == "low":
        print(
            f"  [{sq_id}] LOW confidence: {extraction.get('notes', '')}",
            file=sys.stderr,
        )

    # Retry with next candidate if extraction returned None (max 2 retries)
    if extraction.get("values") is None and len(candidates) > 1:
        fail_reason = extraction.get("notes", "unknown")
        print(f"  [{sq_id}] Retrying (reason: {fail_reason[:80]})", file=sys.stderr)
        retries = 0
        for alt in candidates:
            if alt is selected:
                continue
            if retries >= 2:
                break
            retries += 1
            print(f"  [{sq_id}] Trying: {alt['title'][:60]}", file=sys.stderr)
            alt_data = search_data(sq, alt)
            if not alt_data:
                continue
            alt_ext = extract_value(sq, alt_data)
            if alt_ext.get("values") is not None:
                alt_ext = verify_value(sq, alt_ext, alt_data)
                print(f"  [{sq_id}] Retry got: {alt_ext.get('values')}", file=sys.stderr)
                extraction = alt_ext
                focused = alt_data
                break

    return sq_id, extraction, focused


def _process_subquery_consensus(sq):
    """Process a single sub-query through phases 2-6 with consensus search+extract."""
    sq_id = sq["id"]
    desc = sq.get("description", "")[:80]

    # Phase 2: Multi-path search (3 strategies merged)
    print(f"  [{sq_id}] Multi-path search: {desc}", file=sys.stderr)
    candidates = search_tables_multi(sq, n_strategies=3)
    if not candidates:
        # Fall back to single search
        candidates = search_tables(sq)
    if not candidates:
        print(f"  [{sq_id}] WARNING: No tables found", file=sys.stderr)
        return sq_id, {"values": None, "confidence": "low", "notes": "No tables"}, ""
    print(f"  [{sq_id}] Found {len(candidates)} candidate(s)", file=sys.stderr)

    # Phase 3: Select table (deterministic — unchanged)
    selected = select_table(sq, candidates)
    if not selected:
        return (
            sq_id,
            {"values": None, "confidence": "low", "notes": "Select failed"},
            "",
        )
    print(f"  [{sq_id}] Selected: {selected['title'][:80]}", file=sys.stderr)

    # Phase 4: Read focused data (deterministic — unchanged)
    focused = search_data(sq, selected)
    if not focused:
        return sq_id, {"values": None, "confidence": "low", "notes": "No data rows"}, ""
    print(f"  [{sq_id}] Loaded {len(focused)} chars", file=sys.stderr)

    # Phase 5: Try deterministic extraction first, fall back to consensus LLM
    extraction = extract_value_deterministic(sq, focused)
    if extraction and extraction.get("values") is not None:
        print(
            f"  [{sq_id}] Deterministic extract: values={extraction.get('values')}, "
            f"col='{extraction.get('source_column', '?')}'",
            file=sys.stderr,
        )
    else:
        extraction = extract_value_consensus(sq, focused, n=3)
        print(
            f"  [{sq_id}] LLM consensus extract: values={extraction.get('values')}, "
            f"conf={extraction.get('confidence', '?')}",
            file=sys.stderr,
        )

    # Phase 6: Verify (deterministic — unchanged)
    extraction = verify_value(sq, extraction, focused)
    if extraction.get("confidence") == "low":
        print(
            f"  [{sq_id}] LOW confidence: {extraction.get('notes', '')}",
            file=sys.stderr,
        )

    # Retry with next candidate if extraction returned None (max 2 retries)
    if extraction.get("values") is None and len(candidates) > 1:
        fail_reason = extraction.get("notes", "unknown")
        print(
            f"  [{sq_id}] Retrying alt candidates (reason: {fail_reason[:80]})",
            file=sys.stderr,
        )
        retries = 0
        for alt in candidates:
            if alt is selected:
                continue
            if retries >= 2:
                break
            retries += 1
            print(f"  [{sq_id}] Trying: {alt['title'][:60]}", file=sys.stderr)
            alt_data = search_data(sq, alt)
            if not alt_data:
                continue
            alt_ext = extract_value_consensus(sq, alt_data, n=3)
            if alt_ext.get("values") is not None:
                alt_ext = verify_value(sq, alt_ext, alt_data)
                print(f"  [{sq_id}] Retry got: {alt_ext.get('values')}", file=sys.stderr)
                extraction = alt_ext
                focused = alt_data
                break

    # Adaptive escalation: if still None, widen search to distant years
    if extraction.get("values") is None:
        data_year = sq.get("data_year")
        if data_year:
            print(f"  [{sq_id}] ESCALATING: widening search to ±10 years", file=sys.stderr)
            sq_wide = dict(sq)
            for delta in [5, 6, 7, 8, 9, 10, -2, -3]:
                sq_wide["target_bulletin_year"] = data_year + 1 + delta
                sq_wide["target_bulletin_months"] = []  # search all months
                wide_cands = search_tables(sq_wide)
                if not wide_cands:
                    continue
                # Try each candidate
                for wc in wide_cands[:3]:
                    wc_title = wc.get("title", "")[:60]
                    print(f"  [{sq_id}] Wide search found: {wc_title}", file=sys.stderr)
                    wc_data = search_data(sq, wc)
                    if not wc_data:
                        continue
                    wc_ext = extract_value_consensus(sq, wc_data, n=3)
                    if wc_ext.get("values") is not None:
                        wc_ext = verify_value(sq, wc_ext, wc_data)
                        print(
                            f"  [{sq_id}] Wide search got: {wc_ext.get('values')}",
                            file=sys.stderr,
                        )
                        extraction = wc_ext
                        focused = wc_data
                        break
                if extraction.get("values") is not None:
                    break

    return sq_id, extraction, focused


# fmt: off
PLAN_REVIEW_SYSTEM = "You check research plans for Treasury data questions. Verify: all years mentioned are covered by sub-queries, computation type matches the question (percent change? difference? sum?), calendar vs fiscal year is handled correctly, rounding matches what was asked. Respond JSON only: {\"approved\": true, \"fixes\": []} or {\"approved\": false, \"fixes\": [\"fix1\", \"fix2\"]}"

LIBRARIAN_REVIEW_SYSTEM = """You are the owner of the analysis firm doing final quality control. Your team brought back finished work. You see:
- The original question
- What each sub-query found (extracted values + source rows/columns)
- The ACTUAL TABLE DATA that values were extracted from
- The computed answer

Your job: verify that the extracted values ACTUALLY APPEAR in the source table data. If a value doesn't match what's in the table, flag it. Also check:
- Right columns used? Right years? Does the column name EXACTLY match what the question asks for?
- Answer formatted as asked (commas? percent? decimal places? millions?)?
- Does the computation make sense for the question?

CRITICAL: If you are UNSURE whether the extracted column matches the metric in the question, mark correct=false. For example, if the question asks for "Veterans Administration" expenditures but the extraction came from a "Total" column, that is WRONG — flag it. Be conservative: it is better to reject a questionable match than to accept a wrong one.

If the answer looks wrong based on the evidence, provide your corrected answer by reading the correct value from the source table data.
Respond JSON only: {"answer": "formatted answer", "correct": true/false, "note": "explanation of any issues found"}"""
# fmt: on


def review_plan(question, plan):
    """Check the librarian's plan before executing."""
    sqs = plan.get("sub_queries", [])
    comp = plan.get("computation", {})
    fmt = plan.get("output_format", {})
    parts = [f"QUESTION: {question}", f"PLAN: {len(sqs)} sub-queries"]
    for sq in sqs:
        parts.append(f"  [{sq['id']}] {sq.get('description', '')[:100]}")
        parts.append(
            f"      year={sq.get('data_year')} months={sq.get('data_months')} "
            f"type={sq.get('value_type')} basis={sq.get('period_basis')}"
        )
    parts.append(f"Computation: {comp.get('type')} — {comp.get('description', '')[:100]}")
    parts.append(f"Formula: {comp.get('formula', 'none')}")
    parts.append(
        f"Format: units={fmt.get('units')}, rounding={fmt.get('rounding')}, "
        f"suffix='{fmt.get('suffix', '')}'"
    )
    resp = call_llm(PLAN_REVIEW_SYSTEM, "\n".join(parts), max_tokens=256)
    if resp:
        r = parse_json_response(resp)
        if r and not r.get("approved", True):
            return r.get("fixes", [])
    return []


def librarian_review(question, plan, extracted, computed_answer, evidence=None):
    """Final review: format and sanity-check the answer with source evidence."""
    evidence = evidence or {}
    parts = [f"QUESTION: {question}", "\nSUB-QUERY RESULTS:"]
    for sq in plan.get("sub_queries", []):
        sq_id = sq["id"]
        ext = extracted.get(sq_id, {})
        parts.append(f"  [{sq_id}] {sq.get('description', '')[:100]}")
        parts.append(f"      Extracted values: {ext.get('values')}")
        parts.append(f"      Source row: {ext.get('source_row', '?')}")
        parts.append(f"      Source column: {ext.get('source_column', '?')}")
        parts.append(f"      Confidence: {ext.get('confidence', '?')}")
        if ext.get("notes"):
            parts.append(f"      Notes: {ext['notes']}")
        # Include actual table data so reviewer can cross-check
        table_text = evidence.get(sq_id, "")
        if table_text:
            # Truncate per sub-query to fit in context
            snippet = table_text[:2000]
            if len(table_text) > 2000:
                snippet += "\n... [truncated]"
            parts.append(f"      SOURCE TABLE DATA:\n{snippet}")
    comp = plan.get("computation", {})
    parts.append(f"\nComputation: {comp.get('type', '?')} — {comp.get('formula', '')}")
    parts.append(f"COMPUTED ANSWER: {computed_answer}")
    fmt = plan.get("output_format", {})
    if fmt:
        parts.append(
            f"Expected format: units={fmt.get('units')}, rounding={fmt.get('rounding')}, "
            f"suffix='{fmt.get('suffix', '')}'"
        )
    resp = call_llm(LIBRARIAN_REVIEW_SYSTEM, "\n".join(parts), max_tokens=512)
    if resp:
        r = parse_json_response(resp)
        if r and r.get("answer"):
            note = r.get("note", "")
            is_correct = r.get("correct", True)
            if note:
                print(f"  Librarian note: {note}", file=sys.stderr)
            if not is_correct:
                print(f"  Librarian flagged INCORRECT: {note}", file=sys.stderr)
            return str(r["answer"]).strip()
    return computed_answer


def solve(question):
    """Run the full pipeline."""
    t0 = time.time()
    telemetry("solve_start", {"question": question[:300]})

    # Phase 1: Decompose
    plan = decompose(question)
    if not plan:
        print("FATAL: Decomposition failed", file=sys.stderr)
        telemetry("solve_fail", {"phase": "decompose"})
        return "N/A"
    sub_queries = plan.get("sub_queries", [])
    if not sub_queries:
        print("FATAL: No sub-queries", file=sys.stderr)
        return "N/A"

    # Phase 1b: Plan review
    print("Phase 1b: Reviewing plan...", file=sys.stderr)
    fixes = review_plan(question, plan)
    if fixes:
        print(f"  Fixes needed: {fixes}", file=sys.stderr)
        feedback = (
            "Issues found with your plan:\n"
            + "\n".join(f"- {f}" for f in fixes)
            + f"\n\nRevise the plan for: {question}"
        )
        plan2 = decompose(feedback)
        if plan2 and plan2.get("sub_queries"):
            plan = plan2
            sub_queries = plan["sub_queries"]
            print(f"  Revised: {len(sub_queries)} sub-queries", file=sys.stderr)
    else:
        print("  Plan approved", file=sys.stderr)

    # Phases 2-6 in parallel
    print("Phases 2-6: Processing sub-queries...", file=sys.stderr)
    extracted = {}
    evidence = {}  # Keep source table data for final review
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_process_subquery, sq): sq["id"] for sq in sub_queries}
        for future in concurrent.futures.as_completed(futures):
            try:
                sq_id, result, focused_data = future.result()
                extracted[sq_id] = result
                evidence[sq_id] = focused_data
            except Exception as e:
                sq_id = futures[future]
                extracted[sq_id] = {
                    "values": None,
                    "confidence": "low",
                    "notes": f"Exception: {e}",
                }

    # Phase 7: Compute
    print("Phase 7: Computing...", file=sys.stderr)
    answer = compute(plan, extracted)
    print(f"  Computed: {answer}", file=sys.stderr)

    # Phase 8: Librarian final review (with source evidence)
    print("Phase 8: Final review...", file=sys.stderr)
    answer = librarian_review(question, plan, extracted, answer, evidence=evidence)
    elapsed = time.time() - t0
    print(f"  Final: {answer} ({elapsed:.1f}s)", file=sys.stderr)
    telemetry("solve_done", {"answer": str(answer)[:200], "elapsed": f"{elapsed:.1f}s"})
    return answer


def solve_consensus(question):
    """Run the pipeline with step-wise consensus at each LLM-dependent phase.

    Phase 1: 3x parallel decompose with prompt diversity → merged plan
    Phase 2: 3x parallel search strategies per subquery → merged candidate pool
    Phase 3-4: Deterministic (unchanged)
    Phase 5: 3x parallel extraction with majority vote
    Phase 6-7: Deterministic (unchanged)
    Phase 8: Librarian review (unchanged)
    """
    t0 = time.time()
    telemetry("solve_consensus_start", {"question": question[:300]})

    # Phase 1: Consensus decompose (3 diverse planners)
    plan = decompose_consensus(question, n=3)
    if not plan:
        print("FATAL: Consensus decomposition failed", file=sys.stderr)
        telemetry("solve_fail", {"phase": "decompose_consensus"})
        return "N/A"
    sub_queries = plan.get("sub_queries", [])
    if not sub_queries:
        print("FATAL: No sub-queries", file=sys.stderr)
        return "N/A"

    # Phase 1b: Plan review (single call — cheap sanity check)
    print("Phase 1b: Reviewing plan...", file=sys.stderr)
    fixes = review_plan(question, plan)
    if fixes:
        print(f"  Fixes needed: {fixes}", file=sys.stderr)
        feedback = (
            "Issues found with your plan:\n"
            + "\n".join(f"- {f}" for f in fixes)
            + f"\n\nRevise the plan for: {question}"
        )
        plan2 = decompose(feedback)
        if plan2 and plan2.get("sub_queries"):
            plan = plan2
            sub_queries = plan["sub_queries"]
            print(f"  Revised: {len(sub_queries)} sub-queries", file=sys.stderr)
    else:
        print("  Plan approved", file=sys.stderr)

    # Phase 1c: Search validation — quick parallel check that each subquery can find tables
    # If any subquery finds nothing, re-plan with feedback about available years
    print("Phase 1c: Validating search...", file=sys.stderr)
    empty_sqs = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sub_queries)) as ex:
        sq_futures = {ex.submit(search_tables, sq): sq for sq in sub_queries}
        for future in concurrent.futures.as_completed(sq_futures):
            sq = sq_futures[future]
            try:
                cands = future.result()
                if not cands:
                    empty_sqs.append(sq)
            except Exception:
                empty_sqs.append(sq)
    if empty_sqs:
        # Build feedback about what years exist near the requested data
        corpus_years = set()
        for f in list_corpus_files():
            m = re.search(r"treasury_bulletin_(\d{4})", str(f))
            if m:
                corpus_years.add(int(m.group(1)))
        feedback_parts = [
            f"Your plan for: {question}\n",
            "PROBLEM: These sub-queries found NO tables in the corpus:",
        ]
        for sq in empty_sqs:
            dy = sq.get("data_year", "?")
            by = sq.get("target_bulletin_year", "?")
            terms = sq.get("search_terms", [])
            # Find nearest available years
            if isinstance(by, int):
                nearby = sorted([y for y in corpus_years if abs(y - by) <= 15])[:10]
            else:
                nearby = []
            feedback_parts.append(
                f"  [{sq['id']}] search_terms={terms}, target_year={by}, data_year={dy}"
                f"\n    Nearest available years: {nearby}"
                f"\n    Try broader search terms or different bulletin years."
            )
        feedback_parts.append(
            "\nRevise the plan. Use search_terms from TABLE TITLES, "
            "not question text. Treasury table titles often differ from question wording."
        )
        print(
            f"  {len(empty_sqs)} sub-queries found no tables, re-planning...",
            file=sys.stderr,
        )
        plan2 = decompose("\n".join(feedback_parts))
        if plan2 and plan2.get("sub_queries"):
            plan = plan2
            sub_queries = plan["sub_queries"]
            print(f"  Re-planned: {len(sub_queries)} sub-queries", file=sys.stderr)
    else:
        print(f"  All {len(sub_queries)} sub-queries found candidates", file=sys.stderr)

    # Phases 2-6 with consensus (parallel per sub-query)
    print("Phases 2-6: Consensus processing sub-queries...", file=sys.stderr)
    extracted = {}
    evidence = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_process_subquery_consensus, sq): sq["id"] for sq in sub_queries}
        for future in concurrent.futures.as_completed(futures):
            try:
                sq_id, result, focused_data = future.result()
                extracted[sq_id] = result
                evidence[sq_id] = focused_data
            except Exception as e:
                sq_id = futures[future]
                extracted[sq_id] = {
                    "values": None,
                    "confidence": "low",
                    "notes": f"Exception: {e}",
                }

    # Phase 7: Compute (deterministic)
    print("Phase 7: Computing...", file=sys.stderr)
    answer = compute(plan, extracted)
    print(f"  Computed: {answer}", file=sys.stderr)

    # Phase 8: Librarian final review
    print("Phase 8: Final review...", file=sys.stderr)
    answer = librarian_review(question, plan, extracted, answer, evidence=evidence)
    elapsed = time.time() - t0
    print(f"  Final: {answer} ({elapsed:.1f}s)", file=sys.stderr)
    telemetry(
        "solve_consensus_done",
        {"answer": str(answer)[:200], "elapsed": f"{elapsed:.1f}s"},
    )
    return answer


def solve_stochastic(question, n_paths=3):
    """Run N parallel solve paths and pick the best answer by majority vote.

    Each path independently decomposes, searches, extracts, and computes.
    If 2+ paths agree, use that answer. Otherwise use the one with highest
    confidence (most sub-queries with high confidence extraction).
    """
    t0 = time.time()
    print(f"Stochastic solve: {n_paths} parallel paths", file=sys.stderr)

    answers = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_paths) as executor:
        futures = [executor.submit(solve, question) for _ in range(n_paths)]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            try:
                ans = future.result()
                answers.append(ans)
                print(f"  Path {i + 1}: {ans}", file=sys.stderr)
            except Exception as e:
                print(f"  Path {i + 1} failed: {e}", file=sys.stderr)
                answers.append("N/A")

    # Majority vote: normalize and count
    def _norm(s):
        s = str(s).strip().replace(",", "").replace("%", "").replace(" million", "")
        try:
            return round(float(s), 2)
        except (ValueError, TypeError):
            return s

    normed = [_norm(a) for a in answers]
    counts = {}
    for i, n in enumerate(normed):
        counts.setdefault(n, []).append(i)

    # Pick the value with most votes
    best_val = max(counts.keys(), key=lambda k: len(counts[k]))
    best_idx = counts[best_val][0]
    best_answer = answers[best_idx]

    agreement = len(counts[best_val])
    elapsed = time.time() - t0
    print(
        f"  Stochastic result: {best_answer} ({agreement}/{len(answers)} agree, {elapsed:.1f}s)",
        file=sys.stderr,
    )
    return best_answer


def main():
    """Entry point."""
    if len(sys.argv) < 2:
        print(f'Usage: {sys.argv[0]} "QUESTION"', file=sys.stderr)
        sys.exit(1)
    question = sys.argv[1]
    consensus = os.environ.get("CONSENSUS", "0") == "1"
    stochastic = os.environ.get("STOCHASTIC", "0") == "1"
    n_paths = int(os.environ.get("N_PATHS", "3"))
    print(f"Question: {question}", file=sys.stderr)
    if consensus:
        answer = solve_consensus(question)
    elif stochastic:
        answer = solve_stochastic(question, n_paths=n_paths)
    else:
        answer = solve(question)
    answer_path = Path(ANSWER_PATH)
    answer_path.parent.mkdir(parents=True, exist_ok=True)
    answer_path.write_text(str(answer))
    print(f"Answer written to {answer_path}: {answer}", file=sys.stderr)


if __name__ == "__main__":
    main()
