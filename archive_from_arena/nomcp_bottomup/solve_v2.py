#!/usr/bin/env python3
"""solve_v2: Full pipeline wiring Orchestrator + SearchTask + multi-file monthly aggregation.

Improvements over solve_decompose.py:
  1. Uses richer DECOMPOSE_PROMPT from layer_decompose (Treasury-aware)
  2. Monthly series: searches multiple bulletin files to fill missing months
  3. Single values: uses SearchTask 3-strategy (strict+fuzzy+contextual) + consensus
  4. Uses Orchestrator for validated parallel piece processing

Usage:
    python3 solve_v2.py "What were total defense expenditures in CY 1940?"
    OPENROUTER_API_KEY=... python3 test_pipeline_v2.py
"""

import concurrent.futures
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

# Env setup
_env = HERE.parent.parent / ".env"
if _env.exists():
    for line in _env.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

CORPUS_DIR = os.environ.get("CORPUS_DIR", str(HERE.parent.parent / "corpus"))
os.environ["CORPUS_DIR"] = CORPUS_DIR
ANSWER_PATH = os.environ.get("ANSWER_PATH", "/app/answer.txt")

# Import rich decompose prompt
# Import search/data/compute from solve_decompose
import solve_decompose as sd
from layer_decompose import DECOMPOSE_PROMPT
from layer_decompose import call_llm as ld_call_llm

# Import SearchTask components
from search_task import SearchTask

_search_task = SearchTask()

from consensus_voter import ConsensusVoter

_voter = ConsensusVoter()


# ── 3-Strategy Extraction Prompts ─────────────────────────────────────────────
# Each sees the same table text but reasons differently.
# No max_tokens cap — let the model reason all the way through.

STRICT_EXTRACT = """You extract data from pipe-delimited tables. Be literal and precise.

1. Read the header row to understand what rows and columns represent.
2. Count pipe positions carefully — the Nth cell in a data row maps to the Nth column header.
3. Find the exact row and column matching the request. State what you matched.
4. Strip footnote markers (r, p, *, 3/) from values. Dashes mean zero. "nan" means unavailable.
5. For monthly series: return 12 values in Jan-Dec order, null for any month not found.

OUTPUT JSON only:
{"values": <number or [12 values] or null>, "source_row": "...", "source_column": "...", "confidence": "high/medium/low", "notes": "what you matched and why"}"""


FUZZY_EXTRACT = """You extract data from pipe-delimited tables that were converted from PDF.

Conversion artifacts are common — labels may be wrong, headers may span multiple rows, year labels in multi-level headers may not update at year boundaries. Use the actual sequence of data and surrounding context to figure out what each column really represents rather than trusting labels literally.

Be flexible with matching: ignore footnote markers in labels, allow abbreviations, and consider that the metric you need might be either a row label or a column header depending on how the table is oriented.

For monthly series: return 12 values in Jan-Dec order, null for missing.

OUTPUT JSON only:
{"values": <number or [12 values] or null>, "source_row": "...", "source_column": "...", "confidence": "high/medium/low", "notes": "what you found and any artifacts you corrected"}"""


CONTEXTUAL_EXTRACT = """You extract data from pipe-delimited tables. Reason step by step before answering.

First, read the entire table structure — headers, row labels, data patterns — and explain what the table represents. Identify what the rows are (time periods? metrics?) and what the columns are.

Then locate the specific data requested. If something looks off about the labels (wrong year, missing header), use the surrounding data pattern to figure out the right interpretation.

For monthly series: identify which 12 positions correspond to Jan-Dec of the target year and extract them in order.

Show your reasoning, then give the answer.

OUTPUT JSON only:
{"values": <number or [12 values] or null>, "source_row": "...", "source_column": "...", "confidence": "high/medium/low", "notes": "your step-by-step reasoning"}"""


# ── Decompose ──────────────────────────────────────────────────────────────────


def decompose(question: str, retries: int = 3) -> dict | None:
    """Decompose with rich Treasury-aware prompt. Retries on parse failure."""
    print("Phase 1: Decomposing (rich prompt)...", file=sys.stderr)
    for attempt in range(retries):
        response = ld_call_llm(DECOMPOSE_PROMPT, question, max_tokens=8192)
        if not response:
            continue
        try:
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"^```\w*\n?", "", cleaned)
                cleaned = re.sub(r"\n?```$", "", cleaned)
            plan = json.loads(cleaned)
            if plan.get("sub_queries"):
                n = len(plan["sub_queries"])
                comp = plan.get("computation", {}).get("type", "?")
                print(f"  -> {n} sub-queries, computation={comp}", file=sys.stderr)
                return plan
        except json.JSONDecodeError:
            pass
        if attempt < retries - 1:
            print(f"  Parse failed (attempt {attempt + 1}), retrying...", file=sys.stderr)
    print("  FATAL: Decompose failed", file=sys.stderr)
    return None


# ── Multi-file monthly aggregation ────────────────────────────────────────────

_MONTH_PATS = [
    re.compile(r"\b(jan(?:uary)?)\b", re.I),
    re.compile(r"\b(feb(?:ruary)?)\b", re.I),
    re.compile(r"\b(mar(?:ch)?)\b", re.I),
    re.compile(r"\b(apr(?:il)?)\b", re.I),
    re.compile(r"\b(may)\b", re.I),
    re.compile(r"\b(jun(?:e)?)\b", re.I),
    re.compile(r"\b(jul(?:y)?)\b", re.I),
    re.compile(r"\b(aug(?:ust)?)\b", re.I),
    re.compile(r"\b(sep(?:tember)?)\b", re.I),
    re.compile(r"\b(oct(?:ober)?)\b", re.I),
    re.compile(r"\b(nov(?:ember)?)\b", re.I),
    re.compile(r"\b(dec(?:ember)?)\b", re.I),
]
_NUM_RE = re.compile(r"^\s*[\-\+]?\s*[\d,]+(?:\.\d+)?\s*$")
_FOOTNOTE_RE = re.compile(r"[rpe\*\/]\s*$", re.I)


def _parse_pipe_cell(cell: str) -> float | None:
    """Parse a pipe-delimited table cell into a float. Returns None if not numeric."""
    cell = cell.strip()
    # Strip footnote markers (r, p, e, *, 3/, etc.)
    cell = re.sub(r"\s*[rpe]\s*$", "", cell, flags=re.I)
    cell = re.sub(r"\s*\d+\/\s*$", "", cell)
    cell = re.sub(r"\s*\*+\s*$", "", cell)
    if cell in ("-", "—", "–", "", "nan", "N/A", "n.a."):
        return 0.0  # dash = zero in Treasury tables
    cell = cell.replace(",", "").replace("$", "").strip()
    try:
        v = float(cell)
        return None if _is_year_label(v) else v
    except (ValueError, TypeError):
        return None


def _find_best_column(header_cells: list[str], column_hint: str) -> int:
    """Find the column index best matching column_hint. Returns -1 if none."""
    if not column_hint:
        return -1
    hint_lower = column_hint.lower()
    hint_words = re.findall(r"\w+", hint_lower)
    best_idx, best_score = -1, 0
    for i, cell in enumerate(header_cells):
        cell_lower = cell.lower()
        score = sum(1 for w in hint_words if w in cell_lower)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx if best_score > 0 else -1


def _extract_monthly_deterministic(
    table_text: str,
    data_year: int | None,
    column_hint: str = "",
) -> list[float | None]:
    """Deterministically extract 12 monthly values from pipe-delimited table text.

    Algorithm:
      1. Parse header row → find target column index via column_hint
      2. Scan each data row for month-name label
      3. Extract value from target column (or rightmost numeric if no hint)
      4. Return list of 12 values (None = month not found)

    No LLM calls. Zero variance. Fast.
    """
    results: list[float | None] = [None] * 12
    lines = table_text.splitlines()

    # Find header row (first pipe-delimited row that is NOT a separator)
    header_cells: list[str] = []
    col_idx = -1
    header_line_i = -1
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|") if c.strip()]
        if not cells:
            continue
        # Skip separator rows (all dashes)
        if all(re.match(r"^[-:]+$", c) for c in cells):
            continue
        # This is a header candidate
        header_cells = cells
        header_line_i = i
        col_idx = _find_best_column(cells, column_hint)
        break

    if not header_cells:
        return results

    # Scan data rows for month labels
    year_str = str(data_year) if data_year else ""
    for line in lines[header_line_i + 1 :]:
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [c.strip() for c in stripped.split("|") if c.strip()]
        if not cells or len(cells) < 2:
            continue
        label = cells[0]

        # Check if this row is for data_year (skip other years' rows)
        if year_str and len(label) > 5:
            # Label like "1939-January" or "1940 January" — skip if wrong year
            if re.search(r"\b(18|19|20)\d{2}\b", label):
                if year_str not in label:
                    continue

        # Match month
        month_idx = None
        for m_i, pat in enumerate(_MONTH_PATS):
            if pat.search(label):
                month_idx = m_i
                break
        if month_idx is None:
            continue

        # Extract value: use target column if found, else try columns right-to-left
        value = None
        if 0 <= col_idx < len(cells):
            value = _parse_pipe_cell(cells[col_idx])
        if value is None:
            # Try all columns except label, take last parseable number
            for cell in reversed(cells[1:]):
                value = _parse_pipe_cell(cell)
                if value is not None:
                    break

        # Only set if not already found (first valid value wins per month)
        if value is not None and results[month_idx] is None:
            results[month_idx] = value

    return results


def _derive_column_hint(sq: dict) -> str:
    """Derive a column hint from sub-query when column_hint is empty.

    Treasury table columns often match the last search term (which tends to be
    the specific metric, not the table name). Uses shortest term as best column hint.
    """
    hint = (sq.get("column_hint") or "").strip()
    if hint:
        return hint
    # Use the shortest search term — usually the specific metric keyword
    terms = [t for t in sq.get("search_terms", []) if t]
    if not terms:
        return ""
    return min(terms, key=len)


def _short_search_terms(sq: dict) -> list[str]:
    """Return short (<40 char) search terms, splitting long ones into words."""
    result = []
    for term in sq.get("search_terms", []):
        if len(term) < 40:
            result.append(term)
        else:
            # Split long table-title terms into meaningful keywords
            words = [
                w
                for w in re.findall(r"\b[A-Za-z]{4,}\b", term)
                if w.lower()
                not in {"classified", "major", "general", "function", "functions", "budget"}
            ]
            result.extend(words[:3])
    return result or sq.get("search_terms", [])


def _collect_monthly_from_file(sq: dict, filepath: Path) -> list[float | None]:
    """Deterministically extract monthly values from the best table in a bulletin file.

    Pure Python parsing — NO LLM calls. Fast enough to scan many files.
    """
    search_terms = _short_search_terms(sq)
    candidates = sd.grep_table_metadata(filepath, search_terms)
    if not candidates:
        return []
    selected = sd.select_table(sq, candidates)
    if not selected:
        return []
    focused = sd.search_data(sq, selected)
    if not focused:
        return []

    data_year = sq.get("data_year")
    column_hint = _derive_column_hint(sq)
    return _extract_monthly_deterministic(focused, data_year, column_hint)


def _is_year_label(v) -> bool:
    """True if a value looks like a year label rather than a data value."""
    if v is None:
        return False
    try:
        fv = float(v)
        # Values between 1800-2030 that are exact integers look like years
        return 1800 <= fv <= 2030 and fv == int(fv)
    except (ValueError, TypeError):
        return False


def _merge_monthly_values(existing: list, new_vals: list) -> list:
    """Merge two monthly value lists; fill None slots, cap at 12, filter year labels."""
    result = list(existing[:12])
    for i, v in enumerate(new_vals):
        if i >= 12:
            break
        if v is not None and _is_year_label(v):
            continue  # skip year labels
        if i < len(result):
            if result[i] is None and v is not None:
                result[i] = v
        else:
            result.append(v)
    return result[:12]


def _score_monthly_extraction(vals: list) -> int:
    """Count non-None, non-year-label values in a monthly extraction."""
    return sum(1 for v in vals if v is not None and not _is_year_label(v))


# ── Consensus Extraction ──────────────────────────────────────────────────────


def _parse_extraction(response: str | None) -> dict | None:
    """Parse JSON from an extraction strategy response."""
    if not response:
        return None
    text = response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```$", "", text.strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


def _normalize_extracted(parsed: dict | None):
    """Clean the values field from a parsed extraction response."""
    if not parsed:
        return None
    raw = parsed.get("values")
    if raw is None:
        return None
    if isinstance(raw, list):
        cleaned = []
        for v in raw:
            if v is None or str(v).strip() in ("", "null", "nan", "N/A"):
                cleaned.append(None)
            else:
                try:
                    cleaned.append(float(str(v).replace(",", "").strip()))
                except (ValueError, TypeError):
                    cleaned.append(None)
        return cleaned
    try:
        return float(str(raw).replace(",", "").strip())
    except (ValueError, TypeError):
        return None


def _conf_to_float(s: str) -> float:
    return {"high": 0.9, "medium": 0.65, "low": 0.4}.get(str(s).lower(), 0.5)


def _monthly_position_vote(lists: list[list]) -> list[float | None]:
    """Per-position majority vote across multiple monthly series."""
    merged: list[float | None] = [None for _ in range(12)]
    for i in range(12):
        votes: dict[float, int] = {}
        for lst in lists:
            if lst and i < len(lst) and lst[i] is not None:
                key = round(float(lst[i]), 2)
                votes[key] = votes.get(key, 0) + 1
        if votes:
            best_key = sorted(votes.keys(), key=lambda k: votes[k], reverse=True)[0]
            merged[i] = best_key
    return merged


def extract_value_consensus(subquery: dict, table_text: str) -> dict:
    """Run 3 parallel extraction strategies on raw table text, vote on result.

    Replaces _extract_monthly_deterministic and sd.extract_value.
    Returns {"values": ..., "confidence": "high/medium/low", "notes": "..."}
    """
    if not table_text:
        return {"values": None, "confidence": "low", "notes": "No table data"}

    vtype = subquery.get("value_type", "single")
    user_prompt = sd.build_extract_prompt(subquery, table_text)

    strategies = [
        ("strict", STRICT_EXTRACT),
        ("fuzzy", FUZZY_EXTRACT),
        ("contextual", CONTEXTUAL_EXTRACT),
    ]

    raw_results: dict[str, dict] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(sd.call_llm, system, user_prompt, 8192): name
            for name, system in strategies
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                response = future.result()
                parsed = _parse_extraction(response)
                values = _normalize_extracted(parsed)
                conf = _conf_to_float(parsed.get("confidence", "low") if parsed else "low")
                raw_results[name] = {"values": values, "confidence": conf, "parsed": parsed}
                print(f"    [{name}] {str(values)[:80]} conf={conf}", file=sys.stderr)
            except Exception as e:
                print(f"    [{name}] FAILED: {e}", file=sys.stderr)
                raw_results[name] = {"values": None, "confidence": 0.0, "parsed": None}

    sv = raw_results.get("strict", {}).get("values")
    fv = raw_results.get("fuzzy", {}).get("values")
    cv = raw_results.get("contextual", {}).get("values")

    # Monthly series: per-position vote across all strategies that returned lists
    if vtype in ("monthly_series", "series"):
        lists = [v for v in [sv, fv, cv] if isinstance(v, list)]
        if not lists:
            # All strategies failed — fall back to single LLM call
            fallback = sd.extract_value(subquery, table_text)
            fallback["notes"] = "consensus fallback: no lists returned"
            return fallback
        merged = _monthly_position_vote(lists)
        filled = sum(1 for v in merged if v is not None)
        conf_str = "high" if filled >= 11 else ("medium" if filled >= 7 else "low")
        return {
            "values": merged,
            "confidence": conf_str,
            "notes": f"{len(lists)}/3 strategies returned lists, {filled}/12 months filled",
        }

    # Single value: use ConsensusVoter
    def _to_voter_input(name: str) -> dict:
        v = raw_results.get(name, {}).get("values")
        return {
            "value": v if isinstance(v, (int, float)) else None,
            "confidence": raw_results.get(name, {}).get("confidence", 0.0),
            "method": name,
        }

    vote = _voter.vote(
        _to_voter_input("strict"), _to_voter_input("fuzzy"), _to_voter_input("contextual")
    )
    winning = vote.get("winning_value")
    win_conf = vote.get("winning_confidence", 0.0)
    conf_str = "high" if win_conf >= 0.8 else ("medium" if win_conf >= 0.6 else "low")
    return {
        "values": winning,
        "confidence": conf_str,
        "notes": f"agreement={vote.get('agreement_level')} method={vote.get('winning_method')}",
    }


def _process_monthly_series(sq: dict) -> tuple[str, dict, str]:
    """Find candidate tables, extract monthly series via 3-strategy LLM consensus."""
    sq_id = sq["id"]
    data_year = sq.get("data_year")
    print(f"  [{sq_id}] Monthly series for {data_year}", file=sys.stderr)

    candidates = sd.search_tables(sq)
    if not candidates:
        print(f"  [{sq_id}] No tables found", file=sys.stderr)
        return sq_id, {"values": None, "confidence": "low", "notes": "No tables"}, ""

    # Try candidates until one works — just needs to be in the top 5
    for i, cand in enumerate(candidates[:5]):
        focused = sd.search_data(sq, cand)
        if not focused:
            continue
        print(f"  [{sq_id}] Trying table {i + 1}: {cand['title'][:60]}", file=sys.stderr)

        extraction = extract_value_consensus(sq, focused)
        vals = extraction.get("values")

        if isinstance(vals, list):
            filled = sum(1 for v in vals if v is not None)
            print(
                f"  [{sq_id}] Got {filled}/12 months ({extraction.get('confidence')})",
                file=sys.stderr,
            )
            if filled >= 6:
                return sq_id, extraction, focused
        elif vals is not None:
            # Got a single value instead of series — still usable
            print(f"  [{sq_id}] Got single value: {vals}", file=sys.stderr)
            return sq_id, extraction, focused

    # All candidates failed
    return sq_id, {"values": None, "confidence": "low", "notes": "All candidates failed"}, ""


def _process_single_value(sq: dict) -> tuple[str, dict, str]:
    """Find candidate tables, extract single value via 3-strategy LLM consensus."""
    sq_id = sq["id"]
    desc = sq.get("description", "")[:80]
    print(f"  [{sq_id}] Single value: {desc}", file=sys.stderr)

    candidates = sd.search_tables(sq)
    if not candidates:
        print(f"  [{sq_id}] WARNING: No tables found", file=sys.stderr)
        return sq_id, {"values": None, "confidence": "low", "notes": "No tables"}, ""

    # Try candidates until one works — just needs to be in the top 5
    for i, cand in enumerate(candidates[:5]):
        selected = cand
        focused = sd.search_data(sq, selected)
        if not focused:
            continue
        print(f"  [{sq_id}] Trying table {i + 1}: {selected['title'][:60]}", file=sys.stderr)

        extraction = extract_value_consensus(sq, focused)
        if extraction.get("values") is not None:
            print(
                f"  [{sq_id}] Got: {extraction['values']} ({extraction.get('confidence')})",
                file=sys.stderr,
            )
            return sq_id, extraction, focused

    return sq_id, {"values": None, "confidence": "low", "notes": "All candidates failed"}, ""


# ── Unified subquery processor ─────────────────────────────────────────────────


def _process_subquery_v2(sq: dict) -> tuple[str, dict, str]:
    """Route to monthly or single-value processor based on value_type."""
    value_type = sq.get("value_type", "single")
    if value_type in ("monthly_series", "series"):
        return _process_monthly_series(sq)
    else:
        return _process_single_value(sq)


# ── Main solve function ────────────────────────────────────────────────────────


def solve(question: str) -> str:
    """Full v2 pipeline: decompose → parallel extract → compute → review."""
    t0 = time.time()

    # Phase 1: Decompose
    plan = decompose(question)
    if not plan:
        return "N/A"

    sub_queries = plan.get("sub_queries", [])
    if not sub_queries:
        return "N/A"

    # Phase 1b: Plan review (reuse from solve_decompose)
    print("Phase 1b: Reviewing plan...", file=sys.stderr)
    fixes = sd.review_plan(question, plan)
    if fixes:
        print(f"  Fixes needed: {fixes}", file=sys.stderr)
        feedback = (
            "Issues found with your plan:\n"
            + "\n".join(f"- {f}" for f in fixes)
            + f"\n\nRevise the plan for: {question}"
        )
        plan2 = decompose(feedback, retries=2)
        if plan2 and plan2.get("sub_queries"):
            plan = plan2
            sub_queries = plan["sub_queries"]
    else:
        print("  Plan approved", file=sys.stderr)

    # Phase 2-6: Process sub-queries in parallel
    print(f"Phases 2-6: Processing {len(sub_queries)} sub-queries...", file=sys.stderr)
    extracted = {}
    evidence = {}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_process_subquery_v2, sq): sq["id"] for sq in sub_queries}
        for future in concurrent.futures.as_completed(futures):
            try:
                sq_id, result, focused = future.result()
                extracted[sq_id] = result
                evidence[sq_id] = focused
            except Exception as e:
                sq_id = futures[future]
                extracted[sq_id] = {"values": None, "confidence": "low", "notes": str(e)}

    # Phase 7: Compute
    print("Phase 7: Computing...", file=sys.stderr)
    answer = sd.compute(plan, extracted)
    print(f"  Computed: {answer}", file=sys.stderr)

    # Phase 8: Librarian review
    print("Phase 8: Final review...", file=sys.stderr)
    answer = sd.librarian_review(question, plan, extracted, answer, evidence=evidence)
    elapsed = time.time() - t0
    print(f"  Final: {answer} ({elapsed:.1f}s)", file=sys.stderr)
    return answer


# ── Adapter: Decomposition → Orchestrator ─────────────────────────────────────


def _build_row_identifier(sq: dict) -> list[str]:
    """Build row identifier list from a decomposition sub-query.

    The decomposition gives us 'description' and 'search_terms' but not
    explicit row identifiers. We extract meaningful keywords that would
    appear as row labels in pipe-delimited Treasury tables.
    """
    identifiers = []

    desc = (sq.get("description") or "").strip()
    if desc:
        identifiers.append(desc)

    # Add search terms as secondary identifiers (row labels often match)
    for term in sq.get("search_terms", []):
        if term and term not in identifiers:
            identifiers.append(term)

    # For fiscal year data, the row label is often just the year
    data_year = sq.get("data_year")
    if data_year:
        identifiers.append(str(data_year))

    return identifiers or ["Total"]


def adapt_decomposition_to_orchestrator(decomposition_result: dict, question: str = "") -> dict:
    """Convert LLM decomposition output to Orchestrator-compatible format.

    Decomposition output (from layer_decompose.py):
        {
          "sub_queries": [
            {
              "id": "A",
              "description": "...",
              "search_terms": ["term1", "term2"],
              "target_bulletin_year": YYYY,
              "target_bulletin_months": [MM],
              "data_year": YYYY,
              "data_months": "all_12" | "specific" | "none",
              "value_type": "single" | "monthly_series" | ...,
              "period_basis": "calendar" | "fiscal" | "any"
            }
          ],
          "computation": {"type": "sum", "description": "...", "formula": "..."},
          "output_format": {"units": "millions", ...}
        }

    Orchestrator expects:
        {
          "question": "...",
          "pieces": [
            {
              "piece_id": "A",
              "year": YYYY,
              "period_type": "fiscal" | "calendar",
              "table_id": "...",
              "row_identifier": [...],
              "column_identifier": "...",
              "prefer_months": [MM],
              "value_type": "single" | "monthly_series",
              "search_terms": [...],
            }
          ],
          "computation": "sum",
          "computation_params": {"formula": "..."}
        }
    """
    pieces = []
    for sq in decomposition_result.get("sub_queries", []):
        # table_id: use first search term as best guess; the processor
        # will fall back to grep-based search if exact ID lookup fails
        search_terms = sq.get("search_terms", [])
        table_id = search_terms[0] if search_terms else ""

        # column_identifier: use column_hint if present, else derive from search terms
        column_hint = (sq.get("column_hint") or "").strip()
        if not column_hint and len(search_terms) > 1:
            # Shortest search term is usually the specific metric keyword
            column_hint = min(search_terms, key=len)

        piece = {
            "piece_id": sq["id"],
            "year": sq.get("data_year") or sq.get("target_bulletin_year"),
            "period_type": sq.get("period_basis", "calendar"),
            "table_id": table_id,
            "row_identifier": _build_row_identifier(sq),
            "column_identifier": column_hint,
            "prefer_months": sq.get("target_bulletin_months"),
            "value_type": sq.get("value_type", "single"),
            "search_terms": search_terms,
            # Pass through original sub-query for fallback processing
            "_original_subquery": sq,
        }
        pieces.append(piece)

    computation = decomposition_result.get("computation", {})
    return {
        "question": question,
        "pieces": pieces,
        "computation": computation.get("type", "direct"),
        "computation_params": {
            "formula": computation.get("formula"),
            "description": computation.get("description"),
        },
    }


def _process_piece_with_search(piece: dict) -> dict:
    """Process a single orchestrator piece using grep-based search + consensus extraction.

    This is the bridge between the Orchestrator's piece format and the existing
    search infrastructure (sd.search_tables, sd.search_data, extract_value_consensus).

    Strategy:
      1. Try SearchTask's structured approach (table_id exact lookup) if table_id
         looks like a real table ID (e.g., "FFO-1")
      2. Fall back to grep-based search (sd.search_tables + sd.search_data) using
         the original sub-query, which works with free-text search terms
    """
    piece_id = piece.get("piece_id")
    sq = piece.get("_original_subquery", {})
    value_type = piece.get("value_type", "single")

    # Reconstruct a sub-query dict compatible with sd.search_tables / sd.search_data
    # by merging orchestrator piece fields back with the original sub-query
    search_sq = dict(sq)  # start from original
    search_sq.setdefault("id", piece_id)
    search_sq.setdefault("data_year", piece.get("year"))
    if piece.get("search_terms"):
        search_sq["search_terms"] = piece["search_terms"]
    if piece.get("column_identifier"):
        search_sq.setdefault("column_hint", piece["column_identifier"])

    # --- Attempt 1: Try SearchTask structured approach ---
    table_id = piece.get("table_id", "")
    # Only try exact table ID lookup if it looks like a real ID (contains dash or digits)
    if table_id and (
        re.search(r"[A-Z]+-\d+", table_id.upper())
        or re.search(r"^[A-Z]{2,4}-?\d+", table_id.upper())
    ):
        try:
            st_result = _search_task.execute(piece, verbose=False)
            if st_result.get("success") and st_result.get("value") is not None:
                print(f"  [{piece_id}] SearchTask succeeded: {st_result['value']}", file=sys.stderr)
                return {
                    "value": st_result["value"],
                    "confidence": st_result.get("confidence", 0.8),
                    "success": True,
                    "source": st_result.get("source"),
                    "method": "search_task",
                }
        except Exception as e:
            print(f"  [{piece_id}] SearchTask failed: {e}", file=sys.stderr)

    # --- Attempt 2: Grep-based search with consensus extraction ---
    print(f"  [{piece_id}] Using grep-based search...", file=sys.stderr)

    if value_type in ("monthly_series", "series"):
        _, extraction, focused = _process_monthly_series(search_sq)
    else:
        _, extraction, focused = _process_single_value(search_sq)

    values = extraction.get("values")
    conf_str = extraction.get("confidence", "low")
    conf_float = {"high": 0.9, "medium": 0.65, "low": 0.3}.get(conf_str, 0.5)

    if values is not None:
        # For monthly series that need summing, the Orchestrator's "sum" computation
        # will handle it if the computation type is "sum". But if the piece itself
        # is a monthly_series, return the list so Orchestrator can flatten+sum.
        return {
            "value": values,
            "confidence": conf_float,
            "success": True,
            "source": focused[:200] if focused else None,
            "method": "grep_consensus",
            "notes": extraction.get("notes", ""),
        }
    else:
        return {
            "value": None,
            "confidence": 0.0,
            "success": False,
            "error": extraction.get("notes", "No value found"),
            "method": "grep_consensus",
        }


def _format_orchestrator_answer(orch_result: dict, plan: dict) -> str:
    """Format the Orchestrator's numeric result into a presentable answer string.

    Handles rounding, units, and percent formatting based on the decomposition's
    output_format specification.
    """
    value = orch_result.get("final_value")
    if value is None:
        return "N/A"

    output_fmt = plan.get("output_format", {})
    units = output_fmt.get("units", "")
    rounding = output_fmt.get("rounding")
    suffix = output_fmt.get("suffix", "")

    # Apply rounding
    if rounding is not None:
        try:
            value = round(value, int(rounding))
        except (ValueError, TypeError):
            pass

    # Format number
    if isinstance(value, float) and value == int(value) and rounding in (None, 0):
        formatted = f"{int(value):,}"
    elif isinstance(value, float):
        # Use rounding precision for decimal places
        if rounding is not None:
            formatted = f"{value:,.{int(rounding)}f}"
        else:
            formatted = f"{value:,}"
    else:
        formatted = f"{value:,}"

    if suffix:
        formatted = f"{formatted}{suffix}"

    return formatted


def solve_with_orchestrator(question: str) -> str:
    """Full pipeline using Orchestrator: decompose -> adapt -> orchestrate -> review.

    This is the new path that uses the Wave 2 architecture (SearchTask + Orchestrator)
    while keeping the existing grep-based search as a fallback processor.
    """
    from orchestrator import SearchTaskOrchestrator

    t0 = time.time()

    # Phase 1: Decompose (same as solve())
    plan = decompose(question)
    if not plan:
        return "N/A"

    sub_queries = plan.get("sub_queries", [])
    if not sub_queries:
        return "N/A"

    # Phase 1b: Plan review
    print("Phase 1b: Reviewing plan...", file=sys.stderr)
    fixes = sd.review_plan(question, plan)
    if fixes:
        print(f"  Fixes needed: {fixes}", file=sys.stderr)
        feedback = (
            "Issues found with your plan:\n"
            + "\n".join(f"- {f}" for f in fixes)
            + f"\n\nRevise the plan for: {question}"
        )
        plan2 = decompose(feedback, retries=2)
        if plan2 and plan2.get("sub_queries"):
            plan = plan2
            sub_queries = plan["sub_queries"]
    else:
        print("  Plan approved", file=sys.stderr)

    # Phase 2: Adapt decomposition to orchestrator format
    print("Phase 2: Adapting decomposition for orchestrator...", file=sys.stderr)
    orch_input = adapt_decomposition_to_orchestrator(plan, question)
    n_pieces = len(orch_input["pieces"])
    print(f"  -> {n_pieces} pieces, computation={orch_input['computation']}", file=sys.stderr)

    # Phase 3: Execute via Orchestrator
    print(f"Phase 3: Orchestrator executing {n_pieces} pieces...", file=sys.stderr)
    # Budget: 600s total (arena configurable). Decompose ~10s, review ~10s = 580s for pieces.
    # With 4 parallel workers, each piece gets up to 180s.
    orch = SearchTaskOrchestrator(
        search_task_processor=_process_piece_with_search,
        max_workers=4,
        timeout_per_piece=180,
    )
    orch_result = orch.execute(orch_input)

    if not orch_result.get("success"):
        print(f"  Orchestrator failed: {orch_result.get('error')}", file=sys.stderr)
        print("  Falling back to legacy solve()...", file=sys.stderr)
        return solve(question)

    orch_conf = orch_result.get("final_confidence", 0.0)
    print(
        f"  Orchestrator result: {orch_result['final_value']} (confidence={orch_conf:.2f})",
        file=sys.stderr,
    )

    # Note: low-confidence fallback to legacy disabled for arena (300s budget).
    # Orchestrator answer, even at low confidence, beats timeout/no answer.
    # To enable fallback locally: set ENABLE_LEGACY_FALLBACK=1
    if orch_conf < 0.7 and os.environ.get("ENABLE_LEGACY_FALLBACK"):
        print(
            f"  Low confidence ({orch_conf:.2f}), running legacy for comparison...", file=sys.stderr
        )
        legacy_answer = solve(question)
        if legacy_answer and legacy_answer != "N/A":
            print(
                f"  Legacy answer: {legacy_answer} — using legacy (higher confidence path)",
                file=sys.stderr,
            )
            return legacy_answer

    # Phase 4: Format answer
    answer = _format_orchestrator_answer(orch_result, plan)
    print(f"  Formatted: {answer}", file=sys.stderr)

    # Phase 5: Librarian review using orchestrator evidence
    print("Phase 5: Final review...", file=sys.stderr)
    # Build extracted dict compatible with librarian_review
    extracted = {}
    evidence = {}
    for pr in orch_result.get("piece_results", []):
        pid = pr.get("piece_id")
        extracted[pid] = {
            "values": pr.get("value"),
            "confidence": "high"
            if pr.get("confidence", 0) >= 0.8
            else ("medium" if pr.get("confidence", 0) >= 0.5 else "low"),
            "notes": pr.get("notes", ""),
        }
        evidence[pid] = str(pr.get("source", ""))[:500]

    answer = sd.librarian_review(question, plan, extracted, answer, evidence=evidence)
    elapsed = time.time() - t0
    print(f"  Final: {answer} ({elapsed:.1f}s)", file=sys.stderr)
    return answer


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python3 solve_v2.py "QUESTION"', file=sys.stderr)
        print('       python3 solve_v2.py --orchestrator "QUESTION"', file=sys.stderr)
        sys.exit(1)

    use_orchestrator = False
    question = sys.argv[1]

    if sys.argv[1] == "--orchestrator":
        use_orchestrator = True
        if len(sys.argv) < 3:
            print('Usage: python3 solve_v2.py --orchestrator "QUESTION"', file=sys.stderr)
            sys.exit(1)
        question = sys.argv[2]

    if use_orchestrator:
        print("=== Using Orchestrator path ===", file=sys.stderr)
        answer = solve_with_orchestrator(question)
    else:
        print("=== Using legacy path ===", file=sys.stderr)
        answer = solve(question)

    print(f"\nANSWER: {answer}")

    if answer and answer != "N/A":
        try:
            Path(ANSWER_PATH).write_text(answer)
        except Exception:
            pass
