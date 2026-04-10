#!/usr/bin/env python3
"""
OfficeQA Solver — structured pipeline:
  1. LLM decompose: question → QuestionSpec (per-value data_requests + compute template)
  2. Deterministic: per-request retrieve → union files
  3. LLM extract: grounded per-request value extraction
  4. Python compute: execute template against extracted values
"""

import contextlib
import json
import os
import sqlite3
import sys
import threading
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from compute import ComputeError, format_result, validate_extractions
from compute import execute as compute_execute
from extract import extract_structured
from find import resolve_cells
from retrieve_v2 import retrieve as retrieve_v2
from scout import scout
from verify import verify_answer

sys.stdout.reconfigure(line_buffering=True)
load_dotenv()

MODEL = os.getenv("OFFICEQA_MODEL", "deepseek/deepseek-chat")

client = OpenAI(
    api_key=os.getenv("DEDALUS_API_KEY"),
    base_url=os.getenv("DEDALUS_API_BASE"),
)

# ── Ledger connection for deterministic fast-path ───────────────────────────

_LEDGER_PATH = Path(__file__).parent / "ledger.sqlite"
_FP_TLS = threading.local()


def _fp_conn() -> sqlite3.Connection:
    """Thread-local ledger connection for fast-path queries."""
    conn = getattr(_FP_TLS, "conn", None)
    if conn is None:
        if not _LEDGER_PATH.exists():
            raise FileNotFoundError(f"{_LEDGER_PATH} not found — run build_ledger.py")
        conn = sqlite3.connect(str(_LEDGER_PATH))
        conn.row_factory = sqlite3.Row
        _FP_TLS.conn = conn
    return conn


def _get_table_id_from_entry(entry: dict) -> int | None:
    """Look up the ledger table id from a retrieve_v2 entry's file + element_seq."""
    try:
        row = (
            _fp_conn()
            .execute(
                "SELECT id FROM tables WHERE file = ? AND element_seq = ? LIMIT 1",
                (entry.get("file"), entry.get("element_seq")),
            )
            .fetchone()
        )
    except sqlite3.OperationalError:
        return None
    return row["id"] if row else None


def _build_cells_for_dr(dr: dict, table_id: int) -> list[dict] | None:
    """Build cell specs for resolve_cells() from a data_request + table_id.

    Returns a list of {row_leaf, col_leaf, name} dicts, or None if the
    table structure can't support deterministic resolution for this DR.
    """
    conn = _fp_conn()
    row_hint = dr.get("row_hint", "")
    column_hint = dr.get("column_hint", "")
    granularity = dr.get("granularity", "annual")
    years = dr.get("years") or []
    expected_count = dr.get("expected_count")

    # ── Annual / single-value lookups ──────────────────────────────────
    if granularity in ("annual", "specific_month", "unknown", None) and (
        expected_count is None or expected_count == 1
    ):
        col_leaf = column_hint or ""
        # If column_hint looks like a year, find the actual col_leaf
        if column_hint and column_hint.lstrip("-").isdigit():
            year = int(column_hint)
            col = conn.execute(
                "SELECT col_leaf FROM table_columns WHERE table_id=? AND year_extracted=? LIMIT 1",
                (table_id, year),
            ).fetchone()
            if col:
                col_leaf = col["col_leaf"]
        elif not column_hint and years:
            # No column hint but we have a year — find a matching column
            year = years[0]
            col = conn.execute(
                "SELECT col_leaf FROM table_columns WHERE table_id=? AND year_extracted=? LIMIT 1",
                (table_id, year),
            ).fetchone()
            if col:
                col_leaf = col["col_leaf"]
        return [{"row_leaf": row_hint, "col_leaf": col_leaf, "name": dr["id"]}]

    # ── Monthly all (12 values) ────────────────────────────────────────
    if granularity == "monthly_all":
        target_year = years[0] if years else None
        if target_year is None:
            return None

        # Option A: months as columns (e.g. col_leaf = "Jan.", "Feb.")
        month_cols = conn.execute(
            "SELECT col_leaf, month_extracted FROM table_columns "
            "WHERE table_id=? AND year_extracted=? AND month_extracted IS NOT NULL "
            "ORDER BY month_extracted",
            (table_id, target_year),
        ).fetchall()

        if len(month_cols) >= 10:
            cells = []
            for col in month_cols:
                name = f"m{col['month_extracted']:02d}"
                cells.append({"row_leaf": row_hint, "col_leaf": col["col_leaf"], "name": name})
            return cells

        # Option B: months as rows (e.g. row_leaf = "1940-January")
        month_rows = conn.execute(
            "SELECT row_leaf, month_extracted FROM table_rows "
            "WHERE table_id=? AND year_extracted=? AND month_extracted IS NOT NULL "
            "ORDER BY month_extracted",
            (table_id, target_year),
        ).fetchall()

        if len(month_rows) >= 10:
            col_leaf = column_hint or ""
            if not col_leaf:
                # Try to find a reasonable column (e.g., "Total")
                col = conn.execute(
                    "SELECT col_leaf FROM table_columns WHERE table_id=? LIMIT 1 OFFSET 1",
                    (table_id,),
                ).fetchone()
                if col:
                    col_leaf = col["col_leaf"]
            cells = []
            for row in month_rows:
                name = f"m{row['month_extracted']:02d}"
                cells.append({"row_leaf": row["row_leaf"], "col_leaf": col_leaf, "name": name})
            return cells

        # Can't determine monthly layout
        return None

    # ── Multi-year annual (one value per year) ──────────────────────────
    if granularity == "multi_year_annual" and years:
        cells = []
        for y in years:
            col = conn.execute(
                "SELECT col_leaf FROM table_columns WHERE table_id=? AND year_extracted=? LIMIT 1",
                (table_id, y),
            ).fetchone()
            col_leaf = col["col_leaf"] if col else str(y)
            cells.append({"row_leaf": row_hint, "col_leaf": col_leaf, "name": f"y{y}"})
        return cells

    # Unsupported granularity for deterministic resolution
    return None


def _try_deterministic_fast_path(
    spec: dict, per_dr_entries: dict, verbose: bool = False
) -> dict | None:
    """Try to resolve all data_requests deterministically via resolve_cells().

    Returns an extractions dict matching extract_structured() output if ALL
    data_requests resolve successfully, or None if any DR can't be resolved.
    The caller should fall back to normal LLM extraction when this returns None.
    """
    data_requests = spec.get("data_requests") or []
    if not data_requests:
        return None

    extractions: dict[str, dict] = {}
    notes_parts: list[str] = []

    for dr in data_requests:
        dr_id = dr.get("id", "?")
        source = dr.get("source", "corpus")

        # External/CPI/FX sources can't be resolved from the ledger
        if source in ("external", "cpi", "fx"):
            if verbose:
                print(f"  Fast-path: {dr_id} skipped (source={source})")
            return None

        # Get the first (best) retrieved entry for this DR
        entries = per_dr_entries.get(dr_id) or []
        if not entries:
            if verbose:
                print(f"  Fast-path: {dr_id} no retrieved entries")
            return None

        # Find a table entry (skip prose/footnote entries which have 'content')
        table_entry = None
        for e in entries:
            if e.get("content"):  # prose/footnote — skip
                continue
            if e.get("element_seq") is not None or e.get("html"):
                table_entry = e
                break

        if table_entry is None:
            if verbose:
                print(f"  Fast-path: {dr_id} no table entries")
            return None

        # Look up table_id from the entry
        table_id = _get_table_id_from_entry(table_entry)
        if table_id is None:
            if verbose:
                print(f"  Fast-path: {dr_id} table_id not found")
            return None

        # Build cell specs for resolve_cells()
        cells = _build_cells_for_dr(dr, table_id)
        if cells is None:
            if verbose:
                print(
                    f"  Fast-path: {dr_id} can't build cells (granularity={dr.get('granularity')})"
                )
            return None

        # Call resolve_cells
        try:
            resolved = resolve_cells(table_id, cells)
        except Exception as e:
            if verbose:
                print(f"  Fast-path: {dr_id} resolve_cells error: {e}")
            return None

        values = resolved.get("values", {})

        # Check if all values resolved (non-None)
        unresolved = [k for k, v in values.items() if v is None]
        if unresolved:
            if verbose:
                print(f"  Fast-path: {dr_id} unresolved cells: {unresolved}")
            return None

        # Convert resolved values to extraction format
        granularity = dr.get("granularity", "annual")
        expected_count = dr.get("expected_count") or 1

        if granularity == "monthly_all" and expected_count == 12:
            ordered_values: list[float | None] = []
            ordered_labels: list[str] = []
            for m in range(1, 13):
                name = f"m{m:02d}"
                if name in values and values[name] is not None:
                    ordered_values.append(values[name])
                    ordered_labels.append(f"month {m}")
                else:
                    # Missing month — fast-path fails
                    if verbose:
                        print(f"  Fast-path: {dr_id} incomplete monthly ({len(ordered_values)}/12)")
                    return None
        elif granularity == "multi_year_annual":
            dr_years = dr.get("years") or []
            ordered_values = []
            ordered_labels = []
            for y in dr_years:
                name = f"y{y}"
                if name in values and values[name] is not None:
                    ordered_values.append(values[name])
                    ordered_labels.append(str(y))
                else:
                    if verbose:
                        print(f"  Fast-path: {dr_id} incomplete multi-year values")
                    return None
        else:
            # Single value
            val = values.get(dr_id)
            if val is None:
                if verbose:
                    print(f"  Fast-path: {dr_id} single value is None")
                return None
            ordered_values = [val]
            ordered_labels = [dr.get("row_hint", "")]

        extractions[dr_id] = {
            "values": ordered_values,
            "labels": ordered_labels,
            "source_file": table_entry.get("file", ""),
            "confidence": "deterministic",
        }
        notes_parts.append(f"{dr_id}: resolved deterministically from table {table_id}")

    if verbose:
        n = len(data_requests)
        print(f"  Fast-path: ALL {n} DR{'' if n == 1 else 's'} resolved deterministically ✓")

    return {
        "extractions": extractions,
        "notes": "; ".join(notes_parts),
    }


def llm(system: str, user: str, max_tokens: int = 1000, temperature: float = 0.0) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


def parse_json(raw: str) -> dict | None:
    import re

    raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            return json.loads(raw[: raw.rfind("}") + 1])
        except Exception:
            return None


# ═══════════════════════════════════════════════════════════════════════════════
# LLM CALL 1: STRUCTURED DECOMPOSE → QuestionSpec
# ═══════════════════════════════════════════════════════════════════════════════

DECOMPOSE_SYSTEM = """You decompose U.S. Treasury Bulletin questions into a
structured QuestionSpec that a deterministic pipeline can execute. List EXACTLY
what values are needed, EXACTLY how to compute them (as a Python template), and
EXACTLY how to format the answer. Python runs the computation — you just specify
it.

═══════════════════════════════════════════════════════════════════════════════
THE CORPUS
═══════════════════════════════════════════════════════════════════════════════
U.S. Treasury Bulletins 1930s–2025. One file per issue: treasury_bulletin_YYYY_MM.txt.
  - Published MONTHLY through 1982 (12 issues/year)
  - Published QUARTERLY from 1983 (4 issues/year — typically Mar, Jun, Sep, Dec)

Tables are pipe-delimited: | row_label | val1 | val2 | ...
  - Annual rows:  | 1940 | 1580 |
  - Monthly rows: | 1940-January | 132 |  | 1940-February | 140 | ...

Common table categories:
  - Budget Receipts and Outlays (by function: National defense, Veterans, Net interest, …)
  - Federal Debt (public debt, interest-bearing, marketable securities)
  - International Capital (claims, liabilities, by country)
  - Currency in Circulation (by denomination)
  - Foreign Currency Positions (by currency: JPY, GBP, INR, DEM, CAD)
  - Savings Bonds sales (by series, by state)
  - Treasury auctions (notes, bills, bonds — by maturity date)

═══════════════════════════════════════════════════════════════════════════════
YEARS  (data year, not bulletin year)
═══════════════════════════════════════════════════════════════════════════════
Set `years` to the year(s) the DATA covers — what the question is asking about.
Retrieval handles finding which bulletin files contain that data, so you do NOT
need to guess publication dates, retrospective offsets, or which issue to read.
Just report the data year honestly.

The only exception: if the question phrases the year in terms of an event whose
reporting date differs from the event date (e.g. "2-year notes MATURING July
1984" describes an auction that was held when the notes were ISSUED, two years
earlier), set `years` to the year the underlying event/data actually occurred
and explain the inference in `resolution_notes`.

═══════════════════════════════════════════════════════════════════════════════
FISCAL YEAR vs CALENDAR YEAR
═══════════════════════════════════════════════════════════════════════════════
  CY YYYY = Jan 1 YYYY – Dec 31 YYYY (12 monthly rows)
  FY YYYY (post-1976) = Oct 1 YYYY−1 – Sep 30 YYYY
  FY YYYY (pre-1977)  = Jul 1 YYYY−1 – Jun 30 YYYY
  Transition Quarter (TQ) = Jul–Sep 1976; FY1977 starts Oct 1 1976.

The same metric for CY YYYY vs FY YYYY can differ substantially — the periods
overlap but don't match. Pick the right one based on how the question phrases it.

  "calendar year" / "CY" / "the year YYYY" (bare):
      → granularity "monthly_all", expected_count 12 (sum 12 monthly rows)
  "fiscal year" / "FY":
      → granularity "annual", expected_count 1 (use the annual summary row)

═══════════════════════════════════════════════════════════════════════════════
ROW LABELS
═══════════════════════════════════════════════════════════════════════════════
Tables evolve: agencies get renamed, line items split or combined, categories
restructured across decades. Set `row_hint` to your best single guess at the
exact label as it would have appeared in a bulletin of the relevant era. If you
know the era plausibly has 2–3 different names for the same concept (e.g. due
to a rename, a consolidation, or a naming-convention shift), list those in
`row_hint_alternatives` and retrieval will try them all.

Label hygiene:
  - "Total X" rows already include their sub-items — NEVER ask for "Total X"
    when you want a specific child line.
  - Footnote markers (1/, 2/, r/, p/), "(*)" / "-", and parentheses-for-negative
    appear in values — extraction strips them, so don't worry about them in
    row_hint.
  - row_hint is for the exact label string. Filter descriptions ("all rows
    except …") belong in cohort mode only.

═══════════════════════════════════════════════════════════════════════════════
EXTERNAL DATA SOURCES
═══════════════════════════════════════════════════════════════════════════════
Some questions need values that are NOT in the Treasury Bulletin corpus. Mark
the data_request with `source` so the pipeline routes it to the right fetcher:

  source: "corpus"  (default) — Treasury Bulletin text files
  source: "cpi"                — BLS CPI-U annual average (base 1982-84=100)
                                 Use for inflation adjustment.
                                 real = nominal × (target_cpi / source_cpi)
  source: "fx"                 — Foreign exchange rate at a specific date
                                 Pairs: USD/JPY, USD/GBP, USD/INR, USD/DEM, USD/CAD
  source: "external"           — Anything else (Macrotrends, historical refs)

For CPI/FX, set `years` to the target year(s), leave `row_hint`/`column_hint` blank,
and name the DR something readable (e.g. id "cpi_1940", label "CPI-U annual 1940").

═══════════════════════════════════════════════════════════════════════════════
GRANULARITY ENUM (months are integers 1–12)
═══════════════════════════════════════════════════════════════════════════════
  "annual"             — single annual row for one year, expected_count 1
  "monthly_all"        — 12 monthly rows Jan–Dec for one year, expected_count 12
  "monthly_range"      — months [start_month..end_month] within a SINGLE year,
                         expected_count = end_month − start_month + 1
  "continuous_monthly" — CONTINUOUS span across multiple years, from
                         (start_year, start_month) to (end_year, end_month)
                         INCLUSIVE. Use for phrases like "March 1942 to October
                         1948" or "Jan 1984 through Mar 1987". Set `years` to
                         the list of years covered. expected_count = total
                         months in the span.
  "multi_year_annual"  — one annual row per year in `years`, expected_count len(years)
  "specific_month"     — single named month row, expected_count 1

All month fields use INTEGERS 1–12 (1=Jan, 12=Dec), never month names.
The single set of span fields is: start_year, start_month, end_year, end_month.

═══════════════════════════════════════════════════════════════════════════════
COHORT SELECTION (for max/min/argmax over many rows)
═══════════════════════════════════════════════════════════════════════════════
For "highest-spending department", "country with the largest claims", etc. —
you need values from many rows in the same table, not a single value.

  cohort: true
  expected_count: null
  row_hint: a concise description of the cohort, stating what to INCLUDE and
            what to EXCLUDE (e.g. "all federal department rows, excluding Total
            and All other")

Extraction returns a flat list in `values['<id>']` (same shape as any other DR),
so the template is just `max(values['v1'])` or `min(values['v1'])`. For an
argmax question ("which country had the largest…"), set output_format.type
= "string" and the template can use `values['v1_labels'][values['v1'].index(max(values['v1']))]`.

PREFER SIMPLE DECOMPOSITION over cohort when possible. If the answer is a ratio
A/B (e.g. weighted average = total value / total count), request two separate
DRs for A and B and divide in the template. Do NOT ask for a cohort of
denomination rows and reconstruct the math by parsing row labels.

═══════════════════════════════════════════════════════════════════════════════
CROSS-REFERENCE QUESTIONS
═══════════════════════════════════════════════════════════════════════════════
If the question requires resolving fact A before looking up fact B (e.g. "find
which bureau did X, then use that bureau's report…"), resolve A yourself using
general knowledge and state the resolved fact in `resolution_notes` at the top
level of the spec. Then build data_requests for fact B using the resolved
answer. Do not try to encode the resolution as a corpus lookup.

═══════════════════════════════════════════════════════════════════════════════
UNITS
═══════════════════════════════════════════════════════════════════════════════
Tables declare their scale in their headers: "In thousands", "In millions",
"In billions", "Percent", "Index", etc. Set `output_format.unit` to the unit
the question asks for:
  "millions", "billions", "thousands" — dollar-scale outputs
  "percent"                           — percentage outputs
  null                                — raw dollars, counts, ratios, or when
                                        the question does not specify
Extraction returns values at the source table's native scale; the formatter
converts to output_format.unit at the end. You do NOT need to multiply or
divide inside python_template to reconcile units.

═══════════════════════════════════════════════════════════════════════════════
COMPUTATION  (name the operation — a calculator executes it)
═══════════════════════════════════════════════════════════════════════════════
You do NOT write Python. A separate calculator component takes the values the
pipeline extracts and runs the operation you name here. Your job is to pick the
right operation and list the right data_requests.

Operation enum:
  direct                single-value lookup, result = v1[0]
  sum                   sum of a list
  difference            v2[0] − v1[0]
  abs_difference        abs(v2[0] − v1[0])
  percent_change        (v2[0] − v1[0]) / v1[0] × 100
  abs_percent_change    abs((v2[0] − v1[0]) / v1[0]) × 100
  average               arithmetic mean
  geometric_mean        nth root of the product
  cagr                  compound annual growth rate (%)
  max                   max of a list (use with cohort DR)
  min                   min of a list (use with cohort DR)
  argmax                label of the row with the max value (cohort mode)
  argmin                label of the row with the min value (cohort mode)
  stdev_sample          sample stdev (n−1)
  stdev_population      population stdev (n)
  pearson_correlation   Pearson r between two lists
  linear_regression     OLS slope+intercept; x-axis is `years` by default
  box_cox               (x**lam − 1) / lam (lam in computation_params)
  cpi_adjust            nominal × (target_cpi / base_cpi)
  fx_convert            amount × fx_rate
  kl_divergence         KL(P || Q) between two probability lists
  theil_index           Theil inequality index
  gini                  Gini coefficient
  kurtosis              Fisher kurtosis
  skewness              Fisher-Pearson skewness
  var_parametric        parametric Value at Risk
  ratio                 v1[0] / v2[0]  (use for weighted-average-style A/B)
  custom                anything not in this list — describe it in
                        `computation_notes`

For operations that need parameters, put them in `computation_params`, e.g.
  box_cox:         {"lambda": 0.75}
  var_parametric:  {"confidence": 0.95}
  linear_regression: {"x": "years"}  (default if omitted)

When the question asks for "absolute X", use the abs_* variant.
When it asks for values rounded to N places, set output_format.rounding = N —
the formatter handles rounding, not the calculator.

═══════════════════════════════════════════════════════════════════════════════
MULTI-VALUE ANSWERS
═══════════════════════════════════════════════════════════════════════════════
If the question asks for a list like [slope, intercept] or [val1, val2, val3]:
  output_format.type = "list"
  python_template must end with: result = [a, b, ...]

═══════════════════════════════════════════════════════════════════════════════
OUTPUT — return ONLY this JSON schema, no prose
═══════════════════════════════════════════════════════════════════════════════
{
  "computation": "direct|sum|difference|abs_difference|percent_change|abs_percent_change|average|geometric_mean|cagr|max|min|stdev_sample|stdev_population|linear_regression|pearson_correlation|box_cox|cpi_adjust|fx_convert|custom",
  "data_requests": [
    {
      "id": "v1",
      "label": "<what this value is>",
      "source": "corpus|cpi|fx|external",
      "row_hint": "<exact row label from the corpus>",
      "column_hint": "<column header or year>",
      "years": [1940],
      "granularity": "annual|monthly_all|monthly_range|continuous_monthly|multi_year_annual|specific_month",
      "start_year": null,
      "start_month": null,
      "end_year": null,
      "end_month": null,
      "expected_count": 12,
      "cohort": false
    }
  ],
  "computation_spec": {
    "python_template": "result = sum(values['v1'])"
  },
  "output_format": {
    "type": "number|list|string",
    "unit": "millions|billions|null",
    "rounding": null,
    "as_percent": false
  },
  "resolution_notes": null,
  "question_kind": "value|page_number|date|text"
}"""


def decompose(question: str, scout_hint: str = "", feedback: str = "") -> dict | None:
    """Produce a typed QuestionSpec that binds extraction + computation.

    `scout_hint` is the pre-decompose index peek (see scout.py). `feedback`
    is a message from a prior failed attempt (empty retrieve, verify issue).
    """
    user_parts: list[str] = []
    if scout_hint:
        user_parts.append(scout_hint)
        user_parts.append("")
    user_parts.append(f"QUESTION: {question}")
    if feedback:
        user_parts.append("")
        user_parts.append(
            f"REVIEW FEEDBACK from previous attempt — fix this specifically:\n{feedback}"
        )
    user_msg = "\n".join(user_parts)

    raw = llm(DECOMPOSE_SYSTEM, user_msg, max_tokens=1500)
    spec = parse_json(raw)

    def _is_valid(s: dict | None) -> bool:
        if s is None:
            return False
        if "data_requests" not in s or "computation_spec" not in s:
            return False
        if not isinstance(s["data_requests"], list) or not s["data_requests"]:
            return False
        return "python_template" in s.get("computation_spec", {})

    if _is_valid(spec):
        return spec

    # Fallback when the LLM failed to produce a usable spec — don't drop
    # the question, fall through to a minimal tokenized plan that the
    # retriever can still score. Downstream extract/compute will treat
    # the retrieved tables directly; at worst we degrade to the tokenized
    # baseline, which is still better than returning None.
    import re as _re

    years = sorted({int(y) for y in _re.findall(r"\b(1[89]\d{2}|20[0-3]\d)\b", question)})
    return {
        "computation": "direct",
        "data_requests": [
            {
                "id": "v1",
                "label": question,
                "source": "corpus",
                "row_hint": "",
                "column_hint": "",
                "years": years,
                "target_years": years,
                "granularity": "unknown",
                "expected_count": None,
                "cohort": False,
            }
        ],
        "computation_spec": {"python_template": "result = values['v1']"},
        "output_format": {"type": "number", "unit": None, "rounding": None, "as_percent": False},
        "question_kind": "value",
        "resolution_notes": "decompose fallback (LLM failed)",
        "cpi_needed": False,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def retrieve_for_spec(
    spec: dict,
    question: str,
    verbose: bool = False,
    top_k_per_dr: int = 10,
) -> dict:
    """Per-DR retrieval against the table-level index.

    Returns {dr_id: [retrieve_v2 entries]}. Each entry carries its table HTML
    inline so extract can render it directly — no .txt re-parsing. We wrap
    each DR in its own single-DR plan because retrieve_v2 only honours the
    first data_request.
    """
    per_dr: dict[str, list[dict]] = {}
    data_requests = spec.get("data_requests") or []
    for dr in data_requests:
        dr_id = dr.get("id", "?")
        mini_plan = {"data_requests": [dr]}
        entries = retrieve_v2(
            mini_plan,
            question,
            top_k=top_k_per_dr,
            verbose=verbose,
            dedupe_by_file=True,
        )
        per_dr[dr_id] = entries
    return per_dr


def _total_entries(per_dr: dict) -> int:
    return sum(len(v) for v in per_dr.values())


def _run_extract_and_compute(
    spec: dict,
    per_dr_entries: dict,
    question: str,
    verbose: bool,
    feedback: str = "",
) -> tuple[str, dict | None]:
    """Run extract → validate → compute for one attempt. Returns
    (formatted_answer_or_error, extraction_dict_or_None). The extraction is
    returned so the caller can hand it to verify.

    Tries the deterministic fast-path first: resolve_cells() from find.py
    attempts to look up values directly from the ledger without LLM. If ALL
    data_requests resolve, the LLM extract step is skipped entirely. If ANY
    data_request can't be resolved, falls back to normal LLM extraction.
    """
    # ── Deterministic fast-path ─────────────────────────────────────────
    fast_extraction = _try_deterministic_fast_path(spec, per_dr_entries, verbose=verbose)
    if fast_extraction is not None:
        # All DRs resolved deterministically — skip LLM extraction
        if verbose:
            print("  Fast-path succeeded — skipping LLM extraction")
        extraction = fast_extraction
    else:
        # Fast-path missed — fall back to LLM extraction
        if verbose:
            print("  Fast-path missed — falling back to LLM extraction")
        extraction = extract_structured(
            spec,
            per_dr_entries,
            question,
            verbose=verbose,
            feedback=feedback,
        )

    if extraction is None or "extractions" not in extraction:
        return "EXTRACT_FAILED", None

    warnings = validate_extractions(spec, extraction["extractions"])
    if warnings and verbose:
        print(f"  Extraction warnings: {warnings}")

    for dr in spec.get("data_requests", []):
        vid = dr["id"]
        vals = extraction["extractions"].get(vid, {}).get("values") or []
        if not [v for v in vals if v is not None]:
            return f"NO_VALUES[{vid}]", extraction

    try:
        result = compute_execute(spec, extraction["extractions"], verbose=verbose)
    except ComputeError as e:
        return f"COMPUTE_FAILED: {e}", extraction

    formatted = format_result(result, spec.get("output_format") or {})
    return formatted, extraction


def solve(question: str, verbose: bool = False, use_verify: bool = True) -> str:
    """Structured pipeline with feedback loops:

      scout → decompose → retrieve → extract → compute → verify

    One bounce-back per failure mode:
      - empty retrieve         → re-decompose with "no files matched" hint
      - verify flags extract   → re-extract with the issue as feedback
      - verify flags decompose → re-decompose + retrieve + extract again

    Each phase still fails loudly with an explicit error string so eval
    output tells us exactly which stage broke.
    """
    # Phase 0: scout (deterministic, cheap, grounds decompose)
    hint = scout(question)
    if verbose and hint:
        print(f"  Scout hint:\n{hint[:400]}")

    # Phase 1: decompose (with one retry on empty retrieve)
    spec = decompose(question, scout_hint=hint)
    if spec is None:
        return "DECOMPOSE_FAILED"
    if verbose:
        print(f"  Spec: {json.dumps(spec, indent=2)[:800]}")

    # Phase 2: retrieve — bounce back to decompose if no tables match
    per_dr = retrieve_for_spec(spec, question, verbose=verbose)
    total = _total_entries(per_dr)
    if verbose:
        print(f"  Retrieve: {total} table entries across {len(per_dr)} data_requests")
    if total == 0:
        if verbose:
            print("  No tables matched — retrying decompose with feedback")
        spec = decompose(
            question,
            scout_hint=hint,
            feedback=(
                "Previous decompose produced a spec whose retrieve returned "
                "zero matching tables. Reconsider row_hint, column_hint, and "
                "years. Use the scout candidates above as a guide."
            ),
        )
        if spec is None:
            return "DECOMPOSE_FAILED"
        per_dr = retrieve_for_spec(spec, question, verbose=verbose)
        if _total_entries(per_dr) == 0:
            return "RETRIEVE_EMPTY"

    # Phase 3+4: extract → compute (first attempt)
    answer, extraction = _run_extract_and_compute(spec, per_dr, question, verbose)

    # Bounce-back: if extract came back with nothing, retry extract once with
    # the missing-value ids as feedback, then if still empty, re-decompose.
    if extraction is not None and answer.startswith("NO_VALUES"):
        missing_feedback = (
            f"Previous extract returned empty values for {answer}. The tables "
            "provided may not contain the requested rows. Re-check column and "
            "row labels carefully against the context, and return null only "
            "if the value genuinely isn't present."
        )
        if verbose:
            print(f"  {answer} — retrying extract with feedback")
        answer, extraction = _run_extract_and_compute(
            spec,
            per_dr,
            question,
            verbose,
            feedback=missing_feedback,
        )
        if extraction is not None and answer.startswith("NO_VALUES"):
            if verbose:
                print(f"  {answer} — retrying decompose")
            spec2 = decompose(
                question,
                scout_hint=hint,
                feedback=(
                    f"Previous spec produced empty extractions ({answer}). "
                    "The row_hint or column_hint likely did not match the "
                    "corpus tables. Use the scout hint above to ground them."
                ),
            )
            if spec2 is not None:
                per_dr2 = retrieve_for_spec(spec2, question, verbose=verbose)
                if _total_entries(per_dr2) > 0:
                    answer, extraction = _run_extract_and_compute(
                        spec2,
                        per_dr2,
                        question,
                        verbose,
                    )
                    if extraction is not None:
                        spec = spec2
                        per_dr = per_dr2

    if extraction is None or answer.startswith(("EXTRACT_FAILED", "NO_VALUES", "COMPUTE_FAILED")):
        return answer

    # Phase 5: verify → one bounce-back to extract or decompose
    if not use_verify:
        if verbose:
            print(f"  Answer: {answer}")
        return answer

    verdict = verify_answer(question, spec, extraction["extractions"], answer, verbose=verbose)
    if verdict.get("ok"):
        if verbose:
            print(f"  Answer: {answer}")
        return answer

    issue = verdict.get("issue") or "unspecified problem"
    phase = verdict.get("suggested_phase")

    if phase == "extract":
        if verbose:
            print(f"  Verify flagged extract: {issue}")
        answer2, extraction2 = _run_extract_and_compute(
            spec,
            per_dr,
            question,
            verbose,
            feedback=issue,
        )
        if extraction2 is not None and not answer2.startswith(
            ("EXTRACT_FAILED", "NO_VALUES", "COMPUTE_FAILED")
        ):
            answer = answer2

    elif phase == "decompose":
        if verbose:
            print(f"  Verify flagged decompose: {issue}")
        spec2 = decompose(question, scout_hint=hint, feedback=issue)
        if spec2 is not None:
            per_dr2 = retrieve_for_spec(spec2, question, verbose=verbose)
            if _total_entries(per_dr2) > 0:
                answer2, extraction2 = _run_extract_and_compute(
                    spec2,
                    per_dr2,
                    question,
                    verbose,
                )
                if extraction2 is not None and not answer2.startswith(
                    ("EXTRACT_FAILED", "NO_VALUES", "COMPUTE_FAILED")
                ):
                    answer = answer2

    if verbose:
        print(f"  Answer: {answer}")
    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    verbose = "-v" in sys.argv
    args = [a for a in sys.argv[1:] if a != "-v"]

    if "--eval" in args:
        import concurrent.futures
        import csv

        eval_args = [a for a in args if a != "--eval"]

        n = 0
        parallel = 10
        oracle = "--oracle" in eval_args
        for i, a in enumerate(eval_args):
            if a == "--n" and i + 1 < len(eval_args):
                with contextlib.suppress(ValueError):
                    n = int(eval_args[i + 1])
            if a == "--parallel" and i + 1 < len(eval_args):
                with contextlib.suppress(ValueError):
                    parallel = int(eval_args[i + 1])

        rows = []
        with open("officeqa_full.csv") as f:
            for row in csv.DictReader(f):
                rows.append(row)
        if n:
            rows = rows[:n]

        from reward import fuzzy_match_answer

        def _solve_one(row: dict) -> tuple[dict, str]:
            question = row["question"]
            if oracle:
                # Oracle mode: retrieve per-DR, then filter each DR's entries
                # to only those whose file matches a gold source. This gives
                # extract the right tables while still exercising retrieve_v2's
                # in-file table ranking.
                gold_stems = {
                    f.strip().replace(".txt", "").replace(".json", "")
                    for f in row["source_files"].split("\n")
                    if f.strip()
                }
                spec = decompose(question, scout_hint=scout(question))
                if spec is None:
                    return row, "DECOMPOSE_FAILED"
                per_dr = retrieve_for_spec(spec, question)
                filtered: dict[str, list[dict]] = {}
                for dr_id, entries in per_dr.items():
                    filtered[dr_id] = [
                        e for e in entries if e.get("file", "").replace(".json", "") in gold_stems
                    ]
                extraction = extract_structured(spec, filtered, question, verbose=verbose)
                if extraction is None or "extractions" not in extraction:
                    return row, "EXTRACT_FAILED"
                for dr in spec.get("data_requests", []):
                    vid = dr["id"]
                    vals = extraction["extractions"].get(vid, {}).get("values") or []
                    if not [v for v in vals if v is not None]:
                        return row, f"NO_VALUES[{vid}]"
                try:
                    result = compute_execute(spec, extraction["extractions"], verbose=verbose)
                    return row, format_result(result, spec.get("output_format") or {})
                except ComputeError as e:
                    return row, f"COMPUTE_FAILED: {e}"
            return row, solve(question, verbose=verbose)

        correct, total = 0, 0
        print(
            f"Running {len(rows)} questions across {parallel} workers"
            f"{' (oracle mode)' if oracle else ''}...",
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = [executor.submit(_solve_one, row) for row in rows]
            for fut in concurrent.futures.as_completed(futures):
                row, got = fut.result()
                expected = row["answer"].strip()
                match, rationale = fuzzy_match_answer(expected, got, tolerance=0.01)
                total += 1
                tag = "OK  " if match else "MISS"
                if match:
                    correct += 1
                    print(
                        f"  [{total:3d}/{len(rows)}] {tag} {row['uid']}: {got}",
                        flush=True,
                    )
                else:
                    print(
                        f"  [{total:3d}/{len(rows)}] {tag} {row['uid']}: "
                        f"expected={expected!r} got={got!r} ({rationale})",
                        flush=True,
                    )

        print(f"\nAccuracy: {correct}/{total} = {correct / total * 100:.1f}%")

    elif args:
        question = " ".join(args)
        print(f"\n❓ {question}\n")
        answer = solve(question, verbose=verbose)
        print(f"\n✅ {answer}\n")
    else:
        print("Usage: uv run python solve.py [-v] [--eval [--n N] [--oracle]] <question>")
