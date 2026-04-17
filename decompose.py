#!/usr/bin/env python3
"""Decompose phase: question → QuestionSpec (structured data_requests + compute template).

Called by solve.py as pipeline step 1. Also importable standalone for eval_decompose.py.
"""

from __future__ import annotations

import json
import logging
import re
import time

from find import fetch_vocabulary
from llm_client import MODEL, THINKING_EXTRA_BODY, client, rate_limiter, strip_thinking

logger = logging.getLogger(__name__)


def llm(system: str, user: str, max_tokens: int = 4096, temperature: float = 0.0) -> str:
    """LLM call wrapper with rate limiting and retry/backoff.

    Respects the module-level rate_limiter to enforce API rate limits.
    Retries up to 3 times on rate limit errors (429, 5xx) with exponential backoff.
    """
    _MAX_RETRIES = 3
    _RETRY_BASE = 5  # seconds
    for attempt in range(_MAX_RETRIES):
        try:
            rate_limiter.acquire()  # wait for token before making API call
            resp = client.chat.completions.create(
                model=MODEL,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                extra_body=THINKING_EXTRA_BODY,
            )
            return strip_thinking(resp.choices[0].message.content or "")
        except Exception as e:
            err = str(e)
            is_rate_limit = (
                "429" in err or "504" in err or "502" in err or "503" in err
                or "rate" in err.lower() or "too many" in err.lower()
                or "timeout" in err.lower() or "gateway" in err.lower()
            )
            is_last = attempt == _MAX_RETRIES - 1
            if is_rate_limit and not is_last:
                delay = _RETRY_BASE * (2**attempt)
                logger.warning("Decompose rate limit on attempt %d; retrying in %ds", attempt + 1, delay)
                time.sleep(delay)
                continue
            raise


def parse_json(raw: str) -> dict | None:
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

DECOMPOSE_SYSTEM = """You are a collaborator on a research effort to answer
U.S. Treasury Bulletin questions with precision. Your work here matters: every
spec you produce is handed to a deterministic pipeline that will execute it
without second-guessing. A careful decomposition is worth far more than a fast
one — take the time you need to read the question properly.

It is OK to be uncertain. It is not OK to hide uncertainty. When you are unsure
about a period, a row label, a year, or a computation, say so in
`resolution_notes`. A flagged uncertain spec is strictly better than a
confident wrong one — the pipeline can retry on flagged uncertainty but cannot
recover from fabrication.

You decompose U.S. Treasury Bulletin questions into a structured QuestionSpec
that a deterministic pipeline can execute. List EXACTLY what values are needed,
EXACTLY how to compute them (as a Python template), and EXACTLY how to format
the answer. Python runs the computation — you just specify it.

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
RETRIEVAL HINT  (help the retriever pick the right channel)
═══════════════════════════════════════════════════════════════════════════════
Each data_request may include a `retrieval_hint` object to guide which retrieval
channel is most likely to find the data. Set `channel_preference` to one of:

  "metric_exact"  — the question targets a specific named line item that reads
                    like a ledger row label (e.g. "national defense",
                    "public debt interest", "net interest"). The metric channel
                    does a slug substring search over exact row labels — use this
                    when you have a clean row_hint that should match verbatim.

  "fts_keyword"   — topical or fuzzy question where the answer may span multiple
                    rows, or where a keyword search over table titles and captions
                    is more reliable than a row-label lookup (e.g. "savings bond
                    sales in New York", "foreign currency holdings").

  "prose"         — the answer lives in narrative text, footnotes, or contextual
                    passages rather than table cells (e.g. "according to the
                    footnote", "the report states that …", methodological notes).

  "auto"          — you are uncertain; let retrieval decide (default).

Use `must_match_phrases` for short phrases that MUST appear verbatim in the row
label or section title (e.g. ["national defense"]). Keep to 1–2 phrases max.
Use `avoid_phrases` for terms that would indicate a wrong table or row
(e.g. ["state and local"] when you only want federal figures).

═══════════════════════════════════════════════════════════════════════════════
PERIOD FIELD  (top-level signal for calendar vs fiscal year)
═══════════════════════════════════════════════════════════════════════════════
Set top-level `period` to:
  "CY"  — question clearly asks for calendar-year data ("calendar year 1940",
          "CY 1940", or a bare year like "in 1940" with no fiscal qualifier)
  "FY"  — question explicitly asks for fiscal-year data ("fiscal year 1940",
          "FY 1940", "FY1940")
  null  — ambiguous or period does not apply (e.g. a single date lookup)

This field revives deterministic auto-fixes in a later pipeline stage; setting
it accurately improves reliability on edge cases.

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
                         (not handled by the deterministic fast-path; use LLM extract)
  "continuous_monthly" — CONTINUOUS span across multiple years, from
                         (start_year, start_month) to (end_year, end_month)
                         INCLUSIVE. Use for phrases like "March 1942 to October
                         1948" or "Jan 1984 through Mar 1987". Set `years` to
                         the list of years covered. expected_count = total
                         months in the span.
                         NOTE: the deterministic ledger fast-path does not
                         resolve continuous spans cell-by-cell; those DRs use
                         LLM extraction after retrieval.
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
DERIVED VALUES  (values that don't exist as table cells)
═══════════════════════════════════════════════════════════════════════════════
Some values must be COMPUTED from raw table data because they don't appear as
a direct column:

  - RATIOS: A table may show "interest-bearing debt" and "total debt" as
    separate columns, but not their ratio. Request both amounts as separate
    DRs and compute the ratio in python_template.
    WRONG:  one DR for "ratio of interest-bearing to total debt"
    RIGHT:  DR v1 = "interest-bearing public debt", DR v2 = "total federal
            debt", template = "result = v1[0] / v2[0] * 100"

  - WEIGHTED AVERAGES / PER-UNIT VALUES: A table may show dollar values by
    category but not counts. E.g., currency-in-circulation tables show total
    dollar value per denomination but NOT the number of bills/coins. Request
    a cohort of denomination values and compute counts in python_template:
    template = "pieces = [v / d for v, d in zip(values['v1'],
               [1, 2, 5, 10, 20, 50, 100])]; result = sum(values['v1']) /
               sum(pieces)"

  - PERCENTAGE COLUMNS: If the table already has a "% increase" or "% change"
    column, prefer extracting that directly rather than computing from raw
    amounts — the table's rounded percentage is usually what the gold answer
    expects.

═══════════════════════════════════════════════════════════════════════════════
CROSS-REFERENCE QUESTIONS
═══════════════════════════════════════════════════════════════════════════════
If the question requires resolving fact A before looking up fact B (e.g. "find
which bureau did X, then use that bureau's report…"), resolve A yourself using
general knowledge and state the resolved fact in `resolution_notes` at the top
level of the spec. Then build data_requests for fact B using the resolved
answer. Do not try to encode the resolution as a corpus lookup.

Notable historical facts for cross-reference resolution:
  - Treasury Notes of 1890 were retired/removed from circulation around 1961-
    1962 (table shows them going from "$1 million" to "*" in that period).
  - The Bureau of Government Financial Operations (later renamed Financial
    Management Service/FMS) merged with the Bureau of the Public Debt to form
    the Bureau of the Fiscal Service in 2012.
  - Series E savings bonds were replaced by Series EE in 1980.

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
DATA VINTAGE  (canonical revisions vs. as-originally-reported)
═══════════════════════════════════════════════════════════════════════════════
Treasury Bulletins often revise values in later issues — a 1967-07 table may
republish figures from its 1967-09 successor with small corrections. Most
questions want the best current number; a few explicitly ask for the
originally-reported version.

  vintage: "latest"       — default. Use revised/canonical values when a
                             later bulletin supersedes them.
  vintage: "as_reported"  — set ONLY when the question explicitly asks for
                             preliminary, provisional, originally-reported,
                             or as-of-publication values (e.g. "as
                             originally reported", "preliminary estimate",
                             "as published in the July 1967 bulletin").

If in doubt, leave it as "latest".

═══════════════════════════════════════════════════════════════════════════════
MULTI-VALUE ANSWERS
═══════════════════════════════════════════════════════════════════════════════
If the question asks for a list like [slope, intercept] or [val1, val2, val3]:
  output_format.type = "list"
  python_template must end with: result = [a, b, ...]

═══════════════════════════════════════════════════════════════════════════════
PRE-FLIGHT SELF-CHECK (do this before emitting JSON)
═══════════════════════════════════════════════════════════════════════════════
Before writing the JSON output, silently ask yourself:

1. WHICH ROW? — Write out the exact row_hint you chose. Could any other row in
   the table match? If yes, is the row_hint specific enough to exclude them?
   (e.g. "National defense" vs "Total outlays — National defense and …")

2. EMPTY DR GUARD — If any DR in my python_template had values=[], would the
   template crash (index error, division by zero, empty list to stat function)?
   If yes, redesign: use cohort mode, or add a fallback in the template.

3. CROSS-REFERENCE CHECK — Does the question require knowing the answer to
   sub-question A before I can formulate DR B? If so:
   - If I can resolve A from domain knowledge → resolve it in resolution_notes,
     hard-code the resolved value as row_hint in DR B.
   - If I cannot resolve it without the corpus → use `cohort: true` on DR A
     (fetch all candidates) and encode the selection logic in python_template
     using argmax / argmin or a dict lookup. Never leave it as `computation: direct`
     with a single DR that can't answer the question.

4. UNIT COHERENCE — Does output_format.unit match what the question asks?
   Will format_result correctly convert from the table's native unit?

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
      "row_hint_alternatives": [],
      "column_hint": "<column header or year>",
      "section_hint": "<section header or table title to disambiguate, e.g. 'Budget' vs 'Criminal Cases'>",
      "years": [1940],
      "granularity": "annual|monthly_all|monthly_range|continuous_monthly|multi_year_annual|specific_month",
      "start_year": null,
      "start_month": null,
      "end_year": null,
      "end_month": null,
      "expected_count": 12,
      "cohort": false,
      "retrieval_hint": {
        "channel_preference": "metric_exact|fts_keyword|prose|auto",
        "must_match_phrases": [],
        "avoid_phrases": []
      }
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
  "question_kind": "value|page_number|date|text",
  "vintage": "latest|as_reported",
  "period": "CY|FY|null"
}"""


def _years_from_question(question: str) -> list[int]:
    return sorted({int(y) for y in re.findall(r"\b(1[89]\d{2}|20[0-3]\d)\b", question)})


def _fallback_hints_from_question(question: str, years: list[int]) -> tuple[str, str]:
    """Derive row/column hints when the LLM returned no usable spec.

    Gives retrieval FTS/metric channels substantive tokens instead of empty
    strings, and when a single data year is known, seeds ``column_hint`` so
    the annual branch of ``_build_cells_for_dr`` can resolve a column leaf.
    """
    q = question.strip()
    low = q.lower()
    for prefix in (
        "what were the ",
        "what was the ",
        "what is the ",
        "what are the ",
        "how much ",
        "how many ",
        "in which ",
        "which ",
    ):
        if low.startswith(prefix):
            q = q[len(prefix) :].lstrip()
            low = q.lower()
            break
    if len(q) > 220:
        q = q[:220].rsplit(" ", 1)[0].rstrip()
    col = str(years[0]) if len(years) == 1 else ""
    return q, col


def _granularity_mismatch_feedback(question: str, spec: dict) -> str | None:
    """Detect obvious CY wording paired with single-row annual granularity."""
    ql = question.lower()
    if "calendar year" not in ql and not re.search(r"\bcy\b", ql):
        return None
    for dr in spec.get("data_requests") or []:
        if not isinstance(dr, dict):
            continue
        if str(dr.get("source") or "corpus").lower() not in ("", "corpus"):
            continue
        if dr.get("cohort"):
            continue
        g = (dr.get("granularity") or "annual").strip().lower()
        ec = dr.get("expected_count")
        if g == "annual" and ec in (None, 1):
            return (
                "The question uses calendar-year (CY) wording. For a full calendar-year "
                "TOTAL, Treasury tables are usually summed from 12 monthly rows: use "
                '`granularity` "monthly_all", `expected_count` 12, and `computation` '
                '"sum" unless the question clearly targets one fiscal annual row.'
            )
    return None


def _validate_row_hints_against_ledger(spec: dict, question: str) -> str:
    """Check that each corpus DR's row_hint appears in at least one ledger table.

    Returns a non-empty feedback string if any row_hint has zero ledger matches,
    with actual matching labels to guide the LLM. Empty string = all OK.
    """
    drs = spec.get("data_requests") or []
    if not drs:
        return ""

    from find import _conn  # type: ignore[attr-defined]
    import sqlite3 as _sqlite3

    try:
        conn = _conn()
    except Exception:
        return ""

    problems: list[str] = []
    for dr in drs:
        if not isinstance(dr, dict):
            continue
        if str(dr.get("source") or "corpus").lower() not in ("", "corpus"):
            continue
        rh = (dr.get("row_hint") or "").strip()
        if not rh or len(rh.split()) > 12:
            continue

        # Check if any row in ledger has a metric_slug matching the hint
        slug = re.sub(r"[^\w\s\-]", "", rh.lower()).strip()
        slug_tokens = [t for t in re.findall(r"[a-z0-9]{3,}", slug) if len(t) >= 3]
        if not slug_tokens:
            continue

        like_clauses = " AND ".join(["metric_slug LIKE ?" for t in slug_tokens])
        like_params = [f"%{t}%" for t in slug_tokens]
        try:
            rows = conn.execute(
                f"SELECT DISTINCT metric_slug FROM row_label_lookup WHERE {like_clauses} LIMIT 5",
                like_params,
            ).fetchall()
        except Exception:
            continue

        if not rows:
            # Fetch real labels from the vocabulary to suggest alternatives
            years = [y for y in (dr.get("years") or []) if isinstance(y, int)]
            try:
                vocab = fetch_vocabulary(question, years=years or None)
                real_labels = (vocab.get("row_labels") or [])[:8]
            except Exception:
                real_labels = []
            suggestion = (
                f", ".join(f'"{r}"' for r in real_labels) if real_labels else "(none found)"
            )
            problems.append(
                f"row_hint={rh!r} for DR id={dr.get('id')!r} has NO matches in the "
                f"Treasury ledger. This label does not exist. Real labels from matching "
                f"tables: {suggestion}. Please pick a label verbatim from that list."
            )

    return "\n".join(problems) if problems else ""


def _normalize_row_hint_alternatives(spec: dict) -> None:
    """Ensure each DR has a list[str] ``row_hint_alternatives`` (may be empty)."""
    for dr in spec.get("data_requests") or []:
        if not isinstance(dr, dict):
            continue
        alts = dr.get("row_hint_alternatives")
        if alts is None:
            dr["row_hint_alternatives"] = []
            continue
        if isinstance(alts, str):
            dr["row_hint_alternatives"] = [alts.strip()] if alts.strip() else []
        elif isinstance(alts, list):
            dr["row_hint_alternatives"] = [str(x).strip() for x in alts if str(x).strip()]
        else:
            dr["row_hint_alternatives"] = []


_VALID_CHANNEL_PREFS = frozenset({"metric_exact", "fts_keyword", "prose", "auto"})


def _normalize_retrieval_hint(spec: dict) -> None:
    """Ensure each DR has a well-formed ``retrieval_hint`` and spec has ``period``.

    - Missing ``retrieval_hint`` → default with ``channel_preference: "auto"``.
    - Invalid ``channel_preference`` value → replaced with ``"auto"``.
    - Missing/invalid ``period`` at spec level → ``null``.
    """
    for dr in spec.get("data_requests") or []:
        if not isinstance(dr, dict):
            continue
        hint = dr.get("retrieval_hint")
        if not isinstance(hint, dict):
            dr["retrieval_hint"] = {
                "channel_preference": "auto",
                "must_match_phrases": [],
                "avoid_phrases": [],
            }
            continue
        # Validate / coerce channel_preference
        cp = hint.get("channel_preference")
        if cp not in _VALID_CHANNEL_PREFS:
            hint["channel_preference"] = "auto"
        # Ensure phrase lists are lists
        for key in ("must_match_phrases", "avoid_phrases"):
            val = hint.get(key)
            if not isinstance(val, list):
                hint[key] = [str(val).strip()] if val else []

    # Top-level period field — ensure key is present and valid
    period = spec.get("period")
    if period not in ("CY", "FY", None):
        spec["period"] = None
    else:
        spec.setdefault("period", None)


def _coerce_year_value(y: object) -> int | None:
    """Best-effort int year from JSON scalar; bool is rejected."""
    if isinstance(y, bool):
        return None
    if isinstance(y, int):
        return y
    if isinstance(y, float) and y.is_integer():
        return int(y)
    if isinstance(y, str) and y.strip().lstrip("-").isdigit():
        return int(y.strip())
    return None


def _normalize_decompose_years(spec: dict, question: str) -> None:
    """Coerce each DR's `years` to a list[int] and fill from the question when safe.

    Mirrors validate_decompose.MISSING_YEARS: a single data_request with no
    explicit years but year tokens in the question almost always means the LLM
    forgot to copy them — downstream retrieval keys on `years`.
    """
    drs = spec.get("data_requests")
    if not isinstance(drs, list) or not drs:
        return
    q_years = _years_from_question(question)
    for dr in drs:
        if not isinstance(dr, dict):
            continue
        raw = dr.get("years")
        coerced: list[int] = []
        if isinstance(raw, list):
            for y in raw:
                cy = _coerce_year_value(y)
                if cy is not None:
                    coerced.append(cy)
        dr["years"] = coerced
    if len(drs) == 1 and q_years:
        dr0 = drs[0]
        if not isinstance(dr0, dict):
            return
        if dr0.get("years"):
            return
        src = str(dr0.get("source") or "corpus").lower()
        if src in ("corpus", "cpi", "fx", ""):
            dr0["years"] = list(q_years)


def _format_vocab_block(vocab: dict) -> str:
    """Render a compact bullet list of real row/col labels for the prompt."""
    if not vocab or vocab.get("n_tables", 0) == 0:
        return ""

    row_labels = vocab.get("row_labels") or []
    col_labels = vocab.get("col_labels") or []
    if not row_labels and not col_labels:
        return ""

    lines = [
        "",
        "═══════════════════════════════════════════════════════════════════════════════",
        "ACTUAL LABELS FROM THE CORPUS  (real row/col labels from tables that match this question)",
        "═══════════════════════════════════════════════════════════════════════════════",
        f"Pulled from {vocab['n_tables']} candidate tables in the ledger. Most common first.",
        "When you write `row_hint` and `column_hint`, PREFER a label from these lists",
        "VERBATIM (including punctuation, abbreviations, and capitalization). The LLM is",
        "good at picking from a menu; it is bad at guessing Treasury's exact vocabulary.",
        "Only fall back to your own guess if nothing in the list fits — and explain why",
        "in `resolution_notes`.",
        "",
    ]
    if col_labels:
        lines.append("COLUMN LABELS:")
        for c in col_labels:
            lines.append(f"  • {c}")
        lines.append("")
    if row_labels:
        lines.append("ROW LABELS:")
        for r in row_labels:
            lines.append(f"  • {r}")
        lines.append("")
    return "\n".join(lines)


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

    # Pull real row/col labels from structurally-matching tables in the
    # ledger and inject them as a menu. The LLM picks from this list
    # instead of hallucinating Treasury vocabulary.
    try:
        years = _years_from_question(question)
        vocab = fetch_vocabulary(question, years=years or None)
        vocab_block = _format_vocab_block(vocab)
        if vocab_block:
            user_parts.append(vocab_block)
    except Exception as e:  # never let vocab lookup break decompose
        print(f"  [decompose] fetch_vocabulary failed: {e}", flush=True)

    if feedback:
        user_parts.append("")
        user_parts.append(
            f"REVIEW FEEDBACK from previous attempt — fix this specifically:\n{feedback}"
        )
    user_msg = "\n".join(user_parts)

    raw = llm(DECOMPOSE_SYSTEM, user_msg, max_tokens=3000)
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
        v = spec.get("vintage")
        if v not in ("latest", "as_reported"):
            spec["vintage"] = "latest"

        _normalize_decompose_years(spec, question)
        _normalize_row_hint_alternatives(spec)
        _normalize_retrieval_hint(spec)

        # Reject duplicate-label DRs on binary computations: a `difference`
        # / `ratio` / `percent_change` between two identical things is
        # always meaningless and almost always a decompose mistake (the
        # LLM forgot to differentiate the two periods or columns). Retry
        # once with explicit feedback.
        binary_ops = {
            "difference",
            "abs_difference",
            "percent_change",
            "abs_percent_change",
            "ratio",
        }
        comp = (spec.get("computation") or "").strip().lower()
        drs = spec.get("data_requests") or []
        labels = [(dr.get("label") or "").strip().lower() for dr in drs]
        if (
            not feedback
            and comp in binary_ops
            and len(labels) >= 2
            and len(set(labels)) < len(labels)
        ):
            return decompose(
                question,
                scout_hint=scout_hint,
                feedback=(
                    f"Your previous spec used computation={comp!r} but two "
                    f"data_requests had identical labels. A binary "
                    f"comparison needs two DISTINCT inputs (e.g. different "
                    f"years, columns, or rows). Either differentiate them "
                    f"or switch to a different computation."
                ),
            )

        if "[GRANULARITY_FIX]" not in (feedback or ""):
            gran_fb = _granularity_mismatch_feedback(question, spec)
            if gran_fb:
                return decompose(
                    question,
                    scout_hint=scout_hint,
                    feedback=f"[GRANULARITY_FIX]\n{gran_fb}",
                )

        # Post-decompose row_hint validation: if a row_hint doesn't appear
        # in the ledger at all, retry once with real label suggestions.
        # Only run on first attempt (no feedback yet) to avoid infinite retry.
        if "[ROW_HINT_FIX]" not in (feedback or ""):
            try:
                rh_fb = _validate_row_hints_against_ledger(spec, question)
                if rh_fb:
                    return decompose(
                        question,
                        scout_hint=scout_hint,
                        feedback=f"[ROW_HINT_FIX]\n{rh_fb}",
                    )
            except Exception as e:
                print(f"  [decompose] row_hint validation failed: {e}", flush=True)

        return spec

    # Fallback when the LLM failed to produce a usable spec — don't drop
    # the question, fall through to a minimal tokenized plan that the
    # retriever can still score. Downstream extract/compute will treat
    # the retrieved tables directly; at worst we degrade to the tokenized
    # baseline, which is still better than returning None.
    years = sorted({int(y) for y in re.findall(r"\b(1[89]\d{2}|20[0-3]\d)\b", question)})
    fb_row, fb_col = _fallback_hints_from_question(question, years)
    return {
        "computation": "direct",
        "data_requests": [
            {
                "id": "v1",
                "label": question,
                "source": "corpus",
                "row_hint": fb_row,
                "column_hint": fb_col,
                "row_hint_alternatives": [],
                "years": years,
                "target_years": years,
                "granularity": "unknown",
                "expected_count": None,
                "cohort": False,
                "retrieval_hint": {
                    "channel_preference": "auto",
                    "must_match_phrases": [],
                    "avoid_phrases": [],
                },
            }
        ],
        "computation_spec": {"python_template": "result = values['v1']"},
        "output_format": {"type": "number", "unit": None, "rounding": None, "as_percent": False},
        "question_kind": "value",
        "resolution_notes": "decompose fallback (LLM failed)",
        "cpi_needed": False,
        "vintage": "latest",
        "period": None,
    }
