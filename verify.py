#!/usr/bin/env python3
"""Post-compute verification — arena checklist as an LLM pass.

Ported from officeqa-arena r12/skills/verify. Runs after compute with:
  - the original question
  - the QuestionSpec
  - the extracted values + source files
  - the computed answer

Returns {ok, issue, suggested_phase} so the orchestrator can bounce back to
the right phase on failure. Falls back to ok=True on any error so a broken
verify pass never blocks a valid answer.

Deterministic auto-fixes (auto_fix_units, auto_fix_fy_cy) run BEFORE the
LLM verify call to catch common mistakes without an extra round-trip.
"""

import json
import os
import re

from dotenv import load_dotenv  # type: ignore[import-untyped]
from openai import OpenAI

from compute import UNIT_MULTIPLIERS, parse_unit

load_dotenv()

MODEL = os.getenv("OFFICEQA_MODEL", "deepseek/deepseek-chat")
client = OpenAI(
    api_key=os.getenv("DEDALUS_API_KEY"),
    base_url=os.getenv("DEDALUS_API_BASE"),
)


# Kept in sync with the EXTRACTION CHECKLIST in extract.py so both phases
# reason from the same failure modes. Verify's wording is action-oriented
# (apply the fix post-hoc); extract's wording is prevention-oriented (don't
# produce the error in the first place). If you add a rule here, mirror it
# in extract.EXTRACT_STRUCTURED_SYSTEM.
CHECKLIST = """Treasury Bulletin answer checklist:

1. UNITS. If the question says "dollars" and the table header says "(in
   millions)" or "(in thousands)", scale accordingly. If the question says
   "in millions" but the raw answer is unscaled, divide by 1e6.

2. FISCAL YEAR BOUNDARIES.
   - pre-1977  FY = Jul 1 (YYYY-1) through Jun 30 YYYY
   - post-1976 FY = Oct 1 (YYYY-1) through Sep 30 YYYY
   - CY       = Jan 1 through Dec 31 (sum of 12 monthly rows)

3. TOTAL vs SUB-CATEGORY. "Total X" rows already include all sub-items —
   never double-count by summing sub-items AND the total row.

4. ACTUAL vs ESTIMATED. Treasury tables often show estimates for the
   current year and actuals for prior years. Prefer actuals.

5. ANNUAL vs MONTHLY. A single monthly row is not an annual total.
   "Calendar year YYYY" means the sum of 12 months.

6. COLUMN POSITION. Wide tables with multi-level headers are easy to
   misread. Confirm the column by its header, not its position.
"""


VERIFY_SYSTEM = f"""You are a senior Treasury data analyst reviewing your
intern's work. Be critical and thorough — your intern often makes mistakes
with units, fiscal vs calendar year boundaries, and row/column selection.
Check every detail.

{CHECKLIST}

You receive the question, the extracted values with citations, and the final
computed answer that your intern produced. Check the answer against the
checklist. Flag only real problems — do not second-guess correct work.

Output ONLY valid JSON:
{{
  "ok": true|false,
  "issue": "<short description of the problem, or null>",
  "suggested_phase": "extract"|"decompose"|null
}}

Choose "extract" when the spec was right but the numbers are wrong (misread
value, wrong column, wrong row, units mistake). Choose "decompose" when the
interpretation itself is wrong (fiscal vs calendar confusion, wrong
computation, missing data requests)."""


# ── Deterministic auto-fixes (run before LLM verify) ─────────────────────────


def _parse_numeric_answer(answer: str) -> float | None:
    """Try to parse a formatted numeric answer string into a float.

    Handles commas, negative parentheses, and whitespace.
    Returns None if the answer is not parseable as a number.
    """
    text = answer.strip()
    # Remove commas
    text = text.replace(",", "")
    # Remove surrounding parentheses used for negatives
    if text.startswith("(") and text.endswith(")"):
        text = "-" + text[1:-1]
    try:
        return float(text)
    except (ValueError, TypeError):
        return None


def _format_number(value: float) -> str:
    """Format a float as a human-friendly string.

    - Integers (or near-integers) get comma formatting with no decimals.
    - Others get up to 3 decimal places, trailing zeros stripped.
    """
    # Check if value is effectively an integer
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value)):,}"
    # Format with up to 3 decimal places
    formatted = f"{value:,.3f}".rstrip("0").rstrip(".")
    return formatted


def auto_fix_units(
    answer: str,
    source_unit: str | None,
    target_unit: str | None,
) -> dict | None:
    """Detect unit-scale mismatch and apply correction deterministically.

    Takes the computed answer string, the source table unit (e.g. 'thousands'),
    and the target unit from output_format (e.g. 'millions'). If they differ
    and the answer is numeric, applies the conversion and returns a dict with
    the corrected answer and a description of the issue.

    Returns None if no mismatch is detected or the answer can't be corrected.
    """
    if source_unit is None or target_unit is None:
        return None
    if source_unit == target_unit:
        return None

    src_mult = UNIT_MULTIPLIERS.get(source_unit)
    tgt_mult = UNIT_MULTIPLIERS.get(target_unit)
    if src_mult is None or tgt_mult is None:
        return None

    numeric = _parse_numeric_answer(answer)
    if numeric is None:
        return None

    factor = src_mult / tgt_mult
    corrected_value = numeric * factor
    corrected_str = _format_number(corrected_value)

    issue = (
        f"Unit-scale mismatch: answer appears to be in {source_unit} "
        f"but question asks for {target_unit}. "
        f"Applied conversion (×{factor:.0e})."
    )
    return {"issue": issue, "corrected": corrected_str}


# Patterns that indicate fiscal-year data in extraction labels
_FY_PATTERNS = re.compile(r"\bfiscal\s+year\b|\bfy\b|\bfy\s*\d{4}", re.IGNORECASE)
# Patterns that indicate calendar-year data in extraction labels
_CY_PATTERNS = re.compile(r"\bcalendar\s+year\b|\bcy\b|\bcy\s*\d{4}", re.IGNORECASE)


def auto_fix_fy_cy(
    spec: dict,
    extractions: dict,
) -> dict | None:
    """Detect fiscal-year vs calendar-year confusion deterministically.

    Checks the spec's period (calendar/fiscal) against the labels in
    extraction results. If the spec asks for calendar year but extraction
    labels indicate fiscal year data (or vice versa), flags the mismatch
    for re-extraction.

    Returns None if no FY/CY confusion is detected.
    Returns dict with {flagged, issue, suggested_phase} if a mismatch is found.
    """
    period = spec.get("period")
    if not period:
        return None

    period_lower = period.lower()

    # Collect all labels from all extraction entries
    all_labels: list[str] = []
    for _dr_id, ex in extractions.items():
        if not ex:
            continue
        labels = ex.get("labels") or []
        all_labels.extend(labels)

    if not all_labels:
        return None

    labels_text = " ".join(all_labels)

    has_fy = bool(_FY_PATTERNS.search(labels_text))
    has_cy = bool(_CY_PATTERNS.search(labels_text))

    # Check for period mismatch
    if period_lower in ("calendar", "cy", "calendar_year") and has_fy and not has_cy:
        issue = (
            "FY/CY confusion: spec asks for calendar year but extraction "
            "labels indicate fiscal year data. The intern likely picked "
            "the FY total row instead of summing 12 monthly CY rows."
        )
        return {"flagged": True, "issue": issue, "suggested_phase": "extract"}

    if period_lower in ("fiscal", "fy", "fiscal_year") and has_cy and not has_fy:
        issue = (
            "FY/CY confusion: spec asks for fiscal year but extraction "
            "labels indicate calendar year data. The intern likely summed "
            "monthly rows for CY instead of using the FY annual row."
        )
        return {"flagged": True, "issue": issue, "suggested_phase": "extract"}

    return None


def verify_answer(
    question: str,
    spec: dict,
    extractions: dict,
    answer: str,
    verbose: bool = False,
    source_unit: str | None = None,
) -> dict:
    """Return {ok, issue, suggested_phase}. On any internal failure, returns
    ok=True to avoid false-negative retries.

    Deterministic auto-fixes run BEFORE the LLM verify call:
    1. auto_fix_units: detects unit-scale mismatch, returns corrected answer.
    2. auto_fix_fy_cy: detects FY/CY confusion, flags for re-extraction.
    These avoid unnecessary LLM round-trips for common, mechanically
    detectable errors.
    """
    # ── Deterministic auto-fix: unit scaling ────────────────────────────
    target_unit = (spec.get("output_format") or {}).get("unit")
    # Try to infer source_unit from extraction metadata if not passed
    if source_unit is None:
        source_unit = _infer_source_unit(extractions)

    unit_fix = auto_fix_units(answer, source_unit, target_unit)
    if unit_fix is not None:
        if verbose:
            print(f"  Auto-fix units: {unit_fix['issue']} → {unit_fix['corrected']}")
        return {
            "ok": False,
            "issue": unit_fix["issue"],
            "suggested_phase": "extract",
            "corrected_answer": unit_fix["corrected"],
        }

    # ── Deterministic auto-fix: FY/CY confusion ────────────────────────
    fy_cy_fix = auto_fix_fy_cy(spec, extractions)
    if fy_cy_fix is not None and fy_cy_fix.get("flagged"):
        if verbose:
            print(f"  Auto-fix FY/CY: {fy_cy_fix['issue']}")
        return {
            "ok": False,
            "issue": fy_cy_fix["issue"],
            "suggested_phase": fy_cy_fix["suggested_phase"],
        }

    # ── LLM verify ──────────────────────────────────────────────────────
    parts = [
        f"QUESTION: {question}",
        f"COMPUTED ANSWER: {answer}",
        "",
        f"Computation: {spec.get('computation')}",
        f"Output format: {json.dumps(spec.get('output_format') or {})}",
        "",
        "Extracted values by data_request:",
    ]
    for dr in spec.get("data_requests", []):
        vid = dr["id"]
        ex = extractions.get(vid, {}) or {}
        vals = ex.get("values") or []
        src = ex.get("source_file") or "?"
        preview = vals[:12]
        more = "..." if len(vals) > 12 else ""
        parts.append(
            f"  {vid} [{dr.get('label', '')}] granularity={dr.get('granularity', '?')} "
            f"years={dr.get('years')} src={src} count={len(vals)} values={preview}{more}"
        )
    user_msg = "\n".join(parts) + "\n\nReview the answer. Return the JSON verdict."

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=500,
            temperature=0.0,
            messages=[
                {"role": "system", "content": VERIFY_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        if verbose:
            print(f"  Verify LLM call failed: {e}")
        return {"ok": True, "issue": None, "suggested_phase": None}

    cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
    try:
        verdict = json.loads(cleaned)
    except json.JSONDecodeError:
        try:
            verdict = json.loads(cleaned[: cleaned.rfind("}") + 1])
        except Exception:
            if verbose:
                print(f"  Verify parse failed: {raw[:200]}")
            return {"ok": True, "issue": None, "suggested_phase": None}

    result = {
        "ok": bool(verdict.get("ok", True)),
        "issue": verdict.get("issue"),
        "suggested_phase": verdict.get("suggested_phase"),
    }
    if verbose:
        print(f"  Verify: {result}")
    return result


def _infer_source_unit(extractions: dict) -> str | None:
    """Try to infer the source table unit from extraction metadata.

    Checks each extraction entry for a 'unit' field. Returns the first
    one found, or None if no unit metadata is available.
    """
    for _dr_id, ex in extractions.items():
        if not ex:
            continue
        raw_unit = ex.get("unit")
        if raw_unit:
            parsed = parse_unit(raw_unit)
            if parsed:
                return parsed
    return None
