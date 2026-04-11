#!/usr/bin/env python3
"""Briefing-mode solver — grep-primary, no DB.

Searches raw TXT corpus files, extracts tables, and outputs a structured
"intern's briefing" for the Senior Analyst (MiniMax) to review.

Key features:
- Conflict detection: flags when multiple sources give different values
- Explicit units per evidence source
- Period basis tagging (fiscal vs calendar)

Usage:
    python3 /installed-agent/solve_briefing.py "What were total expenditures for national defense in CY 1940?"
    python3 /installed-agent/solve_briefing.py "question" --keywords "defense expenditures" --year 1940
"""

import os
import re
import sys
import time
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

CORPUS_DIR = os.environ.get("CORPUS_DIR", "/app/corpus")
ANSWER_PATH = os.environ.get("ANSWER_PATH", "/app/answer.txt")

try:
    from solve import deterministic_search, parse_question, search_raw_corpus, verify_answer
except ImportError:

    def search_raw_corpus(**kwargs):
        return {"matches": [], "error": "solve.py not available"}

    def verify_answer(**kwargs):
        return {}

    def deterministic_search(question, limit=5):
        return [], {}

    def parse_question(question):
        return {
            "metric": "",
            "years": [],
            "year": None,
            "period_basis": "calendar",
            "strategies": [],
            "content_words": [],
        }


def _parse_numeric(val_str):
    """Try to parse a string as a number. Returns float or None."""
    if not val_str:
        return None
    cleaned = re.sub(r"[,$ ]", "", str(val_str).replace("\u2212", "-").replace("\u2013", "-"))
    cleaned = cleaned.strip().rstrip("%")
    # Handle parenthetical negatives: (123) -> -123
    m = re.match(r"^\(([0-9,.]+)\)$", cleaned)
    if m:
        cleaned = "-" + m.group(1).replace(",", "")
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


def _detect_period_basis(match_data):
    """Detect whether evidence is fiscal or calendar year data."""
    signals = []
    text = " ".join(
        [
            str(match_data.get("matched_row_vertical", "")),
            str(match_data.get("table_title", "")),
            str(match_data.get("file", "")),
        ]
    ).lower()

    if any(x in text for x in ["fiscal year", "fy", "fiscal "]):
        signals.append("fiscal")
    if any(x in text for x in ["calendar year", "cy", "calendar "]):
        signals.append("calendar")
    if any(
        x in text
        for x in [
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
    ):
        signals.append("has_monthly_data")

    return signals if signals else ["unknown"]


def format_briefing(
    question,
    evidence_items,
    proposed_answer,
    confidence,
    search_strategy,
    caveats,
    conflicts,
    elapsed=0.0,
):
    """Format the intern's briefing for the senior analyst."""
    lines = []
    lines.append("--- INTERN'S RESEARCH BRIEFING ---")
    lines.append(f"QUESTION: {question}")
    lines.append(f"SEARCH TERMS: {search_strategy}")
    lines.append(f"TIME: {elapsed:.1f}s")
    lines.append("")

    # Evidence — each source gets explicit units and period
    lines.append("EVIDENCE FOUND:")
    if evidence_items:
        for i, ev in enumerate(evidence_items[:3]):
            lines.append(f"\n  [{i + 1}] File: {ev.get('file', 'unknown')}")
            lines.append(f"      Table: {ev.get('table_title', '(no title)')}")
            lines.append(f"      Units: {ev.get('units', '(not specified)')}")
            lines.append(f"      Period: {', '.join(ev.get('period_signals', ['unknown']))}")
            if ev.get("matched_row"):
                lines.append(f"      Matched row: {ev['matched_row']}")
            if ev.get("extracted_value"):
                lines.append(f"      Value found: {ev['extracted_value']}")
            if ev.get("vertical_data"):
                lines.append("      Data:")
                for vline in ev["vertical_data"].split("\n")[:25]:
                    if vline.strip():
                        lines.append(f"        {vline}")
            if ev.get("table_data"):
                lines.append("      Table excerpt:")
                for k, v in list(ev["table_data"].items())[:15]:
                    lines.append(f"        {k}: {v}")
    else:
        lines.append("  (none found)")

    lines.append("")

    # Conflict report — the key new feature
    if conflicts:
        lines.append("CONFLICT REPORT:")
        lines.append("  Multiple sources returned DIFFERENT values for this question:")
        for c in conflicts:
            lines.append(f"  - {c}")
        lines.append("  >> Senior analyst: please review which source is authoritative.")
        lines.append("")

    # Proposed answer
    lines.append("INTERN'S PROPOSED ANSWER:")
    lines.append(f"  Value: {proposed_answer}")
    lines.append(f"  Confidence: {confidence}")
    lines.append("")

    if caveats:
        lines.append("CAVEATS FOR REVIEW:")
        for c in caveats:
            lines.append(f"  - {c}")
        lines.append("")

    lines.append("---")
    lines.append("If correct: printf '%s' \"VALUE\" > /app/answer.txt")
    lines.append('If wrong: re-run with --keywords "different terms" --year YYYY')
    lines.append("---")
    return "\n".join(lines)


def run_briefing(question, keywords_override=None, year_override=None):
    """Search the raw TXT corpus and return a briefing."""
    start_time = time.time()

    # Parse question for years
    years_in_q = [int(y) for y in re.findall(r"\b(19\d{2}|20\d{2})\b", question)]
    target_year = year_override or (years_in_q[0] if years_in_q else None)

    # Detect fiscal vs calendar in the question
    q_lower = question.lower()
    wants_calendar = any(x in q_lower for x in ["calendar year", "cy ", "cy1", "cy2"])
    wants_fiscal = any(x in q_lower for x in ["fiscal year", "fy ", "fy1", "fy2"])

    # Extract keywords
    if keywords_override:
        keywords = keywords_override
    else:
        stopwords = {
            "what",
            "were",
            "was",
            "the",
            "total",
            "for",
            "and",
            "how",
            "much",
            "did",
            "many",
            "are",
            "from",
            "with",
            "that",
            "this",
            "which",
            "between",
            "during",
            "about",
            "into",
            "than",
            "its",
            "their",
            "has",
            "had",
            "have",
            "does",
            "been",
            "year",
            "fiscal",
            "calendar",
            "monthly",
            "annual",
        }
        words = re.sub(r"[^\w\s]", "", question.lower()).split()
        keywords = " ".join(
            w for w in words if w not in stopwords and len(w) > 2 and not w.isdigit()
        )

    search_strategy = f"'{keywords}', year={target_year}"
    evidence_items = []
    proposed_answer = None
    confidence = "none"
    caveats = []
    conflicts = []

    # === Search raw TXT corpus (multi-strategy) ===
    try:
        # Use deterministic_search which tries multiple keyword strategies
        # and ranks by bulletin proximity + data completeness
        if keywords_override:
            # Manual override — use single search
            raw_result = search_raw_corpus(keywords=keywords_override, year=target_year, limit=5)
            matches = raw_result.get("results", raw_result.get("matches", []))
        else:
            matches, _params = deterministic_search(question, limit=5)
            search_strategy = (
                f"multi-strategy: {_params.get('strategies', [])[:3]}, year={target_year}"
            )

        # Collect all candidate values for conflict detection
        candidate_values = []  # list of (value_str, value_float, source_file, units)

        for match in matches[:3]:
            period_signals = _detect_period_basis(match)
            units = match.get("units", "(not specified)")

            # matched_row_vertical is a STRING with the best-matching row
            matched_row_vert = match.get("matched_row_vertical", "")
            # vertical_data is a LIST of all rows in vertical format
            all_vert_rows = match.get("vertical_data", [])
            if isinstance(all_vert_rows, str):
                all_vert_rows = [all_vert_rows] if all_vert_rows else []

            # Combine: show matched row first, then other rows
            combined_vertical = matched_row_vert or ""
            if all_vert_rows:
                # Add up to 10 other rows (skip the matched one)
                other_rows = [r for r in all_vert_rows if r != matched_row_vert][:10]
                if other_rows:
                    combined_vertical += "\n\n--- Other rows in this table ---\n" + "\n".join(
                        other_rows
                    )

            td = match.get("table_data", {})
            if isinstance(td, list):
                # table_data is a list of dicts — convert to single dict for display
                td_dict = {}
                for row_d in td[:10]:
                    label = row_d.get("_row_label", "?")
                    vals = {k: v for k, v in row_d.items() if k != "_row_label" and v is not None}
                    td_dict[label] = vals
                td = td_dict
            elif not isinstance(td, dict):
                td = {}

            ev = {
                "file": match.get("file", ""),
                "table_title": match.get("table_title", ""),
                "units": units,
                "period_signals": period_signals,
                "matched_row": "",
                "vertical_data": combined_vertical,
                "table_data": td,
                "extracted_value": None,
            }

            # Extract value from this source
            extracted = None
            vdata = matched_row_vert

            # Try table_data first
            if target_year and isinstance(td, dict):
                for col_name, val in td.items():
                    if str(target_year) in str(col_name):
                        extracted = str(val).strip()
                        break

            # Try vertical data
            if not extracted and vdata and target_year:
                for line in vdata.split("\n"):
                    if str(target_year) in line and ":" in line:
                        parts = line.rsplit(":", 1)
                        if len(parts) == 2:
                            val = parts[1].strip()
                            if val and val not in ("...", "\u2014", "-", "", "None"):
                                extracted = val
                                break

            if extracted:
                ev["extracted_value"] = extracted
                numeric = _parse_numeric(extracted)
                candidate_values.append(
                    (
                        extracted,
                        numeric,
                        match.get("file", "unknown"),
                        units,
                    )
                )

            evidence_items.append(ev)

        # === Conflict detection ===
        if len(candidate_values) >= 2:
            # Check if the numeric values disagree
            numerics = [(v, src, u) for (_, v, src, u) in candidate_values if v is not None]
            if len(numerics) >= 2:
                base_val = numerics[0][0]
                for val, src, units in numerics[1:]:
                    if base_val != 0 and abs(val - base_val) / abs(base_val) > 0.02:
                        conflicts.append(
                            f"{candidate_values[0][0]} (from {candidate_values[0][2]}, {candidate_values[0][3]}) "
                            f"vs {candidate_values[1][0]} (from {src}, {units})"
                        )

        # Pick proposed answer from best match
        if candidate_values:
            proposed_answer = candidate_values[0][0]
            if conflicts:
                confidence = "low — conflicting values found"
            elif len(candidate_values) >= 2:
                # Multiple sources agree
                n0 = candidate_values[0][1]
                n1 = candidate_values[1][1]
                if n0 is not None and n1 is not None and n0 != 0:
                    if abs(n1 - n0) / abs(n0) < 0.02:
                        confidence = "high — multiple sources agree"
                    else:
                        confidence = "medium"
                else:
                    confidence = "medium"
            else:
                confidence = "medium — single source"

        if not matches:
            caveats.append("No matching tables found in corpus.")

    except Exception as e:
        caveats.append(f"Search failed: {e}")

    # === Period mismatch warnings ===
    if evidence_items:
        ev_periods = evidence_items[0].get("period_signals", [])
        if wants_calendar and "fiscal" in ev_periods and "calendar" not in ev_periods:
            caveats.append(
                "PERIOD MISMATCH: Question asks for CALENDAR YEAR but evidence "
                "appears to be FISCAL YEAR data. Pre-1977 FY = Jul-Jun, post-1977 = Oct-Sep."
            )
        if wants_fiscal and "calendar" in ev_periods and "fiscal" not in ev_periods:
            caveats.append(
                "PERIOD MISMATCH: Question asks for FISCAL YEAR but evidence "
                "appears to be CALENDAR YEAR data."
            )

    # === Verification ===
    if proposed_answer and proposed_answer != "(no answer found)":
        try:
            v = verify_answer(
                answer=proposed_answer,
                question=question,
                plan={},
                evidence=str(evidence_items)[:2000],
            )
            for w in v.get("warnings", []):
                caveats.append(f"[{w.get('severity', '?')}] {w.get('message', '')}")
            auto_fixed = v.get("checks", {}).get("auto_fixed_answer")
            if auto_fixed:
                caveats.append(f"Auto-fix suggested: {proposed_answer} -> {auto_fixed}")
                proposed_answer = auto_fixed
        except Exception:
            pass

    if not proposed_answer:
        proposed_answer = "(no answer found)"
        confidence = "none"
        caveats.append("No answer extracted. Try different keywords or grep manually.")

    elapsed = time.time() - start_time

    # Write answer to file if we found one
    if proposed_answer and proposed_answer != "(no answer found)":
        try:
            Path(ANSWER_PATH).write_text(proposed_answer)
        except Exception:
            pass

    return format_briefing(
        question=question,
        evidence_items=evidence_items,
        proposed_answer=proposed_answer,
        confidence=confidence,
        search_strategy=search_strategy,
        caveats=caveats,
        conflicts=conflicts,
        elapsed=elapsed,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Intern's briefing solver (grep-primary)")
    parser.add_argument("question", nargs="?", help="The question to answer")
    parser.add_argument("--keywords", type=str, help="Override search keywords")
    parser.add_argument("--year", type=int, help="Override target year")
    args = parser.parse_args()

    if not args.question:
        print('Usage: python3 solve_briefing.py "QUESTION"')
        sys.exit(1)

    briefing = run_briefing(
        question=args.question,
        keywords_override=args.keywords,
        year_override=args.year,
    )
    print(briefing)


if __name__ == "__main__":
    main()
