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
import sqlite3
import sys
import threading

from dotenv import load_dotenv

from compute import ComputeError, format_result, parse_unit, validate_extractions
from compute import execute as compute_execute
from decompose import decompose
from extract import extract_structured
from find import (
    effective_row_hint_for_bottomup,
    resolve_cells,
    retrieve_bottomup,
    search_cells_bottomup,
)
from ledger_paths import get_ledger_sqlite_path
from retrieve_v2 import retrieve as retrieve_v2
from scout import scout
from verify import verify_answer

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
load_dotenv()

# ── Retry bounds ────────────────────────────────────────────────────────────
# Max retries per phase (decompose / extract / verify). Each retry is one
# additional LLM call.  Initial call + MAX_RETRIES_PER_PHASE retries per phase.
MAX_RETRIES_PER_PHASE = 2
# Absolute cap on total LLM calls per question (decompose + extract + verify
# across all attempts).  Prevents runaway costs on adversarial or ambiguous
# questions.  6 = 2 decompose + 2 extract + 2 verify in the worst case.
MAX_LLM_CALLS = 6

# ── Ledger connection for deterministic fast-path ───────────────────────────

_LEDGER_PATH = get_ledger_sqlite_path()
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

    Deterministic fast-path covers: ``annual`` / ``specific_month`` /
    ``unknown`` (single-value), ``monthly_all``, and ``multi_year_annual``.
    It does **not** cover ``continuous_monthly`` or ``monthly_range`` (span
    logic is left to LLM extraction).
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
) -> tuple[dict[str, dict], list[str]]:
    """Try to resolve data_requests deterministically via resolve_cells().

    Skips DRs whose ``granularity`` is not supported by ``_build_cells_for_dr``
    (e.g. ``continuous_monthly``, ``monthly_range``) and non-corpus sources.

    Returns a tuple of (resolved_extractions, unresolved_dr_ids).
    - resolved_extractions: dict mapping dr_id → extraction dict (same structure
      as extract_structured() output per-DR entries) for DRs that resolved.
    - unresolved_dr_ids: list of dr_ids that could NOT be resolved.

    The caller should merge resolved_extractions with LLM fallback for the
    unresolved DRs. When unresolved_dr_ids is empty, LLM extraction is skipped.
    """
    data_requests = spec.get("data_requests") or []
    if not data_requests:
        return {}, []

    vintage = spec.get("vintage", "latest")
    extractions: dict[str, dict] = {}
    unresolved_ids: list[str] = []
    notes_parts: list[str] = []

    for dr in data_requests:
        dr_id = dr.get("id", "?")
        source = dr.get("source", "corpus")

        # External/CPI/FX sources can't be resolved from the ledger
        if source in ("external", "cpi", "fx"):
            if verbose:
                print(f"  Fast-path: {dr_id} skipped (source={source})")
            unresolved_ids.append(dr_id)
            continue

        # Get the first (best) retrieved entry for this DR
        entries = per_dr_entries.get(dr_id) or []
        if not entries:
            if verbose:
                print(f"  Fast-path: {dr_id} no retrieved entries")
            unresolved_ids.append(dr_id)
            continue

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
            unresolved_ids.append(dr_id)
            continue

        # Look up table_id from the entry
        table_id = _get_table_id_from_entry(table_entry)
        if table_id is None:
            if verbose:
                print(f"  Fast-path: {dr_id} table_id not found")
            unresolved_ids.append(dr_id)
            continue

        # Build cell specs for resolve_cells()
        cells = _build_cells_for_dr(dr, table_id)
        if cells is None:
            if verbose:
                print(
                    f"  Fast-path: {dr_id} can't build cells (granularity={dr.get('granularity')})"
                )
            unresolved_ids.append(dr_id)
            continue

        # Call resolve_cells
        try:
            resolved = resolve_cells(table_id, cells, vintage=vintage)
        except Exception as e:
            if verbose:
                print(f"  Fast-path: {dr_id} resolve_cells error: {e}")
            unresolved_ids.append(dr_id)
            continue

        values = resolved.get("values", {})

        # Check if all values resolved (non-None)
        unresolved_cells = [k for k, v in values.items() if v is None]
        if unresolved_cells:
            # Before giving up, try bottom-up cell search: start from row×col criteria
            # and let the ledger pick the right table + latest bulletin automatically.
            # This recovers from: (a) row_hint that doesn't match any slug/leaf in the
            # top-down retrieved table, (b) year_extracted not indexed for historical
            # columns, (c) multiple bulletins — always picks file_year DESC.
            row_raw = (dr.get("row_hint") or "").strip()
            dr_years = dr.get("years") or []
            col_year = dr_years[0] if dr_years else None
            keywords = dr.get("keywords") or []
            label_s = (dr.get("label") or "").strip()

            topic_parts = [
                str(k).strip() for k in keywords[:6] if isinstance(k, str) and str(k).strip()
            ]
            if row_raw:
                topic_parts.append(row_raw)
            elif not topic_parts and label_s:
                topic_parts.append(label_s)
            topic = " ".join(topic_parts) if topic_parts else label_s

            granularity = dr.get("granularity", "annual")

            eff_row = effective_row_hint_for_bottomup(
                row_raw,
                label=dr.get("label") or "",
                keywords=keywords if keywords else None,
            )

            if eff_row and granularity in ("annual", "specific_month", "unknown", None):
                bu_hits = search_cells_bottomup(
                    row_hint=eff_row,
                    col_year=col_year,
                    topic=topic or eff_row,
                    limit=5,
                )
                if bu_hits:
                    best = bu_hits[0]
                    numeric = best.get("numeric_value")
                    if numeric is not None:
                        if verbose:
                            print(
                                f"  Fast-path: {dr_id} bottomup hit → "
                                f"{numeric} from {best['file']} "
                                f"(row={best['row_leaf']}, col={best['col_leaf']})"
                            )
                        extractions[dr_id] = {
                            "values": [numeric],
                            "labels": [best.get("row_leaf", "")],
                            "source_file": best["file"],
                            "confidence": "bottomup",
                            "bottomup_table_id": best["table_id"],
                            "bottomup_file_year": best["file_year"],
                        }
                        notes_parts.append(
                            f"{dr_id}: resolved via bottom-up search "
                            f"(table {best['table_id']}, {best['file']})"
                        )
                        continue

            if verbose:
                print(f"  Fast-path: {dr_id} unresolved cells: {unresolved_cells}")
            unresolved_ids.append(dr_id)
            continue

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
                    # Missing month — this DR is unresolved
                    if verbose:
                        print(f"  Fast-path: {dr_id} incomplete monthly ({len(ordered_values)}/12)")
                    unresolved_ids.append(dr_id)
                    break
            else:
                # All 12 months resolved
                extractions[dr_id] = {
                    "values": ordered_values,
                    "labels": ordered_labels,
                    "source_file": table_entry.get("file", ""),
                    "confidence": "deterministic",
                }
                notes_parts.append(f"{dr_id}: resolved deterministically from table {table_id}")
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
                    unresolved_ids.append(dr_id)
                    break
            else:
                # All years resolved
                extractions[dr_id] = {
                    "values": ordered_values,
                    "labels": ordered_labels,
                    "source_file": table_entry.get("file", ""),
                    "confidence": "deterministic",
                }
                notes_parts.append(f"{dr_id}: resolved deterministically from table {table_id}")
        else:
            # Single value
            val = values.get(dr_id)
            if val is None:
                if verbose:
                    print(f"  Fast-path: {dr_id} single value is None")
                unresolved_ids.append(dr_id)
                continue
            extractions[dr_id] = {
                "values": [val],
                "labels": [dr.get("row_hint", "")],
                "source_file": table_entry.get("file", ""),
                "confidence": "deterministic",
            }
            notes_parts.append(f"{dr_id}: resolved deterministically from table {table_id}")

    n_resolved = len(extractions)
    n_unresolved = len(unresolved_ids)
    if verbose:
        n = len(data_requests)
        if n_unresolved == 0:
            print(f"  Fast-path: ALL {n} DR{'' if n == 1 else 's'} resolved deterministically ✓")
        else:
            print(
                f"  Fast-path: {n_resolved}/{n} DR{'' if n == 1 else 's'} resolved, "
                f"{n_unresolved} need LLM fallback: {unresolved_ids}"
            )

    return extractions, unresolved_ids


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def retrieve_for_spec(
    spec: dict,
    question: str,
    verbose: bool = False,
    top_k_per_dr: int = 20,
    max_per_file: int = 3,
) -> dict:
    """Per-DR retrieval against the table-level index.

    Returns {dr_id: [retrieve_v2 entries]}. Each entry carries its table HTML
    inline so extract can render it directly — no .txt re-parsing. We wrap
    each DR in its own single-DR plan because retrieve_v2 only honours the
    first data_request.

    `max_per_file` caps how many tables from the same bulletin can appear in
    the top-k. Previously we deduped to 1 per file, which killed the gold
    table when it wasn't the best-scoring table in its file (median intra-file
    rank of the gold table is 13). 3 gives extract a realistic shot at seeing
    the right table while keeping diversity across files.
    """
    per_dr: dict[str, list[dict]] = {}
    data_requests = spec.get("data_requests") or []
    vintage = spec.get("vintage", "latest")
    for dr in data_requests:
        dr_id = dr.get("id", "?")
        mini_plan = {"data_requests": [dr], "vintage": vintage}
        entries = retrieve_v2(
            mini_plan,
            question,
            top_k=top_k_per_dr,
            verbose=verbose,
            max_per_file=max_per_file,
            vintage=vintage,
        )
        per_dr[dr_id] = entries
    return per_dr


def _total_entries(per_dr: dict) -> int:
    return sum(len(v) for v in per_dr.values())


def _determine_source_unit(per_dr_entries: dict) -> str | None:
    """Determine the source unit from retrieval entries.

    Checks the first table entry for each data_request and returns the
    most common unit across DRs. Returns None if no unit info is found.
    """
    from collections import Counter

    units: list[str] = []
    for _dr_id, entries in per_dr_entries.items():
        for e in entries:
            raw_unit = e.get("unit")
            if raw_unit:
                parsed = parse_unit(raw_unit)
                if parsed:
                    units.append(parsed)
                    break  # first table entry with a unit is enough per DR

    if not units:
        return None

    # Return the most common unit (majority vote)
    return Counter(units).most_common(1)[0][0]


def _run_extract_and_compute(
    spec: dict,
    per_dr_entries: dict,
    question: str,
    verbose: bool,
    feedback: str = "",
    llm_counter: dict | None = None,
) -> tuple[str, dict | None]:
    """Run extract → validate → compute for one attempt. Returns
    (formatted_answer_or_error, extraction_dict_or_None). The extraction is
    returned so the caller can hand it to verify.

    Tries the deterministic fast-path first: resolve_cells() from find.py
    attempts to look up values directly from the ledger without LLM. Resolved
    DRs are used directly; unresolved DRs fall through to LLM extraction.
    When all DRs resolve deterministically, the LLM extract step is skipped.

    ``llm_counter`` is an optional mutable dict {"count": int} used by solve()
    to enforce the MAX_LLM_CALLS budget.  When extract_structured() actually
    invokes the LLM (i.e. the fast-path didn't cover all DRs), the counter is
    incremented by 1.
    """
    # ── Deterministic fast-path ─────────────────────────────────────────
    resolved_extractions, unresolved_ids = _try_deterministic_fast_path(
        spec, per_dr_entries, verbose=verbose
    )

    if not unresolved_ids:
        # All DRs resolved deterministically — skip LLM extraction
        if verbose:
            print("  Fast-path succeeded — skipping LLM extraction")
        extraction: dict | None = {"extractions": resolved_extractions, "notes": "deterministic"}
    else:
        # Some or all DRs need LLM fallback
        if resolved_extractions:
            if verbose:
                print(
                    f"  Fast-path partial: {len(resolved_extractions)} resolved, "
                    f"{len(unresolved_ids)} need LLM fallback"
                )
        else:
            if verbose:
                print("  Fast-path missed — falling back to LLM extraction")

        # Filter spec to only unresolved DRs for LLM extraction
        data_requests = spec.get("data_requests") or []
        unresolved_drs = [dr for dr in data_requests if dr.get("id") in unresolved_ids]
        filtered_spec = {**spec, "data_requests": unresolved_drs}
        # Filter per_dr_entries to only unresolved DRs
        filtered_per_dr = {k: v for k, v in per_dr_entries.items() if k in unresolved_ids}

        # Check LLM budget before calling extract
        if llm_counter is not None and llm_counter["count"] >= MAX_LLM_CALLS:
            if verbose:
                print("  LLM budget exhausted — skipping extract")
            return "EXTRACT_FAILED", None

        llm_extraction = extract_structured(
            filtered_spec,
            filtered_per_dr,
            question,
            verbose=verbose,
            feedback=feedback,
        )
        # Count the LLM call if extract_structured actually hit the LLM
        if llm_counter is not None and llm_extraction is not None:
            llm_counter["count"] += 1

        # Merge: deterministic results + LLM results
        merged_extractions = dict(resolved_extractions)  # copy deterministic
        if llm_extraction and "extractions" in llm_extraction:
            merged_extractions.update(llm_extraction["extractions"])
        extraction = {
            "extractions": merged_extractions,
            "notes": llm_extraction.get("notes", "") if llm_extraction else "",
        }

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

    source_unit = _determine_source_unit(per_dr_entries)
    formatted = format_result(result, spec.get("output_format") or {}, source_unit=source_unit)
    return formatted, extraction


def solve(question: str, verbose: bool = False, use_verify: bool = True) -> str:
    """Structured pipeline with feedback loops:

      scout → decompose → retrieve → extract → compute → verify

    One bounce-back per failure mode:
      - empty retrieve         → re-decompose with "no files matched" hint
      - verify flags extract   → re-extract with the issue as feedback
      - verify flags decompose → re-decompose + retrieve + extract again

    Retries are bounded by MAX_RETRIES_PER_PHASE (per phase) and MAX_LLM_CALLS
    (total across all phases for one question).  No infinite loop is possible.

    Each phase still fails loudly with an explicit error string so eval
    output tells us exactly which stage broke.
    """
    # LLM call counter — tracks calls to decompose, extract_structured,
    # and verify_answer to enforce MAX_LLM_CALLS bound.
    llm_calls = {"count": 0}

    def _bump_llm(phase: str) -> bool:
        """Increment LLM counter. Return True if budget remains, False if exceeded."""
        llm_calls["count"] += 1
        if llm_calls["count"] > MAX_LLM_CALLS:
            if verbose:
                print(
                    f"  LLM budget exhausted ({llm_calls['count']}/{MAX_LLM_CALLS}) "
                    f"at {phase} — stopping retries"
                )
            return False
        return True

    # Phase 0: scout (deterministic, cheap, grounds decompose)
    hint = scout(question)
    if verbose and hint:
        print(f"  Scout hint:\n{hint[:400]}")

    # Phase 1: decompose (with one retry on empty retrieve)
    if not _bump_llm("decompose"):
        return "DECOMPOSE_FAILED"
    spec = decompose(question, scout_hint=hint)
    if spec is None:
        return "DECOMPOSE_FAILED"
    if verbose:
        print(f"  Spec: {json.dumps(spec, indent=2)[:800]}")

    # Phase 2: retrieve — bottom-up primary, retrieve_v2 fallback per DR
    per_dr = retrieve_bottomup(spec, question, top_k=5)
    # Fill any DRs that got no bottom-up hits with retrieve_v2
    empty_drs = [dr_id for dr_id, entries in per_dr.items() if not entries]
    if empty_drs:
        if verbose:
            print(
                f"  Bottom-up missed {len(empty_drs)} DR(s) — falling back to retrieve_v2: {empty_drs}"
            )
        fallback = retrieve_for_spec(spec, question, verbose=verbose)
        for dr_id in empty_drs:
            per_dr[dr_id] = fallback.get(dr_id, [])

    total = _total_entries(per_dr)
    if verbose:
        bu_hits = sum(
            1
            for dr_id, entries in per_dr.items()
            if entries and entries[0].get("retrieval_channel") == "bottomup"
        )
        print(
            f"  Retrieve: {total} entries across {len(per_dr)} DRs "
            f"({bu_hits} bottom-up, {len(per_dr) - bu_hits} fallback)"
        )
    if total == 0:
        if not _bump_llm("decompose(retry)"):
            return "RETRIEVE_EMPTY"
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
    # extract_structured is called inside _run_extract_and_compute — we
    # count it as one LLM call when it actually invokes the LLM (i.e. when
    # the deterministic fast-path doesn't cover all DRs).
    answer, extraction = _run_extract_and_compute(
        spec, per_dr, question, verbose, llm_counter=llm_calls
    )

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
            llm_counter=llm_calls,
        )
        if extraction is not None and answer.startswith("NO_VALUES"):
            if not _bump_llm("decompose(no_values_retry)"):
                return answer
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
                        llm_counter=llm_calls,
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

    # Determine source unit from retrieval entries for auto-fix
    source_unit = _determine_source_unit(per_dr)
    if not _bump_llm("verify"):
        # Budget exhausted — return current answer without verification
        if verbose:
            print(f"  Answer (unverified): {answer}")
        return answer
    verdict = verify_answer(
        question,
        spec,
        extraction["extractions"],
        answer,
        verbose=verbose,
        source_unit=source_unit,
    )

    # Auto-fix: if unit correction was applied, use the corrected answer directly
    corrected = verdict.get("corrected_answer")
    if corrected and not verdict.get("ok"):
        if verbose:
            print(f"  Auto-fix applied unit correction: {answer} → {corrected}")
        answer = corrected
        if verbose:
            print(f"  Answer: {answer}")
        return answer

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
            llm_counter=llm_calls,
        )
        if extraction2 is not None and not answer2.startswith(
            ("EXTRACT_FAILED", "NO_VALUES", "COMPUTE_FAILED")
        ):
            answer = answer2

    elif phase == "decompose":
        if not _bump_llm("decompose(verify_retry)"):
            if verbose:
                print(f"  Answer (decompose retry budget exhausted): {answer}")
            return answer
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
                    llm_counter=llm_calls,
                )
                if extraction2 is not None and not answer2.startswith(
                    ("EXTRACT_FAILED", "NO_VALUES", "COMPUTE_FAILED")
                ):
                    answer = answer2

    if verbose:
        print(f"  Answer: {answer}")
    return answer


# ═══════════════════════════════════════════════════════════════════════════════
# ERROR CATEGORIZATION
# ═══════════════════════════════════════════════════════════════════════════════

# Heuristics for classifying wrong answers into failure categories when the
# pipeline returned a numeric/text answer (not an explicit error string).
_UNIT_WORDS = {"thousand", "thousands", "million", "millions", "billion", "billions"}
_FY_CY_WORDS = {"fiscal", "calendar", "fy", "cy"}


def _numeric_ratio(s: str) -> float:
    """Fraction of non-whitespace characters that are digits or commas/periods."""
    cleaned = s.strip().replace(",", "").replace(".", "")
    if not cleaned:
        return 0.0
    return sum(1 for c in cleaned if c.isdigit()) / len(cleaned)


def categorize_error(
    got: str,
    expected: str,
    rationale: str = "",
) -> str:
    """Classify a wrong answer into a failure category.

    Returns one of:
      decompose_failed  — pipeline returned DECOMPOSE_FAILED explicitly
      retrieve_empty    — pipeline returned RETRIEVE_EMPTY explicitly
      extract_failed    — pipeline returned EXTRACT_FAILED/NO_VALUES explicitly
      compute_failed    — pipeline returned COMPUTE_FAILED explicitly
      wrong_table       — answer is numeric but off by >50% (likely wrong table/row)
      unit_scaling      — answer off by ~1000× or ~1e6× (unit conversion missed)
      fy_cy_confusion   — question mentions FY/CY, answer numeric but wrong
      verify_missed     — answer passed verify but is still wrong

    When the pipeline returns an explicit error string (DECOMPOSE_FAILED,
    RETRIEVE_EMPTY, etc.), the category is determined directly. For numeric
    answers that don't match the gold answer, heuristic rules classify the
    likely failure mode.
    """
    got_upper = got.upper().strip()

    # Direct error strings from the pipeline
    if got_upper == "DECOMPOSE_FAILED":
        return "decompose_failed"
    if got_upper == "RETRIEVE_EMPTY":
        return "retrieve_empty"
    if got_upper.startswith("EXTRACT_FAILED"):
        return "extract_failed"
    if got_upper.startswith("NO_VALUES"):
        return "extract_failed"
    if got_upper.startswith("COMPUTE_FAILED"):
        return "compute_failed"

    # For actual answers that are wrong, try heuristic classification.
    # Try to extract numbers from both got and expected for ratio analysis.
    import re as _re

    def _first_number(s: str) -> float | None:
        s = _re.sub(r"[\d,]+\.\d+%", lambda m: m.group().rstrip("%"), s)
        s_clean = s.replace(",", "")
        nums = _re.findall(r"-?\d+\.?\d*", s_clean)
        for n in nums:
            try:
                val = float(n)
                if 1900 <= val <= 2100 and val == int(val):
                    continue  # skip likely years
                return val
            except ValueError:
                continue
        return None

    got_num = _first_number(got)
    exp_num = _first_number(expected)

    if got_num is not None and exp_num is not None and exp_num != 0:
        ratio = got_num / exp_num

        # Unit scaling: off by typical unit conversion factors (×1000 or ×1e6)
        # Common cases: answer in thousands but expected in raw, or vice versa
        if 900 <= ratio <= 1100 or 0.0009 <= ratio <= 0.0011:
            return "unit_scaling"
        if 0.9e6 <= ratio <= 1.1e6 or 0.9e-6 <= ratio <= 1.1e-6:
            return "unit_scaling"
        # Also check for millions ↔ thousands confusion (ratio ~1000)
        if 0.9e3 <= ratio <= 1.1e3 and ratio > 100:
            return "unit_scaling"

        # FY/CY confusion: question mentions fiscal/calendar year and
        # answer is moderately off (FY total ≈ CY total but not exact)
        got_lower = got.lower()
        exp_lower = expected.lower()
        # Check if the question context (from rationale or the answer) hints at FY/CY
        has_fy_cy_context = bool(_FY_CY_WORDS & set(got_lower.split())) or bool(
            _FY_CY_WORDS & set(exp_lower.split())
        )
        if has_fy_cy_context and 0.8 <= abs(ratio) <= 1.25 and abs(ratio) != 1.0:
            return "fy_cy_confusion"

        # Wrong table: answer is numeric but significantly off (>50%)
        if abs(ratio) > 1.5 or (0 < abs(ratio) < 0.67):
            return "wrong_table"

    # If we can't determine a more specific category, check for FY/CY hints
    # in the answer text even without numeric analysis
    if _FY_CY_WORDS & set(got.lower().split()):
        return "fy_cy_confusion"

    # Default: answer passed all pipeline stages but is still wrong
    return "verify_missed"


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
        offset = 0
        parallel = 10
        oracle = "--oracle" in eval_args
        for i, a in enumerate(eval_args):
            if a == "--n" and i + 1 < len(eval_args):
                with contextlib.suppress(ValueError):
                    n = int(eval_args[i + 1])
            if a == "--offset" and i + 1 < len(eval_args):
                with contextlib.suppress(ValueError):
                    offset = int(eval_args[i + 1])
            if a == "--parallel" and i + 1 < len(eval_args):
                with contextlib.suppress(ValueError):
                    parallel = int(eval_args[i + 1])

        rows = []
        with open("officeqa_full.csv") as f:
            for row in csv.DictReader(f):
                rows.append(row)

        if offset:
            rows = rows[offset:]
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
                    source_unit = _determine_source_unit(filtered)
                    return row, format_result(
                        result, spec.get("output_format") or {}, source_unit=source_unit
                    )
                except ComputeError as e:
                    return row, f"COMPUTE_FAILED: {e}"
            return row, solve(question, verbose=verbose)

        correct, total = 0, 0
        error_categories: dict[str, int] = {}
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
                    category = categorize_error(got, expected, rationale)
                    error_categories[category] = error_categories.get(category, 0) + 1
                    print(
                        f"  [{total:3d}/{len(rows)}] {tag} {row['uid']}: "
                        f"expected={expected!r} got={got!r} [{category}] ({rationale})",
                        flush=True,
                    )

        print(f"\nAccuracy: {correct}/{total} = {correct / total * 100:.1f}%")
        if error_categories:
            print("\nError categories:")
            for cat in sorted(error_categories, key=error_categories.get, reverse=True):  # type: ignore[arg-type]
                count = error_categories[cat]
                pct = count / total * 100
                print(f"  {cat}: {count} ({pct:.1f}%)")
            n_categorized = sum(error_categories.values())
            print(f"  Total errors: {n_categorized}/{total}")

    elif args:
        question = " ".join(args)
        print(f"\n❓ {question}\n")
        answer = solve(question, verbose=verbose)
        print(f"\n✅ {answer}\n")
    else:
        print("Usage: uv run python solve.py [-v] [--eval [--n N] [--oracle]] <question>")
