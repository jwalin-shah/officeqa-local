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
import sys

from dotenv import load_dotenv

from compute import ComputeError, format_result, parse_unit, validate_extractions
from compute import execute as compute_execute
from decompose import decompose
from extract import extract_structured
from extract_sql import extract_sql
from find import retrieve_bottomup, try_deterministic_fast_path
from ledger_paths import get_ledger_sqlite_path
from retrieve_v2 import retrieve as retrieve_v2
from scout import scout
from verify import verify_answer

sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
load_dotenv()

# Max retries per phase and total LLM budget per question.
MAX_RETRIES_PER_PHASE = 2
MAX_LLM_CALLS = 6  # decompose + extract + verify, up to 2 retries each


# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════


def retrieve_for_spec(
    spec: dict,
    question: str,
    verbose: bool = False,
    top_k_per_dr: int = 20,
    max_per_file: int = 3,
    llm_counter: dict | None = None,
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

    `llm_counter` is threaded through to retrieve_v2 so the LLM rerank gate
    can check and bump the budget counter.
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
            llm_counter=llm_counter,
            max_llm_calls=MAX_LLM_CALLS,
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
    resolved_extractions, unresolved_ids = try_deterministic_fast_path(
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


def solve(
    question: str, verbose: bool = False, use_verify: bool = True, cached_spec: dict | None = None
) -> str:
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

    Args:
      question: the question to solve
      verbose: print debug output
      use_verify: run verification phase
      cached_spec: optional pre-computed QuestionSpec to skip decompose LLM call
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
    # If a cached spec is provided (e.g., from --cached-decompose eval mode),
    # skip the decompose LLM call and use it directly.
    if cached_spec:
        spec = cached_spec
    else:
        if not _bump_llm("decompose"):
            return "DECOMPOSE_FAILED"
        spec = decompose(question, scout_hint=hint)
        if spec is None:
            return "DECOMPOSE_FAILED"
    if verbose:
        print(f"  Spec: {json.dumps(spec, indent=2)[:800]}")

    # Phase 2: retrieve — always run retrieve_v2 for a diverse candidate pool;
    # merge with bottom-up when it finds exact cell hits (dedup by element_seq).
    bottomup = retrieve_bottomup(spec, question, top_k=5)
    per_dr = retrieve_for_spec(spec, question, verbose=verbose, llm_counter=llm_calls)
    # Prepend any bottom-up hits that aren't already in the retrieve_v2 pool.
    for dr_id, bu_entries in bottomup.items():
        if not bu_entries:
            continue
        existing_seqs = {(e["file"], e["element_seq"]) for e in per_dr.get(dr_id, [])}
        prepend = [e for e in bu_entries if (e["file"], e["element_seq"]) not in existing_seqs]
        if prepend:
            per_dr[dr_id] = prepend + per_dr.get(dr_id, [])

    total = _total_entries(per_dr)
    if verbose:
        bu_hits = sum(
            1
            for dr_id, entries in per_dr.items()
            if entries and entries[0].get("retrieval_channel") == "bottomup"
        )
        print(f"  Retrieve: {total} entries across {len(per_dr)} DRs ({bu_hits} bottom-up-led)")
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
        per_dr = retrieve_for_spec(spec, question, verbose=verbose, llm_counter=llm_calls)
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
                per_dr2 = retrieve_for_spec(spec2, question, verbose=verbose, llm_counter=llm_calls)
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
        per_dr_entries=per_dr,
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
            per_dr2 = retrieve_for_spec(spec2, question, verbose=verbose, llm_counter=llm_calls)
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

    elif phase == "retrieve":
        # v1: log-and-continue — do not spend a retry LLM call on re-retrieval.
        # Measure signal quality first; retrieval retry can be added once we
        # confirm the flag is reliable.
        print(
            f"  [verify_retrieve_flag] verify flagged wrong table: {issue}",
            flush=True,
        )

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
        use_cached_decompose = "--cached-decompose" in eval_args

        # Load cached decompose specs if requested (skips live LLM decompose call)
        _cached_specs: dict[str, dict] = {}
        if use_cached_decompose:
            import json as _jmod

            cache_path = "decompose_eval.full.jsonl"
            if os.path.exists(cache_path):
                with open(cache_path) as _cf:
                    for _line in _cf:
                        _line = _line.strip()
                        if _line:
                            _entry = _jmod.loads(_line)
                            _cached_specs[_entry["uid"]] = _entry.get("spec", {})
                print(f"Loaded {len(_cached_specs)} cached decompose specs", flush=True)

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

        def _extract_years_from_files(source_files_str: str) -> set[int]:
            """Extract available years from gold file names.

            Treasury bulletin filenames are treasury_bulletin_YYYY_MM.json,
            so YYYY is the year of data in that file.
            """
            years = set()
            for fname in source_files_str.split("\n"):
                fname = fname.strip().replace(".txt", "").replace(".json", "")
                # Extract YYYY from treasury_bulletin_YYYY_MM
                parts = fname.split("_")
                if len(parts) >= 3 and parts[0] == "treasury" and parts[1] == "bulletin":
                    try:
                        year = int(parts[2])
                        years.add(year)
                    except (ValueError, IndexError):
                        pass
            return years

        def _enrich_oracle_entries_with_ledger(
            entries: list[dict], source_files_str: str
        ) -> list[dict]:
            """Enrich oracle entries with ledger metadata (years, column headers, row labels).

            Queries ledger.sqlite to populate column_headers, row_labels, years, etc.
            This helps the LLM extract correctly by providing structured table info.
            """
            import sqlite3 as _sqlite3

            ledger_path = get_ledger_sqlite_path()
            if not ledger_path or not os.path.exists(ledger_path):
                return entries  # Ledger not available, return as-is

            # Get file stems to match against ledger
            stems = [
                f.strip().replace(".txt", "").replace(".json", "")
                for f in source_files_str.split("\n")
                if f.strip()
            ]
            file_patterns = [f"{s}.json" for s in stems]

            try:
                conn = _sqlite3.connect(ledger_path)
                for entry in entries:
                    entry_file = entry.get("file", "")
                    if entry_file not in file_patterns:
                        continue

                    # Query ledger for metadata about this file's tables
                    cursor = conn.cursor()

                    # Get all tables in this file
                    cursor.execute(
                        "SELECT id, n_rows, n_cols FROM tables WHERE file = ? ORDER BY element_seq",
                        (entry_file,),
                    )
                    table_rows = cursor.fetchall()

                    if not table_rows:
                        continue

                    # For now, just get metadata from the first table in the file
                    # (oracle mode gives all tables, so we pick first)
                    first_table_id = table_rows[0][0]

                    # Get column headers with years
                    cursor.execute(
                        "SELECT col_path, col_leaf, year_extracted FROM table_columns WHERE table_id = ? ORDER BY col_index",
                        (first_table_id,),
                    )
                    cols = cursor.fetchall()
                    entry["column_headers"] = [
                        col[1] or col[0] for col in cols
                    ]  # leaf or full path
                    col_years = {col[2] for col in cols if col[2]}
                    entry["years"] = sorted(col_years) if col_years else []

                    # Get row labels with years
                    cursor.execute(
                        "SELECT row_path, row_leaf, year_extracted FROM table_rows WHERE table_id = ? AND NOT is_section_header ORDER BY row_index LIMIT 30",
                        (first_table_id,),
                    )
                    rows = cursor.fetchall()
                    entry["row_labels"] = [row[1] or row[0] for row in rows]  # leaf or full path
                    row_years = {row[2] for row in rows if row[2]}
                    if row_years:
                        entry["years"] = sorted(set(entry.get("years", []) | row_years))

                    # Get table dimensions
                    cursor.execute(
                        "SELECT n_rows, n_cols FROM tables WHERE id = ?", (first_table_id,)
                    )
                    dims = cursor.fetchone()
                    if dims:
                        entry["n_rows"] = dims[0]
                        entry["n_cols"] = dims[1]

                conn.close()
            except Exception:
                pass  # Ledger error, continue with unenriched entries

            return entries

        def _load_oracle_entries(source_files_str: str) -> list[dict]:
            """Load all table entries directly from corpus_json gold files.

            Bypasses retrieval entirely — gives extract the actual gold tables
            without any ranking or filtering. Each table becomes one entry with
            the same fields that retrieve_v2 produces.
            """
            import json as _json
            import os as _os

            entries = []
            corpus_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "corpus_json")
            stems = [
                f.strip().replace(".txt", "").replace(".json", "")
                for f in source_files_str.split("\n")
                if f.strip()
            ]
            for stem in stems:
                fpath = _os.path.join(corpus_dir, f"{stem}.json")
                if not _os.path.exists(fpath):
                    continue
                try:
                    with open(fpath) as fh:
                        doc = _json.load(fh)
                    elements = doc.get("document", {}).get("elements", [])
                    # Gather section context before each table
                    current_section = ""
                    for elem in elements:
                        etype = elem.get("type", "")
                        if etype in ("section_header", "title"):
                            current_section = (elem.get("content") or "").strip()
                        if etype != "table":
                            continue
                        html = elem.get("content") or ""
                        if not html:
                            continue
                        entries.append(
                            {
                                "file": f"{stem}.json",
                                "element_id": elem.get("id", ""),
                                "element_seq": len(entries),
                                "page_id": (elem.get("bbox") or [{}])[0].get("page_id", 0)
                                if isinstance(elem.get("bbox"), list)
                                else (elem.get("bbox") or {}).get("page_id", 0),
                                "section": current_section,
                                "title": current_section,
                                "caption": "",
                                "column_headers": [],
                                "row_labels": [],
                                "years": [],
                                "unit": "",
                                "period": "",
                                "n_rows": 0,
                                "n_cols": 0,
                                "retrieval_strategy": "oracle_direct",
                                "retrieval_channel": "oracle",
                                "html": html,
                                "near_content": [],
                                "probe_matched_rows": 0,
                                "probe_best_cells": [],
                                "probe_col_matches": 0,
                                "has_target_year": True,
                                "has_month_data": False,
                                "has_required_granularity": True,
                                "file_year": None,
                                "file_month": None,
                                "signature": "",
                            }
                        )
                except Exception:
                    continue
            return entries

        def _solve_one(row: dict) -> tuple[dict, str]:
            question = row["question"]
            if oracle:
                # Oracle mode: load gold tables directly from corpus_json —
                # no retrieval step. Tests the extract+compute ceiling.
                if use_cached_decompose and row.get("uid") in _cached_specs:
                    spec = _cached_specs[row["uid"]]
                else:
                    spec = decompose(question, scout_hint=scout(question))
                if spec is None:
                    return row, "DECOMPOSE_FAILED"

                # Validate: check if spec asks for years in the gold files
                available_years = _extract_years_from_files(row["source_files"])
                spec_years = set()
                for dr in spec.get("data_requests", []):
                    spec_years.update(dr.get("years") or [])

                # If spec asks for years not in gold files, warn (but continue)
                missing_years = spec_years - available_years
                if missing_years and verbose:
                    print(
                        f"  ⚠️  Spec asks for years {sorted(missing_years)} not in gold files {sorted(available_years)}"
                    )

                # Oracle mode: extract values via SQL queries instead of HTML parsing
                # LLM writes SQL to query the ledger directly against gold file
                file_stem = row["source_files"].strip().replace(".txt", "").replace(".json", "")
                gold_file = f"{file_stem}.json"

                extraction = extract_sql(question, gold_file, spec, verbose=verbose)
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
            # Normal (non-oracle) eval path: use cached_spec if available to skip decompose LLM call
            cached = _cached_specs.get(row.get("uid")) if use_cached_decompose else None
            return row, solve(question, verbose=verbose, cached_spec=cached)

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
