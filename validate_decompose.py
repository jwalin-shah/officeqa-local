#!/usr/bin/env python3
"""Structural sanity check on cached decompose specs.

Flags specs that are *definitely* broken, independent of semantic accuracy:
  - missing or empty data_requests
  - descriptive row_hint (long sentence instead of a literal row label); skipped
    when ``cohort: true`` (cohort hints are intentionally phrased as filters)
  - unparseable computation_spec.python_template
  - missing years on a question that contains explicit year tokens (single DR)
  - ``years`` entries that are not integer-like numbers (rejects bools, strings,
    and non-whole floats)
  - data_request with both row_hint and column_hint empty (orphan request)

Usage:
  uv run python validate_decompose.py [decompose_eval.full.jsonl]
"""

import ast
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b")


def _year_entry_ok(y: object) -> bool:
    """True if ``y`` is a strict numeric integer year (rejects bool and strings)."""
    if isinstance(y, bool):
        return False
    if isinstance(y, int):
        return True
    return isinstance(y, float) and y.is_integer()


def check_spec(row: dict) -> list[str]:
    """Return a list of failure tags for this row. Empty list = all good."""
    fails: list[str] = []
    spec = row.get("spec")
    if spec is None:
        fails.append("NO_SPEC")
        return fails

    drs = spec.get("data_requests") or []
    if not drs:
        fails.append("EMPTY_DATA_REQUESTS")
        return fails

    q = row.get("question") or ""
    q_years = set(int(y) for y in YEAR_RE.findall(q))

    # Descriptive row_hint: more than 8 words, or contains verbs like
    # "excluding", "all", "rows", "aggregates" — these are instructions
    # about rows, not literal row labels.
    descriptive_markers = re.compile(
        r"\b(excluding|all\s+\w+\s+rows|aggregates?|rows?,|including|"
        r"e\.g\.|such\s+as)\b",
        re.IGNORECASE,
    )

    for i, dr in enumerate(drs):
        tag = f"dr{i}"
        row_hint = (dr.get("row_hint") or "").strip()
        col_hint = (dr.get("column_hint") or "").strip()
        label = (dr.get("label") or "").strip()
        years = dr.get("years") or []
        is_cohort = bool(dr.get("cohort"))

        # Orphan: every hint field is empty. Retrieve can't do anything.
        if not row_hint and not col_hint and not label:
            fails.append(f"{tag}:ORPHAN_DR")
            continue

        # Descriptive row_hint — substring match won't work against a
        # paragraph of prose describing which rows to pick.
        if row_hint and not is_cohort:
            word_count = len(row_hint.split())
            if word_count > 8:
                fails.append(f"{tag}:LONG_ROW_HINT({word_count}w)")
            elif descriptive_markers.search(row_hint):
                fails.append(f"{tag}:DESCRIPTIVE_ROW_HINT")

        for y in years:
            if not _year_entry_ok(y):
                fails.append(f"{tag}:NON_INTEGER_YEAR")
                break

        # Year extraction: if the question has explicit years but the DR
        # doesn't surface any, something went sideways. (Not all DRs need
        # years — comparative questions often split years across DRs.)
        if q_years and not years and len(drs) == 1:
            fails.append(f"{tag}:MISSING_YEARS")

    # Unparseable python_template
    tmpl = (spec.get("computation_spec") or {}).get("python_template")
    if tmpl:
        try:
            ast.parse(tmpl)
        except SyntaxError:
            fails.append("BAD_PYTHON_TEMPLATE")

    return fails


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "decompose_eval.full.jsonl")
    rows = [json.loads(line) for line in path.open()]
    print(f"Checking {len(rows)} decomposed specs from {path}\n")

    n_clean = 0
    fail_counts: dict[str, int] = {}
    per_uid_fails: list[tuple[str, list[str]]] = []

    for row in rows:
        fails = check_spec(row)
        if not fails:
            n_clean += 1
        else:
            per_uid_fails.append((row["uid"], fails))
            for f in fails:
                # Bucket by failure *category*, stripping the dr-index prefix
                # so LONG_ROW_HINT(12w) and dr1:LONG_ROW_HINT(8w) group
                key = f.split(":", 1)[-1].split("(", 1)[0]
                fail_counts[key] = fail_counts.get(key, 0) + 1

    print("── Per-failure-category counts ──")
    for k, v in sorted(fail_counts.items(), key=lambda x: -x[1]):
        print(f"  {k:30s} {v:4d}")

    print("\n── Summary ──")
    print(f"  clean:    {n_clean}/{len(rows)} ({n_clean / len(rows) * 100:.0f}%)")
    print(
        f"  flagged:  {len(rows) - n_clean}/{len(rows)} ({(len(rows) - n_clean) / len(rows) * 100:.0f}%)"
    )

    if per_uid_fails[:20]:
        print("\n── First 20 flagged specs (uid → failures) ──")
        for uid, fails in per_uid_fails[:20]:
            print(f"  {uid}  {fails}")


if __name__ == "__main__":
    main()
