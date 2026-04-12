#!/usr/bin/env python3
"""Oracle decompose eval — does the spec point at the gold data?

For every question, open the gold source file(s) directly and check whether
the decomposed spec's row_hint / column_hint / years actually resolve to
something real in those files. No retrieval, no LLM in the loop — this
isolates decompose quality from every other phase.

Input:  decompose_eval.full.jsonl  (cached specs + gold_files per question)
Output: breakdown of decompose errors by category

Buckets per DR:
  OK            — row_hint, column_hint (if set), and year all resolve
                  against at least one gold-file table
  ROW_MISS      — row_hint doesn't substring-match any row_leaf in gold tables
  COL_MISS      — column_hint set but doesn't match any col_leaf
  YEAR_MISS     — no gold-file table has the requested year
  GOLD_NOT_IN_LEDGER — gold file has no tables in the ledger (shouldn't happen)
  COHORT        — cohort DR (descriptive row_hint, skipped)
  EMPTY         — DR has no usable hints

A row is "clean" only if every DR is OK (or COHORT).
"""

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

DB_PATH = "ledger.sqlite"


def norm_file(name: str) -> str:
    """Strip extension + normalise — match between CSV's .txt and ledger's .json."""
    stem = Path(name).stem.lower()
    return stem


def load_gold_tables(conn: sqlite3.Connection, gold_files: list[str]) -> list[int]:
    """Return table_ids for every table whose file matches any gold file."""
    if not gold_files:
        return []
    stems = [norm_file(f) for f in gold_files]
    placeholders = ",".join("?" * len(stems))
    q = f"""
        SELECT id FROM tables
        WHERE LOWER(REPLACE(REPLACE(file, '.json', ''), '.txt', '')) IN ({placeholders})
          AND table_kind = 'data'
    """
    return [r[0] for r in conn.execute(q, stems).fetchall()]


def load_row_leaves(conn: sqlite3.Connection, table_ids: list[int]) -> list[str]:
    if not table_ids:
        return []
    ph = ",".join("?" * len(table_ids))
    rows = conn.execute(
        f"SELECT row_leaf FROM table_rows WHERE table_id IN ({ph}) AND row_leaf IS NOT NULL",
        table_ids,
    ).fetchall()
    return [r[0] for r in rows]


def load_col_leaves(conn: sqlite3.Connection, table_ids: list[int]) -> list[str]:
    if not table_ids:
        return []
    ph = ",".join("?" * len(table_ids))
    rows = conn.execute(
        f"SELECT col_leaf FROM table_columns WHERE table_id IN ({ph}) AND col_leaf IS NOT NULL",
        table_ids,
    ).fetchall()
    return [r[0] for r in rows]


def load_years_covered(conn: sqlite3.Connection, table_ids: list[int]) -> set[int]:
    if not table_ids:
        return set()
    ph = ",".join("?" * len(table_ids))
    years: set[int] = set()
    for tbl in ("table_rows", "table_columns"):
        for r in conn.execute(
            f"SELECT DISTINCT year_extracted FROM {tbl} "
            f"WHERE table_id IN ({ph}) AND year_extracted IS NOT NULL",
            table_ids,
        ):
            years.add(r[0])
    return years


_PUNCT_RE = re.compile(r"[^\w\s]")
_FOOTNOTE_RE = re.compile(r"\b[1-9]\s*/|\b[rpe]\s*/", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def _normalize(s: str) -> str:
    """Lowercase, strip footnote markers and punctuation, collapse whitespace."""
    if not s:
        return ""
    s = _FOOTNOTE_RE.sub(" ", s.lower())
    s = _PUNCT_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


def substring_match(hint: str, haystack: list[str]) -> bool:
    """Match via normalized substring OR token-subset.

    Token-subset: every word in the hint (>=3 chars) appears somewhere in
    the leaf. Handles 'Individual income taxes, net' matching 'Individual
    income taxes' and 'Net interest' matching 'Interest, net'.
    """
    h_norm = _normalize(hint)
    if not h_norm:
        return False
    h_tokens = [t for t in h_norm.split() if len(t) >= 3]
    if not h_tokens:
        return False
    for leaf in haystack:
        leaf_norm = _normalize(leaf)
        if h_norm in leaf_norm:
            return True
        leaf_tokens = set(leaf_norm.split())
        if all(t in leaf_tokens for t in h_tokens):
            return True
    return False


def classify_dr(
    dr: dict,
    row_leaves: list[str],
    col_leaves: list[str],
    years_covered: set[int],
) -> str:
    row_hint = (dr.get("row_hint") or "").strip()
    col_hint = (dr.get("column_hint") or "").strip()
    alts = dr.get("row_hint_alternatives") or []
    years = dr.get("years") or []
    cohort = dr.get("cohort") or False
    source = dr.get("source", "corpus")

    # Non-corpus DRs (CPI, FX, external) can't be checked against the ledger.
    if source != "corpus":
        return "OK"  # trust them

    if cohort:
        return "COHORT"

    if not row_hint and not col_hint and not dr.get("label"):
        return "EMPTY"

    # Row hint check (try alternatives too)
    row_ok = True
    if row_hint:
        row_ok = substring_match(row_hint, row_leaves) or any(
            substring_match(a, row_leaves) for a in alts
        )

    # Column hint check
    col_ok = True
    if col_hint:
        col_ok = substring_match(col_hint, col_leaves)

    # Year check
    year_ok = True
    if years:
        year_ok = any(int(y) in years_covered for y in years)

    if not year_ok:
        return "YEAR_MISS"
    if not row_ok:
        return "ROW_MISS"
    if not col_ok:
        return "COL_MISS"
    return "OK"


def evaluate(records: list[dict]) -> tuple[dict, list]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    dr_counts: Counter = Counter()
    row_clean = 0
    per_row_problems: list[dict] = []

    for rec in records:
        uid = rec["uid"]
        gold_files = rec.get("gold_files") or []
        spec = rec.get("spec")
        if spec is None:
            dr_counts["NO_SPEC"] += 1
            per_row_problems.append({"uid": uid, "fails": ["NO_SPEC"]})
            continue

        table_ids = load_gold_tables(conn, gold_files)
        if not table_ids:
            dr_counts["GOLD_NOT_IN_LEDGER"] += 1
            per_row_problems.append(
                {"uid": uid, "fails": ["GOLD_NOT_IN_LEDGER"], "gold_files": gold_files}
            )
            continue

        row_leaves = load_row_leaves(conn, table_ids)
        col_leaves = load_col_leaves(conn, table_ids)
        years_covered = load_years_covered(conn, table_ids)

        drs = spec.get("data_requests") or []
        dr_tags: list[str] = []
        for dr in drs:
            tag = classify_dr(dr, row_leaves, col_leaves, years_covered)
            dr_tags.append(tag)
            dr_counts[tag] += 1

        # "clean" = every DR OK or COHORT
        if all(t in ("OK", "COHORT") for t in dr_tags):
            row_clean += 1
        else:
            per_row_problems.append(
                {
                    "uid": uid,
                    "fails": dr_tags,
                    "n_tables": len(table_ids),
                    "first_dr": drs[0] if drs else None,
                    "gold_files": gold_files,
                }
            )

    summary = {
        "total_rows": len(records),
        "clean_rows": row_clean,
        "clean_rows_pct": round(row_clean / len(records) * 100, 1) if records else 0.0,
        "dr_counts": dict(dr_counts),
    }
    return summary, per_row_problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default="decompose_eval.full.jsonl")
    ap.add_argument("--uids", type=str, default="")
    ap.add_argument("--out", type=Path, default=Path("decompose_oracle_problems.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()

    path = Path(args.path)
    records = [json.loads(line) for line in path.open()]
    selected = {u.strip() for u in args.uids.split(",") if u.strip()}
    if selected:
        records = [r for r in records if r.get("uid") in selected]
    print(f"Oracle-checking {len(records)} decomposed specs from {path}\n")

    summary, problems = evaluate(records)

    print("── DR classification counts ──")
    total_drs = sum(summary["dr_counts"].values())
    for tag, n in sorted(summary["dr_counts"].items(), key=lambda x: -x[1]):
        pct = n / total_drs * 100 if total_drs else 0
        print(f"  {tag:22s} {n:4d}  ({pct:5.1f}%)")

    print("\n── Row-level summary ──")
    clean = summary["clean_rows"]
    total = summary["total_rows"]
    print(f"  fully clean:    {clean}/{total}  ({clean / total * 100:.1f}%)")
    print(f"  has problem:    {total - clean}/{total}  ({(total - clean) / total * 100:.1f}%)")

    # Show a few examples of each non-OK bucket
    print("\n── Example problems (first 10) ──")
    for p in problems[:10]:
        print(f"  {p['uid']}  {p['fails']}")
        if p.get("first_dr"):
            dr = p["first_dr"]
            print(
                f"    row_hint={dr.get('row_hint')!r}  col_hint={dr.get('column_hint')!r}  years={dr.get('years')}"
            )
            if p.get("gold_files"):
                print(f"    gold_files={p['gold_files']}")

    # Dump full problem list for drilldown
    with open(args.out, "w") as f:
        for p in problems:
            f.write(json.dumps(p) + "\n")
    print(f"\nFull problem list → {args.out}")
    if args.summary_out:
        args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Summary → {args.summary_out}")


if __name__ == "__main__":
    main()
