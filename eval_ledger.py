#!/usr/bin/env python3
"""Measure ledger structural reachability of benchmark gold answers.

For each question in officeqa_full.csv, the benchmark gives us one or more
`source_docs` URLs of the form `...?page=N` plus the matching `source_files`.
We check whether the ledger successfully parsed and stored a `data` table at
that exact (file, page) location — the ingest-quality question, independent
of whether the answer is a single cell lookup or a calculation over many.

Metrics reported:
  reachable@file    — ≥1 data table from the gold source file(s) exists
  reachable@page    — ALL gold (file, page) locations are covered by ≥1 data table
  reachable@page_any — ≥1 gold (file, page) is covered (loose single-location metric)
  cell_present@page — for numeric-gold questions only, the gold value appears
                      as a cell inside a page-matched table (bonus signal;
                      calculation questions won't and shouldn't hit this)

For multi-location questions (e.g. year-over-year comparisons that cite two
bulletins) the strict `reachable@page` requires *every* cited location to be
present — missing one side still makes the question unanswerable. The looser
`reachable@page_any` shows how many questions have at least partial coverage.

Usage:
  uv run python eval_ledger.py
  uv run python eval_ledger.py --n 50        # first 50 questions
  uv run python eval_ledger.py --verbose     # per-question result
"""

import argparse
import csv
import json
import re
import sqlite3
import sys
from pathlib import Path

LEDGER_PATH = Path("ledger.sqlite")
BENCHMARK_PATH = Path("officeqa_full.csv")

URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")
BULLETIN_STEM_RE = re.compile(r"(treasury_bulletin_\d{4}_\d{2})")


def parse_gold_value(raw: str) -> float | None:
    if raw is None:
        return None
    s = raw.strip()
    if not s:
        return None
    s = re.sub(r"[\$%]", "", s)
    s = s.replace(",", "").strip()
    s = re.sub(r"\s*(million|billion|thousand|percent|%)\s*$", "", s, flags=re.I)
    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1].strip()
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def parse_gold_locations(source_docs: str, source_files: str) -> list[tuple[str, int]]:
    """Zip source_docs URLs with source_files stems, extracting ?page=N.

    Each line in source_docs is a URL; each line in source_files is a filename.
    The two columns are order-aligned in the benchmark CSV. Returns a list of
    (file_stem_with_json_ext, page_int) tuples. Lines without ?page=N are
    dropped — that question contributes a file-only match, tracked separately.
    """
    doc_lines = [line.strip() for line in (source_docs or "").split("\n") if line.strip()]
    file_lines = [line.strip() for line in (source_files or "").split("\n") if line.strip()]
    out: list[tuple[str, int]] = []
    # Pair by index; tolerate length mismatches by zipping to the shorter list.
    for url, fname in zip(doc_lines, file_lines, strict=False):
        m = URL_PAGE_RE.search(url)
        if not m:
            continue
        page = int(m.group(1))
        stem = Path(fname).stem  # strip .txt → treasury_bulletin_YYYY_MM
        out.append((f"{stem}.json", page))
    return out


def parse_gold_files(source_files: str) -> set[str]:
    out: set[str] = set()
    for line in (source_files or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        stem = Path(line).stem
        out.add(f"{stem}.json")
    return out


def check_reachable_file(conn: sqlite3.Connection, files: set[str]) -> int:
    """Return count of data tables from any of the given files."""
    if not files:
        return 0
    placeholders = ",".join("?" * len(files))
    row = conn.execute(
        f"SELECT COUNT(*) FROM tables WHERE file IN ({placeholders}) AND table_kind = 'data'",
        tuple(files),
    ).fetchone()
    return row[0]


def check_reachable_page(conn: sqlite3.Connection, locations: list[tuple[str, int]]) -> list[int]:
    """For each (file, page) location, return the count of data tables found.
    A question is reachable@page if any element of the returned list is > 0.
    """
    counts: list[int] = []
    for file, page in locations:
        row = conn.execute(
            "SELECT COUNT(*) FROM tables WHERE file = ? AND page_id = ? AND table_kind = 'data'",
            (file, page),
        ).fetchone()
        counts.append(row[0])
    return counts


def check_cell_present_on_page(
    conn: sqlite3.Connection,
    locations: list[tuple[str, int]],
    gold_value: float,
    tol: float = 0.01,
) -> tuple[bool, float | None]:
    """For numeric-gold questions: check if the gold value appears as a cell
    inside a table at one of the gold (file, page) locations. Tries scale
    variants (×1, ×1000, ÷1000) to bridge unit mismatches. Returns (hit, scale)."""
    if gold_value is None or not locations:
        return False, None

    table_ids: list[int] = []
    for file, page in locations:
        rows = conn.execute(
            "SELECT id FROM tables WHERE file = ? AND page_id = ? AND table_kind = 'data'",
            (file, page),
        ).fetchall()
        table_ids.extend(r[0] for r in rows)

    if not table_ids:
        return False, None

    placeholders = ",".join("?" * len(table_ids))

    for scale in (1.0, 1_000.0, 0.001, 1_000_000.0, 1e-6):
        target = gold_value * scale
        if target == 0:
            lo, hi = -1e-9, 1e-9
        else:
            lo = target - abs(target) * tol
            hi = target + abs(target) * tol
        row = conn.execute(
            f"""
            SELECT 1 FROM cells
            WHERE table_id IN ({placeholders})
              AND numeric_value BETWEEN ? AND ?
            LIMIT 1
            """,
            (*table_ids, lo, hi),
        ).fetchone()
        if row is not None:
            return True, scale

    return False, None


def _parse_uids(raw: str) -> set[str]:
    return {u.strip() for u in raw.split(",") if u.strip()}


def run(
    n: int = 0,
    verbose: bool = False,
    uids: set[str] | None = None,
    out: Path | None = None,
    summary_out: Path | None = None,
) -> None:
    if not LEDGER_PATH.exists():
        print(f"{LEDGER_PATH} does not exist. Run build_ledger.py first.", file=sys.stderr)
        sys.exit(1)

    conn = sqlite3.connect(str(LEDGER_PATH))

    with BENCHMARK_PATH.open() as f:
        rows = list(csv.DictReader(f))
    if uids:
        rows = [row for row in rows if row.get("uid") in uids]
    if n:
        rows = rows[:n]

    print(f"Evaluating structural reachability on {len(rows)} questions...\n")

    total = 0
    parsed_gold = 0
    unparseable_gold = 0
    no_gold_files = 0
    no_page_anchor = 0

    reachable_file = 0
    reachable_page_all = 0  # every gold location covered (strict)
    reachable_page_any = 0  # at least one gold location covered (loose)
    multi_loc_questions = 0
    multi_loc_full = 0  # multi-loc with every side covered
    multi_loc_partial = 0  # multi-loc with some but not all sides
    cell_present_page = 0

    gaps_file: list[tuple[str, str]] = []  # not even reachable@file
    gaps_page: list[tuple[str, str]] = []  # reachable@file but no page hit at all
    gaps_partial: list[tuple[str, str]] = []  # multi-loc with partial coverage
    detailed: list[dict] = []

    for i, row in enumerate(rows, 1):
        total += 1
        uid = row.get("uid", "?")
        question = row.get("question", "")
        gold_raw = row.get("answer", "")
        source_docs = row.get("source_docs", "")
        source_files = row.get("source_files", "")

        gold_value = parse_gold_value(gold_raw)
        gold_files = parse_gold_files(source_files)
        gold_locations = parse_gold_locations(source_docs, source_files)

        if gold_value is None:
            unparseable_gold += 1
        else:
            parsed_gold += 1

        if not gold_files:
            no_gold_files += 1
        if not gold_locations:
            no_page_anchor += 1

        file_hits = check_reachable_file(conn, gold_files)
        page_counts = check_reachable_page(conn, gold_locations)
        n_loc = len(page_counts)
        n_loc_hit = sum(1 for c in page_counts if c > 0)
        any_page_hit = n_loc_hit > 0
        all_page_hit = n_loc > 0 and n_loc_hit == n_loc
        is_multi = n_loc >= 2

        if file_hits > 0:
            reachable_file += 1
        else:
            gaps_file.append((uid, question[:80]))

        if all_page_hit:
            reachable_page_all += 1
        if any_page_hit:
            reachable_page_any += 1
        elif file_hits > 0:
            locs = ", ".join(f"{Path(f).stem}:p{p}" for f, p in gold_locations)
            gaps_page.append((uid, f"{question[:60]}  [{locs}]"))

        if is_multi:
            multi_loc_questions += 1
            if all_page_hit:
                multi_loc_full += 1
            elif any_page_hit:
                multi_loc_partial += 1
                miss = ", ".join(
                    f"{Path(f).stem}:p{p}"
                    for (f, p), c in zip(gold_locations, page_counts, strict=False)
                    if c == 0
                )
                gaps_partial.append((uid, f"{question[:60]}  missing=[{miss}]"))

        cell_hit = False
        scale_hint: float | None = None
        if gold_value is not None and all_page_hit:
            cell_hit, scale_hint = check_cell_present_on_page(conn, gold_locations, gold_value)
            if cell_hit:
                cell_present_page += 1

        if verbose:
            mark = (
                "CELL"
                if cell_hit
                else ("PAGE" if any_page_hit else ("FILE" if file_hits > 0 else "MISS"))
            )
            print(
                f"  [{i:3d}] {uid} {mark:4s}  gold={gold_raw[:14]:14s}  "
                f"file_hits={file_hits:4d}  page_counts={page_counts}  scale={scale_hint}"
            )
        detailed.append(
            {
                "uid": uid,
                "gold_files": sorted(gold_files),
                "gold_locations": gold_locations,
                "reachable_file": file_hits > 0,
                "reachable_page_all": all_page_hit,
                "reachable_page_any": any_page_hit,
                "cell_present_page": cell_hit,
                "page_counts": page_counts,
            }
        )

    conn.close()

    def pct(x: int, d: int) -> str:
        return f"{(x / d * 100):.1f}%" if d else "n/a"

    print("\n─── Results ────────────────────────────────────────────")
    print(f"  total questions:              {total:5d}")
    print(f"  numeric gold parsed:          {parsed_gold:5d}  ({pct(parsed_gold, total)})")
    print(
        f"  unparseable gold:             {unparseable_gold:5d}  ({pct(unparseable_gold, total)})"
    )
    print(f"  questions w/o gold files:     {no_gold_files:5d}")
    print(f"  questions w/o ?page anchor:   {no_page_anchor:5d}")
    print()
    print(f"  reachable@file:               {reachable_file:5d}  ({pct(reachable_file, total)})")
    print(
        f"  reachable@page (strict/all):  {reachable_page_all:5d}  ({pct(reachable_page_all, total)})"
    )
    print(
        f"  reachable@page_any (loose):   {reachable_page_any:5d}  ({pct(reachable_page_any, total)})"
    )
    print(
        f"  cell_present@page (numeric):  {cell_present_page:5d}  "
        f"({pct(cell_present_page, parsed_gold)} of numeric gold)"
    )
    print()
    print(f"  multi-location questions:     {multi_loc_questions:5d}")
    print(
        f"    fully reachable:            {multi_loc_full:5d}  ({pct(multi_loc_full, multi_loc_questions)})"
    )
    print(
        f"    partially reachable:        {multi_loc_partial:5d}  ({pct(multi_loc_partial, multi_loc_questions)})"
    )
    print()
    print("─── Gap: not reachable@file (ingest missed the file entirely) ──")
    for uid, q in gaps_file[:10]:
        print(f"  {uid}: {q}")
    print()
    print("─── Gap: file present but no page hit at all ─────────────────")
    for uid, info in gaps_page[:10]:
        print(f"  {uid}: {info}")
    print()
    print("─── Gap: multi-location partially reachable ──────────────────")
    for uid, info in gaps_partial[:10]:
        print(f"  {uid}: {info}")

    if out:
        out.write_text("\n".join(json.dumps(row) for row in detailed) + "\n")
        print(f"\nWrote detailed results to {out}")

    if summary_out:
        summary = {
            "total_questions": total,
            "reachability_file": round(reachable_file / total * 100, 1) if total else 0.0,
            "reachability_page": (
                round(reachable_page_all / total * 100, 1) if total else 0.0
            ),
            "reachability_page_any": (
                round(reachable_page_any / total * 100, 1) if total else 0.0
            ),
            "cell_present_page_numeric_pct": (
                round(cell_present_page / parsed_gold * 100, 1) if parsed_gold else 0.0
            ),
        }
        summary_out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote summary to {summary_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--uids", type=str, default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--summary-out", type=Path, default=None)
    args = ap.parse_args()
    run(
        n=args.n,
        verbose=args.verbose,
        uids=_parse_uids(args.uids),
        out=args.out,
        summary_out=args.summary_out,
    )
