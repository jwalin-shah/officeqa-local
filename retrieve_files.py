#!/usr/bin/env python3
"""File-level retrieval over files.db — the M2 tool.

Goal: given a question, return the top-K source files most likely to contain
the answer. NOT table-level. NOT cell-level. Just "which files."

Design:
  1. Extract year(s) from the question (hard filter, not soft boost).
     Treasury Bulletin publishes monthly and re-reports prior-year data, so
     the "primary" window is [year, year+2] and "secondary" is [year-1, year+15].
  2. BM25 (FTS5) over files_fts — a per-file FTS built in M1 that concatenates
     title + section headers + table titles + row labels.
  3. Return top-K file names with scores.

Usage:
    uv run python retrieve_files.py "total expenditures for national defense in 1940"
    uv run python retrieve_files.py --eval    # full 246-question file-recall@10
    uv run python retrieve_files.py --eval --k 1,5,10,20
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
import time
from pathlib import Path

DB_PATH = Path("files.db")
BENCHMARK_PATH = Path("officeqa_full.csv")

YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
# FTS5 special chars that must be stripped or quoted
FTS_STRIP_RE = re.compile(r"[^\w\s'-]")

# Lightweight stopwords (don't help FTS, waste query budget)
STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "from",
    "by",
    "with",
    "what",
    "were",
    "was",
    "is",
    "are",
    "did",
    "do",
    "does",
    "how",
    "much",
    "many",
    "total",
    "year",
    "years",
    "calendar",
    "fiscal",
    "during",
    "and",
    "or",
    "that",
    "this",
    "these",
    "those",
    "be",
    "been",
    "being",
    "had",
    "has",
    "have",
    "as",
    "it",
    "its",
}


def extract_years(question: str) -> list[int]:
    """Return all distinct 4-digit years mentioned in the question."""
    return sorted({int(m) for m in YEAR_RE.findall(question)})


def build_fts_query(question: str) -> str:
    """Build an FTS5 query string from the question.

    Strategy: drop stopwords and years, keep content terms, OR-join them
    (FTS5 default is AND — too strict for questions with verbose phrasing).
    """
    # Strip non-alphanumerics, lowercase
    cleaned = FTS_STRIP_RE.sub(" ", question.lower())
    tokens = [t for t in cleaned.split() if t and t not in STOPWORDS and not YEAR_RE.match(t)]
    # Also drop very short tokens (noise)
    tokens = [t for t in tokens if len(t) >= 3]
    if not tokens:
        return ""
    # OR-join for recall. Each token optional; BM25 ranks by how many matched.
    # Use double-quotes for safety (years, hyphens inside tokens).
    return " OR ".join(f'"{t}"' for t in tokens)


def candidate_files_for_years(conn: sqlite3.Connection, years: list[int]) -> set[str] | None:
    """Hard-filter files to those whose pub_year could cover the question years.

    Returns None if no years found (no filter applied).
    Primary window: [year, year+2]  — data reported soon after
    Secondary window: [year-1, year+15] — retrospective tables

    We union both windows (still a strict filter, but not too strict).
    """
    if not years:
        return None
    candidates: set[str] = set()
    for y in years:
        rows = conn.execute(
            "SELECT file FROM files WHERE year BETWEEN ? AND ?",
            (y - 1, y + 15),
        ).fetchall()
        candidates.update(r[0] for r in rows)
    return candidates


def search(conn: sqlite3.Connection, question: str, k: int = 10) -> list[tuple[str, float]]:
    """Return top-K files for the question as [(filename, score), ...].

    Strategy (M2.1): structured intersection over derived indexes.
      1. Extract year(s) → filter tables via table_years (hard filter)
      2. Match row_slugs_fts on question content tokens (short docs, discriminating)
      3. Intersect: tables where BOTH year hit AND row slug hit
      4. Rank: row-slug BM25 + year-proximity bonus
      5. Aggregate to file level (MAX), return top-K
    BM25 over tables_fts is a fallback only if intersection is empty.
    """
    years = extract_years(question)

    # Build row-slug FTS query from content tokens (same as before but for row_slugs_fts)
    fts_query = build_fts_query(question)
    if not fts_query:
        # No usable keywords — fall back to year-only candidates
        candidate_set = candidate_files_for_years(conn, years)
        if candidate_set:
            return [(f, 0.0) for f in sorted(candidate_set)[:k]]
        return []

    # ── Stage 1: tables matching the key phrase via row_slugs_fts ─────
    # Row labels are short ("National defense", "Customs duties") so BM25 on them
    # discriminates properly — common words don't drown out specific ones.
    try:
        slug_hits = conn.execute(
            """
            SELECT file, table_id, bm25(row_slugs_fts) AS score
            FROM row_slugs_fts
            WHERE row_slugs_fts MATCH ?
            ORDER BY score
            LIMIT 5000
            """,
            (fts_query,),
        ).fetchall()
    except sqlite3.OperationalError:
        safe = " OR ".join(w for w in fts_query.split() if w != "OR")
        slug_hits = conn.execute(
            """
            SELECT file, table_id, bm25(row_slugs_fts) AS score
            FROM row_slugs_fts
            WHERE row_slugs_fts MATCH ?
            ORDER BY score
            LIMIT 5000
            """,
            (safe,),
        ).fetchall()

    # Keep best score per (file, table_id) — a table may have multiple matching row labels
    table_scores: dict[tuple[str, str], float] = {}
    for file, table_id, score in slug_hits:
        s = -score
        key = (file, table_id)
        if s > table_scores.get(key, float("-inf")):
            table_scores[key] = s

    # ── Stage 2: hard-intersect with year index ───────────────────────
    if years:
        year_tables: set[tuple[str, str]] = set()
        for y in years:
            for r in conn.execute(
                "SELECT file, table_id FROM table_years WHERE data_year = ?",
                (y,),
            ):
                year_tables.add(tuple(r))
        # Intersect slug hits with year hits
        intersected = {k: v for k, v in table_scores.items() if k in year_tables}
    else:
        intersected = table_scores

    # ── Fallback: if intersection empty, use year-only OR slug-only ───
    if not intersected:
        if years and table_scores:
            # Slug-only (no year in question, or no year match — still usable)
            intersected = table_scores
        elif years:
            # Year-only (no row-label match — return tables in the right year, unscored)
            intersected = {k: 0.0 for k in year_tables}

    # ── Stage 3: aggregate to file level (MAX score per file) ─────────
    file_scores: dict[str, float] = {}
    for (file, _tid), s in intersected.items():
        if s > file_scores.get(file, float("-inf")):
            file_scores[file] = s

    # ── Stage 4: year-proximity bonus — prefer bulletins published right after the data year
    if years:
        primary_files: set[str] = set()
        for y in years:
            for r in conn.execute(
                "SELECT file FROM files WHERE year BETWEEN ? AND ?",
                (y, y + 2),
            ):
                primary_files.add(r[0])
        for f in list(file_scores.keys()):
            if f in primary_files:
                file_scores[f] += 2.0  # modest bump, still lets slug score dominate

    rows = sorted(file_scores.items(), key=lambda x: -x[1])
    # Flip sign so downstream rerank code (which expects bm25-style lower=better) works
    rows = [(f, -s) for f, s in rows]

    # Primary year bonus: boost files within [year, year+2]
    if years:
        primary_window: set[str] = set()
        for y in years:
            for r in conn.execute(
                "SELECT file FROM files WHERE year BETWEEN ? AND ?",
                (y, y + 2),
            ):
                primary_window.add(r[0])

        # Re-rank: primary-window files get a boost (smaller bm25 = better)
        def rerank_key(row: tuple[str, float]) -> float:
            return row[1] - (5.0 if row[0] in primary_window else 0.0)

        rows = sorted(rows, key=rerank_key)

    return [(f, -s) for f, s in rows[:k]]  # flip sign so higher = better


# ── Evaluation ─────────────────────────────────────────────────────


def parse_gold_files(source_files: str) -> set[str]:
    """Return set of gold .json filenames from the CSV source_files column."""
    out: set[str] = set()
    for line in (source_files or "").split("\n"):
        line = line.strip()
        if not line:
            continue
        stem = Path(line).stem
        out.add(f"{stem}.json")
    return out


def evaluate(db_path: Path, ks: list[int], verbose: bool = False) -> dict:
    if not BENCHMARK_PATH.exists():
        print(f"{BENCHMARK_PATH} not found", file=sys.stderr)
        sys.exit(1)

    rows = list(csv.DictReader(BENCHMARK_PATH.open()))
    conn = sqlite3.connect(str(db_path))

    max_k = max(ks)
    hits = {k: 0 for k in ks}
    total = 0
    misses: list[tuple[str, str]] = []

    sys.stdout.reconfigure(line_buffering=True)
    t0 = time.perf_counter()
    for i, row in enumerate(rows, 1):
        uid = row["uid"]
        q = row["question"]
        gold = parse_gold_files(row.get("source_files", ""))
        if not gold:
            continue
        total += 1

        results = search(conn, q, k=max_k)
        retrieved = [f for f, _ in results]

        for k in ks:
            top_k = set(retrieved[:k])
            if gold & top_k:
                hits[k] += 1

        if verbose:
            top1 = retrieved[0] if retrieved else "(none)"
            hit10 = "✓" if gold & set(retrieved[:10]) else "✗"
            print(
                f"  [{i:3d}] {hit10} {uid}  gold={sorted(gold)[0][:30]:30s}  top1={top1[:30]}",
                flush=True,
            )
        elif i % 25 == 0:
            elapsed = time.perf_counter() - t0
            print(
                f"  [{i:3d}/{len(rows)}]  r@10 so far = {hits[10] / total * 100:5.1f}%  ({elapsed:.1f}s)",
                flush=True,
            )

        # Track misses for analysis
        if 10 in hits and not (gold & set(retrieved[:10])) and len(misses) < 5:
            misses.append((uid, q[:80]))

    elapsed = time.perf_counter() - t0
    print(f"\n=== file-level recall ({total} questions, {elapsed:.1f}s) ===")
    for k in ks:
        pct = hits[k] / total * 100 if total else 0
        marker = " ✓" if (k == 10 and pct >= 90) else ""
        print(f"  recall@{k:<3d} = {pct:5.1f}%  ({hits[k]}/{total}){marker}")

    if misses:
        print("\n  miss samples:")
        for uid, q in misses:
            print(f"    {uid}: {q}")

    return {"total": total, "hits": hits}


# ── CLI ────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="Question text to search")
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument(
        "--k", default="1,5,10,20", help="Comma-separated K values for eval (default: 1,5,10,20)"
    )
    ap.add_argument("--eval", action="store_true", help="Run full benchmark eval")
    ap.add_argument("--verbose", action="store_true", help="Per-question trace during eval")
    args = ap.parse_args()

    if not args.db.exists():
        print(f"{args.db} not found. Run build_files_db.py first.", file=sys.stderr)
        sys.exit(1)

    if args.eval:
        ks = [int(x) for x in args.k.split(",")]
        evaluate(args.db, ks, verbose=args.verbose)
    elif args.question:
        conn = sqlite3.connect(str(args.db))
        print(f"Query: {args.question}")
        print(f"Years: {extract_years(args.question)}")
        print(f"FTS:   {build_fts_query(args.question)}")
        print()
        results = search(conn, args.question, k=10)
        for i, (f, score) in enumerate(results, 1):
            print(f"  {i:2d}. {f:40s}  score={score:.3f}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
