#!/usr/bin/env python3
"""Oracle-extract eval with step-by-step timing and debug output."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compute import execute as compute_execute  # noqa: E402
from extract import extract_structured  # noqa: E402
from retrieve_v2 import _load_element_html  # noqa: E402
from reward import score_answer  # noqa: E402
from verify import auto_fix_fy_cy, auto_fix_units  # noqa: E402

LEDGER = HERE / "ledger.sqlite"
URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")


def parse_gold_locs(row: dict) -> list[tuple[str, int]]:
    """Return list of (file.json, page_id) pairs from a CSV row."""
    doc_lines = [ln.strip() for ln in (row.get("source_docs") or "").split("\n") if ln.strip()]
    file_lines = [ln.strip() for ln in (row.get("source_files") or "").split("\n") if ln.strip()]
    locs: list[tuple[str, int]] = []
    for url, fname in zip(doc_lines, file_lines, strict=False):
        m = URL_PAGE_RE.search(url)
        if not m:
            continue
        stem = Path(fname).stem
        locs.append((f"{stem}.json", int(m.group(1))))
    return locs


def build_oracle_entries(conn: sqlite3.Connection, gold_locs: list[tuple[str, int]]) -> list[dict]:
    """Build retrieve_v2-shaped entries for the gold tables."""
    entries: list[dict] = []
    for file, page_id in gold_locs:
        rows = conn.execute(
            """
            SELECT t.id, t.file, t.element_id, t.element_seq, t.page_id,
                   t.file_year, t.file_month, t.section, t.title, t.caption,
                   t.unit, t.period, t.n_rows, t.n_cols
            FROM tables t
            WHERE t.file = ? AND t.page_id = ?
              AND t.parse_ok = 1 AND t.table_kind = 'data'
            ORDER BY t.element_seq
            """,
            (file, page_id),
        ).fetchall()
        if not rows:
            # No data tables — try prose
            prose_rows = conn.execute(
                """
                SELECT id, file, element_seq, page_id, file_year, file_month,
                       section, title, content
                FROM prose
                WHERE file = ? AND page_id = ?
                ORDER BY element_seq
                """,
                (file, page_id),
            ).fetchall()
            for pr in prose_rows:
                if not pr["content"]:
                    continue
                entries.append(
                    {
                        "file": pr["file"],
                        "element_id": f"prose_{pr['id']}",
                        "element_seq": pr["element_seq"],
                        "page_id": pr["page_id"],
                        "file_year": pr["file_year"],
                        "file_month": pr["file_month"],
                        "section": pr["section"] or "",
                        "title": pr["title"] or "",
                        "caption": "",
                        "column_headers": [],
                        "row_labels": [],
                        "years": [],
                        "unit": None,
                        "period": None,
                        "n_rows": 0,
                        "n_cols": 0,
                        "retrieval_strategy": "oracle_prose",
                        "retrieval_channel": "oracle_prose",
                        "html": None,
                        "content": pr["content"],
                    }
                )
        for r in rows:
            cols = conn.execute(
                "SELECT col_path FROM table_columns WHERE table_id = ? ORDER BY col_index",
                (r["id"],),
            ).fetchall()
            labels = conn.execute(
                "SELECT row_path FROM table_rows WHERE table_id = ? ORDER BY row_index LIMIT 80",
                (r["id"],),
            ).fetchall()
            yrs = conn.execute(
                """
                SELECT DISTINCT year FROM (
                    SELECT year_extracted AS year FROM table_columns WHERE table_id = ?
                    UNION
                    SELECT year_extracted AS year FROM table_rows    WHERE table_id = ?
                ) WHERE year IS NOT NULL
                """,
                (r["id"], r["id"]),
            ).fetchall()
            entries.append(
                {
                    "file": r["file"],
                    "element_id": r["element_id"],
                    "element_seq": r["element_seq"],
                    "page_id": r["page_id"],
                    "file_year": r["file_year"],
                    "file_month": r["file_month"],
                    "section": r["section"] or "",
                    "title": r["title"] or "",
                    "caption": r["caption"] or "",
                    "column_headers": [c[0] or "" for c in cols],
                    "row_labels": [lab[0] or "" for lab in labels],
                    "years": sorted(int(y[0]) for y in yrs if y[0] is not None),
                    "unit": r["unit"],
                    "period": r["period"],
                    "n_rows": r["n_rows"],
                    "n_cols": r["n_cols"],
                    "retrieval_strategy": "oracle",
                    "retrieval_channel": "oracle",
                    "html": _load_element_html(r["file"], r["element_seq"]),
                }
            )
    return entries


def process_one(uid: str, row: dict) -> dict:
    """Run full pipeline: decompose → oracle entries → extract → compute → verify."""
    t_start = time.time()
    conn = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    question = row["question"]
    print(f"\n{uid}  Q: {question[:80]}...", flush=True)

    # Decompose
    t0 = time.time()
    rate_limiter.acquire()
    try:
        spec = decompose(question)
        print(
            f"  [decompose]        {time.time() - t0:.2f}s  {len(spec.get('data_requests', []))} DRs",
            flush=True,
        )
    except Exception as e:
        conn.close()
        print(f"  [decompose]        FAIL: {str(e)[:100]}", flush=True)
        return {"uid": uid, "outcome": "decompose_fail", "error": str(e)}

    # Parse gold files
    t0 = time.time()
    gold_locs = parse_gold_locs(row)
    if not gold_locs:
        conn.close()
        print("  [parse_gold_locs]  0 files  SKIP", flush=True)
        return {"uid": uid, "outcome": "skip_no_gold"}
    print(f"  [parse_gold_locs]  {len(gold_locs)} files  {time.time() - t0:.2f}s", flush=True)

    # Build oracle entries from gold files
    t0 = time.time()
    oracle = build_oracle_entries(conn, gold_locs)
    if not oracle:
        conn.close()
        print("  [build_oracle]     0 tables  SKIP", flush=True)
        return {"uid": uid, "outcome": "skip_no_tables"}
    print(f"  [build_oracle]     {len(oracle)} tables/prose  {time.time() - t0:.2f}s", flush=True)

    # Extract from oracle entries
    per_dr_entries = {dr.get("id", "?"): oracle for dr in spec.get("data_requests", [])}
    t0 = time.time()
    try:
        extraction = extract_structured(spec, per_dr_entries, question, verbose=False)
    except Exception as e:
        conn.close()
        print(f"  [extract_structured]  FAIL: {str(e)[:100]}", flush=True)
        return {"uid": uid, "outcome": "extract_fail", "error": str(e)}
    t_extract = time.time() - t0
    print(f"  [extract_structured]  {t_extract:.2f}s", flush=True)

    if not extraction or not extraction.get("extractions"):
        conn.close()
        print("  [result]  EMPTY extractions", flush=True)
        return {
            "uid": uid,
            "outcome": "extract_none",
            "gold": row.get("answer", ""),
        }

    # Compute
    t0 = time.time()
    try:
        raw_answer = compute_execute(spec, extraction.get("extractions", {}))
        print(f"  [compute]          {time.time() - t0:.2f}s  raw={raw_answer}", flush=True)
    except Exception as e:
        conn.close()
        print(f"  [compute]          FAIL: {str(e)[:100]}", flush=True)
        return {"uid": uid, "outcome": "compute_fail", "error": str(e)}

    # Verify
    t0 = time.time()
    try:
        answer = auto_fix_units(raw_answer, spec, extraction.get("extractions", {}))
        answer = auto_fix_fy_cy(answer, spec, extraction.get("extractions", {}))
        print(f"  [verify]           {time.time() - t0:.2f}s  final={answer}", flush=True)
    except Exception as e:
        print(f"  [verify]           FAIL: {str(e)[:100]}", flush=True)
        answer = raw_answer

    # Score
    gold = row.get("answer", "")
    score = score_answer(gold, answer)
    elapsed = time.time() - t_start
    print(f"  [score]            {score:.2f}  gold='{gold}'  elapsed={elapsed:.1f}s", flush=True)

    conn.close()
    return {
        "uid": uid,
        "outcome": "ok" if score == 1.0 else "partial" if score > 0 else "wrong",
        "gold": gold,
        "pred": answer,
        "score": score,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--uids", default=None)
    args = parser.parse_args()

    # Load benchmark
    benchmark = {}
    with open(HERE / "officeqa_full.csv") as f:
        for row in csv.DictReader(f):
            benchmark[row["uid"]] = row

    # Load cached decompose specs
    specs = {}
    with open(HERE / "decompose_eval.full.jsonl") as f:
        for line in f:
            obj = json.loads(line)
            specs[obj.get("uid")] = obj

    # Select uids
    if args.uids:
        uids = args.uids.split(",")
    else:
        all_uids = sorted(benchmark.keys())
        uids = all_uids[args.offset : args.offset + args.n]

    print(f"Testing {len(uids)} questions in oracle-extract mode (with timing)", flush=True)
    print()

    results = []
    for uid in uids:
        row = benchmark[uid]
        result = process_one(uid, row)
        results.append(result)

    # Summary
    print("\n" + "=" * 70, flush=True)
    correct = len([r for r in results if r.get("score") == 1.0])
    partial = len([r for r in results if 0 < r.get("score", 0) < 1.0])
    failed = len([r for r in results if r.get("outcome") in ("extract_fail", "compute_fail")])
    print(
        f"Oracle-extract: {correct}/{len(uids)} correct, {partial} partial, {failed} failed",
        flush=True,
    )
    print(f"Avg time: {sum(r.get('elapsed', 0) for r in results) / len(results):.1f}s", flush=True)


if __name__ == "__main__":
    main()
