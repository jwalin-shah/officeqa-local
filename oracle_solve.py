#!/usr/bin/env python3
"""Oracle-mode solver: question + gold files → answer with verbose timing."""

import csv
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from compute import execute as compute_execute
from decompose import (
    DECOMPOSE_SYSTEM,
    _format_vocab_block,
    _years_from_question,
    decompose,
    fetch_vocabulary,
)
from extract import extract_structured
from llm_client import rate_limiter
from retrieve_v2 import _load_element_html
from reward import score_answer
from verify import verify_answer

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.sqlite"
URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")
MAX_LLM_CALLS = 6


def parse_gold_locs(source_docs: str, source_files: str) -> list[tuple[str, int]]:
    """Extract (file.json, page_id) pairs from benchmark CSV row."""
    doc_lines = [ln.strip() for ln in (source_docs or "").split("\n") if ln.strip()]
    file_lines = [ln.strip() for ln in (source_files or "").split("\n") if ln.strip()]
    locs: list[tuple[str, int]] = []
    for url, fname in zip(doc_lines, file_lines, strict=False):
        m = URL_PAGE_RE.search(url)
        if not m:
            continue
        stem = Path(fname).stem
        locs.append((f"{stem}.json", int(m.group(1))))
    return locs


def build_oracle_entries(conn: sqlite3.Connection, gold_locs: list[tuple[str, int]]) -> list[dict]:
    """Build retrieve_v2-shaped entries from gold files."""
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


def oracle_solve(
    uid: str, question: str, source_docs: str, source_files: str, verbose_prompts: bool = False
) -> dict:
    """Run full pipeline with oracle retrieval. Return {score, gold, pred, timing}."""
    result = {"uid": uid, "question": question}
    t_total = time.time()

    # Decompose with prompt visibility
    t0 = time.time()
    rate_limiter.acquire()
    try:
        # Build decompose prompt for display
        vocab_block = ""
        try:
            years = _years_from_question(question)
            vocab = fetch_vocabulary(question, years=years or None)
            vocab_block = _format_vocab_block(vocab) or ""
        except:
            pass

        decompose_user = f"QUESTION: {question}\n{vocab_block}".strip()

        if verbose_prompts:
            print(f"    [DECOMPOSE SYSTEM PROMPT] ({len(DECOMPOSE_SYSTEM)} chars)", flush=True)
            print(f"    [DECOMPOSE USER PROMPT] ({len(decompose_user)} chars):", flush=True)
            print(f"      {decompose_user[:200]}...", flush=True)

        spec = decompose(question)
        t_decompose = time.time() - t0
        n_drs = len(spec.get("data_requests", []))
        print(f"  [decompose]    {t_decompose:.2f}s  → {n_drs} data_requests", flush=True)
        result["n_data_requests"] = n_drs
        result["decompose_system_len"] = len(DECOMPOSE_SYSTEM)
        result["decompose_user_len"] = len(decompose_user)
    except Exception as e:
        print(f"  [decompose]    FAIL: {str(e)[:80]}", flush=True)
        result["outcome"] = "decompose_fail"
        result["error"] = str(e)
        return result

    # Parse gold files
    t0 = time.time()
    gold_locs = parse_gold_locs(source_docs, source_files)
    if not gold_locs:
        print("  [parse_gold]   0 files → SKIP", flush=True)
        result["outcome"] = "no_gold_files"
        return result
    t_parse = time.time() - t0
    print(f"  [parse_gold]   {t_parse:.2f}s  → {len(gold_locs)} files", flush=True)

    # Oracle retrieval
    t0 = time.time()
    conn = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    oracle = build_oracle_entries(conn, gold_locs)
    if not oracle:
        conn.close()
        print("  [oracle_tables]  0 tables → SKIP", flush=True)
        result["outcome"] = "no_oracle_tables"
        return result
    t_retrieve = time.time() - t0
    print(f"  [oracle_tables] {t_retrieve:.2f}s  → {len(oracle)} tables", flush=True)
    result["n_oracle_tables"] = len(oracle)

    # Extract
    per_dr_entries = {dr.get("id", "?"): oracle for dr in spec.get("data_requests", [])}
    t0 = time.time()
    try:
        extraction = extract_structured(spec, per_dr_entries, question, verbose=False)
        t_extract = time.time() - t0
        if extraction and extraction.get("extractions"):
            n_extracted = len([v for v in extraction["extractions"].values() if v.get("values")])
            print(f"  [extract]      {t_extract:.2f}s  → {n_extracted} extractions", flush=True)
        else:
            print(f"  [extract]      {t_extract:.2f}s  → 0 extractions (EMPTY)", flush=True)
            result["outcome"] = "extract_empty"
            conn.close()
            return result
    except Exception as e:
        print(f"  [extract]      FAIL: {str(e)[:80]}", flush=True)
        result["outcome"] = "extract_fail"
        result["error"] = str(e)
        conn.close()
        return result

    # Compute
    t0 = time.time()
    try:
        raw_answer = compute_execute(spec, extraction.get("extractions", {}))
        t_compute = time.time() - t0
        print(f"  [compute]      {t_compute:.2f}s  → {raw_answer}", flush=True)
    except Exception as e:
        print(f"  [compute]      FAIL: {str(e)[:80]}", flush=True)
        result["outcome"] = "compute_fail"
        result["error"] = str(e)
        conn.close()
        return result

    # Verify
    t0 = time.time()
    try:
        answer = verify_answer(raw_answer, spec, question, extraction.get("extractions", {}))
        t_verify = time.time() - t0
        print(f"  [verify]       {t_verify:.2f}s  → {answer}", flush=True)
    except Exception as e:
        print(f"  [verify]       FAIL: {str(e)[:80]}", flush=True)
        answer = raw_answer

    conn.close()

    # Timing summary
    t_total_elapsed = time.time() - t_total
    result.update(
        {
            "answer": answer,
            "raw_answer": raw_answer,
            "t_decompose": t_decompose,
            "t_parse_gold": t_parse,
            "t_retrieve": t_retrieve,
            "t_extract": t_extract,
            "t_compute": t_compute,
            "t_verify": t_verify,
            "t_total": t_total_elapsed,
        }
    )
    print(f"  [total]        {t_total_elapsed:.2f}s", flush=True)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5, help="Number of questions to test")
    parser.add_argument("--offset", type=int, default=0, help="Start offset")
    parser.add_argument("--uids", default=None, help="Comma-separated UIDs")
    parser.add_argument(
        "--verbose-prompts", action="store_true", help="Show decompose/extract prompts"
    )
    args = parser.parse_args()

    # Load benchmark
    benchmark = {}
    with open(HERE / "officeqa_full.csv") as f:
        for row in csv.DictReader(f):
            benchmark[row["uid"]] = row

    # Select UIDs
    if args.uids:
        uids = args.uids.split(",")
    else:
        all_uids = sorted(benchmark.keys())
        uids = all_uids[args.offset : args.offset + args.n]

    print(f"Oracle Solver: {len(uids)} questions\n", flush=True)

    results = []
    for uid in uids:
        row = benchmark[uid]
        print(f"{uid}  Q: {row['question'][:70]}...", flush=True)

        result = oracle_solve(
            uid,
            row["question"],
            row.get("source_docs", ""),
            row.get("source_files", ""),
            verbose_prompts=args.verbose_prompts,
        )

        if "answer" in result:
            score = score_answer(row.get("answer", ""), result["answer"])
            result["score"] = score
            result["gold"] = row.get("answer", "")
            status = "✓" if score == 1.0 else "✗" if score == 0 else "~"
            print(
                f"    {status}  gold='{row.get('answer', '')}'  pred='{result['answer']}'  score={score:.2f}",
                flush=True,
            )

        results.append(result)
        print()

    # Summary
    print("=" * 70, flush=True)
    correct = len([r for r in results if r.get("score") == 1.0])
    partial = len([r for r in results if 0 < r.get("score", 0) < 1.0])
    print(f"Results: {correct}/{len(uids)} correct, {partial} partial", flush=True)
    if results:
        avg_time = sum(r.get("t_total", 0) for r in results) / len(results)
        print(f"Avg time: {avg_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
