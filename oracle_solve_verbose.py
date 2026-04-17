#!/usr/bin/env python3
"""Oracle solver with full LLM message logging (prompts + responses + tokens)."""

import csv
import json
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
    fetch_vocabulary,
)
from extract import extract_structured
from llm_client import (
    MODEL,
    THINKING_EXTRA_BODY,
    client,
    rate_limiter,
    strip_thinking,
)
from retrieve_v2 import _load_element_html
from verify import verify_answer

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "ledger.sqlite"
URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")


def llm_with_logging(
    system: str, user: str, phase_name: str, max_tokens: int = 4096
) -> tuple[str, dict]:
    """Call LLM and return (response, token_info)."""
    print(f"\n  ═══ {phase_name} SYSTEM PROMPT ═══", flush=True)
    print(f"  [{len(system)} chars]", flush=True)
    print(f"  {system[:500]}{'...' if len(system) > 500 else ''}", flush=True)

    print(f"\n  ═══ {phase_name} USER PROMPT ═══", flush=True)
    print(f"  [{len(user)} chars]", flush=True)
    print(f"  {user[:500]}{'...' if len(user) > 500 else ''}", flush=True)

    rate_limiter.acquire()
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.0,
        extra_body=THINKING_EXTRA_BODY,
    )
    t_api = time.time() - t0

    content = strip_thinking(resp.choices[0].message.content or "")
    usage = getattr(resp, "usage", None)
    token_info = {}
    if usage:
        token_info = {
            "input_tokens": getattr(usage, "prompt_tokens", None),
            "output_tokens": getattr(usage, "completion_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
        }
        print(f"\n  ═══ {phase_name} RESPONSE (API: {t_api:.1f}s) ═══", flush=True)
        print(
            f"  Tokens: in={token_info.get('input_tokens')} out={token_info.get('output_tokens')} total={token_info.get('total_tokens')}",
            flush=True,
        )
    else:
        print(f"\n  ═══ {phase_name} RESPONSE (API: {t_api:.1f}s) ═══", flush=True)
        print("  [No token info]", flush=True)

    print(f"  [{len(content)} chars response]", flush=True)
    print(f"  {content[:500]}{'...' if len(content) > 500 else ''}", flush=True)

    return content, token_info


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


def oracle_solve_verbose(uid: str, question: str, source_docs: str, source_files: str) -> dict:
    """Full pipeline with verbose LLM logging."""
    print(f"\n{'=' * 70}", flush=True)
    print(f"{uid}  Q: {question}", flush=True)
    print(f"{'=' * 70}", flush=True)

    result = {"uid": uid}
    t_total = time.time()

    # Decompose
    print("\n[PHASE: DECOMPOSE]", flush=True)
    t0 = time.time()
    try:
        vocab_block = ""
        try:
            years = _years_from_question(question)
            vocab = fetch_vocabulary(question, years=years or None)
            vocab_block = _format_vocab_block(vocab) or ""
        except:
            pass

        decompose_user = f"QUESTION: {question}\n{vocab_block}".strip()
        raw_decompose, decompose_tokens = llm_with_logging(
            DECOMPOSE_SYSTEM, decompose_user, "DECOMPOSE", max_tokens=3000
        )

        spec = json.loads(raw_decompose)
        t_decompose = time.time() - t0
        print(
            f"\n  [decompose]  {t_decompose:.2f}s  {len(spec.get('data_requests', []))} DRs",
            flush=True,
        )
        result["decompose_tokens"] = decompose_tokens
        result["spec"] = spec
    except Exception as e:
        print(f"  [decompose FAIL]: {str(e)[:200]}", flush=True)
        result["error"] = str(e)
        return result

    # Oracle tables
    print("\n[PHASE: ORACLE TABLES]", flush=True)
    t0 = time.time()
    gold_locs = parse_gold_locs(source_docs, source_files)
    if not gold_locs:
        print("  [no gold files]", flush=True)
        return result

    conn = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    oracle = build_oracle_entries(conn, gold_locs)
    if not oracle:
        conn.close()
        print("  [no tables in gold files]", flush=True)
        return result
    t_oracle = time.time() - t0
    print(f"  [oracle tables]  {t_oracle:.2f}s  {len(oracle)} tables", flush=True)

    # Extract
    print("\n[PHASE: EXTRACT]", flush=True)
    t0 = time.time()
    per_dr_entries = {dr.get("id", "?"): oracle for dr in spec.get("data_requests", [])}
    try:
        extraction = extract_structured(spec, per_dr_entries, question, verbose=False)
        if not extraction or not extraction.get("extractions"):
            print("  [extract returned empty]", flush=True)
            conn.close()
            return result
        t_extract = time.time() - t0
        print(f"  [extract]  {t_extract:.2f}s", flush=True)
    except Exception as e:
        print(f"  [extract FAIL]: {str(e)[:200]}", flush=True)
        conn.close()
        return result

    # Compute
    print("\n[PHASE: COMPUTE]", flush=True)
    t0 = time.time()
    try:
        raw_answer = compute_execute(spec, extraction.get("extractions", {}))
        t_compute = time.time() - t0
        print(f"  [compute]  {t_compute:.2f}s  → {raw_answer}", flush=True)
    except Exception as e:
        print(f"  [compute FAIL]: {str(e)[:200]}", flush=True)
        conn.close()
        return result

    # Verify
    print("\n[PHASE: VERIFY]", flush=True)
    t0 = time.time()
    try:
        answer = verify_answer(raw_answer, spec, question, extraction.get("extractions", {}))
        t_verify = time.time() - t0
        print(f"  [verify]  {t_verify:.2f}s  → {answer}", flush=True)
    except Exception as e:
        print(f"  [verify FAIL]: {str(e)[:200]}", flush=True)
        answer = raw_answer

    conn.close()

    t_total_elapsed = time.time() - t_total
    result.update(
        {
            "answer": answer,
            "t_total": t_total_elapsed,
        }
    )
    print(f"\n[TOTAL]  {t_total_elapsed:.2f}s", flush=True)
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--uids", default=None)
    args = parser.parse_args()

    benchmark = {}
    with open(HERE / "officeqa_full.csv") as f:
        for row in csv.DictReader(f):
            benchmark[row["uid"]] = row

    if args.uids:
        uids = args.uids.split(",")
    else:
        all_uids = sorted(benchmark.keys())
        uids = all_uids[args.offset : args.offset + args.n]

    for uid in uids:
        row = benchmark[uid]
        result = oracle_solve_verbose(
            uid, row["question"], row.get("source_docs", ""), row.get("source_files", "")
        )


if __name__ == "__main__":
    main()
