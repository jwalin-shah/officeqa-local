#!/usr/bin/env python3
"""Measure retrieve_v2.retrieve() recall@K against gold source files.

Loads cached decompose specs from decompose_eval.full.jsonl so we never
hit the LLM. Runs retrieve() with a deeper table pool, then computes file
recall from the first 50 unique files and table recall from the raw table
ranking.
"""

import argparse
import concurrent.futures
import csv
import json
import re
import sqlite3
import sys
import threading
import time
from pathlib import Path

from retrieve_v2 import _bulk_cell_probe, retrieve

sys.stdout.reconfigure(line_buffering=True)

CACHE = Path("decompose_eval.full.jsonl")
CSV_PATH = Path("officeqa_full.csv")
TOP_K = 50
TABLE_TOP_K = 200
WORKERS = 12
URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")
_TLS = threading.local()


def stem(name: str) -> str:
    return Path(name).stem


def parse_gold_locations(source_docs: str, source_files: str) -> set[tuple[str, int]]:
    doc_lines = [line.strip() for line in (source_docs or "").split("\n") if line.strip()]
    file_lines = [line.strip() for line in (source_files or "").split("\n") if line.strip()]
    out: set[tuple[str, int]] = set()
    for url, fname in zip(doc_lines, file_lines, strict=False):
        m = URL_PAGE_RE.search(url)
        if not m:
            continue
        out.add((f"{Path(fname).stem}.json", int(m.group(1))))
    return out


def _conn() -> sqlite3.Connection:
    conn = getattr(_TLS, "conn", None)
    if conn is None:
        conn = sqlite3.connect("ledger.sqlite")
        conn.row_factory = sqlite3.Row
        _TLS.conn = conn
    return conn


def _collect_row_hints(spec: dict) -> list[str]:
    hints: list[str] = []
    for dr in spec.get("data_requests") or []:
        row_hint = (dr.get("row_hint") or "").strip()
        if row_hint:
            hints.append(row_hint)
        for alt in dr.get("row_hint_alternatives") or []:
            alt = str(alt).strip()
            if alt:
                hints.append(alt)
    seen: set[str] = set()
    ordered: list[str] = []
    for hint in hints:
        if hint in seen:
            continue
        seen.add(hint)
        ordered.append(hint)
    return ordered


def run_one(row: dict, csv_row: dict | None) -> dict:
    spec = row.get("spec")
    gold = {stem(f) for f in row.get("gold_files") or []}
    if not spec or not gold:
        return {"uid": row["uid"], "skipped": True, "gold": list(gold)}

    t0 = time.time()
    try:
        debug_meta: dict = {}
        results = retrieve(
            spec,
            row["question"],
            top_k=TABLE_TOP_K,
            load_html=False,
            dedupe_by_file=False,
            debug_meta=debug_meta,
        )
    except Exception as e:
        return {"uid": row["uid"], "error": f"{type(e).__name__}: {e}"}

    conn = _conn()
    retrieved_table_stems = [stem(r["file"]) for r in results]
    retrieved_stems: list[str] = []
    seen_files: set[str] = set()
    for table_stem in retrieved_table_stems:
        if table_stem in seen_files:
            continue
        seen_files.add(table_stem)
        retrieved_stems.append(table_stem)
    ranks = [i + 1 for i, s in enumerate(retrieved_stems) if s in gold]
    first_hit = ranks[0] if ranks else None
    gold_locs = parse_gold_locations(
        (csv_row or {}).get("source_docs", ""),
        (csv_row or {}).get("source_files", ""),
    )
    gold_signatures: set[str] = set()
    gold_titles: set[str] = set()
    for g_file, g_page in gold_locs:
        sig_rows = conn.execute(
            "SELECT signature, title FROM tables WHERE file = ? AND page_id = ? AND table_kind = 'data'",
            (g_file, g_page),
        ).fetchall()
        for sig_row in sig_rows:
            if sig_row[0]:
                gold_signatures.add(sig_row[0])
            if sig_row[1]:
                gold_titles.add(sig_row[1])

    retrieved_locs = [(r["file"], r["page_id"]) for r in results]
    table_ranks = [i + 1 for i, loc in enumerate(retrieved_locs) if loc in gold_locs]
    first_table_hit = table_ranks[0] if table_ranks else None

    retrieved_titles = [r.get("title") for r in results]
    retrieved_signatures = [r.get("signature") for r in results]
    family_ranks = [i + 1 for i, t in enumerate(retrieved_titles) if t and t in gold_titles]
    first_family_hit = family_ranks[0] if family_ranks else None

    table_ids: list[int] = []
    for r in results:
        db_row = conn.execute(
            "SELECT id FROM tables WHERE file = ? AND element_seq = ? LIMIT 1",
            (r["file"], r.get("element_seq")),
        ).fetchone()
        if db_row is not None:
            table_ids.append(int(db_row[0]))
    row_probe = _bulk_cell_probe(conn, table_ids, _collect_row_hints(spec))
    first_row_hit = next(
        (
            idx + 1
            for idx, tid in enumerate(table_ids)
            if row_probe.get(tid, (0, 0))[0] > 0 and row_probe.get(tid, (0, 0))[1] > 0
        ),
        None,
    )
    return {
        "uid": row["uid"],
        "gold": list(gold),
        "gold_signatures": list(gold_signatures),
        # Output field name is legacy; value is up to TOP_K (50) unique file stems.
        "retrieved_top20": retrieved_stems,
        "retrieved_table_top20": retrieved_table_stems,
        "retrieved_signatures": retrieved_signatures,
        "first_hit_rank": first_hit,
        "hit_at_5": first_hit is not None and first_hit <= 5,
        "hit_at_10": first_hit is not None and first_hit <= 10,
        "hit_at_20": first_hit is not None and first_hit <= 20,
        "hit_at_30": first_hit is not None and first_hit <= 30,
        "hit_at_50": first_hit is not None and first_hit <= 50,
        "first_table_hit_rank": first_table_hit,
        "table_hit_at_10": first_table_hit is not None and first_table_hit <= 10,
        "table_hit_at_20": first_table_hit is not None and first_table_hit <= 20,
        "table_hit_at_30": first_table_hit is not None and first_table_hit <= 30,
        "first_family_hit_rank": first_family_hit,
        "family_hit_at_5": first_family_hit is not None and first_family_hit <= 5,
        "family_hit_at_10": first_family_hit is not None and first_family_hit <= 10,
        "family_hit_at_20": first_family_hit is not None and first_family_hit <= 20,
        "family_hit_at_30": first_family_hit is not None and first_family_hit <= 30,
        "family_hit_at_50": first_family_hit is not None and first_family_hit <= 50,
        "first_row_hit_rank": first_row_hit,
        "row_hit_at_10": first_row_hit is not None and first_row_hit <= 10,
        "row_hit_at_20": first_row_hit is not None and first_row_hit <= 20,
        "row_hit_at_30": first_row_hit is not None and first_row_hit <= 30,
        "elapsed_s": round(time.time() - t0, 2),
        **debug_meta,
    }


def _parse_uids(raw: str) -> set[str]:
    return {u.strip() for u in raw.split(",") if u.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", type=Path, default=CACHE)
    ap.add_argument("--out", type=Path, default=Path("retrieve_eval.jsonl"))
    ap.add_argument("--summary-out", type=Path, default=None)
    ap.add_argument("--uids", type=str, default="")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.specs.open()]  # noqa: E741
    with CSV_PATH.open() as f:
        csv_rows = {r["uid"]: r for r in csv.DictReader(f)}
    selected = _parse_uids(args.uids)
    if selected:
        rows = [r for r in rows if r.get("uid") in selected]
    usable = [r for r in rows if r.get("spec") and r.get("gold_files")]
    print(f"Loaded {len(rows)} cached specs; {len(usable)} usable (non-null spec + gold)")

    results: list[dict] = []
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_one, r, csv_rows.get(r["uid"])): r for r in usable}
        done = 0
        for fut in concurrent.futures.as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1  # noqa: SIM113
            if res.get("error"):
                mark = "ERR "
            elif res.get("skipped"):
                mark = "SKIP"
            else:
                rank = res["first_hit_rank"]
                mark = f"@{rank:>2}" if rank else "MISS"
            print(f"  [{done:>3}/{len(usable)}] {mark} {res['uid']}")

    n = sum(1 for r in results if not r.get("error") and not r.get("skipped"))
    h5 = sum(1 for r in results if r.get("hit_at_5"))
    h10 = sum(1 for r in results if r.get("hit_at_10"))
    h20 = sum(1 for r in results if r.get("hit_at_20"))
    h30 = sum(1 for r in results if r.get("hit_at_30"))
    h50 = sum(1 for r in results if r.get("hit_at_50"))
    t10 = sum(1 for r in results if r.get("table_hit_at_10"))
    t20 = sum(1 for r in results if r.get("table_hit_at_20"))
    t30 = sum(1 for r in results if r.get("table_hit_at_30"))
    t50 = sum(1 for r in results if r.get("first_table_hit_rank") is not None and r.get("first_table_hit_rank") <= 50)
    f5 = sum(1 for r in results if r.get("family_hit_at_5"))
    f10 = sum(1 for r in results if r.get("family_hit_at_10"))
    f20 = sum(1 for r in results if r.get("family_hit_at_20"))
    f30 = sum(1 for r in results if r.get("family_hit_at_30"))
    f50 = sum(1 for r in results if r.get("first_family_hit_rank") is not None and r.get("first_family_hit_rank") <= 50)
    r10 = sum(1 for r in results if r.get("row_hit_at_10"))
    r20 = sum(1 for r in results if r.get("row_hit_at_20"))
    r30 = sum(1 for r in results if r.get("row_hit_at_30"))
    avg_pool_before = (
        sum(float(r.get("pool_before_filters", 0)) for r in results if not r.get("error") and not r.get("skipped")) / n
        if n
        else 0.0
    )
    avg_pool_after_year = (
        sum(float(r.get("pool_after_year_filter", 0)) for r in results if not r.get("error") and not r.get("skipped")) / n
        if n
        else 0.0
    )
    avg_pool_after_structure = (
        sum(float(r.get("pool_after_structure_filter", 0)) for r in results if not r.get("error") and not r.get("skipped")) / n
        if n
        else 0.0
    )

    print(f"\nDone in {time.time() - t0:.1f}s")
    print(f"  n      = {n}")
    print(f"  recall@5  = {h5}/{n} ({h5 / n * 100:.1f}%)")
    print(f"  recall@10 = {h10}/{n} ({h10 / n * 100:.1f}%)")
    print(f"  recall@20 = {h20}/{n} ({h20 / n * 100:.1f}%)")
    print(f"  recall@30 = {h30}/{n} ({h30 / n * 100:.1f}%)")
    print(f"  recall@50 = {h50}/{n} ({h50 / n * 100:.1f}%)")
    print(f"  table_hit@10 = {t10}/{n} ({t10 / n * 100:.1f}%)")
    print(f"  table_hit@20 = {t20}/{n} ({t20 / n * 100:.1f}%)")
    print(f"  table_hit@30 = {t30}/{n} ({t30 / n * 100:.1f}%)")
    print(f"  table_hit@50 = {t50}/{n} ({t50 / n * 100:.1f}%)")
    print(f"  family_hit@5  = {f5}/{n} ({f5 / n * 100:.1f}%)")
    print(f"  family_hit@10 = {f10}/{n} ({f10 / n * 100:.1f}%)")
    print(f"  family_hit@20 = {f20}/{n} ({f20 / n * 100:.1f}%)")
    print(f"  family_hit@30 = {f30}/{n} ({f30 / n * 100:.1f}%)")
    print(f"  family_hit@50 = {f50}/{n} ({f50 / n * 100:.1f}%)")
    print(f"  row_hit@10   = {r10}/{n} ({r10 / n * 100:.1f}%)")
    print(f"  row_hit@20   = {r20}/{n} ({r20 / n * 100:.1f}%)")
    print(f"  row_hit@30   = {r30}/{n} ({r30 / n * 100:.1f}%)")
    print(f"  avg pool before filters = {avg_pool_before:.1f}")
    print(f"  avg pool after year     = {avg_pool_after_year:.1f}")
    print(f"  avg pool after structure= {avg_pool_after_structure:.1f}")

    args.out.write_text("\n".join(json.dumps(r) for r in sorted(results, key=lambda x: x["uid"])) + "\n")
    print(f"Wrote detailed results to {args.out}")

    if args.summary_out:
        summary = {
            "total_questions": n,
            "recall_at_5": round(h5 / n * 100, 1) if n else 0.0,
            "recall_at_10": round(h10 / n * 100, 1) if n else 0.0,
            "recall_at_20": round(h20 / n * 100, 1) if n else 0.0,
            "recall_at_30": round(h30 / n * 100, 1) if n else 0.0,
            "recall_at_50": round(h50 / n * 100, 1) if n else 0.0,
            "table_hit_at_10": round(t10 / n * 100, 1) if n else 0.0,
            "table_hit_at_20": round(t20 / n * 100, 1) if n else 0.0,
            "table_hit_at_30": round(t30 / n * 100, 1) if n else 0.0,
            "table_hit_at_50": round(t50 / n * 100, 1) if n else 0.0,
            "family_hit_at_5": round(f5 / n * 100, 1) if n else 0.0,
            "family_hit_at_10": round(f10 / n * 100, 1) if n else 0.0,
            "family_hit_at_20": round(f20 / n * 100, 1) if n else 0.0,
            "family_hit_at_30": round(f30 / n * 100, 1) if n else 0.0,
            "family_hit_at_50": round(f50 / n * 100, 1) if n else 0.0,
            "row_hit_at_10": round(r10 / n * 100, 1) if n else 0.0,
            "row_hit_at_20": round(r20 / n * 100, 1) if n else 0.0,
            "row_hit_at_30": round(r30 / n * 100, 1) if n else 0.0,
            "avg_pool_before_filters": round(avg_pool_before, 1),
            "avg_pool_after_year_filter": round(avg_pool_after_year, 1),
            "avg_pool_after_structure_filter": round(avg_pool_after_structure, 1),
            "elapsed_s": round(time.time() - t0, 2),
        }
        args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Wrote summary to {args.summary_out}")


if __name__ == "__main__":
    main()
