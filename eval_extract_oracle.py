#!/usr/bin/env python3
"""Oracle-mode extraction eval.

Skips retrieval entirely. For each question we build per_dr_entries from the
GOLD tables (file + page_id from officeqa_full.csv), feed them to
extract.extract_structured, then run compute + verify + score against the
gold answer. Answers the question: if retrieval were perfect, how often
does extraction produce the right value?

Usage:
  uv run python eval_extract_oracle.py                       # first 20 uids
  uv run python eval_extract_oracle.py --n 30
  uv run python eval_extract_oracle.py --uids UID0001,UID0042
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from compute import execute as compute_execute  # noqa: E402
from compute import format_result
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
    """Build retrieve_v2-shaped entries for the gold tables.

    When a gold page has no data tables (e.g. chart pages, prose-only pages),
    falls back to prose passages from that page so the LLM can still attempt
    extraction from narrative text.
    """
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
            # No data tables on this page — try prose passages instead.
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


_TLS = threading.local()


def _conn() -> sqlite3.Connection:
    """Per-thread read-only SQLite connection (sqlite3 conns aren't thread-safe)."""
    c = getattr(_TLS, "conn", None)
    if c is None:
        c = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        _TLS.conn = c
    return c


def _process_uid(uid: str, row: dict, spec: dict | None) -> dict:
    """Run the full pipeline on one uid. Returns a dict the caller can use to
    classify the outcome AND write to the result jsonl. Pure-ish: only side
    effect is the per-thread sqlite cursor."""
    if not spec:
        return {"uid": uid, "outcome": "skip_no_spec"}
    gold_locs = parse_gold_locs(row)
    if not gold_locs:
        return {"uid": uid, "outcome": "skip_no_gold"}

    t0 = time.time()
    oracle = build_oracle_entries(_conn(), gold_locs)
    if not oracle:
        return {"uid": uid, "outcome": "skip_no_tables", "gold_locs": gold_locs}

    per_dr_entries = {dr.get("id", "?"): oracle for dr in spec.get("data_requests", [])}
    try:
        extraction = extract_structured(spec, per_dr_entries, row["question"], verbose=False)
    except Exception as e:
        return {"uid": uid, "outcome": "extract_fail", "error": str(e)}

    spec_dr_ids = [dr.get("id") for dr in spec.get("data_requests", [])]

    if not extraction or not extraction.get("extractions"):
        return {
            "uid": uid,
            "outcome": "extract_none",
            "question": row["question"],
            "gold": row.get("answer", ""),
            "gold_locs": gold_locs,
            "n_oracle_tables": len(oracle),
            "spec_dr_ids": spec_dr_ids,
            "per_dr_trace": [],
        }

    extracted = extraction.get("extractions") or {}
    per_dr_trace = []
    for dr_id in spec_dr_ids:
        ex = extracted.get(dr_id) or {}
        raw_vals = ex.get("values") or []
        non_null = [v for v in raw_vals if v is not None]
        per_dr_trace.append(
            {
                "dr_id": dr_id,
                "n_raw": len(raw_vals),
                "n_values": len(non_null),
                "source_file": ex.get("source_file") or ex.get("source"),
                "labels": ex.get("labels") or ex.get("label"),
                "source_unit": ex.get("source_unit"),
                "unit_normalized_from": ex.get("unit_normalized_from"),
                "unit_normalized_to": ex.get("unit_normalized_to"),
                "confidence": ex.get("confidence"),
                "notes": ex.get("notes"),
            }
        )
    missing_drs = [t["dr_id"] for t in per_dr_trace if t["n_values"] == 0]

    try:
        raw_answer = compute_execute(spec, extracted)
        # Determine source_unit for format_result: prefer extraction's
        # normalized/reported unit over the ledger table's unit field.
        # This handles cases where the table header is missing from the
        # ledger but the LLM correctly identified the unit.
        src_unit = None
        from compute import parse_unit

        for ex_val in extracted.values():
            if not isinstance(ex_val, dict):
                continue
            # unit_normalized_to: extraction was already scaled to this unit
            nto = ex_val.get("unit_normalized_to")
            if nto:
                src_unit = nto
                break
            # source_unit: raw unit the LLM read from the table
            su = ex_val.get("source_unit")
            if su:
                from extract import _canonical_unit  # noqa: PLC0415

                src_unit = _canonical_unit(su)
                if src_unit:
                    break
        if src_unit is None:
            for e in oracle:
                if e.get("unit"):
                    src_unit = parse_unit(e["unit"])
                    if src_unit:
                        break
        answer = format_result(raw_answer, spec.get("output_format", {}), src_unit)

        # Deterministic post-compute fixes (no LLM call)
        target_unit = (spec.get("output_format") or {}).get("unit")
        unit_fix = auto_fix_units(str(answer), src_unit, target_unit)
        if unit_fix:
            answer = unit_fix["corrected"]

        fy_cy_flag = auto_fix_fy_cy(spec, extracted)
        # (FY/CY mismatch is flagged for diagnostic purposes; can't re-extract in oracle mode)
    except Exception as e:
        return {
            "uid": uid,
            "outcome": "compute_fail",
            "error": str(e),
            "question": row["question"],
            "gold": row.get("answer", ""),
            "gold_locs": gold_locs,
            "n_oracle_tables": len(oracle),
            "spec_dr_ids": spec_dr_ids,
            "extracted_dr_ids": list(extracted.keys()),
            "per_dr_trace": per_dr_trace,
            "missing_drs": missing_drs,
        }

    gold = row.get("answer", "")
    score = score_answer(gold, str(answer), tolerance=0.01)  # match solve.py's 1% tolerance
    return {
        "uid": uid,
        "outcome": "ok",
        "question": row["question"],
        "gold": gold,
        "predicted": str(answer),
        "score": score,
        "gold_locs": gold_locs,
        "n_oracle_tables": len(oracle),
        "elapsed_s": time.time() - t0,
        "spec_dr_ids": spec_dr_ids,
        "extracted_dr_ids": list(extracted.keys()),
        "per_dr_trace": per_dr_trace,
        "missing_drs": missing_drs,
        "fy_cy_flag": fy_cy_flag,
    }


def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--uids", type=str, default="")
    ap.add_argument("--specs", type=Path, default=HERE / "decompose_eval.full.jsonl")
    ap.add_argument("--csv", type=Path, default=HERE / "officeqa_full.csv")
    ap.add_argument("--out", type=Path, default=HERE / "eval_extract_oracle.jsonl")
    ap.add_argument("--workers", type=int, default=10, help="parallel workers")
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = {r["uid"]: r for r in csv.DictReader(f)}

    spec_cache: dict[str, dict] = {}
    with open(args.specs) as f:
        for line in f:
            d = json.loads(line)
            if d.get("spec"):
                spec_cache[d["uid"]] = d["spec"]

    if args.uids:
        uids = [u.strip() for u in args.uids.split(",") if u.strip()]
    else:
        uids = sorted(spec_cache.keys() & rows.keys())[: args.n]

    print(
        f"Evaluating {len(uids)} questions in oracle-extract mode ({args.workers} workers)",
        flush=True,
    )
    print(f"Specs:  {args.specs.name}  ({len(spec_cache)} cached)")
    print(f"Ledger: {LEDGER.name}\n", flush=True)

    results: list[dict] = []
    n_no_gold = 0
    n_no_spec = 0
    n_no_oracle_tbl = 0
    n_extract_fail = 0
    n_compute_fail = 0
    n_correct = 0
    n_partial = 0
    n_total = 0
    print_lock = threading.Lock()

    def _print_outcome(r: dict) -> None:
        uid = r["uid"]
        out = r["outcome"]
        if out == "ok":
            score = r["score"]
            mark = "✓" if score >= 1.0 else ("~" if score > 0 else "✗")
            line = (
                f"  {uid}  {mark}  gold={r['gold']!r:30}  "
                f"pred={str(r['predicted'])[:30]!r:32}  score={score:.2f}  "
                f"{r['elapsed_s']:.1f}s"
            )
        elif out == "compute_fail":
            line = f"  {uid}  COMPUTE_FAIL: {r.get('error')}  missing_drs={r.get('missing_drs')}"
        elif out == "extract_fail":
            line = f"  {uid}  EXTRACT_FAIL: {r.get('error')}"
        elif out == "extract_none":
            line = f"  {uid}  EXTRACT_NONE"
        elif out == "skip_no_spec":
            line = f"  {uid}  SKIP (no cached spec)"
        elif out == "skip_no_gold":
            line = f"  {uid}  SKIP (no gold locs)"
        elif out == "skip_no_tables":
            line = f"  {uid}  FAIL (gold locs ∉ ledger: {r.get('gold_locs')})"
        else:
            line = f"  {uid}  {out}"
        with print_lock:
            print(line, flush=True)

    valid_uids = [u for u in uids if u in rows]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(_process_uid, uid, rows[uid], spec_cache.get(uid)): uid for uid in valid_uids
        }
        for fut in as_completed(futures):
            r = fut.result()
            _print_outcome(r)
            outcome = r["outcome"]
            if outcome == "skip_no_spec":
                n_no_spec += 1
                continue
            if outcome == "skip_no_gold":
                n_no_gold += 1
                continue
            if outcome == "skip_no_tables":
                n_no_oracle_tbl += 1
                continue

            n_total += 1
            if outcome == "extract_fail" or outcome == "extract_none":
                n_extract_fail += 1
            elif outcome == "compute_fail":
                n_compute_fail += 1
            elif outcome == "ok":
                if r["score"] >= 1.0:
                    n_correct += 1
                elif r["score"] > 0:
                    n_partial += 1
            results.append(r)

    print()
    print("─" * 60)
    print(f"Oracle-extract results (n={n_total})")
    print("─" * 60)
    if n_total:
        print(f"  correct:        {n_correct:3d}  ({n_correct / n_total * 100:4.1f}%)")
        print(f"  partial:        {n_partial:3d}  ({n_partial / n_total * 100:4.1f}%)")
        print(f"  extract_fail:   {n_extract_fail:3d}")
        print(f"  compute_fail:   {n_compute_fail:3d}")
    if n_no_gold:
        print(f"  skip_no_gold:   {n_no_gold}")
    if n_no_spec:
        print(f"  skip_no_spec:   {n_no_spec}")
    if n_no_oracle_tbl:
        print(f"  skip_no_tables: {n_no_oracle_tbl}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    run()
