#!/usr/bin/env python3
"""Stage-attributed evaluation for OfficeQA.

Per UID, records the first concrete stage where the pipeline fails:
- decompose
- retrieve_empty / retrieve_not_in_pool / retrieve_row_miss / retrieve_table_selection
- extract / compute
- verify
- answer_wrong
- ok

Default mode uses cached specs from ``decompose_eval.full.jsonl`` so it is
usable immediately. ``--live-decompose`` is available when you want true
end-to-end attribution including current decompose behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import threading
from collections import Counter
from pathlib import Path

from retrieve_v2 import _bulk_cell_probe
from reward import fuzzy_match_answer
from solve import (
    _run_extract_and_compute,
    categorize_error,
    decompose,
    retrieve_bottomup,
    retrieve_for_spec,
)
from verify import verify_answer

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "officeqa_full.csv"
SPECS_PATH = ROOT / "decompose_eval.full.jsonl"
LEDGER_PATH = ROOT / "ledger.sqlite"
_TLS = threading.local()


def _conn() -> sqlite3.Connection:
    conn = getattr(_TLS, "conn", None)
    if conn is None:
        conn = sqlite3.connect(str(LEDGER_PATH))
        conn.row_factory = sqlite3.Row
        _TLS.conn = conn
    return conn


def _parse_uids(raw: str) -> set[str]:
    return {u.strip() for u in raw.split(",") if u.strip()}


def _load_cached_specs(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            out[row["uid"]] = row
    return out


def _gold_files(row: dict) -> set[str]:
    return {Path(f).stem for f in (row.get("source_files") or "").splitlines() if f.strip()}


def _gold_locations(row: dict) -> set[tuple[str, int]]:
    import re

    url_page_re = re.compile(r"[?&]page=(\d+)")
    docs = [x.strip() for x in (row.get("source_docs") or "").splitlines() if x.strip()]
    files = [x.strip() for x in (row.get("source_files") or "").splitlines() if x.strip()]
    locs: set[tuple[str, int]] = set()
    for url, fname in zip(docs, files, strict=False):
        m = url_page_re.search(url)
        if not m:
            continue
        locs.add((f"{Path(fname).stem}.json", int(m.group(1))))
    return locs


def _dr_row_hints(dr: dict) -> list[str]:
    hints: list[str] = []
    row_hint = (dr.get("row_hint") or "").strip()
    if row_hint:
        hints.append(row_hint)
    for alt in dr.get("row_hint_alternatives") or []:
        alt = str(alt).strip()
        if alt:
            hints.append(alt)
    return hints


def _entry_table_id(entry: dict) -> int | None:
    row = (
        _conn()
        .execute(
            "SELECT id FROM tables WHERE file = ? AND element_seq = ? LIMIT 1",
            (entry.get("file"), entry.get("element_seq")),
        )
        .fetchone()
    )
    return int(row[0]) if row is not None else None


def _retrieval_signals(spec: dict, per_dr: dict[str, list[dict]], gold_row: dict) -> dict:
    gold_files = _gold_files(gold_row)
    gold_locs = _gold_locations(gold_row)
    total_entries = sum(len(v) for v in per_dr.values())

    first_file_hit_rank: int | None = None
    first_table_hit_rank: int | None = None
    first_row_hit_rank: int | None = None

    for dr in spec.get("data_requests") or []:
        dr_id = dr.get("id")
        if not dr_id:
            continue
        entries = per_dr.get(dr_id) or []

        local_file_rank = next(
            (i + 1 for i, e in enumerate(entries) if Path(e["file"]).stem in gold_files),
            None,
        )
        local_table_rank = next(
            (i + 1 for i, e in enumerate(entries) if (e["file"], e.get("page_id")) in gold_locs),
            None,
        )
        if local_file_rank is not None:
            first_file_hit_rank = min(first_file_hit_rank or local_file_rank, local_file_rank)
        if local_table_rank is not None:
            first_table_hit_rank = min(first_table_hit_rank or local_table_rank, local_table_rank)

        table_ids = [tid for tid in (_entry_table_id(e) for e in entries) if tid is not None]
        probe = _bulk_cell_probe(_conn(), table_ids, _dr_row_hints(dr))
        local_row_rank = next(
            (
                i + 1
                for i, tid in enumerate(table_ids)
                if probe.get(tid, (0, 0))[0] > 0 and probe.get(tid, (0, 0))[1] > 0
            ),
            None,
        )
        if local_row_rank is not None:
            first_row_hit_rank = min(first_row_hit_rank or local_row_rank, local_row_rank)

    return {
        "total_entries": total_entries,
        "first_file_hit_rank": first_file_hit_rank,
        "first_table_hit_rank": first_table_hit_rank,
        "first_row_hit_rank": first_row_hit_rank,
    }


def _classify_retrieval(signals: dict) -> tuple[str, str]:
    if signals["total_entries"] == 0:
        return "retrieve_empty", "no retrieved entries"
    if signals["first_file_hit_rank"] is None:
        return "retrieve_not_in_pool", "gold file never appeared in retrieved pool"
    if signals["first_row_hit_rank"] is None:
        return "retrieve_row_miss", "retrieved files did not expose a row-hint match"
    if signals["first_table_hit_rank"] is None:
        return "retrieve_table_selection", "row-plausible table found, but gold table not retrieved"
    return "retrieve_ok", "retrieval has gold file/table and row hit"


def _build_per_dr(spec: dict, question: str) -> dict[str, list[dict]]:
    per_dr = retrieve_bottomup(spec, question, top_k=5)
    empty = [dr_id for dr_id, entries in per_dr.items() if not entries]
    if empty:
        fallback = retrieve_for_spec(spec, question, verbose=False)
        for dr_id in empty:
            per_dr[dr_id] = fallback.get(dr_id, [])
    return per_dr


def _one(row: dict, cached_spec_row: dict | None, live_decompose: bool) -> dict:
    uid = row["uid"]
    question = row["question"]
    expected = row["answer"].strip()

    if live_decompose:
        spec = decompose(question)
        spec_source = "live"
    else:
        spec = (cached_spec_row or {}).get("spec")
        spec_source = "cached"

    if spec is None:
        return {
            "uid": uid,
            "stage": "decompose_failed",
            "detail": "spec missing",
            "spec_source": spec_source,
        }

    per_dr = _build_per_dr(spec, question)
    retrieval = _retrieval_signals(spec, per_dr, row)
    retrieve_stage, retrieve_detail = _classify_retrieval(retrieval)
    if retrieve_stage != "retrieve_ok":
        return {
            "uid": uid,
            "stage": retrieve_stage,
            "detail": retrieve_detail,
            "spec_source": spec_source,
            "retrieval": retrieval,
        }

    try:
        answer, extraction = _run_extract_and_compute(
            spec,
            per_dr,
            question,
            verbose=False,
            llm_counter={"count": 0},
        )
    except Exception as e:
        return {
            "uid": uid,
            "stage": "extract_exception",
            "detail": f"{type(e).__name__}: {e}",
            "spec_source": spec_source,
            "retrieval": retrieval,
        }

    if extraction is None or answer.startswith(("EXTRACT_FAILED", "NO_VALUES")):
        return {
            "uid": uid,
            "stage": "extract_failed",
            "detail": answer,
            "spec_source": spec_source,
            "retrieval": retrieval,
        }
    if answer.startswith("COMPUTE_FAILED"):
        return {
            "uid": uid,
            "stage": "compute_failed",
            "detail": answer,
            "spec_source": spec_source,
            "retrieval": retrieval,
        }

    try:
        verdict = verify_answer(question, spec, extraction["extractions"], answer, verbose=False)
    except Exception as e:
        verdict = {"ok": True, "issue": f"verify_exception: {type(e).__name__}: {e}"}

    match, rationale = fuzzy_match_answer(expected, answer, tolerance=0.01)
    if match:
        return {
            "uid": uid,
            "stage": "ok",
            "detail": "answer matched gold",
            "answer": answer,
            "spec_source": spec_source,
            "retrieval": retrieval,
        }

    if not verdict.get("ok", True):
        return {
            "uid": uid,
            "stage": f"verify_{verdict.get('suggested_phase') or 'flagged'}",
            "detail": verdict.get("issue"),
            "answer": answer,
            "expected": expected,
            "spec_source": spec_source,
            "retrieval": retrieval,
            "error_category": categorize_error(answer, expected, rationale),
        }

    return {
        "uid": uid,
        "stage": "answer_wrong",
        "detail": rationale,
        "answer": answer,
        "expected": expected,
        "spec_source": spec_source,
        "retrieval": retrieval,
        "error_category": categorize_error(answer, expected, rationale),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--uids", type=str, default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--live-decompose", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "attribution_eval.jsonl")
    ap.add_argument("--summary-out", type=Path, default=ROOT / "attribution_eval.summary.json")
    args = ap.parse_args()

    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    selected = _parse_uids(args.uids)
    if selected:
        rows = [r for r in rows if r["uid"] in selected]
    if args.n:
        rows = rows[: args.n]

    cached = _load_cached_specs(SPECS_PATH)
    results = [_one(row, cached.get(row["uid"]), args.live_decompose) for row in rows]

    args.out.write_text("\n".join(json.dumps(r) for r in results) + "\n")

    stage_counts = Counter(r["stage"] for r in results)
    error_counts = Counter(r.get("error_category") for r in results if r.get("error_category"))
    summary = {
        "total_questions": len(results),
        "stage_counts": dict(stage_counts),
        "error_categories": dict(error_counts),
    }
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"Wrote details to {args.out}")
    print(f"Wrote summary to {args.summary_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
