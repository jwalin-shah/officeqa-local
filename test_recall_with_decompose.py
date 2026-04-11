#!/usr/bin/env python3
"""Measure retrieval recall using cached decompose specs instead of the
raw question. Compares against the tokenized-question baseline to quantify
how much decompose hints actually move file-level and table-level recall.

Usage:
  uv run python test_recall_with_decompose.py [decompose_eval.v2.jsonl]
"""

import csv
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from retrieve_v2 import retrieve

URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")


def parse_gold(row: dict) -> tuple[set[str], set[tuple[str, int]]]:
    """Return (gold_files, gold_locations) for a benchmark row."""
    doc_lines = [
        line.strip() for line in (row.get("source_docs") or "").split("\n") if line.strip()
    ]
    file_lines = [
        line.strip() for line in (row.get("source_files") or "").split("\n") if line.strip()
    ]
    locs: set[tuple[str, int]] = set()
    for url, fname in zip(doc_lines, file_lines, strict=False):
        m = URL_PAGE_RE.search(url)
        if not m:
            continue
        stem = Path(fname).stem
        locs.add((f"{stem}.json", int(m.group(1))))
    files = {f"{Path(f).stem}.json" for f in file_lines}
    return files, locs


def run():
    spec_path = Path(sys.argv[1] if len(sys.argv) > 1 else "decompose_eval.v2.jsonl")
    decomp_rows: dict[str, dict] = {}
    for line in spec_path.open():
        r = json.loads(line)
        decomp_rows[r["uid"]] = r

    print(f"Loaded {len(decomp_rows)} decomposed specs from {spec_path}", flush=True)

    with open("officeqa_full.csv") as f:
        csv_rows = {r["uid"]: r for r in csv.DictReader(f)}

    subset = [csv_rows[uid] for uid in sorted(decomp_rows) if uid in csv_rows]
    print(f"Evaluating {len(subset)} questions\n", flush=True)

    def eval_once(use_decompose: bool) -> dict:
        file_at: dict[int, int] = {1: 0, 5: 0, 10: 0, 30: 0}
        tbl_at: dict[int, int] = {1: 0, 5: 0, 10: 0, 30: 0}
        total = 0
        per_q_t: list[float] = []
        misses: list[tuple[str, set[str], list[str]]] = []

        for row in subset:
            uid = row["uid"]
            gold_files, gold_locs = parse_gold(row)
            if not gold_files:
                continue

            t0 = time.time()
            if use_decompose:
                spec = decomp_rows[uid].get("spec") or {}
                results = retrieve(
                    spec,
                    row["question"],
                    top_k=30,
                    dedupe_by_file=False,
                    load_html=False,
                )
            else:
                plan = {
                    "data_requests": [
                        {
                            "label": row["question"],
                            "row_hint": "",
                            "column_hint": "",
                            "years": [],
                        }
                    ]
                }
                results = retrieve(
                    plan,
                    row["question"],
                    top_k=30,
                    dedupe_by_file=False,
                    load_html=False,
                )
            per_q_t.append(time.time() - t0)

            # Table-level: raw result order
            ret_locs = [(e["file"], e["page_id"]) for e in results]
            # File-level: dedup in rank order so we measure "first K distinct
            # files seen", not "first K tables". Otherwise a single bad file
            # contributing 10 tables eats the top-10 slots and file-level
            # recall becomes identical to table-level recall.
            seen: set[str] = set()
            uniq_files: list[str] = []
            for e in results:
                if e["file"] not in seen:
                    seen.add(e["file"])
                    uniq_files.append(e["file"])

            total += 1
            for k in (1, 5, 10, 30):
                if any(f in gold_files for f in uniq_files[:k]):
                    file_at[k] += 1
                if any(loc in gold_locs for loc in ret_locs[:k]):
                    tbl_at[k] += 1

            if not any(f in gold_files for f in uniq_files[:10]):
                misses.append((uid, gold_files, uniq_files[:5]))

        return {
            "total": total,
            "file_at": file_at,
            "tbl_at": tbl_at,
            "latency_p50": sorted(per_q_t)[len(per_q_t) // 2] * 1000 if per_q_t else 0,
            "latency_p95": sorted(per_q_t)[int(len(per_q_t) * 0.95)] * 1000 if per_q_t else 0,
            "misses": misses,
        }

    print("── Baseline (tokenized question, no decompose) ──", flush=True)
    base = eval_once(use_decompose=False)
    print(f"  total={base['total']}")
    for k in (1, 5, 10, 30):
        pct_f = base["file_at"][k] / base["total"] * 100
        pct_t = base["tbl_at"][k] / base["total"] * 100
        print(
            f"  @{k:<2d}   file={base['file_at'][k]:2d}/{base['total']} ({pct_f:4.1f}%)   "
            f"table={base['tbl_at'][k]:2d}/{base['total']} ({pct_t:4.1f}%)"
        )
    print(f"  latency p50={base['latency_p50']:.0f}ms  p95={base['latency_p95']:.0f}ms\n")

    print("── With decompose hints ──", flush=True)
    deco = eval_once(use_decompose=True)
    print(f"  total={deco['total']}")
    for k in (1, 5, 10, 30):
        pct_f = deco["file_at"][k] / deco["total"] * 100
        pct_t = deco["tbl_at"][k] / deco["total"] * 100
        delta_f = deco["file_at"][k] - base["file_at"][k]
        delta_t = deco["tbl_at"][k] - base["tbl_at"][k]
        print(
            f"  @{k:<2d}   file={deco['file_at'][k]:2d}/{deco['total']} ({pct_f:4.1f}%, Δ={delta_f:+d})   "
            f"table={deco['tbl_at'][k]:2d}/{deco['total']} ({pct_t:4.1f}%, Δ={delta_t:+d})"
        )
    print(f"  latency p50={deco['latency_p50']:.0f}ms  p95={deco['latency_p95']:.0f}ms\n")

    print("── Decompose-mode file-level misses (top-5 retrieved vs gold) ──", flush=True)
    for uid, gold, got in deco["misses"]:
        print(f"  {uid}  gold={sorted(gold)}")
        print(f"         got ={got}")


if __name__ == "__main__":
    run()
