#!/usr/bin/env python3
"""Oracle+Tools full benchmark with complete trace logging to JSONL."""

import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from compute import execute as compute_execute
from decompose import decompose
from extract_with_tools import extract_structured_with_tools
from llm_client import rate_limiter
from reward import score_answer
from verify import verify_answer

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "officeqa_full.csv"
OUTPUT = HERE / "oracle_tools_eval.jsonl"
URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")


def parse_gold_files(source_files: str) -> list[str]:
    """Extract file stems from CSV row."""
    file_lines = [ln.strip() for ln in (source_files or "").split("\n") if ln.strip()]
    return [Path(fname).stem for fname in file_lines]


def oracle_solve_tools(uid: str, question: str, source_files: str) -> dict:
    """Solve with full trace logging."""
    result = {
        "uid": uid,
        "question": question,
        "gold_files": parse_gold_files(source_files),
    }
    t_total = time.time()

    # Decompose
    t0 = time.time()
    rate_limiter.acquire()
    try:
        spec = decompose(question)
        t_decompose = time.time() - t0
        result["decompose_spec"] = spec
        result["t_decompose"] = t_decompose
        n_drs = len(spec.get("data_requests", []))
        print(f"{uid}  decompose: {t_decompose:.1f}s, {n_drs} DRs", flush=True)
    except Exception as e:
        result["outcome"] = "decompose_fail"
        result["error"] = str(e)
        return result

    # Get gold file names
    gold_files = result["gold_files"]
    if not gold_files:
        result["outcome"] = "skip_no_gold_files"
        return result

    # Extract with tools
    t0 = time.time()
    per_dr_entries = {dr.get("id", "?"): [] for dr in spec.get("data_requests", [])}

    try:
        extraction = extract_structured_with_tools(
            spec,
            per_dr_entries,
            question,
            gold_files,
            verbose=False,
        )
        t_extract = time.time() - t0
        result["t_extract"] = t_extract

        if extraction and extraction.get("extractions"):
            result["extraction"] = extraction
            n_ext = len([v for v in extraction["extractions"].values() if v.get("values")])
            print(f"  extract: {t_extract:.1f}s, {n_ext} extractions", flush=True)
        else:
            result["outcome"] = "extract_empty"
            result["t_total"] = time.time() - t_total
            return result
    except Exception as e:
        result["outcome"] = "extract_fail"
        result["error"] = str(e)
        result["t_total"] = time.time() - t_total
        return result

    # Compute
    t0 = time.time()
    try:
        raw_answer = compute_execute(spec, extraction.get("extractions", {}))
        t_compute = time.time() - t0
        result["t_compute"] = t_compute
        result["raw_answer"] = raw_answer
        print(f"  compute: {t_compute:.1f}s → {raw_answer}", flush=True)
    except Exception as e:
        result["outcome"] = "compute_fail"
        result["error"] = str(e)
        result["t_total"] = time.time() - t_total
        return result

    # Verify
    t0 = time.time()
    try:
        answer = verify_answer(raw_answer, spec, question, extraction.get("extractions", {}))
        t_verify = time.time() - t0
        result["t_verify"] = t_verify
        result["answer"] = answer
        print(f"  verify: {t_verify:.1f}s → {answer}", flush=True)
    except Exception as e:
        print(f"  verify FAIL: {str(e)[:50]}", flush=True)
        result["answer"] = raw_answer
        result["verify_error"] = str(e)

    result["t_total"] = time.time() - t_total
    result["outcome"] = "ok"
    return result


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=246, help="Number of questions")
    parser.add_argument("--offset", type=int, default=0, help="Start offset")
    parser.add_argument("--uids", default=None, help="Comma-separated UIDs")
    args = parser.parse_args()

    benchmark = {}
    with open(BENCHMARK) as f:
        for row in csv.DictReader(f):
            benchmark[row["uid"]] = row

    if args.uids:
        uids = args.uids.split(",")
    else:
        all_uids = sorted(benchmark.keys())
        uids = all_uids[args.offset : args.offset + args.n]

    print(f"Oracle+Tools Benchmark: {len(uids)} questions → {OUTPUT} (5 workers)\n", flush=True)

    results = {}
    lock = __import__("threading").Lock()

    def process_uid(uid):
        row = benchmark[uid]
        result = oracle_solve_tools(uid, row["question"], row.get("source_files", ""))

        if "answer" in result:
            gold = row.get("answer", "")
            score = score_answer(gold, result["answer"])
            result["score"] = score
            result["gold"] = gold

        return uid, result

    with open(OUTPUT, "w") as out, ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(process_uid, uid): uid for uid in uids}
        for future in as_completed(futures):
            uid, result = future.result()
            results[uid] = result
            with lock:
                out.write(json.dumps(result) + "\n")
                out.flush()

    # Reorder results by UID for summary
    results = [results[uid] for uid in uids if uid in results]

    # Summary
    print("\n" + "=" * 70, flush=True)
    correct = len([r for r in results if r.get("score") == 1.0])
    partial = len([r for r in results if 0 < r.get("score", 0) < 1.0])
    print(f"Results: {correct}/{len(uids)} correct ({100 * correct / len(uids):.1f}%)", flush=True)
    print(f"         {partial} partial", flush=True)
    if results:
        avg_time = sum(r.get("t_total", 0) for r in results) / len(results)
        print(f"Avg time: {avg_time:.1f}s/question", flush=True)
    print(f"Output: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
