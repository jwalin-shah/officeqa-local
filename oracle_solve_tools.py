#!/usr/bin/env python3
"""Oracle solver with LLM-driven tool calls (SQLite + JSON reads)."""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path

# Load .env file manually
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

sys.stdout.reconfigure(line_buffering=True)

from compute import execute as compute_execute
from decompose import decompose
from extract_simple import extract_from_files
from llm_client import rate_limiter
from reward import score_answer
from verify import verify_answer

HERE = Path(__file__).resolve().parent
BENCHMARK = HERE / "officeqa_full.csv"
CACHED_SPECS = HERE / "decompose_eval.full.jsonl"
URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")

# Load cached decompose specs
_cached_specs = {}
if CACHED_SPECS.exists():
    with open(CACHED_SPECS) as f:
        for line in f:
            try:
                obj = json.loads(line)
                if "uid" in obj and "spec" in obj:
                    _cached_specs[obj["uid"]] = obj["spec"]
            except:
                pass
    print(f"Loaded {len(_cached_specs)} cached decompose specs", flush=True)


def parse_gold_files(source_files: str) -> list[str]:
    """Extract file stems from CSV row."""
    file_lines = [ln.strip() for ln in (source_files or "").split("\n") if ln.strip()]
    return [Path(fname).stem for fname in file_lines]


def oracle_solve_tools(uid: str, question: str, source_files: str) -> dict:
    """Solve with LLM-driven tool access."""
    result = {"uid": uid}
    t_total = time.time()

    print(f"\n{'=' * 70}", flush=True)
    print(f"{uid}  Q: {question}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Decompose (use cached spec if available, else call LLM)
    print("\n[DECOMPOSE]", flush=True)
    t0 = time.time()

    if uid in _cached_specs:
        spec = _cached_specs[uid]
        print("  [cached]  loaded pre-computed spec", flush=True)
    else:
        rate_limiter.acquire()
        try:
            spec = decompose(question)
        except Exception as e:
            print(f"  FAIL: {str(e)[:100]}", flush=True)
            result["outcome"] = "decompose_fail"
            return result

    t_decompose = time.time() - t0
    n_drs = len(spec.get("data_requests", []))
    print(f"  {t_decompose:.2f}s  → {n_drs} data_requests", flush=True)

    # Get gold file names
    gold_files = parse_gold_files(source_files)
    if not gold_files:
        print("  No gold files", flush=True)
        return result
    print(f"  Gold files: {gold_files}", flush=True)

    # Extract from gold files
    print("\n[EXTRACT]", flush=True)
    t0 = time.time()

    try:
        extraction = extract_from_files(spec, question, gold_files, verbose=False)
        t_extract = time.time() - t0

        if extraction and extraction.get("extractions"):
            n_ext = len([v for v in extraction["extractions"].values() if v.get("values")])
            print(f"  {t_extract:.2f}s  → {n_ext} extractions", flush=True)
        else:
            print(f"  {t_extract:.2f}s  → EMPTY", flush=True)
            return result
    except Exception as e:
        print(f"  FAIL: {str(e)[:100]}", flush=True)
        result["outcome"] = "extract_fail"
        return result

    # Compute
    print("\n[COMPUTE]", flush=True)
    t0 = time.time()
    try:
        raw_answer = compute_execute(spec, extraction.get("extractions", {}))
        t_compute = time.time() - t0
        print(f"  {t_compute:.2f}s  → {raw_answer}", flush=True)
    except Exception as e:
        print(f"  FAIL: {str(e)[:100]}", flush=True)
        result["outcome"] = "compute_fail"
        return result

    # Verify
    print("\n[VERIFY]", flush=True)
    t0 = time.time()
    try:
        answer_str = str(raw_answer) if raw_answer is not None else ""
        verify_result = verify_answer(question, spec, extraction.get("extractions", {}), answer_str)
        answer = (
            verify_result.get("corrected_answer", answer_str)
            if verify_result.get("corrected_answer")
            else answer_str
        )
        t_verify = time.time() - t0
        print(f"  {t_verify:.2f}s  → {answer}", flush=True)
    except Exception as e:
        print(f"  FAIL: {str(e)[:100]}", flush=True)
        answer = str(raw_answer) if raw_answer is not None else ""

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
    parser.add_argument("--n", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--uids", default=None)
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

    print(f"Oracle+Tools Mode: {len(uids)} questions\n", flush=True)

    results = []
    for uid in uids:
        row = benchmark[uid]
        result = oracle_solve_tools(uid, row["question"], row.get("source_files", ""))

        if "answer" in result:
            gold = row.get("answer", "")
            score = score_answer(gold, result["answer"])
            result["score"] = score
            result["gold"] = gold
            status = "✓" if score == 1.0 else "✗" if score == 0 else "~"
            print(f"  {status}  gold='{gold}'  pred='{result['answer']}'  {score:.2f}", flush=True)

        results.append(result)

    # Summary
    print("\n" + "=" * 70, flush=True)
    correct = len([r for r in results if r.get("score") == 1.0])
    print(f"Results: {correct}/{len(uids)} correct", flush=True)
    if results:
        avg_time = sum(r.get("t_total", 0) for r in results) / len(results)
        print(f"Avg time: {avg_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
