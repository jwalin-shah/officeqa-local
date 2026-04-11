#!/usr/bin/env python3
"""Local eval harness for OfficeQA Arena.

Uses src/agent.run_agent_loop with the same MCP tool surface as the
Arena submission path. Results may differ from Arena in timing (no
sandbox overhead) but tool routing and accuracy should match.

Usage:
    python3 scripts/eval.py --cases data/officeqa_full.csv --db data/officeqa_corpus.sqlite3
    python3 scripts/eval.py --cases data/officeqa_full.csv --db data/officeqa_corpus.sqlite3 --subset UID0004,UID0023
    python3 scripts/eval.py --cases data/officeqa_full.csv --db data/officeqa_corpus.sqlite3 --parallel 2 --save-traces traces/
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load .env if present
_env_file = ROOT / ".env"
if _env_file.exists():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env_file)
    except ImportError:
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

from src.agent import run_agent_loop  # noqa: E402
from src.reward import fuzzy_match_answer, score_answer  # noqa: E402

# ---------------------------------------------------------------------------
# Failure classification
# ---------------------------------------------------------------------------


def classify_failure(result: dict) -> str:
    """Classify a failed result into a structured failure bucket."""
    if result.get("score", 0) >= 1.0:
        return "correct"

    predicted = str(result.get("predicted", "")).strip()
    log = result.get("log", [])

    if not predicted:
        return "timeout"

    import re

    pred_num = exp_num = None
    expected = str(result.get("expected", "")).strip()
    try:
        pred_num = float(re.sub(r"[,$%]", "", predicted))
    except (ValueError, TypeError):
        pass
    try:
        exp_num = float(re.sub(r"[,$%]", "", expected))
    except (ValueError, TypeError):
        pass

    # Unit scale error: off by ~1000x or ~1000000x
    if pred_num is not None and exp_num is not None and exp_num != 0:
        ratio = pred_num / exp_num
        if 900 < ratio < 1100 or 0.0009 < ratio < 0.0011:
            return "units_scale"
        if 9e5 < ratio < 1.1e6 or 9e-7 < ratio < 1.1e-6:
            return "units_scale"

    # Check tool call patterns
    empty_query_rows = 0
    total_tool_calls = 0
    for entry in log:
        tool_calls = entry.get("tool_calls", [])
        total_tool_calls += len(tool_calls)
        for tc in tool_calls:
            name = tc.get("name", "")
            rf = tc.get("result_full", {})
            if isinstance(rf, str):
                try:
                    rf = json.loads(rf)
                except Exception:
                    rf = {}
            if name == "query_table_rows" and isinstance(rf, dict) and rf.get("count", -1) == 0:
                empty_query_rows += 1

    if empty_query_rows >= 3 and total_tool_calls > 0 and empty_query_rows / total_tool_calls > 0.3:
        return "over_filter"

    if pred_num is not None and exp_num is not None and exp_num != 0:
        pct_diff = abs(pred_num - exp_num) / abs(exp_num) * 100
        if pct_diff < 20:
            return "math_compute"
        if pct_diff < 50:
            return "wrong_row_col"

    if pred_num is not None and exp_num is not None:
        return "retrieval_wrong_table"

    return "unknown"


# ---------------------------------------------------------------------------
# Case loading
# ---------------------------------------------------------------------------


def load_cases(path: Path) -> list[dict[str, str]]:
    """Load test cases from JSON, JSONL, or CSV."""
    text = path.read_text(encoding="utf-8").strip()

    if path.suffix == ".csv":
        import csv

        cases = []
        for row in csv.DictReader(text.splitlines()):
            uid = row.get("uid") or row.get("question_id") or row.get("id") or ""
            instruction = row.get("instruction") or row.get("question") or row.get("input") or ""
            expected = row.get("expected_answer") or row.get("expected") or row.get("answer") or ""
            if instruction:
                cases.append({"uid": uid, "instruction": instruction, "expected_answer": expected})
        return cases

    if text.startswith("["):
        raw = json.loads(text)
    else:
        raw = [json.loads(line) for line in text.splitlines() if line.strip()]

    cases = []
    for item in raw:
        uid = (
            item.get("uid")
            or item.get("question_id")
            or item.get("case_id")
            or item.get("id")
            or ""
        )
        instruction = item.get("instruction") or item.get("question") or item.get("input") or ""
        expected = item.get("expected_answer") or item.get("expected") or item.get("answer") or ""
        if instruction:
            cases.append({"uid": uid, "instruction": instruction, "expected_answer": expected})
    return cases


# ---------------------------------------------------------------------------
# MCP tools loader
# ---------------------------------------------------------------------------


def load_mcp_tools(db_path: str):
    from server import load_tools

    return load_tools(db_path=db_path)


# ---------------------------------------------------------------------------
# Single-case evaluator
# ---------------------------------------------------------------------------


def evaluate_case(
    case: dict[str, str],
    tools_obj: object,
    model: str,
    max_iterations: int,
    verbose: bool,
) -> dict:
    uid = case.get("uid", "unknown")
    instruction = case["instruction"]
    expected = case.get("expected_answer", "")

    print(f"\n{'=' * 60}")
    print(f"Case: {uid}")
    print(f"Q:    {instruction[:120]}...")
    print(f"Exp:  {expected}")

    if hasattr(tools_obj, "reset_budgets"):
        tools_obj.reset_budgets()

    t0 = time.time()
    try:
        raw_answer, log = run_agent_loop(
            instruction=instruction,
            tools_obj=tools_obj,
            model=model,
            max_iterations=max_iterations,
            verbose=verbose,
        )
    except Exception as exc:
        elapsed = time.time() - t0
        print(f"  ERROR: {exc}")
        return {
            "uid": uid,
            "predicted": "",
            "expected": expected,
            "score": 0.0,
            "rationale": f"Agent error: {exc}",
            "elapsed_s": round(elapsed, 1),
        }

    elapsed = time.time() - t0
    predicted = raw_answer or ""

    if expected:
        score = score_answer(expected, predicted)
        _, rationale = fuzzy_match_answer(expected, predicted)
    else:
        score = -1.0
        rationale = "No expected answer provided"

    marker = "PASS" if score == 1.0 else ("SKIP" if score < 0 else "FAIL")
    print(f"  Ans:  {predicted}")
    print(f"  {marker}  score={score}  ({rationale})  [{elapsed:.1f}s]")

    result = {
        "uid": uid,
        "predicted": predicted,
        "expected": expected,
        "score": score,
        "rationale": rationale,
        "elapsed_s": round(elapsed, 1),
        "iterations": len(log),
        "tool_calls": sum(len(e.get("tool_calls", [])) for e in log),
    }
    if score == 0.0:
        bucket = classify_failure({**result, "log": log})
        result["failure_bucket"] = bucket
        print(f"  Bucket: {bucket}")
    return result


# ---------------------------------------------------------------------------
# Parallel worker
# ---------------------------------------------------------------------------

_print_lock = threading.Lock()


def _evaluate_case_worker(case, db_path, model, max_iterations, verbose):
    tools_obj = load_mcp_tools(db_path)
    result = evaluate_case(case, tools_obj, model, max_iterations, verbose)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="OfficeQA Arena local eval")
    parser.add_argument("--cases", required=True, help="Path to test cases (JSON/JSONL/CSV)")
    parser.add_argument(
        "--db", default="", help="Path to SQLite corpus DB (auto-detected if omitted)"
    )
    parser.add_argument("--model", default=os.environ.get("OFFICEQA_MODEL", "minimax/minimax-m2.5"))
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--output", default="", help="Path to write results JSONL")
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--save-traces", default="", help="Dir for per-case traces")
    parser.add_argument("--subset", default="", help="Comma-separated UIDs to eval")
    args = parser.parse_args()

    cases = load_cases(Path(args.cases))
    if not cases:
        print("No cases loaded.")
        sys.exit(1)

    if args.subset:
        subset_uids = {u.strip() for u in args.subset.split(",")}
        cases = [c for c in cases if c["uid"] in subset_uids]
        if not cases:
            print(f"No cases matched subset: {args.subset}")
            sys.exit(1)

    # Auto-detect DB
    db_path = args.db
    if not db_path:
        for candidate in [
            ROOT / "data" / "officeqa_corpus.sqlite3",
            Path("/app/corpus/officeqa_corpus.sqlite3"),
        ]:
            if candidate.exists():
                db_path = str(candidate)
                break
    if not db_path:
        print("No DB found. Pass --db or set OFFICEQA_SQLITE_DB.")
        sys.exit(1)

    print(f"Local eval: {len(cases)} case(s), model={args.model}, db={db_path}")

    traces_dir = None
    if args.save_traces:
        traces_dir = Path(args.save_traces)
        traces_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    if args.parallel <= 1:
        tools_obj = load_mcp_tools(db_path)
        for case in cases:
            result = evaluate_case(case, tools_obj, args.model, args.max_iterations, args.verbose)
            results.append(result)
            if traces_dir:
                uid = result.get("uid", "unknown")
                (traces_dir / f"{uid}.json").write_text(json.dumps(result, indent=2, default=str))
            # Write incrementally to output
            if args.output:
                with open(args.output, "a") as f:
                    f.write(
                        json.dumps({k: v for k, v in result.items() if k != "log"}, default=str)
                        + "\n"
                    )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = {
                pool.submit(
                    _evaluate_case_worker,
                    case,
                    db_path,
                    args.model,
                    args.max_iterations,
                    args.verbose,
                ): case
                for case in cases
            }
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                results.append(result)
                if traces_dir:
                    uid = result.get("uid", "unknown")
                    (traces_dir / f"{uid}.json").write_text(
                        json.dumps(result, indent=2, default=str)
                    )

    # Summary
    scored = [r for r in results if r["score"] >= 0]
    total = len(scored)
    correct = sum(1 for r in scored if r["score"] == 1.0)
    print(f"\n{'=' * 60}")
    print(
        f"Results: {correct}/{total} correct ({correct / total * 100:.1f}%)"
        if total
        else "No scored results"
    )

    buckets: dict[str, list[str]] = {}
    for r in results:
        b = r.get("failure_bucket")
        if b:
            buckets.setdefault(b, []).append(r["uid"])
    if buckets:
        print("Failure buckets:")
        for bucket, uids in sorted(buckets.items(), key=lambda x: -len(x[1])):
            print(f"  {bucket:25s} {len(uids):3d}  {', '.join(uids[:8])}")


if __name__ == "__main__":
    main()
