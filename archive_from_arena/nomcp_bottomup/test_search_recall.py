#!/usr/bin/env python3
"""Test search recall: what % of questions find the right source file via decompose + search."""

import argparse
import csv
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure we can import from the same package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Set CORPUS_DIR before importing solve_decompose (it reads at import time)
os.environ.setdefault("CORPUS_DIR", "/app/corpus")
# Set API key if not already set
if not os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("LLM_API_KEY"):
    raise SystemExit("Set OPENROUTER_API_KEY or LLM_API_KEY for LLM-backed recall tests.")
os.environ.setdefault("OPENROUTER_API_KEY", os.environ.get("LLM_API_KEY", ""))

from bottomup.solve_decompose import decompose, search_tables, search_tables_multi

DATA_CSV = Path(__file__).resolve().parent.parent.parent / "data" / "officeqa_full.csv"


def load_questions(csv_path, limit=None):
    """Load questions from CSV. Returns list of dicts."""
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
            if limit and len(rows) >= limit:
                break
    return rows


def test_one(row, mode="single"):
    """Run decompose + search for one question. Returns result dict."""
    uid = row["uid"]
    question = row["question"]
    expected_file = row["source_files"].strip()
    difficulty = row.get("difficulty", "unknown").strip()

    result = {
        "uid": uid,
        "expected_file": expected_file,
        "difficulty": difficulty,
        "found": False,
        "match_idx": None,
        "n_candidates": 0,
        "searched_files": [],
        "error": None,
    }

    try:
        # Phase 1: Decompose
        plan = decompose(question)
        if not plan or "sub_queries" not in plan:
            result["error"] = "decompose_failed"
            return result

        # Phase 2: Search across all sub-queries, collect all candidates
        all_candidates = []
        search_fn = search_tables_multi if mode == "multi" else search_tables
        for sq in plan["sub_queries"]:
            cands = search_fn(sq)
            all_candidates.extend(cands)

        # Deduplicate by (source_file, title)
        seen = set()
        deduped = []
        for c in all_candidates:
            key = (c.get("source_file", ""), c.get("title", "")[:60])
            if key not in seen:
                seen.add(key)
                deduped.append(c)

        result["n_candidates"] = len(deduped)
        result["searched_files"] = list(dict.fromkeys(c.get("source_file", "") for c in deduped))

        # Check if any candidate matches expected source file(s)
        # expected_file may contain multiple files separated by newlines
        expected_basenames = set(
            os.path.basename(f.strip()) for f in expected_file.split("\n") if f.strip()
        )
        for i, c in enumerate(deduped):
            cand_basename = os.path.basename(c.get("source_file", ""))
            if cand_basename in expected_basenames:
                result["found"] = True
                result["match_idx"] = i + 1  # 1-indexed
                break

    except Exception as e:
        result["error"] = str(e)

    return result


def summarize_files(files, max_show=5):
    """Compact representation of searched files."""
    if not files:
        return "(none)"
    if len(files) <= max_show:
        return ", ".join(files)
    return ", ".join(files[:max_show]) + f" ... +{len(files) - max_show} more"


def main():
    parser = argparse.ArgumentParser(description="Test search recall on OfficeQA questions")
    parser.add_argument(
        "--limit", type=int, default=50, help="Number of questions to test (default: 50)"
    )
    parser.add_argument(
        "--mode",
        choices=["single", "multi"],
        default="single",
        help="Search mode: single (search_tables) or multi (search_tables_multi)",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )
    parser.add_argument("--csv", type=str, default=str(DATA_CSV), help="Path to CSV file")
    args = parser.parse_args()

    print(f"Loading questions from {args.csv} ...", file=sys.stderr)
    questions = load_questions(args.csv, limit=args.limit)
    print(
        f"Testing {len(questions)} questions, mode={args.mode}, workers={args.workers}\n",
        file=sys.stderr,
    )

    results = []
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(test_one, row, args.mode): row for row in questions}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)

            # Print per-question result immediately
            if r["found"]:
                print(
                    f"  [{r['uid']}] FOUND | expected={r['expected_file']} "
                    f"| found in candidate #{r['match_idx']} of {r['n_candidates']}"
                )
            elif r["error"]:
                print(f"  [{r['uid']}] ERROR | expected={r['expected_file']} | error={r['error']}")
            else:
                print(
                    f"  [{r['uid']}] MISS  | expected={r['expected_file']} "
                    f"| searched: {summarize_files(r['searched_files'])}"
                )

    elapsed = time.time() - t0

    # Sort results by UID for consistent display
    results.sort(key=lambda r: r["uid"])

    # Summary
    total = len(results)
    found = sum(1 for r in results if r["found"])
    errors = sum(1 for r in results if r["error"])

    # By difficulty
    by_diff = {}
    for r in results:
        d = r["difficulty"]
        if d not in by_diff:
            by_diff[d] = {"total": 0, "found": 0}
        by_diff[d]["total"] += 1
        if r["found"]:
            by_diff[d]["found"] += 1

    print(f"\n{'=' * 60}")
    print(f"  Search recall: {found}/{total} ({100 * found / total:.0f}%)")
    if errors:
        print(f"  Errors: {errors}/{total}")

    diff_parts = []
    for d in sorted(by_diff.keys()):
        v = by_diff[d]
        pct = 100 * v["found"] / v["total"] if v["total"] else 0
        diff_parts.append(f"{d}={v['found']}/{v['total']} ({pct:.0f}%)")
    print(f"  By difficulty: {', '.join(diff_parts)}")
    print(f"  Time: {elapsed:.1f}s ({elapsed / total:.1f}s/question)")
    print(f"  Mode: {args.mode}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
