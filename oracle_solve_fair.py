#!/usr/bin/env python3
"""Oracle-fair solver: know which FILES have data, but model finds tables within them."""

import csv
import json
import re
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

from compute import execute as compute_execute
from decompose import decompose
from extract import extract_structured
from llm_client import rate_limiter
from reward import score_answer
from verify import verify_answer

HERE = Path(__file__).resolve().parent
CORPUS = HERE / "corpus_json"
BENCHMARK = HERE / "officeqa_full.csv"
URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")


def parse_gold_files(source_docs: str, source_files: str) -> list[str]:
    """Extract file stems (no .json) from benchmark CSV row."""
    file_lines = [ln.strip() for ln in (source_files or "").split("\n") if ln.strip()]
    files = []
    for fname in file_lines:
        stem = Path(fname).stem
        files.append(stem)
    return files


def load_file_as_text(file_stem: str) -> str:
    """Load a corpus JSON file and return as text representation."""
    file_path = CORPUS / f"{file_stem}.json"
    if not file_path.exists():
        return ""

    try:
        with open(file_path) as f:
            data = json.load(f)

        # Convert JSON structure to readable text (tables + prose)
        lines = []
        lines.append(f"FILE: {file_stem}")
        lines.append("")

        # Process elements in order
        for elem in data.get("elements", []):
            elem_type = elem.get("type", "")

            if elem_type == "table":
                title = elem.get("title", "")
                caption = elem.get("caption", "")
                rows = elem.get("rows", [])

                if title:
                    lines.append(f"TABLE: {title}")
                if caption:
                    lines.append(f"CAPTION: {caption}")

                # Render table as pipe-delimited
                if rows:
                    for row in rows[:50]:  # Limit to first 50 rows
                        cells = row.get("cells", [])
                        cell_text = " | ".join(str(c.get("value", "")) for c in cells)
                        lines.append(f"| {cell_text} |")
                lines.append("")

            elif elem_type in ("text", "prose"):
                content = elem.get("content", "")
                if content:
                    lines.append(f"PROSE: {content[:500]}")
                    lines.append("")

            elif elem_type == "footnote":
                content = elem.get("content", "")
                if content:
                    lines.append(f"FOOTNOTE: {content[:200]}")
                    lines.append("")

        return "\n".join(lines)
    except Exception as e:
        return f"[Error loading {file_stem}: {str(e)}]"


def oracle_fair_solve(uid: str, question: str, source_files: str) -> dict:
    """Solve with gold files as unstructured context."""
    result = {"uid": uid, "question": question}
    t_total = time.time()

    print(f"\n{'=' * 70}", flush=True)
    print(f"{uid}  Q: {question}", flush=True)
    print(f"{'=' * 70}", flush=True)

    # Decompose
    print("\n[DECOMPOSE]", flush=True)
    t0 = time.time()
    rate_limiter.acquire()
    try:
        spec = decompose(question)
        t_decompose = time.time() - t0
        n_drs = len(spec.get("data_requests", []))
        print(f"  {t_decompose:.2f}s  → {n_drs} data_requests", flush=True)
    except Exception as e:
        print(f"  FAIL: {str(e)[:100]}", flush=True)
        result["outcome"] = "decompose_fail"
        return result

    # Load gold files as unstructured text
    print("\n[LOAD GOLD FILES]", flush=True)
    t0 = time.time()
    gold_files = parse_gold_files("", source_files)
    if not gold_files:
        print("  No gold files", flush=True)
        return result

    file_content = []
    for file_stem in gold_files:
        content = load_file_as_text(file_stem)
        if content:
            file_content.append(content)
    t_load = time.time() - t0
    total_chars = sum(len(c) for c in file_content)
    print(f"  {t_load:.2f}s  → {len(gold_files)} files, {total_chars:,} chars", flush=True)

    if not file_content:
        print("  No file content loaded", flush=True)
        return result

    # Create synthetic retrieve_v2 entries from file content
    # (treat entire file as a single "pseudo-table" entry)
    oracle_entries = [
        {
            "file": f"{file_stem}.json",
            "element_id": "gold_file_content",
            "element_seq": i,
            "page_id": 0,
            "file_year": None,
            "file_month": None,
            "section": "Gold File Content",
            "title": f"Content from {file_stem}",
            "caption": "",
            "column_headers": [],
            "row_labels": [],
            "years": [],
            "unit": None,
            "period": None,
            "n_rows": 0,
            "n_cols": 0,
            "retrieval_strategy": "oracle_fair",
            "retrieval_channel": "oracle_fair",
            "html": None,
            "content": content,  # Raw file content for LLM to search
        }
        for i, (file_stem, content) in enumerate(zip(gold_files, file_content))
    ]

    # Extract
    print("\n[EXTRACT]", flush=True)
    t0 = time.time()
    per_dr_entries = {dr.get("id", "?"): oracle_entries for dr in spec.get("data_requests", [])}
    try:
        extraction = extract_structured(spec, per_dr_entries, question, verbose=False)
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
        answer = verify_answer(raw_answer, spec, question, extraction.get("extractions", {}))
        t_verify = time.time() - t0
        print(f"  {t_verify:.2f}s  → {answer}", flush=True)
    except Exception as e:
        print(f"  FAIL: {str(e)[:100]}", flush=True)
        answer = raw_answer

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
    parser.add_argument("--n", type=int, default=5)
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

    print(f"Oracle-Fair Mode: {len(uids)} questions\n", flush=True)

    results = []
    for uid in uids:
        row = benchmark[uid]
        result = oracle_fair_solve(uid, row["question"], row.get("source_files", ""))

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
