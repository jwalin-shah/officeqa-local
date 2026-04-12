import csv
import json
import os
import random
import sys
from pathlib import Path

# Add current dir to sys.path to import local modules
sys.path.append(os.getcwd())

from extract import build_context_from_entries
from solve import _run_extract_and_compute, decompose, retrieve_for_spec, scout


def audit():
    # 1. Load questions
    questions = []
    with open("officeqa_full.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            questions.append(row)

    # 2. Pick 30 random questions
    # Use a fixed seed for reproducibility if needed, or just random
    random.seed(42)
    sampled_questions = random.sample(questions, 30)  # nosec B311  # reproducible audit sample

    results = []

    # Failure counters
    decompose_errors = 0
    retrieval_misses = 0
    extraction_failures = 0
    total_audited = 0

    for i, q_row in enumerate(sampled_questions):
        uid = q_row["uid"]
        question = q_row["question"]
        gold_files = [f.strip() for f in q_row["source_files"].split("\n") if f.strip()]
        gold_stems = {Path(f).stem for f in gold_files}

        print(f"\n[{i + 1}/30] Auditing {uid}: {question[:100]}...")

        # Step 1: Decompose
        try:
            hint = scout(question)
            spec = decompose(question, scout_hint=hint)
            if not spec:
                print(f"  Decompose failed for {uid}")
                decompose_errors += 1
                results.append({"uid": uid, "status": "decompose_error"})
                continue
        except Exception as e:
            print(f"  Decompose exception for {uid}: {e}")
            decompose_errors += 1
            results.append({"uid": uid, "status": "decompose_error", "error": str(e)})
            continue

        # Evaluate hints
        drs = spec.get("data_requests", [])
        hints_ok = True
        for dr in drs:
            row_hint = dr.get("row_hint", "")
            # Heuristic check for vocabulary - if it's too long or conversational
            if len(row_hint.split()) > 10 or "please find" in row_hint.lower():
                hints_ok = False
                break

        if not hints_ok:
            print(f"  Poor hints for {uid}: row_hint='{row_hint}'")
            # We'll still continue but note it

        # Step 2: Retrieval
        try:
            per_dr = retrieve_for_spec(spec, question, top_k_per_dr=20)

            all_retrieved_files = set()
            for _dr_id, entries in per_dr.items():
                for entry in entries:
                    all_retrieved_files.add(Path(entry["file"]).stem)

            gold_in_top20 = any(stem in all_retrieved_files for stem in gold_stems)

            if not gold_in_top20:
                print(
                    f"  Retrieval miss for {uid}. Gold files: {gold_stems}. Top 20 stems: {list(all_retrieved_files)[:5]}..."
                )
                retrieval_misses += 1
                results.append(
                    {
                        "uid": uid,
                        "status": "retrieval_miss",
                        "gold_files": list(gold_stems),
                        "retrieved": list(all_retrieved_files),
                        "spec": spec,
                    }
                )
                continue
        except Exception as e:
            print(f"  Retrieval exception for {uid}: {e}")
            retrieval_misses += 1
            results.append({"uid": uid, "status": "retrieval_error", "error": str(e)})
            continue

        # Step 3: Extraction
        try:
            # Simulate solve.py's extract/compute
            # We want to see if it fails with NO_VALUES despite gold being present
            answer, extraction = _run_extract_and_compute(spec, per_dr, question, verbose=False)

            if answer.startswith("NO_VALUES"):
                print(f"  Extraction failure (NO_VALUES) for {uid}")
                extraction_failures += 1

                # Inspect context
                # Find which DR failed
                failed_dr_id = answer.split("[")[1].split("]")[0] if "[" in answer else None
                failed_entries = per_dr.get(failed_dr_id, []) if failed_dr_id else []

                # Check if gold file was in the failed DR's entries
                gold_in_failed_dr = any(Path(e["file"]).stem in gold_stems for e in failed_entries)

                context = ""
                if gold_in_failed_dr:
                    # Get the context specifically for the gold file
                    gold_entries = [e for e in failed_entries if Path(e["file"]).stem in gold_stems]
                    context = build_context_from_entries(gold_entries)

                results.append(
                    {
                        "uid": uid,
                        "status": "extraction_failure",
                        "answer": answer,
                        "gold_in_failed_dr": gold_in_failed_dr,
                        "context_sample": context[:1000]
                        if context
                        else "No gold entry in failed DR",
                        "spec": spec,
                    }
                )
            else:
                print(f"  Success for {uid}: {answer}")
                results.append({"uid": uid, "status": "success", "answer": answer})

        except Exception as e:
            print(f"  Extraction exception for {uid}: {e}")
            extraction_failures += 1
            results.append({"uid": uid, "status": "extraction_error", "error": str(e)})

        total_audited += 1

    # Report Heatmap
    print("\n" + "=" * 40)
    print("FAILURE HEATMAP")
    print("=" * 40)
    if total_audited > 0:
        print(
            f"Decompose Error:     {decompose_errors / total_audited * 100:5.1f}% ({decompose_errors})"
        )
        print(
            f"Retrieval Miss:      {retrieval_misses / total_audited * 100:5.1f}% ({retrieval_misses})"
        )
        print(
            f"Extraction Failure:  {extraction_failures / total_audited * 100:5.1f}% ({extraction_failures})"
        )
        print(
            f"Success/Other:       {(total_audited - decompose_errors - retrieval_misses - extraction_failures) / total_audited * 100:5.1f}%"
        )
    else:
        print("No questions successfully audited.")
    print("=" * 40)

    # Save detailed results
    with open("audit_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    audit()
