#!/usr/bin/env python3
"""
Evaluate agent performance on OfficeQA benchmark.
"""

import csv
import sys
from pathlib import Path

from agent import TreasuryAgent


def evaluate_benchmark(limit=None):
    """Run agent on benchmark and report scores."""

    agent = TreasuryAgent()
    dataset_path = Path("officeqa_full.csv")

    if not dataset_path.exists():
        print("❌ officeqa_full.csv not found")
        return

    # Load dataset
    questions = []
    with open(dataset_path) as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            questions.append(row)

    print(f"\n🧪 Testing on {len(questions)} questions...\n")

    correct = 0
    total = len(questions)

    for i, q in enumerate(questions, 1):
        uid = q["uid"]
        question = q["question"]
        expected = q["answer"].strip()
        difficulty = q.get("difficulty", "unknown")

        print(f"[{i}/{total}] {uid} ({difficulty})")
        print(f"  Q: {question[:80]}...")

        try:
            answer = agent.answer_question(question)

            # Simple exact match for now (can improve)
            is_correct = expected.lower() in answer.lower()

            if is_correct:
                correct += 1
                print("  ✓ Correct")
            else:
                print(f"  ✗ Got: {answer[:80]}...")
                print(f"    Expected: {expected}")
        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()

    # Report
    accuracy = correct / total * 100
    print("\n📊 Results:")
    print(f"  Correct: {correct}/{total}")
    print(f"  Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    evaluate_benchmark(limit)
