#!/usr/bin/env python3
"""
A/B testing framework for prompt strategies.

Test different system prompts, skills configurations, and parameters
against a small set of questions to find the best approach.

Usage:
    # Compare default vs chain-of-verification prompt
    python3 nomcp/experiment.py --corpus /path/to/corpus \
        --variants default,cove \
        --uids UID0001,UID0002,UID0003,UID0007,UID0011

    # List available prompt variants
    python3 nomcp/experiment.py --list-variants

    # Test a single variant on smoke set
    python3 nomcp/experiment.py --corpus /path/to/corpus --variants cove --smoke
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Import the runner
sys.path.insert(0, str(Path(__file__).parent))
import tempfile

from run_all import NOMCP_DIR, PROJECT_ROOT, load_questions, pre_build_index, run_single_task

# ── Prompt Variants ──────────────────────────────────────────────────

VARIANTS = {}


def register_variant(
    name, description, system_prompt_override=None, extra_skills=None, temperature=0.0
):
    """Register a prompt variant for testing."""
    VARIANTS[name] = {
        "name": name,
        "description": description,
        "system_prompt_override": system_prompt_override,
        "extra_skills": extra_skills or "",
        "temperature": temperature,
    }


# ── Default: current system.j2 ──────────────────────────────────────
register_variant(
    "default",
    "Current system.j2 prompt (two-pass reasoning)",
)

# ── CoVe: Chain of Verification ─────────────────────────────────────
register_variant(
    "cove",
    "Chain-of-Verification: model must verify each step before proceeding",
    system_prompt_override="""\
You are a Treasury Data Analyst. Answer questions using Treasury Bulletin data.

## MANDATORY PROTOCOL: Chain of Verification (CoVe)

For EVERY question, follow these steps strictly:

### Step 1: DECOMPOSE the question
Before searching, write down:
- What METRIC is being asked about? (e.g., "national defense expenditures")
- What TIME PERIOD? (fiscal year, calendar year, specific months?)
- What UNITS? (millions, thousands, nominal dollars?)
- What OPERATION? (direct lookup, sum, percent change, difference?)

### Step 2: SEARCH
```
python3 /installed-agent/solve.py "FULL QUESTION"
```

### Step 3: VERIFY the source
After getting results, confirm:
- [ ] Table title matches the metric I need
- [ ] Column header matches what I'm extracting
- [ ] Units in table header match what the question asks
- [ ] Year/period type matches (fiscal vs calendar)
- [ ] Row label matches exactly (not a subtotal or different category)

### Step 4: EXTRACT and cross-check
Read the value. Then verify:
- [ ] Is this value in the right order of magnitude for the era?
- [ ] If I can see a total row AND components, do they add up?
- [ ] Did I strip footnote markers (3/, *, r/) from the number?

### Step 5: COMPUTE (if needed)
```
python3 -c "print(round(...))"
```
Verify: Does the result make intuitive sense?

### Step 6: SUBMIT
```
echo -n "ANSWER" > /app/answer.txt
```

## FISCAL YEAR RULES
- Pre-1977: FY = Jul(Y-1) to Jun(Y). Row "1940" = FY1940 = Jul 1939-Jun 1940.
- Post-1977: FY = Oct(Y-1) to Sep(Y).
- Calendar year = Jan-Dec. NOT the same as fiscal year row.

## ANSWER FORMAT: numeric only. Commas if data uses them. % for percentages.

{{ instruction }}
""",
)

# ── Structured Table-as-Thought ──────────────────────────────────────
register_variant(
    "table_thought",
    "Table-as-Thought: model organizes reasoning in a structured table format",
    system_prompt_override="""\
You are a Treasury Data Analyst. Answer questions using Treasury Bulletin data.

## REASONING FORMAT: Table-as-Thought

Structure ALL your reasoning as a table before submitting:

| Step | Action | Result | Verified? |
|------|--------|--------|-----------|
| 1. Parse question | Identify metric, year, period type, units | ... | ... |
| 2. Search | Run solve.py | Found N tables | ... |
| 3. Select table | Pick best match by title/year | Table X, line Y | ... |
| 4. Verify source | Check title, units, column, row | All match / Mismatch: ... | Y/N |
| 5. Extract value | Read specific cell | VALUE | ... |
| 6. Compute | Apply formula if needed | RESULT | ... |
| 7. Format answer | Apply rounding, units, commas | FINAL | ... |

## SEARCH COMMAND
```
python3 /installed-agent/solve.py "FULL QUESTION"
```

## COMPUTATION (always use python3)
```
python3 -c "print(round(...))"
```

## SUBMIT
```
echo -n "ANSWER" > /app/answer.txt
```

## KEY RULES
- Fiscal year != Calendar year. Pre-1977: FY=Jul-Jun. Post-1977: FY=Oct-Sep.
- Always check table units before using values.
- Strip footnotes (3/, *, r/) and commas from numbers.
- A wrong answer > no answer. ALWAYS submit.

{{ instruction }}
""",
)

# ── Aggressive: Minimal reasoning, fast execution ────────────────────
register_variant(
    "fast",
    "Minimal prompt, 3 tool calls max, optimized for speed/cost",
    system_prompt_override="""\
Treasury Data Analyst. Answer financial questions from Treasury Bulletin files.

1. python3 /installed-agent/solve.py "QUESTION"
2. Read output. Extract value. If math needed: python3 -c "print(...)"
3. echo -n "ANSWER" > /app/answer.txt

FY rules: Pre-1977: Jul-Jun. Post-1977: Oct-Sep. Calendar year: Jan-Dec.
Strip footnotes (3/, *). Use commas in answer if data has them. % for percentages.

{{ instruction }}
""",
)

# ── Retry-on-doubt: Model re-searches if uncertain ──────────────────
register_variant(
    "retry",
    "Model explicitly told to re-search if first result seems wrong",
    system_prompt_override="""\
You are a Treasury Data Analyst. Answer questions using Treasury Bulletin data.

## PROTOCOL

### Step 1: Search
```
python3 /installed-agent/solve.py "FULL QUESTION"
```

### Step 2: Evaluate confidence
After reading results, rate your confidence:
- HIGH: Table title, units, and row clearly match. Proceed to compute/submit.
- MEDIUM: Partial match. Run a targeted search to verify:
  ```
  python3 /installed-agent/search.py find "SPECIFIC KEYWORD" YEAR
  python3 /installed-agent/search.py table FILENAME LINE_NUMBER
  ```
- LOW: No good match. Try broader terms or adjacent years.

### Step 3: Compute (if needed)
```
python3 -c "print(round(...))"
```

### Step 4: Self-check
Before submitting, verify:
- Calendar vs fiscal year: which does the question ask for?
- Units: does the table's unit match what's asked?
- Value magnitude: does it make sense for the era?

### Step 5: Submit
```
echo -n "ANSWER" > /app/answer.txt
```

## FISCAL YEAR RULES
- Pre-1977: FY = Jul(Y-1)-Jun(Y). Post-1977: FY = Oct(Y-1)-Sep(Y).
- Calendar year = Jan-Dec.
- Row labeled "1940" = fiscal year total (usually).
- For calendar year: find "Calendar yr." row or sum Jan-Dec months.

ALWAYS submit an answer. A wrong answer scores higher than no answer.

{{ instruction }}
""",
    temperature=0.1,  # Slightly more creative for retry strategies
)

# ── Optimized: Short prompt tuned for MiniMax M2.5 quirks ────────────
# Based on research: M2.5 over-thinks with verbose prompts, prefers
# decision-tree format, 88% hallucination rate needs structural enforcement
_optimized_prompt = (
    (NOMCP_DIR / "prompts" / "system_optimized.j2").read_text()
    if (NOMCP_DIR / "prompts" / "system_optimized.j2").exists()
    else None
)

if _optimized_prompt:
    register_variant(
        "optimized",
        "Short decision-tree prompt tuned for MiniMax M2.5 (recommended)",
        system_prompt_override=_optimized_prompt,
    )

# ── Decompose: Mandatory checklist before searching ──────────────────
_decompose_prompt = (
    (NOMCP_DIR / "prompts" / "system_decompose.j2").read_text()
    if (NOMCP_DIR / "prompts" / "system_decompose.j2").exists()
    else None
)
if _decompose_prompt:
    register_variant(
        "decompose",
        "Mandatory checklist decomposition before any tool call",
        system_prompt_override=_decompose_prompt,
    )

# ── Multi-search: Runs multiple searches for cross-verification ──────
_multi_search_prompt = (
    (NOMCP_DIR / "prompts" / "system_multi_search.j2").read_text()
    if (NOMCP_DIR / "prompts" / "system_multi_search.j2").exists()
    else None
)
if _multi_search_prompt:
    register_variant(
        "multi_search",
        "Multiple searches for cross-verification of values",
        system_prompt_override=_multi_search_prompt,
    )

# ── Expert: Senior auditor persona with professional standards ───────
_expert_prompt = (
    (NOMCP_DIR / "prompts" / "system_expert.j2").read_text()
    if (NOMCP_DIR / "prompts" / "system_expert.j2").exists()
    else None
)
if _expert_prompt:
    register_variant(
        "expert",
        "Senior government auditor persona with strict professional standards",
        system_prompt_override=_expert_prompt,
    )


def run_variant(
    variant_name: str, tasks: list, api_key: str, corpus_dir: str, shared_index_dir: str
) -> list:
    """Run a set of tasks with a specific prompt variant."""
    variant = VARIANTS[variant_name]
    agent_dir = str(NOMCP_DIR)

    # If variant has a custom system prompt, temporarily override the template
    original_template = (NOMCP_DIR / "prompts" / "system.j2").read_text()
    if variant["system_prompt_override"]:
        (NOMCP_DIR / "prompts" / "system.j2").write_text(variant["system_prompt_override"])

    try:
        results = []
        for task in tasks:
            result = run_single_task(task, api_key, corpus_dir, agent_dir, shared_index_dir)
            results.append(result)
        return results
    finally:
        # Restore original template
        (NOMCP_DIR / "prompts" / "system.j2").write_text(original_template)


def main():
    parser = argparse.ArgumentParser(description="OfficeQA prompt A/B testing")
    parser.add_argument("--corpus", type=str, default=os.environ.get("CORPUS_DIR", "/app/corpus"))
    parser.add_argument(
        "--cases", type=str, default=str(PROJECT_ROOT / "data" / "officeqa_full.csv")
    )
    parser.add_argument("--uids", type=str, default=None)
    parser.add_argument(
        "--variants", type=str, default="default", help="Comma-separated variant names to test"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Use smoke test UIDs (UID0001,UID0002,UID0007)"
    )
    parser.add_argument(
        "--hard10", action="store_true", help="Use curated 10 hardest questions from hard10.jsonl"
    )
    parser.add_argument("--list-variants", action="store_true")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    if args.list_variants:
        print("Available prompt variants:\n")
        for name, v in VARIANTS.items():
            print(f"  {name:15s}  {v['description']}")
        return

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: Set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(1)

    corpus_dir = os.path.abspath(args.corpus)

    if args.hard10:
        # Load from hard10.jsonl
        hard10_path = NOMCP_DIR / "hard10.jsonl"
        tasks = []
        with open(hard10_path) as f:
            for line in f:
                if line.strip():
                    d = json.loads(line)
                    tasks.append(
                        {
                            "uid": d["uid"],
                            "question": d["question"],
                            "answer": d["expected_answer"]
                            if isinstance(d["expected_answer"], str)
                            else str(d["expected_answer"]),
                            "difficulty": d.get("difficulty", "hard"),
                            "source_files": "",
                        }
                    )
    elif args.smoke:
        uid_list = ["UID0001", "UID0002", "UID0007"]
        tasks = load_questions(args.cases, uid_list)
    elif args.uids:
        uid_list = [u.strip() for u in args.uids.split(",")]
        tasks = load_questions(args.cases, uid_list)
    else:
        tasks = load_questions(args.cases, None)
    variant_names = [v.strip() for v in args.variants.split(",")]

    # Validate variants
    for v in variant_names:
        if v not in VARIANTS:
            print(
                f"ERROR: Unknown variant '{v}'. Use --list-variants to see options.",
                file=sys.stderr,
            )
            sys.exit(1)

    print(f"Tasks: {len(tasks)}", file=sys.stderr)
    print(f"Variants: {variant_names}", file=sys.stderr)
    print(f"Total API calls: ~{len(tasks) * len(variant_names) * 3} (est.)", file=sys.stderr)

    # Pre-build index
    shared_index_dir = tempfile.mkdtemp(prefix="officeqa_index_")
    pre_build_index(corpus_dir, str(NOMCP_DIR), shared_index_dir)

    # Run each variant
    all_results = {}
    for vname in variant_names:
        print(f"\n{'=' * 60}", file=sys.stderr)
        print(f"VARIANT: {vname} — {VARIANTS[vname]['description']}", file=sys.stderr)
        print(f"{'=' * 60}\n", file=sys.stderr)

        results = run_variant(vname, tasks, api_key, corpus_dir, shared_index_dir)
        all_results[vname] = results

    # Summary comparison
    print(f"\n{'=' * 60}")
    print("A/B TEST RESULTS")
    print(f"{'=' * 60}")
    print(
        f"{'Variant':<18s} {'Correct':>8s} {'Total':>6s} {'Rate':>7s} {'Avg Time':>9s} {'Avg Calls':>10s}"
    )
    print(f"{'-' * 60}")

    for vname in variant_names:
        results = all_results[vname]
        correct = sum(1 for r in results if r["correct"])
        total = len(results)
        rate = correct / total * 100 if total else 0
        avg_time = sum(r["elapsed_s"] for r in results) / total if total else 0
        avg_calls = sum(r["tool_calls"] for r in results) / total if total else 0
        print(
            f"{vname:<18s} {correct:>8d} {total:>6d} {rate:>6.1f}% {avg_time:>8.1f}s {avg_calls:>9.1f}"
        )

    # Per-task comparison
    if len(variant_names) > 1:
        print("\nPer-task breakdown:")
        print(f"{'UID':<10s}", end="")
        for vname in variant_names:
            print(f" {vname:>12s}", end="")
        print()

        for i, task in enumerate(tasks):
            uid = task["uid"]
            print(f"{uid:<10s}", end="")
            for vname in variant_names:
                r = all_results[vname][i]
                status = "PASS" if r["correct"] else "FAIL"
                print(f" {status:>12s}", end="")
            print(f"  expected={task['answer']}")

    # Save detailed results
    if args.output:
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
