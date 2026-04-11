#!/usr/bin/env python3
"""Layer test with REAL corpus data — bottom-up.

Starts at the simplest level: give the LLM one real table from the corpus,
ask it to extract ONE value. Then escalate complexity.

Level 1: Extract a single value from a single table
Level 2: Sum 12 monthly values from a single table
Level 3: Handle a table where months span two calendar years
Level 4: Find values across two tables

Usage:
    OPENROUTER_API_KEY=... python3 layer_real.py
"""

import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

CORPUS_DIR = Path(
    os.environ.get("CORPUS_DIR", str(Path(__file__).resolve().parent.parent.parent / "corpus"))
)
API_KEY = os.environ.get("OPENROUTER_API_KEY", os.environ.get("LLM_API_KEY", ""))
MODEL = os.environ.get("SOLVER_MODEL", "minimax/minimax-m2.5")


def call_llm(system_prompt, user_prompt, max_tokens=2048):
    if not API_KEY:
        print("ERROR: No API key", file=sys.stderr)
        sys.exit(1)
    payload = json.dumps(
        {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
    ).encode()
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/officeqa-arena",
    }
    for attempt in range(3):
        try:
            req = urllib.request.Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=payload,
                headers=headers,
            )
            with urllib.request.urlopen(req, timeout=90) as resp:
                result = json.loads(resp.read().decode())
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            print(f"  LLM attempt {attempt + 1} failed: {e}", file=sys.stderr)
            if attempt < 2:
                time.sleep(2**attempt)
    return None


def read_lines(fname, start, end):
    """Read lines from a corpus file."""
    fpath = CORPUS_DIR / fname
    lines = fpath.read_text(errors="replace").splitlines()
    return "\n".join(lines[start - 1 : end])


EXTRACT_PROMPT = """You are a data extraction specialist. You receive raw data from a U.S. Treasury Bulletin table and a specific question about one value.

Read the table carefully. The table uses pipe-delimited columns. Headers tell you what each column means.

Rules:
- Read the column headers to understand the table structure
- Find the EXACT row and column matching the question
- Output ONLY the value, nothing else
- If the value has commas (like 1,559), include them
- If the value has footnote markers (like "5,240 3/"), return just the number: 5,240

Output format:
VALUE: <the number>"""


# ============================================================================
# TESTS — real corpus data, escalating complexity
# ============================================================================

TESTS = []

# ── Level 1: Single value, single table, clear headers ──

# FY 1940 total national defense from the annual rows
TESTS.append(
    {
        "level": 1,
        "uid": "REAL_L1_01",
        "question": "In this table, what is the value in the 'National defense' column for fiscal year 1940?",
        "data_file": "treasury_bulletin_1941_06.txt",
        "data_lines": (630, 656),
        "expected": "1,559",
        "description": "FY1940 total national defense — row '1940', column 'National defense'",
    }
)

# Single month value
TESTS.append(
    {
        "level": 1,
        "uid": "REAL_L1_02",
        "question": "In this table, what is the value in the 'National defense' column for the month of October (1940)?",
        "data_file": "treasury_bulletin_1941_06.txt",
        "data_lines": (630, 656),
        "expected": "287",
        "description": "Oct 1940 national defense",
    }
)

# Different column
TESTS.append(
    {
        "level": 1,
        "uid": "REAL_L1_03",
        "question": "In this table, what is the 'Total' expenditure for fiscal year 1939?",
        "data_file": "treasury_bulletin_1941_06.txt",
        "data_lines": (630, 656),
        "expected": "8,432",
        "description": "FY1939 total",
    }
)

# ── Level 2: Sum monthly values from a single table ──

# Sum May-Dec 1940 from this table (8 months visible)
TESTS.append(
    {
        "level": 2,
        "uid": "REAL_L2_01",
        "question": "In this table, what is the SUM of 'National defense' expenditures for the months May through December 1940? List each monthly value and the total.",
        "data_file": "treasury_bulletin_1941_06.txt",
        "data_lines": (630, 656),
        "expected": "2,049",  # 154+153+177+200+219+287+376+473 = 2039... let me recheck
        "expected_values": [154, 153, 177, 200, 219, 287, 376, 473],
        "description": "Sum of May-Dec 1940 national defense (8 months from this table)",
    }
)

# ── Level 3: Understand fiscal year vs calendar year in the table ──

TESTS.append(
    {
        "level": 3,
        "uid": "REAL_L3_01",
        "question": "This table shows monthly expenditures. The rows labeled '1940-May' through 'December' are May 1940 through December 1940. The rows 'January 1941' through 'May' are January 1941 through May 1941. What are ALL the National defense values for CALENDAR YEAR 1940 months that appear in this table (May through December 1940)? List them.",
        "data_file": "treasury_bulletin_1941_06.txt",
        "data_lines": (630, 656),
        "expected_values": [154, 153, 177, 200, 219, 287, 376, 473],
        "expected": "2,039",
        "description": "CY 1940 months from fiscal-year-oriented table",
    }
)
# Fix: recompute 154+153+177+200+219+287+376+473 = 2039

# Update L2 expected
TESTS[3]["expected"] = "2,039"

# ── Level 4: Full real question with real evidence ──

# Now give the FULL question text with the real table
TESTS.append(
    {
        "level": 4,
        "uid": "REAL_L4_01",
        "question": "Using specifically only the reported values for all individual calendar months in 1940, what was the total sum value of expenditures for the U.S. national defense? This table shows May-December 1940. What is the national defense value for EACH of these months and their sum?",
        "data_file": "treasury_bulletin_1941_06.txt",
        "data_lines": (630, 656),
        "expected_values": [154, 153, 177, 200, 219, 287, 376, 473],
        "expected": "2,039",
        "description": "Full question with real table — partial CY 1940",
    }
)


def parse_response(response, test):
    """Extract value(s) from LLM response."""
    # Try to find VALUE: line
    value_match = re.search(r"VALUE:\s*([^\n]+)", response, re.I)
    answer = value_match.group(1).strip() if value_match else None

    # Try to find individual values listed
    numbers = re.findall(r"(?:^|\s|:|\|)(\d[\d,]*\.?\d*)", response)
    found_values = []
    for n in numbers:
        try:
            found_values.append(float(n.replace(",", "")))
        except ValueError:
            pass

    return answer, found_values


def check_answer(expected, got):
    if not got or not expected:
        return False

    def norm(s):
        s = str(s).replace(",", "").replace("%", "").strip()
        try:
            return float(s)
        except ValueError:
            return s

    ne, ng = norm(expected), norm(got)
    if isinstance(ne, float) and isinstance(ng, float):
        if ne == 0:
            return abs(ng) < 0.5
        return abs(ne - ng) / abs(ne) < 0.02
    return ne == ng


def check_values(expected_vals, found_vals):
    """Check if all expected values appear in the found values."""
    if not expected_vals:
        return True, []
    missing = []
    for ev in expected_vals:
        if not any(abs(fv - ev) < 0.5 for fv in found_vals):
            missing.append(ev)
    return len(missing) == 0, missing


def run_tests():
    print("Real Corpus Layer Test — Bottom-Up")
    print(f"Corpus: {CORPUS_DIR}")
    print(f"Model: {MODEL}")
    print(f"Tests: {len(TESTS)}")
    print("=" * 70)

    results = []

    for tc in TESTS:
        uid = tc["uid"]
        level = tc["level"]
        print(f"\n--- {uid} [Level {level}] ---")
        print(f"  {tc['description']}")
        print(f"  Q: {tc['question'][:100]}...")

        # Read real data
        data = read_lines(tc["data_file"], tc["data_lines"][0], tc["data_lines"][1])
        print(
            f"  Data: {len(data)} chars from {tc['data_file']}:{tc['data_lines'][0]}-{tc['data_lines'][1]}"
        )

        # Show first few lines of data
        for dl in data.split("\n")[:4]:
            print(f"    {dl[:120]}")
        if data.count("\n") > 4:
            print(f"    ... ({data.count(chr(10)) - 3} more lines)")

        # Build prompt
        user_prompt = f"QUESTION: {tc['question']}\n\nDATA:\n{data}"

        response = call_llm(EXTRACT_PROMPT, user_prompt, max_tokens=1024)
        if not response:
            print("  FAIL: No LLM response")
            results.append({"uid": uid, "level": level, "pass": False, "error": "no_response"})
            continue

        print("  LLM response:")
        for line in response.strip().split("\n"):
            print(f"    {line[:120]}")

        answer, found_values = parse_response(response, tc)

        # Check answer
        answer_ok = check_answer(tc["expected"], answer) if tc.get("expected") else True
        if answer_ok:
            print(f"  ANSWER: OK (got='{answer}', expected='{tc.get('expected')}')")
        else:
            print(f"  ANSWER: FAIL (got='{answer}', expected='{tc.get('expected')}')")

        # Check individual values
        values_ok = True
        missing = []
        if tc.get("expected_values"):
            values_ok, missing = check_values(tc["expected_values"], found_values)
            if values_ok:
                print(f"  VALUES: OK (found all {len(tc['expected_values'])} expected values)")
            else:
                print(f"  VALUES: FAIL (missing: {missing})")

        passed = answer_ok and values_ok
        status = "PASS" if passed else "FAIL"
        print(f"  >>> {status}")

        results.append(
            {
                "uid": uid,
                "level": level,
                "pass": passed,
                "answer_ok": answer_ok,
                "values_ok": values_ok,
                "got_answer": answer,
                "expected": tc.get("expected"),
                "missing_values": missing,
                "response_preview": response[:300],
            }
        )
        time.sleep(1)

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    passed = sum(1 for r in results if r["pass"])
    total = len(results)
    print(f"Passed: {passed}/{total} ({100 * passed / total:.0f}%)")
    print()

    by_level = {}
    for r in results:
        by_level.setdefault(r["level"], []).append(r)

    for level in sorted(by_level):
        level_results = by_level[level]
        level_pass = sum(1 for r in level_results if r["pass"])
        print(f"  Level {level}: {level_pass}/{len(level_results)}")
        for r in level_results:
            status = "PASS" if r["pass"] else "FAIL"
            detail = (
                f" got='{r.get('got_answer', '')}' exp='{r.get('expected', '')}'"
                if not r["pass"]
                else ""
            )
            if r.get("missing_values"):
                detail += f" missing={r['missing_values']}"
            print(f"    {r['uid']}: {status}{detail}")

    out_path = Path(__file__).parent / "layer_real_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return passed == total


if __name__ == "__main__":
    run_tests()
