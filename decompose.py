#!/usr/bin/env python3
"""
V3: Decompose all 246 questions with critical fixes for:
1. CY Annual Total Mismatch (42 cases)
2. Sum/Total Computation Error (25 cases)
3. Regression Mismatch (6 cases)
4. Period Type Tagging (6 cases)
"""

import json
import os
import sys
import urllib.request
import re
import csv
import time
import concurrent.futures
from threading import Semaphore

API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MAX_CONCURRENT = 24
semaphore = Semaphore(MAX_CONCURRENT)

def load_questions():
    """Load all 246 questions from CSV"""
    questions = []
    with open('data/officeqa_full.csv', 'r') as f:
        for row in csv.DictReader(f):
            questions.append(row)
    return questions

def call_llm(question):
    """Call LLM with improved v3 prompt"""
    if not API_KEY:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        return None

    # CRITICAL RULES LEARNED FROM V1/V2 ANALYSIS
    prompt = """You are a precise question decomposition expert. Analyze this Treasury question and output ONLY valid JSON.

QUESTION: """ + question + """

===== CRITICAL RULES =====

1. CALENDAR YEAR TOTAL RULE:
   If period_type = "calendar" AND question asks for "total" or "sum":
     → computation: "sum"
     → value_format: "monthly_series" (NOT "annual_total")

   WHY: Treasury Bulletins only provide annual totals for Fiscal Years.
   To get Calendar Year totals, you MUST sum 12 individual months (Jan-Dec).

   Examples:
   ✓ "total expenditures in calendar year 1776" → sum, monthly_series
   ✓ "sum of...in 2050" (no "fiscal" mentioned) → sum, monthly_series
   ✗ "total in FY 1776" → sum, annual_total (fiscal year row exists)

2. SUM/TOTAL DETECTION RULE:
   If question asks you to combine multiple values into a single total or aggregate:
     → computation: "sum" (typically, not direct)

   Treasury totals are computed by summing component parts. If a question asks for
   "total" or "combined" amount, it's asking you to add up multiple rows/months.

   Exception: If it's clearly asking for a pre-computed total that exists as a single row,
   use "direct". Default to "sum".

   Examples:
   ✓ "total expenditures in 2050" → sum (add up all components)
   ✓ "sum of 12 months in 1776" → sum (add months)
   ✓ "combined holdings by all banks" → sum (add bank totals)

3. REGRESSION RULE:
   If question asks you to fit a line through data points or use a model to predict/project:
     → computation: "linear_regression"

   This includes questions asking for slope, intercept, fitting a model, forecasting,
   or using historical data to estimate future values.

   Examples:
   ✓ "fit an OLS regression to predict 2050" → linear_regression
   ✓ "what is the slope and intercept" → linear_regression
   ✓ "forecast the value using 2040-2048 data" → linear_regression

4. PERIOD TYPE RULE:
   - If "FY" or "fiscal year" explicitly in text → period_type: "fiscal"
   - If "calendar year" or "CY" explicitly in text → period_type: "calendar"
   - If only a year like "in 1776" with no explicit period → DEFAULT to "calendar"
     (most Treasury questions are calendar year unless explicitly fiscal)

   Examples:
   ✓ "FY 2050" → fiscal
   ✓ "fiscal year 1776" → fiscal
   ✓ "calendar year 2050" → calendar
   ✓ "in 1776" (no qualifier) → calendar

5. EXTREME VALUE RULE:
   If question asks for the largest or smallest value across a dataset:
     → computation: "max" or "min"

   Examples:
   ✓ "what was the highest amount owed" → max
   ✓ "which year had the lowest interest rate" → min
   ✓ "the greatest holding was" → max

===== UNCERTAINTY HANDLING =====
If you are uncertain about period_type, computation, or data availability:
- Still output the best answer
- Add to notes: "UNCERTAIN: [what you're unsure about]" so solver can review
- Do NOT leave fields blank or output null

Example:
{
  "period_type": "calendar",
  "notes": "UNCERTAIN: Question doesn't explicitly state period type, defaulting to calendar. Could also be fiscal."
}

===== OUTPUT FIELDS (ALL REQUIRED) =====

Output ONLY valid JSON with these fields:
{
  "data_year": <year or [list of years]>,
  "topic": "<data category>",
  "period_type": "<fiscal or calendar>",
  "computation": "<standard type OR custom description>",
  "value_format": "<annual_total|monthly_series|quarterly_series|multi_year>",
  "years_needed": [<list of years>],
  "search_terms": [<keywords>],
  "notes": "<special handling, units, rounding requirements, uncertainties>"
}

COMPUTATION field can be:
- STANDARD: sum, direct, difference, percent_change, geometric_mean, average, linear_regression, max, min, count, ratio, median, correlation, standard_deviation
- CUSTOM: If the question requires a specialized statistical operation not listed above, describe it concisely.

  Examples of good custom descriptions:
  ✓ "Gini coefficient calculation"
  ✓ "Pearson correlation of X vs Y, then multiply by total"
  ✓ "VaR calculation: sqrt(w1²*σ1² + w2²*σ2² + 2*w1*w2*ρ) * z-score"
  ✓ "sum then trimmed_mean (remove top/bottom 10%)"
  ✓ "Box-Cox transformation with lambda=0.75"

  Do NOT invent computation names. Use actual statistical/financial terms. If unsure,
  describe the mathematical operation required.

===== VALIDATION CHECKLIST =====
Before outputting, check:
□ period_type is "fiscal" or "calendar" (required; if uncertain, still pick best guess + note)
□ computation is filled (standard type or custom description; never empty)
□ If calendar year + "total" → computation typically is "sum" AND value_format is "monthly_series"
□ If "regression/fit/forecast" → computation typically is "linear_regression"
□ If "highest"/"lowest" → computation typically is "max" or "min"
□ All required years are in years_needed
□ notes include any uncertainties or special requirements
□ If using custom computation, verify it's a real statistical/financial term, not made-up

Output ONLY valid JSON. Your response must start with { and end with }.
"""

    payload = json.dumps({
        "model": "minimax/minimax-m2.5",
        "messages": [
            {
                "role": "system",
                "content": "You are a precise JSON output expert. Output ONLY valid JSON, nothing else."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.1,
        "max_tokens": 1500,
    }).encode()

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=payload,
            headers=headers
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read().decode())

        if "choices" in result and len(result["choices"]) > 0:
            return result["choices"][0]["message"]["content"]
        return None
    except Exception as e:
        print(f"Error calling LLM: {e}", file=sys.stderr)
        return None

def parse_json_response(text):
    """Extract JSON from LLM response"""
    if not text:
        return None

    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    # Try direct JSON parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting JSON object
    try:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
    except:
        pass

    return None

def decompose_question(idx, question):
    """Decompose a single question (with concurrency limiting)"""
    with semaphore:
        response = call_llm(question)

        if not response:
            return {
                "id": idx,
                "question": question,
                "error": "LLM call failed"
            }

        result = parse_json_response(response)

        if result:
            return {
                "id": idx,
                "question": question,
                "result": result
            }
        else:
            return {
                "id": idx,
                "question": question,
                "error": "JSON parse failed",
                "raw_response": response[:200]
            }

def main():
    questions = load_questions()
    print(f"Starting V3 decomposition of {len(questions)} questions...", file=sys.stderr)
    print(f"API Key present: {bool(API_KEY)}", file=sys.stderr)
    print(f"Concurrency: {MAX_CONCURRENT} workers", file=sys.stderr)

    results: list = [None] * len(questions)
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
        # Submit all tasks
        futures = {
            executor.submit(decompose_question, idx, q['question']): idx
            for idx, q in enumerate(questions)
        }

        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            idx = result['id']
            results[idx] = result

            completed += 1
            if completed % 10 == 0:
                print(f"  [{completed:3d}/246] Processed...", file=sys.stderr)

    # Save results
    output_file = 'decomposition_results_v3.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved to {output_file}", file=sys.stderr)

    # Quick stats
    successful = sum(1 for r in results if r and 'result' in r)
    failed = sum(1 for r in results if r and 'error' in r)

    print(f"\nStats:", file=sys.stderr)
    print(f"  Successful: {successful}/{len(questions)}", file=sys.stderr)
    print(f"  Failed: {failed}/{len(questions)}", file=sys.stderr)

if __name__ == "__main__":
    main()
