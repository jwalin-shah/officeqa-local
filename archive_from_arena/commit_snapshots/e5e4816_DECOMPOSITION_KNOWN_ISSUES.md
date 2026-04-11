# Decomposition Known Issues Guide

**For use by Search/Retrieval/Solving layer**

This document describes known decomposition errors that downstream systems should account for.

## Critical Issues to Handle

### 1. CY Annual Total Mismatch (16 cases)
**Problem**: Questions asking for Calendar Year totals are marked as `value_format: annual_total`, but Treasury Bulletins don't provide pre-calculated CY annual totals.

**Affected Question IDs**: [19, 44, 60, 63, 66, ...]

**Detection Rule**:
```
if decomposition.period_type == "calendar"
   and "total" in question
   and decomposition.value_format == "annual_total"
   and decomposition.computation not in special_cases
   → OVERRIDE: search for monthly_series and sum 12 months
```

**Action**: When search finds this pattern, automatically search for monthly values (Jan-Dec) instead of looking for a pre-calculated annual total.

**Example**:
- Question: "What was total expenditures in calendar year 1940?"
- Decomposition says: `value_format: annual_total`
- Correct action: Search for 12 individual monthly values and sum them

---

### 2. Sum/Total Computation Mismatch (66 cases)
**Problem**: Questions explicitly asking to "sum" or "total" values are sometimes marked as `computation: direct` instead of `computation: sum`.

**Affected Question IDs**: [3, 8, 14, 23, 31, ...]

**Detection Rule**:
```
if "total" in question or "sum" in question
   and decomposition.computation in ["direct", "average", "other_single_value"]
   and NOT (regression, VaR, smoothing, polynomial, or other complex computation)
   → Check if solver should sum multiple components instead of extracting single value
```

**Action**: When extracting data, verify if the extracted value is a pre-calculated total, or if you need to sum component rows (e.g., sum of different departments, sum of all months).

**Example**:
- Question: "What was the total expenditures across all departments?"
- Decomposition says: `computation: direct`
- Correct action: Don't just extract "total row" — sum all individual department rows to verify/calculate total

---

### 3. Period Type Mismatch (7 cases)
**Problem**: Questions containing both "FY" and year numbers are sometimes marked as `period_type: calendar` when they should be `period_type: fiscal`.

**Affected Question IDs**: [7, 51, 140, 167, 194, ...]

**Detection Rule**:
```
if "FY" in question or "fiscal year" in question
   and decomposition.period_type == "calendar"
   → Likely mismatch, check both fiscal and calendar data
```

**Action**: If marked as calendar but question explicitly mentions "FY", search both fiscal year and calendar year sections. Fiscal year is typically the correct interpretation when both are mentioned.

**Example**:
- Question: "What was the total in FY 1991?"
- Decomposition says: `period_type: calendar`
- Correct action: Search fiscal year 1991 data primarily, calendar year 1991 as secondary

---

## How to Implement in Solver Prompts

Add this guidance to your search/extraction prompts:

```
KNOWN DECOMPOSITION ISSUES:
If you encounter these patterns, apply corrections:

1. CY totals marked as annual_total → Search for monthly values and sum
2. "Total" questions marked as direct → Verify by summing components
3. FY questions marked as calendar → Search fiscal year data primarily

These account for ~17% error rate in decompositions. Use them as hints
when initial extraction seems wrong.
```

---

## Impact Assessment

| Issue | Count | Severity | Impact on Accuracy |
|-------|-------|----------|-------------------|
| CY Annual Total | 16 | HIGH | Will fail extraction if ignored |
| Sum/Total Computation | 66 | MEDIUM | May extract wrong value |
| Period Type | 7 | MEDIUM | May search wrong section |

**Total affected**: ~89 questions (36% of dataset)
**Total unaffected**: ~157 questions (64% of dataset)

---

## Testing Recommendation

1. Run solver with and without corrections
2. Measure improvement in accuracy for affected question IDs
3. Use this to quantify decomposition error impact on final scores
4. Return to decomposition refinement if impact is >5% of total score
