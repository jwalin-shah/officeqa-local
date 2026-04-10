# Extraction Scrutiny: Issue Analysis & Fix Recommendations

## Issue 1: Month Detection False Positives (`_detect_month_index`)

### Severity: **Real bug — HIGH**

The `_detect_month_index()` function at extract.py:175 has a prefix-matching fallback (line 187):

```python
for month_name, idx in _CALENDAR_MONTHS.items():
    if h.startswith(month_name):
        return idx
```

`_CALENDAR_MONTHS` includes short keys like `"mar"`, `"may"`, `"jun"`, `"jul"`, `"aug"`, `"oct"`, `"nov"`, `"dec"`. Any column header starting with these strings will be falsely labeled as a month. Concrete false-positive examples from Treasury Bulletin tables:
- `"Marketable securities"` → starts with `"mar"` → falsely labeled month 3
- `"May-dated"` or `"Margin"` → false matches
- `"October issue"` or `"Notes outstanding"` → `"oct"`, `"nov"` false matches
- `"August reductions"` or `"December bonds"` → `"aug"`, `"dec"` false matches

This directly corrupts the vertical serialization output: `html_to_vertical_text()` at line 270 calls `_annotate_month()` on every column header, which calls `_detect_month_index()`. A table with a "Marketable securities" column would render as `Marketable securities (month 3): 1234`, confusing the LLM.

### Fix Complexity: **Small — ~5-line change**

Replace the loose prefix match with a word-boundary check. The column header must **be** a month name (possibly with trailing punctuation or year), not merely start with one:

```python
def _detect_month_index(col_header: str) -> int | None:
    h = col_header.strip().lower()
    # Direct exact match
    if h in _CALENDAR_MONTHS:
        return _CALENDAR_MONTHS[h]
    # Check if header starts with a month name followed by a non-alpha char
    # (e.g., "Jan. 1940", "Jan 1940", "January") — NOT "Marketable"
    for month_name, idx in _CALENDAR_MONTHS.items():
        if h.startswith(month_name) and (len(h) == len(month_name) or not h[len(month_name)].isalpha()):
            return idx
    return None
```

The key addition is `not h[len(month_name)].isalpha()` — if the character immediately after the month prefix is a letter, it's part of a longer word (e.g., "mar" + "ketable") and should NOT match.

### Impact: **Medium-high** — Prevents incorrect month annotations on non-month columns in wide tables. Without this fix, any non-month column starting with a month abbreviation gets a wrong `(month N)` annotation, which could mislead the LLM during extraction. Affects vertical serialization (tables ≥8 columns).

---

## Issue 2: All-or-Nothing Fast-Path (per-DR fallback)

### Severity: **Real architectural gap — MEDIUM**

`_try_deterministic_fast_path()` in solve.py:175 returns `None` as soon as ANY data request fails to resolve (lines 199, 220, 227, 236, 253). The call site at solve.py:734 then falls through to full LLM extraction for ALL DRs — discarding any DRs that *were* successfully resolved deterministically.

### Fix Complexity: **Medium — ~30-line refactor**

The fix involves:

1. **Change `_try_deterministic_fast_path` to return partial results**: Instead of returning `None` on first failure, collect resolved DRs and return `(resolved_dict, unresolved_dr_ids)`.

2. **Modify the call site in `_run_extract_and_compute`**:
   - Call fast-path, get partial results + unresolved list
   - If all resolved → skip LLM (current happy path)
   - If some resolved, some not → call `extract_structured()` with only the unresolved DRs, then merge
   - If none resolved → full LLM fallback (current unhappy path)

3. **Filter DRs for LLM extraction**: `extract_structured()` already takes the full spec; we'd need to either filter `spec["data_requests"]` to only unresolved DRs, or pass a subset of `per_dr_entries`.

Specific changes:
- `_try_deterministic_fast_path()`: Instead of `return None` on each failure, do `continue` and track which DRs resolved vs. not. Return a tuple `(extractions_dict, unresolved_dr_ids)`.
- Call site (~solve.py:734): Branch on `len(unresolved_dr_ids) == 0` vs. partial vs. full fallback.

### Impact: **Low-medium for accuracy, important for correctness** — In practice, most questions have 1-2 DRs, and if DR #1 fails fast-path, the LLM handles it fine. The per-DR fallback is more important for multi-DR questions (e.g., "compare defense and education spending") where one DR resolves deterministically and the other doesn't. This is a correctness/efficiency concern more than an accuracy blocker.

---

## Issue 3: CY Row Filtering Scope

### Severity: **Theoretical concern — LOW**

`filter_cy_rows()` at extract.py:988 is gated on `granularity == "monthly_all"` (line 999, called at line 1260). The scrutiny concern is that fiscal-year questions with `monthly_all` granularity would also get CY filtering (removal of FY summary rows).

**Analysis of actual behavior**: `filter_cy_rows()` removes:
- Bare year labels: `"1940"` (matches `^\d{4}$`)
- FY labels: `"Fiscal year 1940"`, `"FY 1940"` (matches `^(Fiscal\s+year|FY|Fiscal)\s+\d{4}`)

For a fiscal-year monthly question (e.g., "What were the monthly receipts for FY 1940?"), the decompose would set `granularity="monthly_all"`. The filtering would remove summary rows labeled "Fiscal year 1940" or "1940" — which is actually **correct behavior** for monthly extraction regardless of CY vs FY. When extracting 12 monthly values, you always want to suppress the annual/FY summary row so the LLM picks the individual months, not the total.

The concern says "over-filtering fiscal-monthly requests by suppressing fiscal summary rows", but for `monthly_all`, suppressing summary rows is the intended behavior for *both* CY and FY monthly extraction. You want the 12 individual months, not the annual total — whether calendar or fiscal.

**The only scenario where this could be a problem**: If someone asked "What was the fiscal year 1940 total?" and the decompose incorrectly set `granularity="monthly_all"` instead of `"annual"`. But that's a decompose bug, not a filtering bug.

### Fix Complexity: **1-line (optional, for explicitness only)**

If desired for defensive correctness, add `period_type` check:
```python
if ctx and granularity == "monthly_all" and spec.get("period_type") != "fiscal":
    ctx = filter_cy_rows(ctx, granularity)
```

But this is arguably wrong — even for fiscal monthly, you still want to suppress summary rows. The current behavior is correct for the actual use case.

### Impact: **Negligible** — The current gating is functionally correct. Fixing would add code complexity for no accuracy gain. The scrutiny review's concern appears to be based on a misunderstanding of what `filter_cy_rows` actually removes.

---

## Issue 4: Oracle Eval n=50 Not Completed

### Severity: **Process gap — LOW**

`baseline_metrics.json` records `oracle_extraction.n = 20`. The validation contract requires n=50. This is just running a command and recording the result.

### Fix Complexity: **Trivial — run command + update JSON**

```bash
uv run python extract.py --test-oracle --n 50
```
Then update `baseline_metrics.json` with the result.

### Impact: **Zero on accuracy** — This is purely a validation process requirement. The code is identical whether n=20 or n=50; it just needs a larger sample for confidence in the metric.

---

## Priority Ranking

| Priority | Issue | Type | Fix Size | Accuracy Impact |
|----------|-------|------|----------|-----------------|
| **P1** | #1: Month detection false positives | Real bug | ~5 lines | Medium-high |
| **P2** | #2: Per-DR fast-path fallback | Architecture gap | ~30 lines | Low-medium |
| **P3** | #4: Oracle eval n=50 | Process gap | Run command | Zero |
| **P4** | #3: CY row filtering scope | Theoretical | 1 line (optional) | Negligible |

### Recommended approach:
1. **Fix #1 immediately** — single function change, prevents real data corruption in vertical serialization
2. **Fix #2 next** — moderate refactor but important for scrutiny pass; the all-or-nothing behavior is explicitly called out as blocking
3. **Run #4** — just execute the command and update metrics
4. **#3 can be closed as "by design"** — or add a 1-line guard with a comment explaining the rationale. The scrutiny review's concern doesn't hold up under analysis, but adding `period_type` context could make intent clearer.
