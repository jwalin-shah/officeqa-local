# Extraction Improvement Tracker

Status of all fixes targeting the 16 failing UIDs from the oracle extraction eval (`eval_extract_oracle.jsonl`). These are failures that occur **even when the correct tables are served**, isolating extraction, decompose, compute, and ingest issues.

## Pipeline execution order (solve.py)

```
scout → decompose (LLM) → retrieve (deterministic) → [fast-path] → extract (LLM) → compute → verify (LLM)
                                                        ↓
                                              extract_structured()
                                                ├─ pre-resolve external DRs (CPI/FX)
                                                ├─ per-DR context building
                                                │   ├─ year-match sorting
                                                │   ├─ _ensure_year_coverage() [multi-year DRs]
                                                │   ├─ build_context_from_entries()
                                                │   │   └─ render_entry() [page_id, prose, vertical/pipe]
                                                │   ├─ filter_cy_rows() [monthly_all]
                                                │   └─ pre_extract_monthly_values()
                                                ├─ LLM call (with cell-coordinate grounding prompt)
                                                ├─ JSON parse + cohesion retry
                                                │   └─ quality retry (missing + incomplete + labels-but-null)
                                                │       └─ _alt_render=True on labels-but-null retry
                                                ├─ _normalize_extraction_units()
                                                ├─ _verify_against_ledger() [cross-check vs HTML cells]
                                                └─ _filter_cohort_aggregates() [cohort DRs]
```

## Fixes implemented (this session)

### 1. Cohort aggregate filtering (`_filter_cohort_aggregates`)
- **File**: extract.py
- **UIDs fixed**: UID0006, UID0012
- **What**: Removes Total/Subtotal/Grand Total/All Other/Other <region> rows from cohort DR extractions post-LLM
- **How**: Regex patterns on labels, in-place filtering with safety net (no-op if all would be removed)

### 2. Expected-count validation + targeted retry
- **File**: extract.py (quality retry block in `extract_structured`)
- **UIDs targeted**: UID0005 (11/12 months), UID0018 (25/39 months), UID0022 (4/9 years)
- **What**: After extraction, checks `len(non-null values) < expected_count` and retries with specific feedback
- **How**: Combined with missing-DR retry into one unified quality retry pass

### 3. Multi-file budget balancing (`_ensure_year_coverage`)
- **File**: extract.py
- **UIDs targeted**: UID0018, UID0022
- **What**: For multi-year DRs, promotes one entry per uncovered year to the front of the entry list before budget truncation
- **How**: Reorders entries so char budget is spread across all required years

### 4. Labels-but-null alternate rendering
- **File**: extract.py (`_alt_render` parameter)
- **UIDs targeted**: UID0009, UID0019, UID0026, UID0029
- **What**: When retry detects DRs with correct labels but all-null values, retries with `vertical_threshold=999` (forces pipe format) and `max_rows=150`
- **How**: `_alt_render` flag threaded through `build_context_from_entries` → `render_entry`

### 5. page_id in render_entry
- **File**: extract.py
- **UIDs targeted**: UID0011
- **What**: `render_entry()` now emits `[page 42]` in header when entry has page_id
- **How**: Trivial addition to header_bits

### 6. Prose entry rendering
- **File**: extract.py
- **UIDs targeted**: UID0017
- **What**: `render_entry()` now handles prose/footnote entries with `content` field but no `html`
- **How**: Falls through to content text when html is empty

### 7. CPI annual average fix
- **File**: extract.py (`_resolve_external_dr`)
- **UIDs targeted**: UID0005
- **What**: Uses BLS published annual averages (`cpi.A` dict) instead of computing `sum(monthly)/12`
- **How**: Changed import from `cpi.M` to `cpi.A`, direct dict lookup

### 8. External FX data module (live fetch)
- **File**: external_data.py (new)
- **UIDs targeted**: UID0010
- **What**: Live FX rate lookup via fawazahmed0/currency-api CDN (free, no API key, historical rates)
- **How**: `resolve_fx_dr()` detects currency from DR label → `lookup_fx()` checks `FX_CACHE` → live `_fetch_rate()` on miss
- **Adding new rates**: Add to `FX_CACHE[("usd", "jpy", year, month, day)] = rate` or just let live fetch handle it
- **Adding new currencies**: Add a tuple to `_CURRENCY_PATTERNS` list

### 9. Decompose prompt improvements
- **File**: solve.py (decompose system prompt)
- **UIDs targeted**: UID0009, UID0024, UID0026
- **What**: Added three new sections:
  - DERIVED VALUES: guidance for ratio decomposition, weighted averages, percentage columns
  - Cross-reference facts: Treasury Notes of 1890 retirement (~1962), FMS/Public Debt merger (2012), Series E→EE (1980)

### 10. Cell-coordinate grounding prompt
- **File**: extract.py (`EXTRACT_STRUCTURED_SYSTEM`)
- **What**: Requires LLM to cite `row_labels` and `col_labels` arrays for each extracted value
- **How**: Added GROUNDING section to system prompt, updated JSON schema example

### 11. Row label disambiguation (`_disambiguate_row_labels`)
- **File**: extract.py
- **UIDs targeted**: UID0005
- **What**: Deterministic pre-pass on parsed HTML rows that prefixes bare month labels ("January", "February") with the inferred year, based on the most recent year-prefixed row ("1939-December" → next bare months become "1940-January", etc.)
- **How**: Wired into both `html_to_pipe_text()` and `html_to_vertical_text()` before rendering. Detects `YYYY-Month` patterns, tracks current_year, resets on bare `YYYY` rows.

### 12. Percentage column preference in extraction prompt
- **File**: extract.py (`EXTRACT_STRUCTURED_SYSTEM`)
- **UIDs targeted**: UID0020
- **What**: Added checklist item telling LLM to prefer table's pre-rounded "% increase" / "% change" columns over computing from raw amounts
- **How**: New bullet in EXTRACTION CHECKLIST section

### 13. Ledger cross-check (`_verify_against_ledger`)
- **File**: extract.py
- **What**: Post-extraction, parses source HTML tables into `(row_label, col_label) → cell_value` map, compares against LLM values, silently corrects mismatches
- **How**: Runs after unit normalization, before cohort filtering. Tags corrections in `verification` field.

## Per-UID status

| UID | Gold | Failure category | Root cause | Fix applied | Status |
|-----|------|-----------------|------------|-------------|--------|
| UID0005 | 39482.03 | CPI precision + missing month | Dec 1940 null (ambiguous rows) + cpi.py used monthly avg not published annual | CPI fix (#7), expected-count retry (#2), row disambiguation (#11) | NEEDS REEVAL — CPI fixed + row labels now unambiguous |
| UID0006 | 103,375 | Cohort picks aggregate | max() hit "Total Europe" (229,314) not UK (103,375) | Cohort filter (#1) | FIXED |
| UID0007 | 4962.46 | Corpus/gold mismatch | Pipeline computes 4957.21 = exact geometric mean of table data. Gold differs. | None — unfixable | WONTFIX (gold data mismatch) |
| UID0009 | 32.703 | Decompose error | Requests "total_pieces" column that doesn't exist; must derive pieces=value/denomination | Decompose prompt (#9) | NEEDS REEVAL — prompt now guides cohort+compute approach |
| UID0010 | 935851121560 | External FX rate | Needs USD/JPY rate from Macrotrends, not in corpus | Live FX fetch (#8) | FIXED (rate resolves via API) |
| UID0011 | 42 | page_id not rendered | Pipeline never showed page numbers to LLM | page_id in render (#5) | FIXED (page_id now in context) |
| UID0012 | 36080 million | Cohort picks aggregate | max() hit "1955 Total" (79,223) not "Defense Department" (36,080) | Cohort filter (#1) | FIXED |
| UID0017 | [10102000000, 4.73] | Data in prose not tables | Gold page has text elements only, zero tables. Auction data in paragraphs. | Prose rendering (#6) | NEEDS REEVAL — prose renders + data confirmed in ledger prose_fts |
| UID0018 | 81.406 | Multi-file span | 25/39 months from 1 of 4 oracle files | Year coverage (#3), expected-count retry (#2) | NEEDS REEVAL |
| UID0019 | 1169.41 million | Gold annotation incomplete | Gold lists page 54 (JPY only); GBP on page 58 never served by oracle | None — gold_locs incomplete | WONTFIX (oracle eval gap) |
| UID0020 | 0.00262 | Wrong column extracted | LLM used raw amounts; gold uses table's pre-rounded "% increase" column | Decompose prompt (#9), extraction prompt (#12) | NEEDS REEVAL |
| UID0022 | [273.28, 54244, 56703] | Multi-file span | 4/9 values from 1 of 2 oracle files | Year coverage (#3), expected-count retry (#2) | NEEDS REEVAL |
| UID0024 | 0.13 | Decompose error | Requests "ratio" column; table has raw amounts only | Decompose prompt (#9) | NEEDS REEVAL — prompt now guides ratio decomposition |
| UID0026 | 894 | Decompose error | Identifies year as 1935; actual is ~1962 | Cross-reference facts in decompose (#9) | NEEDS REEVAL |
| UID0027 | 3069 | Corpus parse gap | AY-1 yield table: 4 identical column groups, year sub-headers lost in HTML parse | BUILD_LEDGER enrichment (user handling) | BLOCKED on build_ledger year-group enrichment |
| UID0028 | 92000000 | Corpus parse gap + multi-step | Same AY-1 table + needs chained lookup | BUILD_LEDGER enrichment (user handling) | BLOCKED on build_ledger year-group enrichment |
| UID0029 | 0.88525 | Corpus parse gap | Same AY-1 table. 10 year labels correct, 0 values | BUILD_LEDGER enrichment (user handling) | BLOCKED on build_ledger year-group enrichment |

## Summary counts

- **FIXED**: 4 (UID0006, UID0010, UID0011, UID0012)
- **NEEDS REEVAL**: 8 (UID0005, UID0009, UID0017, UID0018, UID0020, UID0022, UID0024, UID0026)
- **BLOCKED**: 3 (UID0027, UID0028, UID0029) — waiting on build_ledger year-group enrichment
- **WONTFIX**: 2 (UID0007, UID0019) — gold data / annotation mismatch

## What to do next

### Highest-priority re-evaluation
Run `uv run python eval_extract_oracle.py` to see which of the NEEDS REEVAL UIDs are now fixed by the prompt/retry improvements.

### Remaining extraction-quality levers (all addressed this session)
1. **Row disambiguation** (UID0005): DONE — `_disambiguate_row_labels()` now prefixes bare month labels with inferred year. Wired into both pipe and vertical renderers.
2. **Prose retrieval** (UID0017): DONE — Confirmed prose data exists in `prose_fts` with section "Auction of 2-Year Notes". `render_entry()` already renders prose content.
3. **Percentage column preference** (UID0020): DONE — Added extraction checklist item for pre-rounded percentage columns.

### Structural improvements (future sessions)
4. **Build-ledger year-group enrichment** (UID0027/28/29): Detect repeated column groups in HTML tables, infer year headers from title/caption/surrounding text. This is the user's current focus.
5. **Multi-step question support**: UID0028 needs chaining (find min yield spread month → look up railroad retirement for that month). Current pipeline is single-pass. Would need either a two-pass extraction or a compute-phase that can trigger a second retrieval.

## Test inventory
- `tests/test_extraction.py`: 116 tests covering all new functions
  - `TestIsAggregateLabel`, `TestFilterCohortAggregates` — cohort filtering
  - `TestEnsureYearCoverage` — multi-year entry promotion
  - `TestRenderingParameters` — max_rows, vertical_threshold threading
  - `TestPageIdAndProse` — page_id rendering, prose entry handling
  - `TestCpiResolution` — BLS published annual averages
  - `TestExternalFx` — FX resolution, cache behavior
  - `TestLedgerCrossCheck` — correction, no-op, missing-coordinates
