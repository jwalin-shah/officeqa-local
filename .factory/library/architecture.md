# Architecture

**What belongs here:** Architectural decisions, patterns, component relationships.

---

## Pipeline Flow
```
Question → scout() → decompose() → retrieve_for_spec() → [_try_deterministic_fast_path] → extract_structured() → execute() → verify_answer() → Answer
                                         ↑                        ↑                              ↑
                                    [bounce-back on empty]  [bounce-back on verify fail]  [fast-path: skip LLM if all DRs resolve]
```

## Component Responsibilities
- **solve.py** — orchestrator. Manages bounce-back loops (empty retrieve → re-decompose, verify fail → re-extract or re-decompose). Parallel eval via ThreadPoolExecutor. **Deterministic fast-path**: before LLM extraction, attempts resolve_cells() for each data_request. If ALL DRs resolve, skips LLM entirely; if ANY fails, falls back to normal LLM extraction.
- **scout.py** — deterministic. Calls retrieve_from_question() top-5 to ground decompose with real corpus labels.
- **retrieve_v2.py** — deterministic. Two-channel funnel: FTS (broad) + metric substring (precise). Union + rerank. No LLM.
- **extract.py** — LLM. Per-data-request context assembly. Renders HTML tables to pipe-delimited text. Returns structured JSON extractions.
- **compute.py** — deterministic. Executes python_template from QuestionSpec. Safe sandbox with restricted builtins.
- **verify.py** — LLM. Post-compute verification checklist. Returns {ok, issue, suggested_phase}.
- **find.py** — deterministic. resolve_cells() for direct cell lookup by row/col labels. **Wired into solve.py as deterministic fast-path** before LLM extraction.

## Ledger Schema (ledger.sqlite)
- `tables` — one row per source table (94K tables)
- `table_columns` — columns with parsed year/month (893K)
- `table_rows` — rows with row_path, indent level, metric_slug (2.9M)
- `cells` — raw parsed cells with flags (~19M after is_missing deletion)
- `metrics` — VIEW over cells/table_rows/table_columns/tables UNION ALL metrics_synthesized (~21.4M rows)
- `metrics_synthesized` — ~10K CY/FY synthetic total rows (is_synthesized=1)
- `row_label_lookup` / `col_label_lookup` — dedup'd lookup tables for metric channel
- `prose` / `footnotes` / `page_metadata` — non-table elements
- FTS: `tables_fts`, `prose_fts`, `footnotes_fts`

DB size: ~2.2GB (reduced from 7.8GB after metrics VIEW migration + is_missing cell deletion)

## Arena Reference Architecture
The arena's best systems used: deterministic ingestion → structured extraction via sub-agents → deterministic computation. Our pipeline follows this pattern. Key arena techniques ported:
- ✅ Vertical serialization (extract)
- ✅ Synonym expansion + multi-strategy search (retrieve)
- ✅ Deterministic fast-path cell resolution (extract → solve.py)
- ✅ Pre-extracted monthly values (extract) — pre-extracts 12 monthly values for CY sum questions
- ✅ CY row filtering (extract) — suppresses annual/FY rows from extraction context
- Mentor/review verification pattern (verify) — pending

## Deterministic Fast-Path Details
The fast-path in `_run_extract_and_compute()` attempts to resolve all data_requests using `resolve_cells()` from find.py before falling back to LLM extraction.

**Flow:**
1. For each DR, check if source is corpus (skip external/cpi/fx)
2. Get the best retrieved table entry → look up table_id from ledger
3. Build cell specs based on granularity (annual: 1 cell, monthly_all: 12 cells, multi_year_annual: N cells)
4. Call `resolve_cells(table_id, cells)` — does exact row_leaf/col_leaf matching
5. If ALL DRs resolve with non-None values → return extractions dict, skip LLM
6. If ANY DR has unresolved values → return None, fall back to LLM

**Key helpers in solve.py:**
- `_fp_conn()` — thread-local ledger connection for fast-path queries
- `_get_table_id_from_entry(entry)` — maps retrieve_v2 entry to ledger table ID
- `_build_cells_for_dr(dr, table_id)` — builds cell specs for resolve_cells()
- `_try_deterministic_fast_path(spec, per_dr_entries, verbose)` — main entry point

**Limitations:**
- Requires exact row_leaf/col_leaf match — fuzzy matches fall through to LLM
- External/CPI/FX source DRs automatically fail (can't resolve from ledger)
- Prose/footnote entries are skipped (no table to resolve from)
- Currently the fast-path is all-or-nothing: if any DR fails, all DRs go to LLM

## Monthly Pre-Extraction & CY Row Filtering

Two extraction improvements in `extract.py` that reduce LLM extraction errors for calendar-year sum questions:

**CY Row Filtering** (`filter_cy_rows()`): When `granularity='monthly_all'`, suppresses annual total rows (bare year like "1940") and fiscal-year summary rows ("Fiscal year 1940") from the rendered context. Prevents the LLM from picking the wrong total when it should sum 12 monthly values. Works on both pipe-delimited and vertical format.

**Monthly Pre-Extraction** (`pre_extract_monthly_values()`): When `granularity='monthly_all'` AND `computation='sum'`, parses the rendered table context for 12 monthly values and prepends an annotation:
```
PRE-EXTRACTED MONTHLY VALUES for CY YYYY: [v1, v2, ..., v12] (Count: 12 — sum for calendar year total)
```
Two parsing strategies: (1) vertical format with `(month N): value` annotations, (2) pipe format with months as rows. Row-hint matching ensures the correct category is extracted when multiple ROW entries exist.

**Integration in `extract_structured()`**: Both features are applied per-DR in the context-building loop. Row filtering happens first, then pre-extraction is attempted on the filtered context. The annotation is prepended to the context before sending to the LLM.
