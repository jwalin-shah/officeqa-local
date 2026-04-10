# Architecture

**What belongs here:** Architectural decisions, patterns, component relationships.

---

## Pipeline Flow
```
Question → scout() → decompose() → retrieve_for_spec() → extract_structured() → execute() → verify_answer() → Answer
                                         ↑                        ↑
                                    [bounce-back on empty]  [bounce-back on verify fail]
```

## Component Responsibilities
- **solve.py** — orchestrator. Manages bounce-back loops (empty retrieve → re-decompose, verify fail → re-extract or re-decompose). Parallel eval via ThreadPoolExecutor.
- **scout.py** — deterministic. Calls retrieve_from_question() top-5 to ground decompose with real corpus labels.
- **retrieve_v2.py** — deterministic. Two-channel funnel: FTS (broad) + metric substring (precise). Union + rerank. No LLM.
- **extract.py** — LLM. Per-data-request context assembly. Renders HTML tables to pipe-delimited text. Returns structured JSON extractions.
- **compute.py** — deterministic. Executes python_template from QuestionSpec. Safe sandbox with restricted builtins.
- **verify.py** — LLM. Post-compute verification checklist. Returns {ok, issue, suggested_phase}.
- **find.py** — deterministic. resolve_cells() for direct cell lookup by row/col labels. Not wired into main pipeline yet.

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
The arena's best systems used: deterministic ingestion → structured extraction via sub-agents → deterministic computation. Our pipeline follows this pattern. Key arena techniques to port:
- Vertical serialization (extract)
- Synonym expansion + multi-strategy search (retrieve)
- Deterministic fast-path cell resolution (extract)
- Mentor/review verification pattern (verify)
- Pre-extracted monthly values (extract)
