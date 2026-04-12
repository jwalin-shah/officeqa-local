# officeqa-local

Answer U.S. Treasury Bulletin questions from a 697-file text corpus, targeting >75% accuracy on the 246-question OfficeQA benchmark.

## Shared Agent Rules

All coding agents on this repo should read [AGENTS.md](AGENTS.md) first.

Traversal defaults:
- use `llm-tldr` for structure, symbol search, call graphs, and context gathering
- use `rtk` for compact file reads, trees, grep, diff, pytest, and error output
- fall back to raw `rg`, `sed`, `git`, `sqlite3`, or Python only when exact output is required

## Run it
```bash
uv run python solve.py "What were the total expenditures for national defense in 1940?"
uv run python solve.py --eval --n 10     # evaluate first 10 questions
uv run python solve.py --eval --n 10 --offset 50  # evaluate questions 50-59 (resume/parallel)
uv run python solve.py --eval --oracle   # evaluate with gold source files
uv run python solve.py --eval            # full 246-question benchmark
uv run python batch_test.py               # batch evaluation with detailed metrics
```

## Test it
```bash
uv run pytest                            # unit tests (tests/ directory)
uv run python test_solve.py              # validation script
uv run python build_ledger.py            # build ledger tables from corpus for offline reference
uv run python retrieve.py --test-phase-a # test retrieval with pre-cached decompose
uv run python extract.py --test-oracle --n 20  # test extraction with gold files
uv run python eval_decompose.py          # evaluate decompose phase output
```

**Local agent worktrees:**
```bash
./scripts/setup_agents.py                 # initialize agent infrastructure and task specs
./scripts/doctor_agents.py                # verify agent setup and diagnose issues
./scripts/setup_worktrees.sh              # create git worktrees for siloed agent stages
./scripts/link_worktree_artifacts.sh      # link ledger.sqlite and cached decompose to worktrees
./scripts/refresh_worktrees_from_main.sh  # refresh worktrees from latest main branch
DRY_RUN=1 ./scripts/agent_iterate.sh     # iterate agent rounds (use DRY_RUN=1 first)
OFFICEQA_WT_ROOT="$PWD/.agent-worktrees" ./scripts/cursor_agent_once.sh retrieval  # run Cursor Agent on a stage
./scripts/rebuild_ledger_background.sh   # rebuild ledger asynchronously in background
```

## Architecture
Pipeline: scout → decompose → retrieve → [_try_deterministic_fast_path] → extract → compute → verify.

**Scout phase** (deterministic): initial analysis of the question and corpus to understand scope and key signals.

**Decompose phase** (LLM): parses question into a `QuestionSpec` with per-value `data_requests` (keywords, year hints) and a `compute_template` (Python expression tree for final aggregation).

**Retrieve phase** (deterministic): three-channel funnel with progressive synonym expansion over `ledger.sqlite`:
- **FTS channel** — FTS5 `tables_fts` over title/section/caption/columns/rows with year filtering. Good for topical/fuzzy questions.
- **Metric channel** — substring lookup over `metrics.metric_slug` (normalized row label). Good when question's noun phrase matches a row label cleanly ("national defense" → exact slug hit). Mirrors arena's master_ledger retrieval.
- **Prose/footnote channel** — supplementary context from `prose_fts` and `footnotes_fts` for ~6% of questions requiring non-table data. Weighted lower than direct table hits.
- **Progressive expansion** — if initial queries don't reach threshold, fallback stages try synonym/partial matches via staged thresholds before hitting top-N limit. Tracks which strategy retrieved each row via `ChannelTrace`.

Each channel returns top-N candidates; union is reranked by file_year proximity and strategy bonuses. Reranking also applies **period-aware scoring**: tables matching the decomposed year_mode (calendar vs fiscal) receive +0.10 boost; non-matching receive -0.06 penalty (scaled by 0.2× to act as a tiebreaker rather than dominating FTS+metric scores). Multi-request hint extraction processes all `data_requests` from decompose (not just the first) to drive parallel retrieval strategies. Different retrieval strategies (primary_exact, synonym_exact, partial_match, exact_year_filter, exact_year_window, synonym_year_window, synonym_unrestricted, year_shifted, exact_unrestricted) receive weighted bonuses to boost matches from higher-confidence retrieval paths. Prose/footnote entries are supplementary (up to ~PF_MAX_SLOTS slots beyond the table top_k limit) and returned alongside table entries. All channels backed by the same lossless cell store:
- **tables** — source tables with signature hash for deduplication
- **table_columns** — parsed headers with year/month annotations
- **table_rows** — row paths with indent levels for hierarchy
- **cells** — normalized cell values with flags (missing, zero, footnote, revised, preliminary)
- **prose** — prose passages with FTS index for supplementary context
- **footnotes** — footnote text with FTS index for reference data
- **metrics** (VIEW) — union of cells and metrics_synthesized, covering all cell data and synthetic computed totals
- **metrics_synthesized** — ~10K CY/FY synthetic total rows (computed aggregates, is_synthesized=1)
- **facts** (view) — cells joined to context with resolved data_year
- **canonical_facts** (view) — deduped facts by (signature, row_path, col_path, data_year), keeping latest-published
- **supersessions** (view) — facts that were superseded by newer data, for audit

Ledger optimizations: **metrics converted to VIEW** (7.8GB → 2.2GB reduction), is_missing cells deleted, metric_slug added to table_rows for retrieval efficiency. Old `corpus_index.pkl + BM25 + sentence-transformer` retriever retired. Includes thread-safe caching with bounded retry loops (MAX_LLM_CALLS=6).

See [Local ledger state](project_ledger_state.md) for current retrieval schema.

**Deterministic fast-path** (optional): before LLM extraction, attempts `resolve_cells()` for each `data_request`. If ALL data requests resolve directly from the ledger, skips LLM extraction entirely. Falls back to normal LLM extraction if any request cannot be resolved.

**Extract phase** (LLM): grounded per-request value extraction from retrieved context. Includes pre-extraction optimization for monthly values via `pre_extract_monthly_values()` and CY row filtering via `filter_cy_rows()` to disambiguate calendar vs fiscal year rows before LLM processing. Outputs structured values.

**Compute phase** (Python): enum dispatcher with operation name aliases normalizes LLM variation ("mean" → "average", "std_dev" → "stdev_sample", "linreg" → "linear_regression", etc.). Dispatches known statistical operations explicitly: `average`, `count`, `sum`, `min`, `max`, `stdev_sample`, `stdev_pop`, `median`, `percent_change`, `percent_of`, `ratio`, `difference`, `pearson_correlation`, `linear_regression`, `coefficient_of_variation`, `geometric_mean`, `harmonic_mean`, `gini`, `theil_index`, `kl_divergence`. Falls back to safe template evaluation (restricted builtins, explicit imports) for custom expressions. Includes unit conversion utilities (`parse_unit()`, `convert_unit()`) to handle table header units (thousands, millions, billions, percent) and apply scaling when extracted values use a different unit than the template expects. Conversion formula: `converted = value × (source_multiplier / target_multiplier)`.

**Verify phase** (optional post-compute): deterministic auto-fixes for extracted unit scaling (thousands/millions/billions/percent) and fiscal/calendar year disambiguation based on table context. Improves reliability on edge cases without additional LLM calls.

**Primary corpus is `corpus_json/`** — 697 parsed bulletin JSONs with typed elements
(`title`, `section_header`, `table`, `footnote`, `text`, ...). Tables are stored as HTML,
so column headers and row labels fall out of an HTML parse. Originals were copied from
`~/archive/officeqa/treasury_bulletins_parsed/jsons/`. The `corpus/` `.txt` files are
legacy fallback; prefer the JSON corpus for all new code.

### File reference
- `solve.py` — entry point (orchestrates scout → decompose → retrieve → extract → compute → verify pipeline via OpenAI client with DeepSeek). Implements deterministic fast-path: before LLM extraction, attempts to resolve all data_requests directly via `resolve_cells()` from find.py; if successful, skips LLM entirely. Supports `--eval [--n N] [--offset OFFSET] [--oracle] [--parallel P]` for batch evaluation.
- `scout.py` — initial analysis/scouting of the question and corpus.
- `build_ledger.py` — builds ledger.sqlite from corpus_json/ with cell normalization, deduplication, and views for fact retrieval. Table header parsing uses only contiguous header rows from the top (stops at first non-header row) to avoid joining sub-section headers into column paths. Mid-table `<th>` rows (embedded section headers) are kept as data rows, not silently dropped. Handles Type-B rolling-series tables (e.g., "March 1979 through February 1980") via title date-range parsing to infer year_extracted for month-only columns.
- `ledger_paths.py` — utility for managing ledger database file paths across different environments and configurations.
- `migrate_metrics_to_view.py` — utility for migrating ledger data.
- `retrieve.py` — retrieval interface.
- `retrieve_v2.py` — table-level retrieval implementation (three-channel FTS + metric + prose/footnote funnel with progressive synonym expansion, period-aware reranking scaled to gentle tiebreaker strength, and strategy-weighted ranking). Includes thread-safe caching with bounded retry loops (MAX_LLM_CALLS=6).
- `extract.py` — structured per-request value extraction via LLM, with pre-extraction helpers for monthly values (`pre_extract_monthly_values()`) and CY row filtering (`filter_cy_rows()`). Post-extraction: unit normalization, ledger cross-check (`_verify_against_ledger`), cohort aggregate filtering (`_filter_cohort_aggregates`). Quality retry for missing/incomplete/labels-but-null DRs with expected-count validation and alternate rendering. Multi-year budget balancing via `_ensure_year_coverage()`. Renders prose/footnote entries alongside tables.
- `external_data.py` — external data lookups (FX rates via fawazahmed0 CDN API with static fallback cache). Used by `_resolve_external_dr()` for `source: "fx"` data_requests.
- `compute.py` — enum dispatcher with operation name alias normalization to handle LLM variation, dispatches known statistical operations (average, count, sum, min, max, stdev_sample, stdev_pop, median, percent_change, percent_of, ratio, difference, pearson_correlation, linear_regression, coefficient_of_variation, geometric_mean, harmonic_mean, gini, theil_index, kl_divergence), plus safe Python evaluation (restricted builtins, explicit imports) for custom expressions. Includes unit conversion utilities for handling table header units (thousands, millions, billions, percent) and normalizing extracted values to the target unit.
- `verify.py` — post-compute answer verification with deterministic auto-fixes for units and fiscal/calendar year disambiguation.
- `reward.py` — benchmark scoring.
- `find.py` — deterministic cell lookup by row/col labels; implements `resolve_cells()` for direct ledger queries. Wired into solve.py as the deterministic fast-path before LLM extraction.
- `deep_dive_audit.py` — diagnostic audit tool for analyzing system state and performance across the pipeline.
- `test_solve.py` — validation script.
- `test_recall_with_decompose.py` — test retrieval recall with pre-cached decompose output.
- `eval_decompose.py` — evaluation utilities for decompose phase output.
- `eval_retrieve.py` — evaluation utilities for retrieval phase output.
- `validate_decompose.py` — validation script for decompose phase output.
- `eval_ledger.py` — evaluation utilities for ledger-based reference.
- `batch_test.py` — batch evaluation runner with detailed metrics.
- `build_index.py` — legacy corpus indexing (replaced by ledger approach).
- `cpi.py` — CPI-U data 1930-2026.
- `tests/conftest.py` — pytest fixtures for test setup.
- `tests/test_build_ledger_row_year_propagate.py` — tests for ledger row year propagation logic.
- `tests/test_extract_quality_retry.py` — tests for extraction quality retry mechanisms.
- `scripts/setup_agents.py` — initialize agent infrastructure and standardize task specs.
- `scripts/doctor_agents.py` — diagnostic tool to verify agent configuration and troubleshoot setup issues.
- `scripts/setup_worktrees.sh` — create git worktrees in `.agent-worktrees/` for isolated agent stages.
- `scripts/link_worktree_artifacts.sh` — symlink ledger.sqlite and decompose specs to worktrees.
- `scripts/refresh_worktrees_from_main.sh` — refresh existing worktrees with latest code from main branch.
- `scripts/agent_iterate.sh` — drive iterative agent rounds across worktrees (supports DRY_RUN=1).
- `scripts/agent_task_specs/` — task specifications (decompose.txt, eval.txt, extract.txt, fastpath.txt, ingest.txt, retrieval.txt) for siloed agent stages; used by agent invocation harness.
- `scripts/cursor_agent_once.sh` — invoke Cursor Agent on a single stage.
- `scripts/rebuild_ledger_background.sh` — background script for rebuilding ledger.sqlite asynchronously without blocking main workflow.
- `scripts/_invoke_agent.py` — agent invocation helper.
- `AGENTS.md` — agent architecture and rules for coordinating LLM agents across siloed stages.
- `corpus_json/` — 697 parsed bulletin JSONs (primary corpus, gitignored).
- `corpus/` — 697 `.txt` OCR fallback (legacy).
- `officeqa_full.csv` — 246 benchmark questions with gold answers and source files.
- `decompose_eval.full.jsonl` — cached decompose specs for all 246 questions; used offline for retrieval/extraction testing without live LLM calls.
- `ledger.sqlite` — SQLite database built from corpus_json/; stores normalized cells, tables, prose, footnotes, and derived views for retrieval.

### Archive
Historical arena code has been removed from the repo (was `archive_from_arena/`, removed 2026-04-11 as part of repo cleanup). If you need to reference it, `git log --all --full-history -- archive_from_arena/` will find the last commit that contained it.

## Key conventions
- LLM: OpenAI client library with DeepSeek via Dedalus (OpenAI-compatible endpoint). Model via
  `OFFICEQA_MODEL` env var (defaults to `deepseek/deepseek-chat`). API config in `.env`.
- **Always test retrieval without the live LLM decompose call.** Use `retrieve_from_question()`
  or `decompose_eval.full.jsonl` (pre-cached specs). Never put an LLM call in the inner loop of a benchmark
  sweep — it turns 5-second tests into 5-minute tests and mixes decompose noise into
  retrieval measurements.
- **Real-time per-question logging during tests.** Every test loop should print one
  line per question as it processes, not every-N batch updates. Use `flush=True` and
  `sys.stdout.reconfigure(line_buffering=True)` so output appears under background tasks.
- Keep prompts simple — arena proved simpler prompts beat complex multi-phase pipelines.
- Don't build structured table parsers for the `.txt` corpus — use `corpus_json/` instead.
- Stdlib tools only: `difflib.SequenceMatcher`, `html.parser`, `clean_value()`, `to_num()`.
- Safe Python evaluation in compute phase: restricted builtins, explicit imports.

## CI
GitHub Actions runs **Ruff** and **pytest** on pushes and PRs to `main`. Integration tests requiring `ledger.sqlite` are skipped in CI (marked with `pytest.mark.skipif(not os.path.exists(...))`). Run the full suite locally after building the ledger.

## Planning & documentation
Deeper architecture and workstream docs are in:
- `PLAN.md` — lossless ledger design and evolution strategy
- `docs/TARGET_ARCHITECTURE.md` — system design targets and constraints
- `docs/EXECUTION_ROADMAP.md` — implementation milestones and timeline
- `docs/WORKSTREAMS.md` — workstream ownership and dependencies
- `docs/INGESTION_AND_BENCHMARK_STRATEGY.md` — corpus integrity and benchmark approach
- `docs/AGENT_ORCHESTRATION.md` — agent worktrees, siloed-stage fakes, merge rules, and Cursor Agent wiring

## Lint / type check
```bash
uv run ruff check .
uv run pyright
```
