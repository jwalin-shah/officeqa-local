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
uv run python extract.py --test-oracle --n 20  # test extraction with gold files
uv run python eval_decompose.py          # evaluate decompose phase output
```

**Local agent worktrees:**
```bash
./scripts/setup_agents.py                 # initialize agent infrastructure and task specs
./scripts/doctor_agents.py                # verify agent setup and diagnose issues
./scripts/setup_worktrees.sh              # create git worktrees in `.agent-worktrees/` for isolated agent stages
./scripts/link_worktree_artifacts.sh      # symlink ledger.sqlite and cached decompose to worktrees
./scripts/refresh_worktrees_from_main.sh  # refresh worktrees from latest main branch
DRY_RUN=1 ./scripts/agent_iterate.sh     # iterate agent rounds (use DRY_RUN=1 first)
OFFICEQA_WT_ROOT="$PWD/.agent-worktrees" ./scripts/cursor_agent_once.sh retrieval  # run Cursor Agent on a stage
./scripts/rebuild_ledger_background.sh   # rebuild ledger asynchronously in background
```

## Architecture
Pipeline: scout → decompose → retrieve → [_try_deterministic_fast_path] → extract → compute → verify.

**Scout phase** (deterministic): initial analysis of the question and corpus to understand scope and key signals.

**Decompose phase** (LLM): parses question into a `QuestionSpec` with:
- `period` — top-level signal ("CY" for calendar year, "FY" for fiscal year, or inferred from question)
- per-value `data_requests` — each with keywords, year hints, and optional `retrieval_hint` to guide channel selection
- `retrieval_hint` — optional guidance with `channel_preference` ("metric_exact" for exact row labels, "fts_keyword" for topical/fuzzy, "prose" for narrative context, "auto" to let retrieval decide), plus `must_match_phrases` and `avoid_phrases` for filtering
- `compute_template` — Python expression tree for final aggregation
- output format specification

**Retrieve phase** (deterministic): three-channel funnel with progressive synonym expansion over `ledger.sqlite`:
- **FTS channel** — FTS5 `tables_fts` over title/section/caption/columns/rows with year filtering. Good for topical/fuzzy questions.
- **Metric channel** — substring lookup over `metrics.metric_slug` (normalized row label). Good when question's noun phrase matches a row label cleanly ("national defense" → exact slug hit). Mirrors arena's master_ledger retrieval.
- **Prose/footnote channel** — supplementary context from `prose_fts` and `footnotes_fts` for ~6% of questions requiring non-table data. Weighted lower than direct table hits.
- **Progressive expansion** — if initial queries don't reach threshold, fallback stages try synonym/partial matches via staged thresholds before hitting top-N limit. Tracks which strategy retrieved each row via `ChannelTrace`. Respects `retrieval_hint` channel_preference when present.

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

**Core pipeline** (what `solve.py` calls, in order):
- `solve.py` — pipeline orchestrator: scout → decompose → retrieve → extract → compute → verify. Handles retry loops and LLM budget (MAX_LLM_CALLS=6). Supports `--eval [--n N] [--offset OFFSET] [--oracle] [--parallel P]`.
- `scout.py` — deterministic pre-decompose index peek; grounds decompose with real corpus signals.
- `decompose.py` — LLM call 1: question → `QuestionSpec` (period, data_requests with retrieval hints, compute template, output format). Contains `DECOMPOSE_SYSTEM` prompt and all decompose helpers.
- `retrieve_v2.py` — three-channel FTS + metric + prose/footnote funnel against `ledger.sqlite`. Progressive synonym expansion, period-aware reranking, strategy-weighted scoring, channel-preference routing from retrieval hints.
- `find.py` — deterministic ledger lookups: `resolve_cells()`, bottom-up cell search, `try_deterministic_fast_path()` (tries to skip LLM extraction entirely), vocabulary fetch for decompose.
- `extract.py` — LLM call 2: grounded per-request value extraction from retrieved tables. Pre-extraction helpers for monthly values and CY/FY row filtering. Quality retry, ledger cross-check, cohort filtering.
- `compute.py` — safe Python evaluation of the compute template. Dispatches named operations (sum, average, percent_change, linear_regression, …). Unit conversion utilities.
- `verify.py` — LLM call 3: post-compute answer check with deterministic auto-fixes for unit scaling and fiscal/calendar year errors.

**Data and scoring:**
- `build_ledger.py` — builds `ledger.sqlite` from `corpus_json/`. Run once (or after corpus changes).
- `ledger_paths.py` — resolves ledger path across environments.
- `external_data.py` — FX rate lookups (fawazahmed0 CDN + static fallback).
- `cpi.py` — BLS CPI-U data 1930-2026.
- `reward.py` — benchmark scoring logic.

**Evaluation and diagnostics:**
- `eval_decompose.py` — run decompose() on benchmark questions and inspect specs.
- `eval_decompose_oracle.py` — check whether decomposed spec points at gold data.
- `eval_retrieve.py` — measure retrieval recall against gold source files.
- `eval_ledger.py` — check ledger structural reachability of gold answers.
- `eval_attribution.py` — stage-attributed failure analysis (where does the pipeline break?).
- `eval_bottomup.py` — measure bottom-up cell search recall.
- `eval_direct_oracle.py` / `eval_extract_oracle.py` / `eval_hybrid_oracle.py` — oracle upper-bound evals for each stage.
- `eval_suite.py` — coordinated eval runner for commit-to-commit comparisons.
- `batch_test.py` — parallel benchmark runner (wraps solve.py).
- `compare_eval_runs.py` — diff two eval run JSONL files.
- `analyze_retrieval_run.py` — analyze a single retrieval eval run.
- `analyze_retrieve_misses.py` — bucket retrieval misses by failure mode.
- `deep_dive_audit.py` — cross-phase diagnostic audit.
- `validate_decompose.py` — structural sanity check on cached decompose specs.

**Bench/validation scripts (root level):**
- `test_solve.py` — smoke tests for pipeline entry points (no LLM calls).
- `test_recall_with_decompose.py` — retrieval recall with pre-cached decompose specs.
- `test_variants.py` — variant configuration testing.
- `sql_solve.py` — SQL-based solver variant for exploring pure query-language approaches to question decomposition and retrieval.

**Migrations (already applied, in `scripts/migrations/`):**
- `migrate_metrics_to_view.py`, `migrate_fill_column_years.py`, `migrate_propagate_year.py`

**Tests:**
- `tests/conftest.py` — pytest fixtures for test setup.
- `tests/test_build_ledger_row_year_propagate.py` — tests for ledger row year propagation logic.
- `tests/test_extract_quality_retry.py` — tests for extraction quality retry mechanisms.
- `tests/test_attribution_eval.py` — tests for answer attribution evaluation.
- `tests/test_retrieval_analysis.py` — tests for retrieval analysis diagnostics.
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
- `runs/` — eval run outputs directory (gitignored); stores JSONL and JSON artifacts from systematic evaluation and comparison runs.
- `.agent-worktrees/` — agent worktree directories created by setup_worktrees.sh (tracked in git; contains isolated stage code and artifacts).

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
