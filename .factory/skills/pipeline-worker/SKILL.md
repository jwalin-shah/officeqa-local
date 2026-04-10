---
name: pipeline-worker
description: Implements features for the OfficeQA pipeline (retrieval, extraction, compute, verification, ledger, tests)
---

# Pipeline Worker

NOTE: Startup and cleanup are handled by `worker-base`. This skill defines the WORK PROCEDURE.

## When to Use This Skill

Use for any feature that modifies the OfficeQA pipeline codebase:
- Retrieval improvements (retrieve_v2.py)
- Extraction improvements (extract.py, find.py)
- Compute/verification changes (compute.py, verify.py)
- Ledger/DB optimization (build_ledger.py)
- Test infrastructure (tests/)
- Eval harness changes (batch_test.py, eval scripts)
- Pipeline orchestration (solve.py)

## Work Procedure

### Step 1: Understand the Feature

1. Read the feature description, preconditions, expectedBehavior, and verificationSteps carefully
2. Read `AGENTS.md` for mission boundaries and conventions
3. Read `.factory/library/architecture.md` for pipeline component relationships
4. Identify which files need modification — check current state of each file before editing
5. If the feature references arena techniques, read the specific arena file mentioned (e.g., `/Users/jwalinshah/archive/officeqa-arena/archive/scripts/build_master_ledger_v2.py`)

### Step 2: Write Tests First (TDD)

1. Create or update test file(s) in `tests/` directory
2. Write failing tests that cover the expected behavior:
   - At least 3 test cases per behavioral requirement
   - Cover happy path, edge cases, and error handling
   - For retrieval: test with known question/answer pairs from officeqa_full.csv
   - For extraction: test with mock tables and expected outputs
   - For compute: test with known templates and values
3. Run `uv run pytest tests/test_<component>.py -v` to confirm tests FAIL (red phase)

### Step 3: Implement

1. Make the minimum changes needed to pass the tests
2. Follow existing code patterns and conventions:
   - Match the style of surrounding code
   - Use existing helper functions (`clean_value`, `to_num`, etc.)
   - Keep prompts simple (arena lesson: simpler beats complex)
3. For features porting arena techniques:
   - Reference the specific arena implementation
   - Adapt to our schema (cells/table_rows/table_columns, not master_ledger)
   - Preserve existing functionality — additions, not replacements
4. Run `uv run pytest` to confirm tests PASS (green phase)

### Step 4: Run Deterministic Eval

1. Run `uv run ruff check .` — fix any lint errors
2. Run `uv run pyright` — fix any type errors
3. Run `uv run pytest --tb=short -q` — all tests must pass
4. If the feature affects retrieval: run `uv run python retrieve_v2.py --test` and record recall@1/5/10
5. If the feature affects the ledger: run `uv run python eval_ledger.py` and record reachability
6. Compare against baseline (in `baseline_metrics.json` if it exists)
7. **Do NOT run LLM eval** unless the feature's verificationSteps explicitly require it

### Step 5: Manual Verification

1. For retrieval changes: pick 3 questions from officeqa_full.csv and manually trace the retrieval results to verify they include the gold source files
2. For extraction changes: run `uv run python solve.py "What were the total expenditures for national defense in 1940?"` and verify the answer is reasonable
3. For DB changes: run SQL queries to verify row counts and data integrity
4. Record what you checked and what you observed

### Step 6: Commit

1. Stage only the files you modified
2. Write a clear commit message describing what was changed and why
3. Include metric deltas if applicable (e.g., "recall@10: 45% → 52%")

## Example Handoff

```json
{
  "salientSummary": "Added synonym dictionary with 15 arena-proven term pairs to retrieve_v2.py and wired synonym expansion into both FTS and metric channels. Retrieval recall@10 improved from 45.1% to 51.6% (+6.5pp) on 246-question benchmark.",
  "whatWasImplemented": "SYNONYMS dict in retrieve_v2.py with 15 pairs (expenditures/outlays, receipts/revenue, defense/military, veterans/VA, social_security/social_insurance, deficit/shortfall, grants/grants-in-aid, debt/public_debt, interest/net_interest, medicare/health, surplus/excess, tax/taxation, medicaid/medical_assistance, spending/disbursements, obligations/liabilities). _expand_synonyms() applied in both _fts_channel (query token expansion) and _metric_channel (slug matching alternatives). Progressive: synonyms only fire when exact terms yield <3 candidates.",
  "whatWasLeftUndone": "",
  "verification": {
    "commandsRun": [
      {"command": "uv run pytest tests/test_retrieval.py -v", "exitCode": 0, "observation": "12 tests passed including 4 new synonym expansion tests"},
      {"command": "uv run ruff check .", "exitCode": 0, "observation": "no lint errors"},
      {"command": "uv run pyright", "exitCode": 0, "observation": "0 errors"},
      {"command": "uv run python retrieve_v2.py --test", "exitCode": 0, "observation": "recall@1=23.2% recall@5=41.8% recall@10=51.6% recall@20=62.4% (baseline was 45.1% @10)"}
    ],
    "interactiveChecks": [
      {"action": "Traced UID0003 (veterans expenditures 1942) through retrieval", "observed": "With synonyms, 'veterans administration' matches tables previously missed. Gold file now in top-5."},
      {"action": "Traced UID0015 (national defense outlays 1938) through retrieval", "observed": "'outlays' synonym expands to 'expenditures', finds correct Analysis of General Expenditures table at rank 2."},
      {"action": "Ran solve.py on 'What were the total expenditures for national defense in 1940?'", "observed": "Returns '2,602' which matches expected answer. Pipeline completed without errors."}
    ]
  },
  "tests": {
    "added": [
      {"file": "tests/test_retrieval.py", "cases": [
        {"name": "test_synonym_expansion_basic", "verifies": "expenditures expands to include outlays"},
        {"name": "test_synonym_expansion_bidirectional", "verifies": "outlays also expands to expenditures"},
        {"name": "test_synonyms_only_on_insufficient_results", "verifies": "synonyms don't fire when exact terms get >=3 hits"},
        {"name": "test_metric_channel_uses_synonyms", "verifies": "metric_slug search tries synonym alternatives"}
      ]}
    ]
  },
  "discoveredIssues": []
}
```

## When to Return to Orchestrator

- Feature depends on a schema change that requires rebuilding ledger.sqlite (>5 min, may OOM on 8GB)
- LLM API returns persistent errors (rate limit, auth failure)
- Pre-cached decompose specs (decompose_eval.full.jsonl) are missing or corrupted
- Feature requires modifying corpus_json/ or officeqa_full.csv (off-limits)
- Existing tests fail on the UNCHANGED codebase (pre-existing issue)
- Feature scope is significantly larger than described (>3 files to modify)
