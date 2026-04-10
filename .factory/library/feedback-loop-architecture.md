# Feedback Loop & Retry Architecture

## Constants
- `MAX_RETRIES_PER_PHASE = 2` — max retries per phase (decompose/extract/verify) in solve.py
- `MAX_LLM_CALLS = 6` — absolute cap on total LLM calls per question

## Bounce-back paths in solve.py
1. **Empty retrieve → re-decompose**: When retrieve returns 0 entries, decompose is re-called with feedback "zero matching tables". Budget checked via `_bump_llm("decompose(retry)")`.
2. **NO_VALUES → retry extract → re-decompose**: When extract returns empty values, first retry extract with feedback, then re-decompose if still empty. Budget checked before each retry.
3. **verify ok=False + suggested_phase='extract' → re-extract**: Re-runs `_run_extract_and_compute()` with verify's issue as feedback. No budget check at this level (extract_structured's LLM call is counted via `llm_counter`).
4. **verify ok=False + suggested_phase='decompose' → full re-run**: Re-runs decompose→retrieve→extract→compute. Budget checked via `_bump_llm("decompose(verify_retry)")`.

## LLM call counting
- `llm_calls = {"count": 0}` is a mutable dict created per `solve()` invocation
- `_bump_llm(phase)` increments counter and returns False when budget exhausted
- `_run_extract_and_compute()` accepts `llm_counter` param and increments it when `extract_structured()` is called (i.e. when deterministic fast-path doesn't cover all DRs)
- When budget is exhausted, solve returns the current best answer (no crash)

## Thread-safety for parallel eval
- `retrieve_v2._LEDGER_TLS` — thread-local sqlite3 connection (each thread gets its own)
- `solve._FP_TLS` — thread-local sqlite3 connection for deterministic fast-path
- `retrieve_v2._JSON_ELEMENT_CACHE_LOCK` — threading.Lock protecting shared HTML cache dict
- The `_JSON_ELEMENT_CACHE` dict is shared across threads but writes are protected by lock with double-check pattern
