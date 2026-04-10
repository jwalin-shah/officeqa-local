# User Testing

**What belongs here:** Testing surface, resource costs, validation approach.

---

## Validation Surface

This is a CLI-only data pipeline. No web UI, no server, no browser testing needed.

### Surface 1: CLI Eval Scripts (Deterministic)
- **Tool:** Direct command execution
- **Commands:**
  - `uv run python retrieve_v2.py --test` — retrieval recall metrics
  - `uv run python eval_ledger.py` — ledger structural reachability
  - `uv run pytest` — unit tests
- **Characteristics:** Fast (seconds to minutes), no LLM calls, deterministic, safe to run repeatedly
- **Resource cost:** Low — SQLite queries + Python computation only

### Surface 2: LLM Eval Scripts
- **Tool:** Direct command execution
- **Commands:**
  - `uv run python solve.py --eval --n N` — end-to-end pipeline eval
  - `uv run python extract.py --test-oracle --n N` — oracle extraction eval
- **Characteristics:** Slow (minutes), costs money (LLM API calls), non-deterministic at temperature>0
- **Resource cost:** Medium — LLM API calls + SQLite queries, bound by API rate limits

### Surface 3: Unit Tests
- **Tool:** `uv run pytest`
- **Characteristics:** Fast, deterministic, no external calls
- **Resource cost:** Negligible

## Validation Concurrency

**Machine:** 8GB RAM, 8 cores (Apple Silicon)
**Baseline utilization:** ~6GB used at baseline (OS + apps)
**Available headroom:** ~2GB * 0.7 = ~1.4GB usable

Given the resource constraints:
- **CLI eval scripts:** Max 1 concurrent (SQLite locks, shared DB)
- **LLM eval:** Max 1 concurrent (shared API client, thread safety)
- **Unit tests:** Max 1 concurrent (default pytest behavior)

**Max concurrent validators: 1** — resource-constrained machine, single SQLite DB, limited RAM headroom.

## Flow Validator Guidance: CLI

**Surface:** Direct command execution (no browser, no TUI)
**Tool:** Execute shell commands via the Execute tool directly — no skill invocation needed.

**Isolation rules:**
- All validators share the same `ledger.sqlite` — run sequentially, never concurrently
- Do NOT modify `ledger.sqlite`, `corpus_json/`, `corpus/`, or `officeqa_full.csv`
- Do NOT run `build_ledger.py` (modifies DB) — use read-only queries only
- Working directory: `/Users/jwalinshah/projects/officeqa-local`
- All commands run with `uv run python ...` or `uv run pytest ...`

**Evidence collection:**
- Capture command stdout/stderr as evidence
- For each assertion, record the specific command output that proves pass/fail

**Common patterns:**
- DB checks: `uv run python -c "import sqlite3; ..."`
- File checks: `ls -lh ledger.sqlite`, `cat baseline_metrics.json`
- Test runs: `uv run pytest --tb=short -v`
- Retrieval eval: `uv run python retrieve_v2.py --test`
- Ledger eval: `uv run python eval_ledger.py`

## Testing Strategy

1. After each code change: run `uv run pytest` and `uv run python retrieve_v2.py --test` (deterministic, fast)
2. After milestone completion: run `uv run python solve.py --eval --n 20` (sample LLM eval)
3. Final milestone 4: run full `uv run python solve.py --eval` (246 questions)
