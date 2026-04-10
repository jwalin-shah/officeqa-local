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

## Testing Strategy

1. After each code change: run `uv run pytest` and `uv run python retrieve_v2.py --test` (deterministic, fast)
2. After milestone completion: run `uv run python solve.py --eval --n 20` (sample LLM eval)
3. Final milestone 4: run full `uv run python solve.py --eval` (246 questions)
