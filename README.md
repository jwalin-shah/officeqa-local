# OfficeQA Local

Ledger-backed benchmark harness for answering U.S. Treasury Bulletin questions on the 246-question OfficeQA benchmark.

The active system is the structured pipeline in `solve.py`, backed by `ledger.sqlite` and `corpus_json/`. The older grep-oriented scripts are still present in the repo, but they are not the architectural center of the project anymore.

## Current Architecture

Active path:

```text
question
  -> scout
  -> decompose
  -> retrieve
  -> deterministic fast-path
  -> extract
  -> compute
  -> verify
  -> answer
```

Core modules:

- `solve.py` — current orchestrator
- `retrieve_v2.py` — canonical retrieval path over `ledger.sqlite`
- `find.py` — deterministic cell lookup / bottom-up search
- `extract.py` — grounded extraction
- `compute.py` — deterministic math / formatting
- `verify.py` — post-compute checks and auto-fixes
- `build_ledger.py` — ingestion into the ledger

## Setup

```bash
cp .env.example .env
source .venv/bin/activate
```

Primary runtime dependencies for full local runs:

- `corpus_json/`
- `ledger.sqlite`
- API credentials in `.env`

## Common Commands

Run one question:

```bash
uv run python solve.py "What were the total expenditures for national defense in 1940?"
```

Run tests:

```bash
uv run pytest
```

Check shared agent wiring:

```bash
UV_CACHE_DIR=/tmp/uv-officeqa uv run python -m scripts.doctor_agents
UV_CACHE_DIR=/tmp/uv-officeqa uv run python -m scripts.setup_agents
```

Run selected eval flows:

```bash
uv run python solve.py --eval --n 10
uv run python solve.py --eval
uv run python eval_decompose.py
uv run python extract.py --test-oracle --n 20
```

## Planning Docs

- [Target Architecture](docs/TARGET_ARCHITECTURE.md)
- [Execution Roadmap](docs/EXECUTION_ROADMAP.md)
- [Workstreams And Ownership](docs/WORKSTREAMS.md)
- [Ingestion Integrity And Benchmark Strategy](docs/INGESTION_AND_BENCHMARK_STRATEGY.md)
- [Agent orchestration and git worktrees](docs/AGENT_ORCHESTRATION.md)
- [Orchestration plan (doc reconciliation + wave model)](docs/ORCHESTRATION_PLAN.md)
- [Lossless Ledger Plan](PLAN.md)
- [Current Project Architecture / Commands](CLAUDE.md)

**Note on phase names:** `PLAN.md` uses phases 0–5 for the **ledger** (ingestion, metrics, losslessness). `docs/EXECUTION_ROADMAP.md` uses phases 0–8 for the **solver and harness**. When assigning work, use workstream names from `WORKSTREAMS.md`, not bare “phase 2,” so agents do not confuse the two.

## CI

GitHub Actions runs **Ruff** and **pytest** on pushes and pull requests to `main` (no `ledger.sqlite` in the runner, so integration tests in `test_feedback_loop.py` and `test_ledger.py` are skipped there). Run the full suite locally after `build_ledger.py`. Pyright is still `uv run pyright` on your machine until legacy entrypoints are excluded or cleaned up.

### Local worktrees + agent rounds

```bash
./scripts/setup_worktrees.sh
./scripts/link_worktree_artifacts.sh
DRY_RUN=1 ./scripts/agent_iterate.sh
```

See [Agent orchestration](docs/AGENT_ORCHESTRATION.md) for siloed-stage fakes, merge rules, and wiring `OFFICEQA_AGENT` for iterative check loops on your machine.

One-shot **Cursor Agent** in a worktree (requires local `cursor-agent` login):

```bash
OFFICEQA_WT_ROOT="$PWD/.agent-worktrees" ./scripts/cursor_agent_once.sh retrieval
```

Use `CURSOR_AGENT_MODE=plan` for a read-only pass first.

**Cursor app (open worktree + paste spec):** see [scripts/agent_task_specs/README.md](scripts/agent_task_specs/README.md) for the assignment table and step-by-step “Open Folder → attach `.txt`” flow.

## Repo Notes

- `retrieve_v2.py` is the canonical retrieval implementation.
- `ledger.sqlite` and evaluation artifacts are large; keep a clear distinction between code and generated outputs.
- The project should first become excellent on OfficeQA before being generalized to other benchmarks.

---

## Related

**[OfficeQA Arena](https://github.com/jwalin-shah/officeqa-arena)** — Competition entry using this pipeline for the Sentient Arena benchmark.
