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
- [Lossless Ledger Plan](PLAN.md)
- [Current Project Architecture / Commands](CLAUDE.md)

## Repo Notes

- `retrieve_v2.py` is the canonical retrieval implementation.
- `ledger.sqlite` and evaluation artifacts are large; keep a clear distinction between code and generated outputs.
- The project should first become excellent on OfficeQA before being generalized to other benchmarks.
