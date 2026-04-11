# Execution Roadmap

This roadmap assumes the target architecture in [TARGET_ARCHITECTURE.md](TARGET_ARCHITECTURE.md).

The order is deliberate: first make the system excellent on OfficeQA, then generalize.

## Program Goals

Primary goal:
- make the ledger-backed harness reliably strong on OfficeQA

Secondary goal:
- make the architecture reusable across adjacent table-grounded benchmarks

Non-goal for now:
- building a generic multi-agent research platform before OfficeQA is solved

## Phase 0: Lock The Architecture

Deliverables:
- target architecture doc
- canonical `QuestionState`
- stage contracts
- verifier contracts
- workstream ownership

Exit criteria:
- everyone is building against the same mental model
- new code can be classified into a stage

## Phase 1: Establish The Measurement Harness

Goal:
- make stage-level progress measurable

Deliverables:
- standard benchmark subsets:
  - `smoke`
  - `retrieval_dev`
  - `extraction_dev`
  - `full_eval`
- per-stage metrics
- reproducible eval scripts that do not require live LLM calls when avoidable

Required metrics:
- decompose validity rate
- row_hint coverage
- retrieval recall@k
- retrieval row-hit recall
- deterministic fast-path coverage
- oracle extraction accuracy
- end-to-end accuracy
- failure-mode distribution

Exit criteria:
- each stage can be measured independently
- regressions are attributable to a stage, not just to the full pipeline

## Phase 2: Retrieval First

Goal:
- improve evidence quality before tuning downstream stages

Focus:
- candidate generation
- reranking
- row/year intersection checks
- summary-table prioritization
- bottom-up search policy

Key work:
- add deterministic rerank features using actual ledger evidence
- promote tables that contain a resolvable row/year intersection
- make retrieval traces auditable
- compare table recall and row-hit recall before and after changes

Exit criteria:
- meaningful improvement in recall@k and row-hit recall
- reduced dependence on extraction over weak evidence

## Phase 3: Decompose Tightening

Goal:
- reduce ambiguity before retrieval

Focus:
- `row_hint`
- years
- granularity
- period basis
- binary-op correctness

Key work:
- post-decompose ledger validation
- retry decompose when `row_hint` is implausible
- enforce menu selection discipline when vocab is available

Exit criteria:
- row_hint coverage clears the quality bar
- fewer fallback specs
- lower retrieval miss rate caused by bad decomposition

## Phase 4: Extraction Hardening

Goal:
- make extraction accurate when retrieval is already good

Focus:
- oracle extraction performance
- ledger-assisted extraction checks
- alternate evidence retry strategy
- null/label mismatch handling

Key work:
- inject resolved ledger hints where safe
- retry against different candidate tables, not just different renderings
- strengthen extraction verification against ledger facts

Exit criteria:
- oracle extraction meaningfully improves
- fewer unit and period mistakes survive into verify

## Phase 5: Fast-Path Expansion

Goal:
- replace LLM work with deterministic logic wherever possible

Focus:
- `continuous_monthly`
- better multi-year support
- stronger partial-resolution handling

Key work:
- formalize the fast-path contract
- separate strict deterministic resolution from opportunistic bottom-up rescue
- add explicit tests for both modes

Exit criteria:
- fast-path coverage increases
- test expectations match the intended contract

## Phase 6: Orchestrator Policy Tuning

Goal:
- make retries disciplined instead of accidental

Focus:
- retry boundaries
- stage-local repair
- evidence preservation

Key work:
- standardize retry reasons
- make the orchestrator consume verifier outputs, not ad hoc strings
- ensure retries do not discard useful prior evidence without logging why

Exit criteria:
- phase transitions are coherent
- retry behavior is testable and explainable

## Phase 7: Integration And Cleanup

Goal:
- converge the codebase around the modern harness

Focus:
- archive legacy scripts
- split oversized modules
- unify duplicated constants
- update README and operational docs

Exit criteria:
- one obvious supported path through the codebase
- fewer overlapping entrypoints

## Phase 8: Generalize To Other Benchmarks

Only start this after OfficeQA is strong and stable.

Generalize by parameterizing:
- question schema
- evidence schema
- benchmark adapters
- scorer interfaces
- ingestion contracts

Do not generalize by weakening the OfficeQA-specific constraints that currently make the system grounded.

## Recommended First 6 Work Items

1. Define `QuestionState` and stage verifier result schemas.
2. Replace the legacy eval path with a canonical harness-driven evaluation path.
3. Improve retrieval reranking using row/year intersection evidence.
4. Add post-decompose `row_hint` validation against the ledger.
5. Formalize fast-path contracts and fix the current regression ambiguity.
6. Add a small offline golden set with cached stage artifacts.
