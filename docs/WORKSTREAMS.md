# Workstreams And Ownership

This file defines the major workstreams, what good looks like, and how multiple tools or models should collaborate without stepping on each other.

## Collaboration Model

Use a hub-and-spoke workflow:

- one planner/editor-in-chief owns the master plan
- one owner per workstream owns implementation
- one reviewer per workstream reviews findings or diffs
- integration into main happens through one integrator at a time

Do not have multiple models make overlapping edits in the same files without explicit ownership.

## Workstream 1: Retrieval

Scope:
- `retrieve_v2.py`
- bottom-up retrieval policy
- reranking
- retrieval metrics and traces

Goal:
- improve table recall and row-hit recall

Success metrics:
- recall@k
- file recall@k
- row/year intersection hit rate
- fewer extraction attempts on weak evidence

Recommended owner:
- Codex or Cursor for implementation

Recommended reviewer:
- Claude or Gemini for design review and heuristic critique

## Workstream 2: Decompose

Scope:
- decompose prompt
- spec schema
- post-decompose validation

Goal:
- make specs more grounded and less ambiguous

Success metrics:
- valid spec rate
- row_hint coverage
- row_hint plausibility against ledger
- fewer retrieval misses caused by poor specs

Recommended owner:
- Claude for prompt and schema reasoning
- Codex for validation code and tests

Recommended reviewer:
- Gemini

## Workstream 3: Extraction

Scope:
- table rendering
- extraction prompt behavior
- ledger-assisted extraction checks
- quality retry logic

Goal:
- improve extraction on good evidence

Success metrics:
- oracle extraction accuracy
- null-rate reduction
- citation/provenance completeness
- fewer verify-stage corrections

Recommended owner:
- Claude for prompt strategy
- Codex for deterministic assists, context shaping, and tests

Recommended reviewer:
- Gemini

## Workstream 4: Fast Path

Scope:
- `resolve_cells`
- deterministic resolution contracts
- continuous monthly support
- partial-resolution semantics

Goal:
- expand deterministic coverage and make behavior explicit

Success metrics:
- fast-path coverage
- deterministic exactness
- reduction in unnecessary LLM extraction

Recommended owner:
- Codex

Recommended reviewer:
- Claude

## Workstream 5: Evaluation Harness

Scope:
- benchmark runners
- golden fixtures
- stage metrics
- offline regression coverage

Goal:
- make progress measurable and reproducible

Success metrics:
- reproducible offline smoke tests
- one canonical eval path
- stage metrics reported consistently

Recommended owner:
- Codex

Recommended reviewer:
- Claude

## Workstream 6: Ingestion Integrity

Scope:
- `build_ledger.py`
- migrations
- losslessness checks
- provenance and audit tables

Goal:
- ensure nothing important is lost during ingestion

Success metrics:
- reconciliation counts by element type
- no silent drops
- reproducible rebuilds
- audit reports for parser rescue paths

Recommended owner:
- Codex

Recommended reviewer:
- Gemini or Claude

## Suggested Tool Roles

Use tools by role, not by taste.

- `Codex`
  - code changes
  - refactors
  - tests
  - integration
- `Claude`
  - architecture review
  - prompt design
  - decomposition strategy
  - extraction strategy
- `Gemini`
  - broad synthesis
  - second-opinion review
  - comparative design critique
- `Cursor`
  - fast local coding
  - repo navigation
  - interactive implementation support
- `Droid` or autonomous runners
  - batch experiments
  - repeated eval sweeps
  - artifact generation

## Delegation Template

Every delegated task should specify:

1. Goal
2. Files in scope
3. What not to touch
4. Exact expected output
5. Acceptance criteria
6. Reviewer

Example:

```text
Goal: Improve summary-table prioritization in retrieval reranking.
Files: retrieve_v2.py, tests/test_retrieval.py
Do not touch: extract.py, solve.py
Output: patch + before/after metrics on retrieval_dev subset
Acceptance: no test regressions; recall@10 improves on subset
Reviewer: Claude
```

## Merge Discipline

- one person or one model integrates to the main branch at a time
- each workstream should prefer short-lived branches
- merge only after stage metrics or tests are attached
- keep a changelog of benchmark-impacting changes
