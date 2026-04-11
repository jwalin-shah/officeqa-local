# Ingestion Integrity And Benchmark Strategy

This document answers two related questions:

1. How do we guarantee we are not losing important data during ingestion?
2. How do we adapt this harness to other benchmarks without weakening it?

## Part 1: Ingestion Integrity

The ledger is only valuable if it preserves the corpus faithfully enough to support grounded reasoning. A structured database that silently drops hard cases is worse than a raw-text fallback because it creates false confidence.

## Ingestion Guarantees

The ingestion pipeline should aim to guarantee:

- every source file is accounted for
- every source element type is counted
- every dropped element is categorized and logged
- every parser rescue path is auditable
- table provenance is preserved down to `(file, element_seq)` or equivalent source coordinates
- derived layers never overwrite the raw layer

## Required Data Layers

### 1. Raw Source Layer

Purpose:
- preserve the source corpus exactly enough for audit and replay

Requirements:
- source file identity
- file checksum or equivalent version id
- source element ids / order
- raw HTML or raw content where present

### 2. Normalized Structural Layer

Purpose:
- make tables, rows, columns, cells, prose, footnotes, and page metadata queryable

Requirements:
- tables linked back to raw source coordinates
- row and column hierarchy preserved
- unit and footnote flags preserved
- parse-status fields for ambiguous cases

### 3. Derived Facts Layer

Purpose:
- expose retrieval-friendly and compute-friendly abstractions

Requirements:
- facts and metrics are derived from the normalized layer
- every derived fact can be traced back to normalized cells
- no derived layer is the only place where important information exists

## Anti-Loss Controls

The ingestion harness should produce a reconciliation report on every rebuild:

- files ingested
- elements by type
- tables parsed
- cells emitted
- prose emitted
- footnotes emitted
- page metadata emitted
- elements dropped by category
- parser fallback counts
- parser failure counts

Suggested outputs:
- `ingestion_report.json`
- `ingestion_diff_vs_previous.json`
- `ingestion_failures_sample.jsonl`

## Recommended Integrity Tests

### Source Reconciliation

For a sample or full run, verify:
- source element count by type
- emitted object count by destination table
- unexplained drop count is zero or explicitly justified

### Table Preservation

For sampled source tables, verify:
- row count preserved
- column count preserved
- header structure preserved
- footnote markers preserved
- numeric values preserved

### Provenance Traceability

For sampled derived facts, verify:
- fact -> cell -> table -> source file path is resolvable

### Deterministic Rebuild

For the same corpus version, verify:
- rebuild outputs are stable or differences are explained by intended schema changes

## Recommendation: Keep Raw And Derived Separate

Do not try to make one table serve all purposes.

Use:
- raw/structural layers for truth
- derived fact/metric layers for retrieval speed

This is how you avoid losing fidelity while still making queries fast.

## Part 2: Benchmark Strategy

## Immediate Recommendation

First make the harness kill OfficeQA.

That means:
- get the architecture stable
- get ingestion guarantees explicit
- get retrieval/extraction/verification working together
- get stage-level measurement solid

Do not start with a broad benchmark abstraction layer before OfficeQA is strong.

## Why OfficeQA First

OfficeQA is already hard in the right ways:
- historical tables
- revisions
- annual vs monthly ambiguity
- fiscal vs calendar ambiguity
- units
- footnotes
- multi-step computations

If the harness can do this well with citations and proof, it will be much easier to adapt to related benchmarks later.

## When To Generalize

Generalize only after these conditions are met:

1. `QuestionState` is stable.
2. stage contracts are explicit.
3. ingestion reports are trustworthy.
4. evaluation is reproducible.
5. the modern harness is clearly better than legacy entrypoints.

## How To Adapt To Other Benchmarks

Adapt by creating benchmark adapters, not by weakening the core harness.

The portable parts should be:
- orchestrator
- state model
- verifier pattern
- evaluation harness
- ingestion audit pattern

The benchmark-specific parts should be:
- corpus adapter
- question/spec adapter
- scorer
- retrieval heuristics tuned to the corpus

## Recommended Abstraction Boundary

Keep these interfaces explicit:

```python
BenchmarkAdapter:
    load_questions()
    score_answer(question, prediction)
    build_question_context(question)
    normalize_gold(question)

CorpusAdapter:
    ingest()
    retrieve(spec)
    resolve_cells(request)
    render_evidence(entries)
```

The OfficeQA implementation should be the first concrete adapter, not the first generic interface casualty.

## Suggested Rollout

### Stage A
- solve OfficeQA with one canonical harness

### Stage B
- extract benchmark-agnostic contracts from what actually worked

### Stage C
- add a second benchmark only after the OfficeQA path is stable

### Stage D
- compare what truly generalizes versus what should remain benchmark-specific

## Bottom Line

Do not try to generalize the system before the core harness is winning on OfficeQA.

The right order is:

1. make ingestion trustworthy
2. make the benchmark harness auditable
3. make OfficeQA strong
4. then abstract the interfaces needed for reuse
