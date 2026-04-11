# Retrieval Eval Results (Post-Retrieval Milestone)

## Metrics Comparison (baseline → current)

| Metric | Baseline | Current | Delta |
|--------|----------|---------|-------|
| recall@1 | 6.5% | 7.7% | +1.2pp |
| recall@5 | 19.1% | 20.7% | +1.6pp |
| recall@10 | 25.6% | 27.6% | +2.0pp |
| recall@20 | 31.7% | 35.0% | +3.3pp |
| recall@30 | 35.8% | 39.0% | +3.2pp |
| file-level recall@10 | 62.2% | 62.2% | 0.0pp |

## Edge Case Guarantees (VAL-RETR-008)

1. **Zero results → empty list**: `retrieve()` returns `[]` (not None, not exception) when both FTS and metric channels return no results.
2. **No parseable years → unfiltered FTS**: When a question has no 4-digit years, `_fts_channel_trace` uses `exact_unrestricted` strategy (no year filtering).

## Retrieval Improvements Contributing to Gains

- Synonym expansion (12 groups) — FTS + metric channels
- Multi-strategy progressive broadening (5 FTS strategies, 3 metric strategies)
- Prose/footnote supplementary channel (~1.2% of questions)
- Row hint alternatives consumed from decompose specs
- Period-aware scoring (calendar/fiscal boost/penalty)
- Multi-request processing (all data_requests, not just first)

## Test Coverage

7 edge case tests added in `tests/test_retrieval.py`:
- `test_retrieve_zero_results_returns_empty_list`
- `test_retrieve_zero_results_both_channels_empty`
- `test_retrieve_zero_results_null_plan`
- `test_retrieve_no_parseable_years_uses_unfiltered_fts`
- `test_retrieve_from_question_no_years_still_returns_results`
- `test_fts_channel_no_years_uses_unrestricted_strategy`
