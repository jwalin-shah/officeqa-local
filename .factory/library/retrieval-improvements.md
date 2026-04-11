# Retrieval Improvements Notes

## row_hint_alternatives (retrieve_v2.py)

`_extract_hints()` returns a list of dicts (one per data_request) with:
- `metric`, `col_hint`, `row_hint`, `row_hint_alternatives`, `years`

The metric channel now runs for EACH data_request including each
`row_hint_alternative`. Results are merged (deduped by table id).

## Multi-request processing

`retrieve()` processes ALL data_requests in a plan. Each DR's metric
channel is called separately with its own row_hint and alternatives.
FTS channel uses aggregated query text from all DRs.

`solve.py`'s `retrieve_for_spec()` still creates per-DR mini_plans and
calls `retrieve()` for each — this provides per-DR isolation of year
filters. The multi-request support in `retrieve()` is used when the
function is called directly with a full plan.

## Period-aware scoring

`_period_aware_score_delta(year_mode, table_period)` returns:
- +0.5 when period matches year_mode (fiscal↔fiscal, calendar↔calendar)
- -0.3 when period mismatches (fiscal↔calendar, calendar↔fiscal)
- 0.0 when year_mode is "unknown" or table_period is empty/None

Applied in reranking with a 0.2x scaling factor to prevent the period
signal from dominating channel scores. Many tables contain both FY and
CY data despite being labeled with one period, so the raw -0.3 penalty
would incorrectly demote correct tables.

## Recall metrics (as of this feature)

recall@1=7.7%, recall@5=20.7%, recall@10=27.6%, recall@20=35.0%,
recall@30=39.0% (baseline was @10=25.6%, @30=35.8%)
