**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# ConsensusVoter Component

## Overview

The `ConsensusVoter` class compares results from 3 parallel subsearch strategies and picks the best value based on agreement level and confidence weighting.

**Strategies:**
- **Strict**: Exact matching (highest confidence when it works)
- **Fuzzy**: Keyword matching (moderate confidence, broader coverage)
- **Contextual**: Structure-based matching (moderate confidence, semantic understanding)

## Design Philosophy

1. **Consensus over confidence**: Agreement from multiple strategies is more reliable than a single high-confidence result
2. **Confidence boosting**: When strategies agree, boost the confidence; when they disagree, penalize it
3. **Tolerance for rounding**: Values within 1% are treated as equivalent (handles floating-point precision)
4. **Transparent voting**: Full audit trail of voting details and reasoning

## Quick Start

### Basic Usage

```python
from consensus_voter import ConsensusVoter

voter = ConsensusVoter()

result = voter.vote(
    strict_result={'value': 1350576.0, 'confidence': 0.99, 'method': 'strict'},
    fuzzy_result={'value': 1350576.0, 'confidence': 0.85, 'method': 'fuzzy'},
    contextual_result={'value': 1350576.0, 'confidence': 0.88, 'method': 'contextual'}
)

print(f"Winning value: {result['winning_value']}")
print(f"Confidence: {result['winning_confidence']}")
print(f"Agreement: {result['agreement_level']}")
print(f"Reasoning: {result['reasoning']}")
```

### Convenience Function

```python
from consensus_voter import vote_three_results

result = vote_three_results(
    strict_result={...},
    fuzzy_result={...},
    contextual_result={...}
)
```

## Output Format

```python
{
    # Winning result
    'winning_value': 1350576.0,           # The agreed-upon value
    'winning_confidence': 0.96,           # Boosted confidence
    'winning_method': 'strict',           # Which strategy won

    # Agreement details
    'agreement_level': 'full',            # 'full', 'partial', 'none'
    'num_agreeing': 3,                    # Number of agreeing strategies

    # Voting breakdown
    'voting_details': {
        'strict': {'value': 1350576.0, 'confidence': 0.99, 'method': 'strict'},
        'fuzzy': {'value': 1350576.0, 'confidence': 0.85, 'method': 'fuzzy'},
        'contextual': {'value': 1350576.0, 'confidence': 0.88, 'method': 'contextual'}
    },

    # Diagnostics
    'reasoning': 'All 3 strategies agreed on value 1350576.0, boosting confidence from 0.99 to 0.96',
    'risk_flags': []                      # Any concerns (empty if none)
}
```

## Agreement Levels

### Full Agreement
**Condition**: All valid strategies return the same value (within tolerance)

**Confidence Boost**: 15% increase, capped at 0.99
- Example: 0.99 base → 0.99 final (already at cap)
- Example: 0.85 base → 0.9775 → 0.977 final

**Use case**: Highest confidence result, safe to use

### Partial Agreement
**Condition**: 2 out of 3 strategies agree

**Confidence Boost**: 5% increase
- Example: 0.99 base → 1.0395 → 0.99 final (capped)
- Example: 0.85 base → 0.8925 final

**Use case**: Good confidence, investigate the disagreement

### No Agreement
**Condition**: All 3 strategies return different values (or only 1 valid)

**Confidence Penalty**: 5% decrease
- Example: 0.99 base → 0.9405 final
- Example: 0.70 base → 0.665 final

**Use case**: Use with caution, consider re-searching

## Value Comparison Tolerance

Values are considered "agreeing" if they're within **1% relative difference**:

```
tolerance = 0.01 (1%)
percent_diff = |value1 - value2| / max(abs(value1), abs(value2), 1)
agreed = percent_diff <= tolerance
```

**Examples:**
- 1,000,000 vs 1,005,000 → 0.5% diff → AGREE
- 1,000,000 vs 1,010,000 → 1% diff → AGREE (at tolerance)
- 1,000,000 vs 1,020,000 → 2% diff → DISAGREE
- 100 vs 101 → 1% diff → AGREE
- 100 vs 102 → 2% diff → DISAGREE

## Edge Cases

### Case 1: All Results None
```python
result = voter.vote(
    strict_result=None,
    fuzzy_result=None,
    contextual_result=None
)
# Output:
# {
#     'winning_value': None,
#     'winning_confidence': 0.0,
#     'winning_method': None,
#     'agreement_level': 'none',
#     'num_agreeing': 0,
#     'risk_flags': ['all_results_none']
# }
```

### Case 2: One Valid Result
```python
result = voter.vote(
    strict_result={'value': 1350576.0, 'confidence': 0.99, 'method': 'strict'},
    fuzzy_result=None,
    contextual_result=None
)
# Output:
# {
#     'winning_value': 1350576.0,
#     'winning_confidence': 0.9405,      # 0.99 * 0.95 (penalty for no agreement)
#     'winning_method': 'strict',
#     'agreement_level': 'none',
#     'num_agreeing': 1,
#     'risk_flags': []
# }
```

### Case 3: Two Valid Results (One None)
```python
result = voter.vote(
    strict_result={'value': 1350576.0, 'confidence': 0.99, 'method': 'strict'},
    fuzzy_result={'value': 1350576.0, 'confidence': 0.85, 'method': 'fuzzy'},
    contextual_result=None
)
# Output:
# {
#     'winning_value': 1350576.0,
#     'winning_confidence': 0.99,        # 0.99 * 1.15 capped at 0.99
#     'winning_method': 'strict',
#     'agreement_level': 'full',         # 2 of 2 valid results agree
#     'num_agreeing': 2,
#     'risk_flags': []
# }
```

### Case 4: Close Values (Within Tolerance)
```python
result = voter.vote(
    strict_result={'value': 1000000.0, 'confidence': 0.95, 'method': 'strict'},
    fuzzy_result={'value': 1005000.0, 'confidence': 0.80, 'method': 'fuzzy'},  # 0.5% diff
    contextual_result={'value': 1000000.0, 'confidence': 0.85, 'method': 'contextual'}
)
# Output:
# {
#     'winning_value': 1000000.0,        # Or 1005000.0, grouped together
#     'winning_confidence': 0.99,        # Full agreement boost
#     'agreement_level': 'full',
#     'num_agreeing': 3,
#     'risk_flags': []
# }
```

### Case 5: Significant Disagreement
```python
result = voter.vote(
    strict_result={'value': 1000000.0, 'confidence': 0.99, 'method': 'strict'},
    fuzzy_result={'value': 2000000.0, 'confidence': 0.85, 'method': 'fuzzy'},
    contextual_result={'value': 3000000.0, 'confidence': 0.70, 'method': 'contextual'}
)
# Output:
# {
#     'winning_value': 1000000.0,        # Best confidence (0.99)
#     'winning_confidence': 0.9405,      # Penalty: 0.99 * 0.95
#     'winning_method': 'strict',
#     'agreement_level': 'none',
#     'num_agreeing': 1,
#     'risk_flags': ['no_agreement_all_differ', 'multiple_values_found: ...']
# }
```

## Risk Flags

The voter identifies potential concerns and includes them in `risk_flags`:

- **`all_results_none`**: All three strategies failed to return a value
- **`no_agreement_all_differ`**: All three strategies returned different values
- **`low_confidence`**: Final confidence is below 0.50
- **`multiple_values_found`**: Different strategies found different values (disagreement)

## Helper Methods

### `check_agreement(value1, value2, tolerance=0.01)`

Compare if two numeric values agree.

```python
voter = ConsensusVoter()
agreement = voter.check_agreement(1000000.0, 1005000.0)
# Returns: 'close' (within 1% tolerance)

agreement = voter.check_agreement(1000000.0, 1020000.0)
# Returns: 'disagree' (2% difference)

agreement = voter.check_agreement(100.0, 100.0)
# Returns: 'exact'
```

### `boost_confidence_for_agreement(agreement_level, individual_confidences)`

Calculate boosted confidence based on agreement.

```python
voter = ConsensusVoter()

# Full agreement: 0.85 * 1.15 = 0.9775
boosted = voter.boost_confidence_for_agreement('full', [0.85, 0.90, 0.80])

# Partial agreement: 0.95 * 1.05 = 0.9975 → 0.99
boosted = voter.boost_confidence_for_agreement('partial', [0.95, 0.88])

# No agreement: 0.99 * 0.95 = 0.9405
boosted = voter.boost_confidence_for_agreement('none', [0.99])
```

### `rank_results(results)`

Rank results by confidence (for debugging/inspection).

```python
voter = ConsensusVoter()
results = [
    {'value': 1000.0, 'confidence': 0.70},
    {'value': 2000.0, 'confidence': 0.95},
    {'value': 3000.0, 'confidence': 0.80},
]
ranked = voter.rank_results(results)
# Returns: [
#   ({'value': 2000.0, 'confidence': 0.95}, 0.95),
#   ({'value': 3000.0, 'confidence': 0.80}, 0.80),
#   ({'value': 1000.0, 'confidence': 0.70}, 0.70),
# ]
```

## Integration Example

```python
from consensus_voter import ConsensusVoter
from search_module import run_strict_search, run_fuzzy_search, run_contextual_search

def find_table_value(query, table_name, year):
    """Find a value using 3 parallel search strategies."""

    # Run 3 searches in parallel
    strict_result = run_strict_search(query, table_name, year)
    fuzzy_result = run_fuzzy_search(query, table_name, year)
    contextual_result = run_contextual_search(query, table_name, year)

    # Vote on results
    voter = ConsensusVoter()
    consensus = voter.vote(strict_result, fuzzy_result, contextual_result)

    # Use result
    if consensus['winning_value'] is None:
        # All strategies failed
        return None, "No value found"

    if consensus['agreement_level'] == 'none':
        # No agreement - log warning
        print(f"WARNING: Disagreement detected. Using {consensus['winning_method']}")

    return consensus['winning_value'], consensus['reasoning']
```

## Testing

Run the comprehensive test suite:

```bash
python3 test_consensus_voter.py
```

This runs 13 test cases covering:
- Full agreement (all 3 agree)
- Partial agreement (2 out of 3)
- No agreement (all different)
- Close values (within tolerance)
- All None results
- One valid result
- Two valid results
- Confidence boost math
- Value comparison
- Result ranking
- Output structure
- Large value differences

## Performance

- Single vote: ~0.1-0.2ms (pure Python, no I/O)
- Memory: <1KB per vote
- Thread-safe: Yes (no shared state)

## Design Decisions

### Why Tolerance-Based Grouping?

Floating-point arithmetic and rounding differences are inevitable when dealing with Treasury data across different table formats. A 1% tolerance handles:
- Rounding variations (999.9 vs 1000.0)
- Different decimal precision (1350576.0 vs 1350576)
- Unit conversions in some cases

### Why Confidence Boost?

Consensus is more reliable than individual high confidence. A single 0.99 result could be lucky; three independent approaches agreeing is strong signal. The boost math:
- **Full agreement**: 15% boost (0.85→0.9775) — very strong signal
- **Partial agreement**: 5% boost (0.85→0.8925) — good signal
- **Penalty**: 5% penalty (0.99→0.9405) — uncertainty

### Why Cap at 0.99?

Perfect confidence (1.0) is never justified. Always leave room for:
- Hidden data issues
- Parsing errors
- Tolerance mismatches

## Future Enhancements

1. **Weighted voting**: Different weights for different strategies
2. **Temporal voting**: Use historical agreement rates to weight strategies
3. **Domain-aware tolerance**: Different tolerances for different value types
4. **Explanation generation**: Detailed, multi-paragraph reasoning
5. **Learning from feedback**: Adjust boosting factors based on downstream accuracy
