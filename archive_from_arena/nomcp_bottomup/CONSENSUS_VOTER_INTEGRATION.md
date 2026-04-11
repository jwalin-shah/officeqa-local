**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# ConsensusVoter Integration Guide

## Quick Integration Checklist

- [x] **Component**: `consensus_voter.py` (13KB, 1 class)
- [x] **Tests**: `test_consensus_voter.py` (13 test cases, all passing)
- [x] **Documentation**: `CONSENSUS_VOTER.md` (comprehensive)
- [x] **Dependencies**: None (pure Python, no external libs)

## Integration Points

### 1. Replace Hardcoded Consensus Logic

**Before** (current solve_decompose.py approach):
```python
# Groups values by similarity, picks best group
groups = {}
for val, r in scalars:
    matched = False
    for key in groups:
        if abs(key - val) < 0.01 * max(abs(key), 1):
            groups[key].append((val, r))
            matched = True
            break
    if not matched:
        groups[val] = [(val, r)]

best_group = max(groups.values(), key=len)
winner_val, winner_result = best_group[0]
```

**After** (with ConsensusVoter):
```python
from consensus_voter import ConsensusVoter

voter = ConsensusVoter()
result = voter.vote(
    strict_result=strategy_a_result,
    fuzzy_result=strategy_b_result,
    contextual_result=strategy_c_result
)

winning_value = result['winning_value']
winning_confidence = result['winning_confidence']
reasoning = result['reasoning']
risk_flags = result['risk_flags']
```

### 2. Use Consensus Voter in search_raw_corpus

If implementing 3-strategy search voting:

```python
from consensus_voter import ConsensusVoter

def search_corpus_with_voting(query, year, month):
    """Search using 3 strategies, vote on best result."""

    # Run 3 searches in parallel
    strict_result = search_strict(query, year, month)
    fuzzy_result = search_fuzzy(query, year, month)
    contextual_result = search_contextual(query, year, month)

    # Vote on results
    voter = ConsensusVoter()
    consensus = voter.vote(
        strict_result=strict_result,
        fuzzy_result=fuzzy_result,
        contextual_result=contextual_result
    )

    # Log voting details for debugging
    if consensus['risk_flags']:
        log_warning(f"Search voting risk: {consensus['risk_flags']}")

    return {
        'value': consensus['winning_value'],
        'confidence': consensus['winning_confidence'],
        'method': consensus['winning_method'],
        'details': consensus['voting_details'],
        'reasoning': consensus['reasoning']
    }
```

### 3. Use Consensus Voter in extract_value_consensus

Refactor the current approach to use ConsensusVoter:

```python
def extract_value_consensus(subquery, table_text, n=3):
    """Phase 5 consensus: Run N extraction calls with diverse prompts, use voter."""
    if not table_text:
        return {"values": None, "confidence": "low", "notes": "No table data"}

    # Run 3 extractions with different prompts/temperatures
    results = []
    for i in range(n):
        result = run_extraction_variant(i, subquery, table_text)
        if result and result.get("values") is not None:
            results.append(result)

    if not results:
        return {"values": None, "confidence": "low", "notes": "All extraction variants failed"}
    if len(results) == 1:
        return results[0]

    # Use ConsensusVoter for final value selection
    from consensus_voter import ConsensusVoter

    voter = ConsensusVoter()
    consensus = voter.vote(
        strict_result=results[0] if len(results) > 0 else None,
        fuzzy_result=results[1] if len(results) > 1 else None,
        contextual_result=results[2] if len(results) > 2 else None
    )

    return {
        'values': consensus['winning_value'],
        'confidence': 'high' if consensus['agreement_level'] == 'full' else 'medium',
        'notes': consensus['reasoning'],
        'voting_details': consensus['voting_details']
    }
```

### 4. Use in solve_consensus Pipeline

```python
def solve_consensus(question):
    """Run pipeline with ConsensusVoter at value extraction phase."""

    # Phase 1: Decompose
    plan = decompose_consensus(question, n=3)

    # Phase 2-4: Search (produce 3 parallel sets of tables)
    strict_tables = search_strict_tables(plan)
    fuzzy_tables = search_fuzzy_tables(plan)
    contextual_tables = search_contextual_tables(plan)

    # Phase 5: Extract with voting
    final_values = []
    for subquery in plan['subqueries']:
        from consensus_voter import ConsensusVoter
        voter = ConsensusVoter()

        strict_val = extract_from_tables(subquery, strict_tables)
        fuzzy_val = extract_from_tables(subquery, fuzzy_tables)
        contextual_val = extract_from_tables(subquery, contextual_tables)

        consensus = voter.vote(strict_val, fuzzy_val, contextual_val)
        final_values.append(consensus['winning_value'])

    # Phase 6: Compute answer
    return compute_answer(plan, final_values)
```

## Confidence Score Interpretation

| Confidence | Meaning | Action |
|------------|---------|--------|
| 0.95-0.99 | Excellent | Use confidently, minimal validation |
| 0.85-0.94 | Good | Use, with light spot-checking |
| 0.70-0.84 | Moderate | Use, but recommend verification |
| 0.50-0.69 | Weak | Consider re-search or flag for review |
| < 0.50 | Very Low | Reject or escalate |

## Debugging Consensus Failures

When `agreement_level == 'none'` or risk flags are present:

```python
result = voter.vote(strict, fuzzy, contextual)

if result['risk_flags']:
    print(f"Risk flags: {result['risk_flags']}")
    print(f"Voting details:")
    for method, detail in result['voting_details'].items():
        print(f"  {method}: value={detail['value']}, conf={detail['confidence']}")
    print(f"Reasoning: {result['reasoning']}")

    # Consider re-running search with different parameters
    if 'no_agreement_all_differ' in result['risk_flags']:
        print("→ Consider expanding search year window or adjusting search terms")
    if 'low_confidence' in result['risk_flags']:
        print("→ Consider using a different table or search strategy")
```

## Performance Metrics

```python
import time
from consensus_voter import ConsensusVoter

voter = ConsensusVoter()

start = time.time()
result = voter.vote(strict, fuzzy, contextual)
elapsed_ms = (time.time() - start) * 1000

print(f"Voting time: {elapsed_ms:.2f}ms")
# Expected: <0.2ms for typical results
```

## Testing Your Integration

```python
# In your integration tests:

from consensus_voter import ConsensusVoter

def test_search_voting_integration():
    """Test that search produces voteable results."""

    voter = ConsensusVoter()

    # Run 3 searches
    strict = search_strict_strategy(query, year, month)
    fuzzy = search_fuzzy_strategy(query, year, month)
    contextual = search_contextual_strategy(query, year, month)

    # Vote
    result = voter.vote(strict, fuzzy, contextual)

    # Verify output
    assert result['winning_value'] is not None
    assert 0.0 <= result['winning_confidence'] <= 1.0
    assert result['agreement_level'] in ('full', 'partial', 'none')
    assert len(result['risk_flags']) >= 0

    print(f"✓ Voting passed: {result['agreement_level']} agreement, "
          f"confidence {result['winning_confidence']:.2f}")
```

## Migration Path

### Phase 1: Add ConsensusVoter (No Breaking Changes)
- Copy `consensus_voter.py` to codebase
- Add as optional component (don't remove old logic yet)
- Run tests to verify

### Phase 2: Hybrid Mode (Validation)
- Run both old and new voting logic in parallel
- Compare results
- Log divergences for 100+ queries
- Verify new approach matches or improves on old

### Phase 3: Replace (Full Migration)
- Replace old hardcoded voting with ConsensusVoter
- Remove old consensus logic
- Benchmark performance improvement

### Phase 4: Optimize (Tuning)
- Adjust tolerance values based on corpus analysis
- Tune confidence boost factors if needed
- Consider strategy-specific weights

## Files

| File | Purpose | Size |
|------|---------|------|
| `consensus_voter.py` | Main component (1 class, 2 functions) | 13KB |
| `test_consensus_voter.py` | Test suite (13 test cases) | 11KB |
| `CONSENSUS_VOTER.md` | Full documentation | 11KB |
| `CONSENSUS_VOTER_INTEGRATION.md` | This file (integration guide) | 6KB |

## Support

For questions or issues:

1. Check `CONSENSUS_VOTER.md` for detailed documentation
2. Run `test_consensus_voter.py` to verify the component works
3. Review edge case examples in documentation
4. Check integration examples in this file

## Key Design Principles

1. **Consensus > Confidence**: Agreement from multiple strategies beats single high confidence
2. **Transparent**: Every result includes reasoning and voting breakdown
3. **Defensive**: Handles all edge cases (None, single result, disagreement)
4. **Lightweight**: Pure Python, no external dependencies
5. **Testable**: Comprehensive test suite with edge cases
