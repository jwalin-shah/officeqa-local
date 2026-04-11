**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# Orchestrator: Multi-Piece Extraction Coordinator

## Overview

The **Orchestrator** is a production-ready system for managing parallel extraction of multiple decomposition pieces. It coordinates the processing of decomposition pieces and combines results using flexible computation rules.

## Quick Links

- **Start Here**: [ORCHESTRATOR_QUICK_START.md](ORCHESTRATOR_QUICK_START.md) (30-second intro)
- **Full Docs**: [ORCHESTRATOR.md](ORCHESTRATOR.md) (comprehensive reference)
- **Summary**: [ORCHESTRATOR_SUMMARY.md](ORCHESTRATOR_SUMMARY.md) (implementation details)
- **Integration**: [orchestrator_integration.py](orchestrator_integration.py) (how to integrate with solve_decompose)

## Files

| File | Purpose |
|------|---------|
| `orchestrator.py` | Core implementation (481 lines) |
| `test_orchestrator.py` | 21 comprehensive tests (all passing) |
| `orchestrator_integration.py` | Integration guide for solve_decompose |
| `ORCHESTRATOR.md` | Full documentation (18 sections) |
| `ORCHESTRATOR_QUICK_START.md` | Quick reference guide |
| `ORCHESTRATOR_SUMMARY.md` | Implementation summary |
| `README_ORCHESTRATOR.md` | This file |

## What It Does

Takes a decomposition with multiple pieces and:

1. **Validates** the decomposition (checks for required fields)
2. **Parallelizes** processing of each piece independently
3. **Collects** results with timeout handling
4. **Combines** results using computation rules (sum, average, difference, ratio, etc.)
5. **Calculates** confidence with pessimistic strategy (uses minimum + bonuses/penalties)
6. **Returns** final answer with complete transparency

## Example

```python
from orchestrator import SearchTaskOrchestrator

# Define how to process each piece
def extract_value(piece):
    """Search tables and extract value from a single decomposition piece."""
    piece_id = piece.get('piece_id')
    year = piece.get('year')

    # ... search for table, extract value ...

    return {
        'value': 1350576.0,
        'confidence': 0.99,
        'success': True,
        'source': 'treasury_bulletin_1995_12.txt/FFO-1'
    }

# Define the decomposition
decomposition = {
    'question_id': 123,
    'question': 'What was total receipts in FY 1995 and FY 1996?',
    'pieces': [
        {'piece_id': 1, 'year': 1995, 'description': 'FY 1995 receipts'},
        {'piece_id': 2, 'year': 1996, 'description': 'FY 1996 receipts'},
    ],
    'computation': 'sum'
}

# Execute
orch = SearchTaskOrchestrator(extract_value, max_workers=4)
result = orch.execute(decomposition)

# Result:
# {
#   'final_value': 2763732.0,
#   'final_confidence': 0.98,
#   'success': True,
#   'piece_results': [
#     {'piece_id': 1, 'value': 1350576.0, 'confidence': 0.99, ...},
#     {'piece_id': 2, 'value': 1413156.0, 'confidence': 0.98, ...}
#   ],
#   'steps': [...]
# }
```

## Key Features

### 1. Parallel Processing
```python
# Automatically parallelizes N pieces
# 4 pieces with 2 workers: ~50% of sequential time
orch = SearchTaskOrchestrator(processor, max_workers=4)
```

### 2. Flexible Computation Rules
```python
# 8 built-in operations
computation ∈ {direct, sum, average, difference, ratio, product, min, max}

# Examples:
'sum'        → add all pieces
'average'    → mean of pieces
'difference' → first - second
'ratio'      → first / second
```

### 3. Pessimistic Confidence
```python
# Confidence = min(piece_confidences)
# + 0.1 bonus if pieces agree
# - 0.2 penalty if many pieces fail

# Example:
# Piece 1: 0.99, Piece 2: 0.98
# final_confidence = 0.98 (uses minimum)
```

### 4. Error Handling
```python
# All errors are graceful:
# - Invalid decomposition → error
# - Processor exception → piece fails, others continue
# - Timeout → piece fails, others continue
# - Partial failures → return value with confidence penalty
# - All pieces fail → return None with error
```

### 5. Complete Transparency
```python
result['steps']           # Detailed step log
result['piece_results']   # Individual piece results
result['final_confidence']  # Why confidence is X
```

## Testing

All tests pass:

```bash
python3 -m pytest test_orchestrator.py -v

# 21 tests, 0.23 seconds
# Coverage: validation, computation rules, confidence, parallelization, errors
```

## Integration with solve_decompose

### Current approach (manual parallelization)
```python
def solve(question):
    plan = decompose(question)
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_process_subquery, sq): sq['id'] for sq in plan['sub_queries']}
        for future in concurrent.futures.as_completed(futures):
            sq_id, result, _ = future.result()
            extracted[sq_id] = result
    answer = compute(plan, extracted)
    return answer
```

### With Orchestrator (cleaner separation)
```python
from orchestrator_integration import solve_with_orchestrator

def solve(question):
    return solve_with_orchestrator(
        question,
        decompose_fn=decompose,
        search_tables_fn=search_tables,
        select_table_fn=select_table,
        search_data_fn=search_data,
        extract_value_fn=extract_value,
        compute_fn=compute,
        max_workers=4
    )
```

**Benefits**:
- ✅ Cleaner code (orchestration separated from extraction)
- ✅ Reusable (works with any processor)
- ✅ Better tested (21 tests)
- ✅ More flexible (custom computation rules)
- ✅ More transparent (detailed logging)
- ✅ Better error handling (graceful degradation)

## Usage Patterns

### Pattern 1: Simple Processor Function
```python
def processor(piece):
    # Extract value from piece
    return {'value': 1000.0, 'confidence': 0.95, 'success': True}

orch = SearchTaskOrchestrator(processor)
result = orch.execute(decomposition)
```

### Pattern 2: Custom Orchestrator Class
```python
class MyOrchestrator(Orchestrator):
    def _process_piece(self, piece):
        # Custom extraction logic
        return {'value': ..., 'confidence': ..., 'success': ...}

orch = MyOrchestrator(max_workers=4)
result = orch.execute(decomposition)
```

### Pattern 3: With Existing Pipeline
```python
from orchestrator_integration import ProcessPieceAdapter

adapter = ProcessPieceAdapter(
    search_tables_fn=search_tables,
    select_table_fn=select_table,
    search_data_fn=search_data,
    extract_value_fn=extract_value,
)
processor = adapter.create_processor()
orch = SearchTaskOrchestrator(processor)
result = orch.execute(decomposition)
```

## API Reference

### Orchestrator

```python
class Orchestrator:
    def __init__(self, max_workers=4, timeout_per_piece=180):
        """Initialize orchestrator."""

    def execute(self, decomposition) -> Dict:
        """Execute extraction for all pieces."""

    def _process_piece(self, piece) -> Dict:
        """Override to implement custom extraction logic."""
```

### SearchTaskOrchestrator

```python
class SearchTaskOrchestrator(Orchestrator):
    def __init__(self, search_task_processor, max_workers=4, timeout_per_piece=180):
        """Initialize with a processor function."""
```

### Convenience Function

```python
def execute_decomposition(decomposition, search_task_processor, max_workers=4) -> Dict:
    """One-shot decomposition executor."""
```

## Input Format

```python
{
    'question_id': 123,                     # Unique identifier
    'question': 'What was...',              # Original question
    'pieces': [                             # Sub-problems
        {
            'piece_id': 1,                  # Required: unique ID
            'year': 1995,                   # Custom fields as needed
            'description': 'FY 1995 receipts',
            # ... any other fields for processor
        }
    ],
    'computation': 'sum'                    # How to combine pieces
}
```

## Output Format

```python
{
    'question_id': 123,
    'final_value': 2763732.0,               # Result after computation
    'final_confidence': 0.98,               # Confidence (0.0-1.0)
    'success': True,                        # Whether computation succeeded
    'computation': 'sum',                   # Computation rule used
    'piece_results': [                      # Individual piece results
        {
            'piece_id': 1,
            'value': 1350576.0,
            'confidence': 0.99,
            'success': True,
            'source': '...',
            'notes': '...'
        },
        # ... more pieces
    ],
    'steps': [                              # Detailed execution log
        'Starting orchestrator for question 123',
        'Validation passed: 2 unique pieces',
        'Piece 1: value=1350576.0, confidence=0.99',
        # ... more steps
    ],
    'error': None  # Error message if failure
}
```

## Performance

- **Parallelization**: Effective with 2+ workers
- **Execution time**: 0.23s for 21 tests (4 pieces each)
- **Memory**: Minimal (stores piece results only)
- **Timeout handling**: Non-blocking per-piece timeouts

## Production Ready

- ✅ 21 comprehensive tests (all passing)
- ✅ Full error handling and graceful degradation
- ✅ Type hints throughout
- ✅ Complete docstrings
- ✅ Comprehensive documentation
- ✅ Detailed logging
- ✅ Tested with 4-10 parallel pieces

## Next Steps

1. **Start**: Read [ORCHESTRATOR_QUICK_START.md](ORCHESTRATOR_QUICK_START.md)
2. **Learn**: Read [ORCHESTRATOR.md](ORCHESTRATOR.md)
3. **Explore**: Run tests: `python3 -m pytest test_orchestrator.py -v`
4. **Integrate**: Follow [orchestrator_integration.py](orchestrator_integration.py)
5. **Use**: Implement with one of the patterns above

## FAQ

**Q: How many workers should I use?**
A: 2-4 typically. More parallelism but higher contention.

**Q: Can I use async/await?**
A: ThreadPoolExecutor works with any function. Simpler than async.

**Q: What if a piece times out?**
A: Marked as failed, doesn't block others. Result includes confidence penalty.

**Q: Can pieces depend on each other?**
A: No. Pieces are independent. Use custom subclass for dependencies.

**Q: How do I debug failures?**
A: Check `result['steps']` and `result['piece_results']` for details.

## Support

For detailed information:
- **Quick reference**: [ORCHESTRATOR_QUICK_START.md](ORCHESTRATOR_QUICK_START.md)
- **Full documentation**: [ORCHESTRATOR.md](ORCHESTRATOR.md)
- **Implementation details**: [ORCHESTRATOR_SUMMARY.md](ORCHESTRATOR_SUMMARY.md)
- **Integration guide**: [orchestrator_integration.py](orchestrator_integration.py)
- **Tests**: [test_orchestrator.py](test_orchestrator.py)

## Summary

The Orchestrator provides a clean, flexible, well-tested system for managing parallel extraction pipelines. Ready for production use.
