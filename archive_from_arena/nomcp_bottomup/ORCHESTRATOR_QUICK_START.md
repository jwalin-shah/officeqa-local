**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# Orchestrator Quick Start

## 30-Second Overview

The **Orchestrator** parallelizes the processing of multiple decomposition pieces and combines results using computation rules.

```python
from orchestrator import SearchTaskOrchestrator

def my_processor(piece):
    # Search and extract value from piece
    return {
        'value': 1000.0,
        'confidence': 0.95,
        'success': True
    }

decomposition = {
    'question_id': 123,
    'pieces': [
        {'piece_id': 1, 'year': 1995},
        {'piece_id': 2, 'year': 1996}
    ],
    'computation': 'sum'  # Add pieces together
}

orch = SearchTaskOrchestrator(my_processor)
result = orch.execute(decomposition)
# result['final_value'] = sum of extracted values
# result['final_confidence'] = confidence in result
```

## Files

| File | Purpose |
|------|---------|
| `orchestrator.py` | Core implementation (3 classes, 8 methods) |
| `test_orchestrator.py` | 21 comprehensive tests (all pass) |
| `orchestrator_integration.py` | Integration guide for solve_decompose |
| `ORCHESTRATOR.md` | Full documentation |

## Key Classes

### Orchestrator
Base class. Override `_process_piece()` for custom logic.

### SearchTaskOrchestrator
Wrapper that uses a processor function. Most common choice.

### ProcessPieceAdapter
Helper to wrap existing pipeline functions. Used for integration.

## Common Patterns

### Pattern 1: Simple Processor Function

```python
def search_and_extract(piece):
    # ... search tables, extract value ...
    return {'value': 1000.0, 'confidence': 0.95, 'success': True}

orch = SearchTaskOrchestrator(search_and_extract)
result = orch.execute(decomposition)
```

### Pattern 2: Custom Orchestrator Class

```python
class MyOrchestrator(Orchestrator):
    def _process_piece(self, piece):
        # Custom logic
        return {'value': ..., 'confidence': ..., 'success': ...}

orch = MyOrchestrator()
result = orch.execute(decomposition)
```

### Pattern 3: Integrate with Existing Pipeline

```python
from orchestrator_integration import solve_with_orchestrator

result = solve_with_orchestrator(
    question=question,
    decompose_fn=decompose,
    search_tables_fn=search_tables,
    select_table_fn=select_table,
    search_data_fn=search_data,
    extract_value_fn=extract_value,
    compute_fn=compute,
    max_workers=4
)
```

## Computation Rules

```
'direct'      → Return first value
'sum'         → Add all values
'average'     → Mean of values
'difference'  → First - Second
'ratio'       → First / Second
'product'     → Multiply all
'min'         → Minimum
'max'         → Maximum
```

## Input Format

```python
{
    'question_id': 123,
    'question': 'What was...',
    'pieces': [
        {
            'piece_id': 1,           # Required: unique ID
            'description': 'FY 1995',
            'year': 1995,            # Custom fields
            'table': 'FFO-1',        # as needed
            # ... any other fields for processor
        }
    ],
    'computation': 'sum'             # How to combine pieces
}
```

## Output Format

```python
{
    'final_value': 2763732.0,
    'final_confidence': 0.98,        # min(piece_confidences)
    'success': True,
    'computation': 'sum',
    'piece_results': [
        {
            'piece_id': 1,
            'value': 1350576.0,
            'confidence': 0.99,
            'success': True
        },
        # ... more pieces
    ],
    'steps': [
        'Starting orchestrator for question 123',
        'Piece 1: value=1350576.0, confidence=0.99',
        # ... more steps
    ]
}
```

## Configuration

```python
# Defaults: 4 workers, 180 second timeout per piece
orch = SearchTaskOrchestrator(processor)

# Custom settings
orch = SearchTaskOrchestrator(
    processor,
    max_workers=8,                  # More parallelism
    timeout_per_piece=300           # 5 minutes per piece
)
```

## Confidence Strategy

**Pessimistic approach**: Use minimum confidence from all pieces.

- `Base`: `min(piece_confidences)`
- `+0.1 bonus`: If pieces agree (within 1%)
- `-0.2 penalty`: If many pieces fail

Example:
```
Piece 1: 1350576.0, confidence 0.99
Piece 2: 1413156.0, confidence 0.98
→ final_confidence = min(0.99, 0.98) = 0.98
→ values agree (0.02% difference) → +0.1
→ final_confidence = 1.0 (capped)
```

## Testing

```bash
python3 -m pytest test_orchestrator.py -v

# All 21 tests pass in 0.23 seconds
# Coverage: validation, all computation rules, confidence, parallelization, errors
```

## Performance

- **Parallelization**: 4 tasks with 2 workers takes ~50% of sequential time
- **Timeout**: Slow pieces don't block others, marked as failed
- **Memory**: Minimal (stores piece results in memory)

## Error Handling

All errors are graceful:

```
Validation error     → return error result
Processor exception  → piece marked as failed
Timeout             → piece marked as failed
All pieces fail     → return None with error message
Some pieces fail    → return value from successful pieces
```

## Integration Steps

1. **Review**: Read `ORCHESTRATOR.md` for full documentation
2. **Test**: Run `test_orchestrator.py` to see it in action
3. **Choose**: Pick one of the patterns above
4. **Implement**: Create processor or subclass Orchestrator
5. **Test**: Call `orch.execute(decomposition)` with your data

## Example: Real Integration

```python
# Before: solve_decompose.py
def solve(question):
    plan = decompose(question)
    extracted = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(_process_subquery, sq): sq['id'] for sq in plan['sub_queries']}
        for future in concurrent.futures.as_completed(futures):
            sq_id, result, _ = future.result()
            extracted[sq_id] = result
    answer = compute(plan, extracted)
    return answer

# After: with Orchestrator
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

## Debugging Tips

1. **Check steps**: `result['steps']` shows exactly what happened
2. **Check piece_results**: See success/failure for each piece
3. **Check final_confidence**: If too low, look at piece confidences
4. **Enable logging**: Already logged to stderr at INFO level
5. **Test processor**: Mock processor with hardcoded values first

## FAQ

**Q: Why not use async/await?**
A: ThreadPoolExecutor works with any function. Async adds complexity.

**Q: Can I cancel running pieces?**
A: Not directly. Use timeout_per_piece to limit individual piece time.

**Q: How many workers should I use?**
A: 2-4 typically. More workers = more parallelism but more contention.

**Q: Can pieces depend on each other?**
A: No. Orchestrator assumes pieces are independent. Use custom subclass if you need dependencies.

**Q: What's the max decomposition size?**
A: Tested with 4 pieces. Should work with 100+. Memory is the limit.

## Next Steps

- Run `python3 test_orchestrator.py` to see examples
- Read `ORCHESTRATOR.md` for comprehensive documentation
- Follow `orchestrator_integration.py` for solve_decompose integration
- Implement your processor using the pattern above
