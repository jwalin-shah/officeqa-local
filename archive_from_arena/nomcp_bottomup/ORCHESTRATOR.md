**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# Orchestrator: Multi-Piece Extraction Coordinator

## Overview

The **Orchestrator** is the top-level coordinator for the extraction pipeline. It manages the parallel processing of multiple decomposition pieces and combines results using computation rules.

**Purpose**: Replace the manual parallelization logic in `solve()` with a reusable, testable orchestration framework.

**Key Innovation**: Cleanly separate orchestration logic from search/extract logic, enabling:
- Better composability
- Easier testing
- Flexible computation rules
- Transparent confidence calculation
- Detailed step logging

## Architecture

```
Question
    ↓
Decomposer (Phase 1)
    ↓ Returns: [piece_1, piece_2, ..., piece_N]
    ↓
Orchestrator.execute(decomposition)
    ├─ Validate pieces (unique IDs, required fields)
    ├─ Parallel process each piece:
    │  ├─ Piece 1 → SearchTask → {value: 1350576.0, confidence: 0.99}
    │  ├─ Piece 2 → SearchTask → {value: 1413156.0, confidence: 0.98}
    │  └─ Piece N → SearchTask → {value: X, confidence: Y}
    ├─ Collect results with timeouts
    ├─ Apply computation rule (sum, average, difference, etc.)
    ├─ Calculate final confidence (pessimistic: min of pieces)
    └─ Return final answer with transparency
    ↓
Output
{
  'final_value': 2763732.0,
  'final_confidence': 0.98,
  'success': True,
  'piece_results': [...],
  'steps': [...]
}
```

## Usage

### Basic Usage

```python
from orchestrator import SearchTaskOrchestrator

def search_and_extract(piece):
    """Process one decomposition piece."""
    # ... search tables, extract value ...
    return {
        'value': 1350576.0,
        'confidence': 0.99,
        'success': True,
        'source': 'treasury_bulletin_1995_12.txt/FFO-1'
    }

# Prepare decomposition
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
orch = SearchTaskOrchestrator(search_and_extract, max_workers=4)
result = orch.execute(decomposition)

print(f"Final value: {result['final_value']}")
print(f"Confidence: {result['final_confidence']}")
```

### Convenience Function

```python
from orchestrator import execute_decomposition

result = execute_decomposition(
    decomposition,
    search_task_processor=search_and_extract,
    max_workers=4
)
```

### Custom Orchestrator

Subclass `Orchestrator` to customize piece processing:

```python
from orchestrator import Orchestrator

class MyOrchestrator(Orchestrator):
    def _process_piece(self, piece):
        # Custom logic here
        ...
        return {
            'value': 1000.0,
            'confidence': 0.95,
            'success': True
        }

orch = MyOrchestrator(max_workers=4)
result = orch.execute(decomposition)
```

## Input Format

### Decomposition Dict

```python
{
    'question_id': 123,                    # Unique question identifier
    'question': 'What was total...',       # Original question text
    'pieces': [                            # List of sub-problems
        {
            'piece_id': 1,                 # Unique within decomposition
            'description': 'FY 1995...',   # What this piece extracts
            'year': 1995,                  # Custom fields per pipeline
            'table': 'FFO-1',
            'search_terms': ['receipts'],
            'column_hint': 'Total',
            # ... any other fields needed by processor
        },
        {
            'piece_id': 2,
            'description': 'FY 1996...',
            'year': 1996,
            # ...
        }
    ],
    'computation': 'sum'                   # How to combine pieces
}
```

### Piece Structure

Each piece must have:
- `piece_id`: Unique integer identifier
- Other fields as needed by the processor

Recommended fields:
- `description`: What this piece extracts
- `year`: Data year
- `table`: Expected table name
- `search_terms`: Search keywords
- `column_hint`: Expected column name
- `value_type`: 'single' or 'monthly_series'

## Output Format

### Success Case

```python
{
    'question_id': 123,
    'final_value': 2763732.0,
    'final_confidence': 0.98,
    'success': True,
    'computation': 'sum',
    'piece_results': [
        {
            'piece_id': 1,
            'value': 1350576.0,
            'confidence': 0.99,
            'success': True,
            'source': 'treasury_bulletin_1995_12.txt/FFO-1',
            'notes': 'Extracted from FFO-1 table'
        },
        {
            'piece_id': 2,
            'value': 1413156.0,
            'confidence': 0.98,
            'success': True,
            'source': 'treasury_bulletin_1996_12.txt/FFO-1',
            'notes': 'Extracted from FFO-1 table'
        }
    ],
    'steps': [
        'Starting orchestrator for question 123',
        'Validation passed: 2 unique pieces',
        'Piece 1: value=1350576.0, confidence=0.99',
        'Piece 2: value=1413156.0, confidence=0.98',
        'Computation: sum([1350576.0, 1413156.0]) = 2763732.0, confidence=0.98',
        'Final result: 2763732.0 (confidence: 0.98)'
    ]
}
```

### Partial Success

If some pieces fail, Orchestrator still returns a value if at least one piece succeeded:

```python
{
    'question_id': 123,
    'final_value': 1350576.0,  # Only piece 1
    'final_confidence': 0.79,  # Penalized for failures
    'success': True,
    'piece_results': [
        {'piece_id': 1, 'value': 1350576.0, 'confidence': 0.99, 'success': True},
        {'piece_id': 2, 'value': None, 'confidence': 0.0, 'success': False, 'error': 'No tables found'}
    ]
}
```

### Complete Failure

If all pieces fail:

```python
{
    'question_id': 123,
    'final_value': None,
    'final_confidence': 0.0,
    'success': False,
    'error': 'All pieces returned None',
    'piece_results': [
        {'piece_id': 1, 'value': None, 'success': False, 'error': '...'},
        {'piece_id': 2, 'value': None, 'success': False, 'error': '...'}
    ]
}
```

## Computation Rules

### Supported Operations

| Rule | Behavior | Example |
|------|----------|---------|
| `direct` | Return first value | `[1995, 1996] → 1995` |
| `sum` | Add all values | `[1995, 1996] → 3991` |
| `average` | Mean of all values | `[1900, 2000] → 1950` |
| `difference` | First minus second | `[2000, 1000] → 1000` |
| `ratio` | First divided by second | `[2000, 1000] → 2.0` |
| `product` | Multiply all values | `[2, 3, 4] → 24` |
| `min` | Minimum value | `[1995, 1996] → 1995` |
| `max` | Maximum value | `[1995, 1996] → 1996` |

### Examples

```python
# Sum of two years
decomposition = {
    'pieces': [
        {'piece_id': 1, 'year': 1995},
        {'piece_id': 2, 'year': 1996}
    ],
    'computation': 'sum'
}
# Result: sum of piece values

# Difference between two quarters
decomposition = {
    'pieces': [
        {'piece_id': 1, 'quarter': 'Q1'},
        {'piece_id': 2, 'quarter': 'Q2'}
    ],
    'computation': 'difference'  # Q1 - Q2
}

# Ratio of two years
decomposition = {
    'pieces': [
        {'piece_id': 1, 'year': 1996},
        {'piece_id': 2, 'year': 1995}
    ],
    'computation': 'ratio'  # 1996 / 1995
}
```

## Confidence Calculation

The Orchestrator uses a **pessimistic** confidence strategy:

1. **Base confidence**: `min(piece_confidences)`
   - If piece 1 has 0.99 and piece 2 has 0.98, use 0.98

2. **Agreement bonus**: +0.1 if all values agree (within 1% relative tolerance)
   - If both pieces are very close, boost confidence

3. **Disagreement penalty**: -0.2 if less than 50% of pieces succeeded
   - If many pieces failed, reduce confidence in the final answer

### Examples

```
Case 1: Both pieces agree
  Piece 1: value=1000.0, confidence=0.99
  Piece 2: value=1005.0, confidence=0.95  (0.5% difference)
  → Final: confidence = min(0.99, 0.95) + 0.1 = 1.0

Case 2: Pieces disagree
  Piece 1: value=1000.0, confidence=0.99
  Piece 2: value=2000.0, confidence=0.90  (100% difference)
  → Final: confidence = min(0.99, 0.90) = 0.90

Case 3: One piece fails
  Piece 1: value=1000.0, confidence=0.99
  Piece 2: value=None, success=False
  → Final: confidence = 0.99 - 0.2 = 0.79  (penalized)
```

## Integration with solve_decompose

### Current Pipeline

```python
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
```

### Refactored with Orchestrator

```python
from orchestrator_integration import solve_with_orchestrator

def solve(question):
    return solve_with_orchestrator(
        question=question,
        decompose_fn=decompose,
        search_tables_fn=search_tables,
        select_table_fn=select_table,
        search_data_fn=search_data,
        extract_value_fn=extract_value,
        compute_fn=compute,
        verify_value_fn=verify_value,
        max_workers=4
    )
```

See `orchestrator_integration.py` for full integration guide.

## Testing

### Run Tests

```bash
cd nomcp/bottomup
python3 -m pytest test_orchestrator.py -v
```

### Test Coverage

- ✅ Validation (empty, missing fields, duplicates)
- ✅ All computation rules (direct, sum, average, difference, ratio, min, max)
- ✅ Confidence calculation (minimum, agreement bonus, disagreement penalty)
- ✅ Parallel execution timing (4 tasks with 2 workers)
- ✅ Error handling (all pieces fail, partial failures, timeouts)
- ✅ Step logging

### Example Test

```python
def test_execute_sum():
    orch = MockOrchestrator()
    decomp = {
        'question_id': 1,
        'pieces': [
            {'piece_id': 1, 'year': 1995},
            {'piece_id': 2, 'year': 1996}
        ],
        'computation': 'sum'
    }
    result = orch.execute(decomp)
    assert result['success']
    assert result['final_value'] == 3991.0
```

## Performance

### Parallelization

The Orchestrator uses `ThreadPoolExecutor` for efficient parallel processing:

```
Sequential: 4 tasks × 0.1s = 0.4s
Parallel (2 workers): 0.2s
Parallel (4 workers): 0.1s
```

### Timeout Handling

- Default: 180 seconds per piece
- Configurable: `Orchestrator(timeout_per_piece=300)`
- Failed pieces: Collected and reported, don't block others

## Error Handling

### Validation Errors

```python
# Invalid decomposition
decomp = {}  # Missing pieces
result = orch.execute(decomp)
assert not result['success']
assert 'No decomposition pieces' in result['error']
```

### Piece Processor Errors

```python
# Processor raises exception
def bad_processor(piece):
    raise ValueError("Oops")

orch = SearchTaskOrchestrator(bad_processor)
result = orch.execute(decomp)
# Piece is marked as failed, doesn't crash orchestrator
assert piece_result['success'] == False
assert 'Oops' in piece_result['error']
```

### Timeout Errors

```python
# Processor is too slow
def slow_processor(piece):
    time.sleep(10)
    return {...}

orch = SearchTaskOrchestrator(slow_processor, timeout_per_piece=1)
result = orch.execute(decomp)
# Piece is marked as timeout
assert 'Timeout' in piece_result['error']
```

## Advanced Usage

### Custom Processor with State

```python
class StatefulProcessor:
    def __init__(self):
        self.cache = {}

    def __call__(self, piece):
        piece_id = piece['piece_id']
        if piece_id in self.cache:
            return self.cache[piece_id]

        result = self._process(piece)
        self.cache[piece_id] = result
        return result

    def _process(self, piece):
        # Actual logic
        ...

processor = StatefulProcessor()
orch = SearchTaskOrchestrator(processor)
result = orch.execute(decomp)
```

### Multiple Computation Variants

```python
# Try sum first, fall back to average
decomp['computation'] = 'sum'
result = orch.execute(decomp)

if not result['success']:
    decomp['computation'] = 'average'
    result = orch.execute(decomp)
```

### Batch Processing

```python
questions = [...100 questions...]
results = []

for question in questions:
    plan = decompose(question)
    decomp = {
        'question_id': plan['question_id'],
        'pieces': convert_to_pieces(plan['sub_queries']),
        'computation': plan.get('computation', 'direct')
    }
    result = orch.execute(decomp)
    results.append(result)

# Aggregated metrics
success_rate = sum(1 for r in results if r['success']) / len(results)
avg_confidence = sum(r['final_confidence'] for r in results) / len(results)
```

## Logging

The Orchestrator logs to `stderr` with INFO level. Each step is recorded:

```
2026-04-04 01:49:50,785 - orchestrator - INFO - Starting orchestrator for question 123
2026-04-04 01:49:50,785 - orchestrator - INFO - Decomposition: 2 pieces
2026-04-04 01:49:50,785 - orchestrator - INFO - Validation passed: 2 unique pieces
2026-04-04 01:49:50,787 - orchestrator - INFO - Piece 1: value=1350576.0, confidence=0.99
2026-04-04 01:49:50,787 - orchestrator - INFO - Piece 2: value=1413156.0, confidence=0.98
2026-04-04 01:49:50,787 - orchestrator - INFO - Computation: sum([...]) = 2763732.0
2026-04-04 01:49:50,787 - orchestrator - INFO - Final result: 2763732.0 (confidence: 0.98)
```

All steps are also included in the result under `result['steps']` for programmatic access.

## FAQ

**Q: Why use `min(confidences)` instead of average?**
A: The pessimistic approach is safer. If we average, a low-confidence piece gets masked by high-confidence pieces. With min, we're conservative.

**Q: What if I have 10 pieces?**
A: Orchestrator handles any number. With 4 workers, 10 pieces take ~2.5x the time of a single piece (parallel pipelining).

**Q: Can I customize timeout per piece?**
A: Yes: `Orchestrator(timeout_per_piece=300)` for 5 minutes per piece.

**Q: What if a piece returns a list (e.g., monthly values)?**
A: The processor can return `{'value': [v1, v2, ...], ...}`. The Orchestrator treats it as a single value (doesn't decompose further).

**Q: How do I integrate with the existing pipeline?**
A: Use `orchestrator_integration.py`. It wraps `_process_subquery` logic in a processor function for the Orchestrator.

## Files

- `orchestrator.py` - Core Orchestrator class
- `test_orchestrator.py` - Comprehensive test suite (21 tests)
- `orchestrator_integration.py` - Integration guide and helpers for solve_decompose
- `ORCHESTRATOR.md` - This file

## Next Steps

1. Review `test_orchestrator.py` to understand usage patterns
2. Read `orchestrator_integration.py` for integration with solve_decompose
3. Optionally run `python3 orchestrator.py` to see a live example
4. Integrate into solve_decompose.py per integration guide
