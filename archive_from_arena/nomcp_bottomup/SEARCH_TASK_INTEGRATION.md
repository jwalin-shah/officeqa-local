**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# SearchTask Integration Guide

## Overview

`SearchTask` is the primary orchestration layer that integrates all extraction components into a unified pipeline. It processes individual decomposition pieces and returns fully extracted values with confidence scores and source attribution.

## Component Integration Map

```
SearchTask.execute()
├─ FileLocator.locate_file()
│  └─ Returns: file path + metadata
├─ For each file:
│  ├─ TableFinder.find_table()
│  │  └─ TableFinder.extract_rows()
│  ├─ DataNormalizer.clean_rows()
│  │  ├─ .normalize_row_label()
│  │  └─ .normalize_numeric_value()
│  ├─ TableValidator.validate()
│  │  ├─ .check_structural_integrity()
│  │  ├─ .check_completeness()
│  │  └─ .check_data_quality()
│  └─ For each subsearch strategy (A, B, C in parallel):
│     ├─ RowMatcher.find_row_*()
│     │  ├─ .find_row_strict()
│     │  ├─ .find_row_fuzzy()
│     │  └─ .find_row_contextual()
│     ├─ ColumnFinder.find_column()
│     └─ ValueParser.extract_value()
├─ ConsensusVoter.vote()
│  └─ Returns: consensus result
└─ Return final result
```

## Data Flow

### Input
```python
{
    'piece_id': 1,
    'year': 1995,
    'period_type': 'fiscal',
    'table_id': 'FFO-1',
    'row_identifier': ['Total', 'Receipts'],
    'column_identifier': 'Total receipts',
    'prefer_months': [12, 6],
    'row_index': None
}
```

### Processing Steps

1. **Validation**: Check all required fields present
2. **File Location**: Find file(s) for the given year
3. **Table Extraction**:
   - Find table in file
   - Extract rows
   - Normalize data
   - Validate table structure
4. **Parallel Subsearches** (A, B, C):
   - Find matching row
   - Find matching column
   - Extract and parse value
5. **Consensus Voting**: Select best result
6. **Output**: Return result with metadata

### Output
```python
{
    'piece_id': 1,
    'value': 1350576.0,
    'confidence': 0.95,
    'success': True,
    'source': {
        'file': 'treasury_bulletin_1995_12.txt',
        'table': 'FFO-1',
        'row_identifier': ['Total', 'Receipts'],
        'column': 'Total > Receipts',
        'method': 'strict',
        'agreement_level': 'full'
    },
    'steps': [...],
    'file_results': [...],
    'voting_details': {...},
    'errors': []
}
```

## Subsearch Strategy Details

### Strategy A: Strict Matching
```python
def _subsearch_strict(rows, row_id, col_id):
    row = RowMatcher.find_row_strict(rows, row_id)      # Exact match
    col_name, col_conf = ColumnFinder.find_column(rows, col_id)
    value = ValueParser.extract_value(row, col_id)
    confidence = min(0.99, col_conf, value_conf)
```

**Confidence**: High (0.95+) when all match exactly
**Coverage**: Narrow, only unambiguous cases
**Use when**: Row/column identifiers are precise

### Strategy B: Fuzzy Matching
```python
def _subsearch_fuzzy(rows, row_id, col_id):
    row, row_conf = RowMatcher.find_row_fuzzy(rows, row_id)  # Keywords
    col_name, col_conf = ColumnFinder.find_column(rows, col_id)
    value = ValueParser.extract_value(row, col_id)
    confidence = min(row_conf, col_conf, value_conf) * 0.95
```

**Confidence**: Moderate (0.70-0.90)
**Coverage**: Broader, handles variations
**Use when**: Identifiers may have typos or formatting variations

### Strategy C: Contextual Matching
```python
def _subsearch_contextual(rows, row_id, col_id, row_index=None):
    row, row_conf = RowMatcher.find_row_contextual(rows, row_id, row_index)
    col_name, col_conf = ColumnFinder.find_column(rows, col_id)
    value = ValueParser.extract_value(row, col_id)
    confidence = min(row_conf, col_conf, value_conf) * 0.90
```

**Confidence**: Moderate (0.70-0.85)
**Coverage**: Good for complex structures
**Use when**: Table structure knowledge available

## Voting Mechanism

```
Strategy Results:
  Strict:      1350576.0 (confidence: 0.95)
  Fuzzy:       1350576.0 (confidence: 0.92)
  Contextual:  1350576.0 (confidence: 0.88)

Agreement Analysis:
  - All 3 agree on 1350576.0
  - Agreement group size: 3/3 = 100%
  - Agreement level: 'full'

Final Decision:
  - Winning value: 1350576.0
  - Base confidence: 0.95 (from Strict)
  - Boost applied: +0.0 (already high)
  - Final confidence: 0.95
```

Voting logic:
1. Group by value (which strategies agree?)
2. Find largest group
3. Select value from largest group
4. Use confidence from best method in group
5. Apply agreement bonus

## Multi-File Handling

When year is ambiguous (multiple files available):

```
SearchTask.execute(piece)
├─ locate_file(year=1995) → candidates = [june, sept, dec]
├─ Process june:
│  └─ Result: value=1350576, confidence=0.92
├─ Process sept:
│  └─ Result: value=1350576, confidence=0.90
├─ Process dec:
│  └─ Result: value=1350576, confidence=0.95
└─ Return best: value=1350576, confidence=0.95, source=dec
```

Each file is processed independently, best result returned.

## Error Handling Strategy

| Scenario | Result | Confidence | Notes |
|----------|--------|-----------|-------|
| Missing field | Fail early | 0.0 | Input validation catches |
| File not found | Try all candidates | 0.0 | Returns None if all fail |
| Table not found | Try next file | 0.0 | Skipped, not fatal |
| Bad table structure | Try next file | 0.0 | Validation catches |
| Row not found | Try next file | 0.0 | All subsearch fails |
| Column not found | Try next file | 0.0 | All subsearch fails |
| Value parsing fails | Try next file | 0.0 | No valid value |
| All strategies fail | Use any valid | 0.0+ | Fallback to partial match |

## Usage Patterns

### Pattern 1: Simple Case
```python
task = SearchTask()
piece = {
    'piece_id': 1,
    'year': 1995,
    'table_id': 'FFO-1',
    'row_identifier': ['Total', 'Receipts'],
    'column_identifier': 'Total receipts'
}
result = task.execute(piece)
assert result['success']
value = result['value']
confidence = result['confidence']
```

### Pattern 2: With Month Preference
```python
piece = {
    'piece_id': 2,
    'year': 1995,
    'table_id': 'FFO-1',
    'row_identifier': ['Total'],
    'column_identifier': 'receipts',
    'prefer_months': [12, 6, 3]  # Try Dec first, then Jun, then Mar
}
result = task.execute(piece)
```

### Pattern 3: With Row Position Hint
```python
piece = {
    'piece_id': 3,
    'year': 1995,
    'table_id': 'FFO-1',
    'row_identifier': ['Total'],
    'column_identifier': 'receipts',
    'row_index': 5  # Expected at row 5 in table
}
result = task.execute(piece)
```

### Pattern 4: Verbose Mode for Debugging
```python
result = task.execute(piece, verbose=True)
# Prints step-by-step progress
```

## Configuration & Tuning

### Confidence Thresholds
Edit in subsearch methods:
```python
# _subsearch_strict
confidence = min(row_conf, col_conf, value_conf)  # No penalty

# _subsearch_fuzzy
confidence = min(...) * 0.95  # 5% penalty for fuzzy

# _subsearch_contextual
confidence = min(...) * 0.90  # 10% penalty for contextual
```

### Table Validation Thresholds
In `TableValidator`:
```python
MIN_ROWS = 5                    # Minimum rows to accept table
MIN_COMPLETE_ROWS_PCT = 50      # % rows with all columns
MAX_NULL_PCT_OVERALL = 50       # Max missing values
MAX_NULL_PCT_COLUMN = 70        # Max per-column missing
MIN_NUMERIC_COLUMNS = 1         # Need at least 1 numeric column
```

### Row/Column Matching Thresholds
In `ColumnFinder.score_column_match()`:
```python
0.99  # Exact match
0.85  # All keywords present
0.65  # >50% keywords
0.60  # Some keywords present
0.00  # No match
```

## Performance Characteristics

- **Single file**: 100-300ms
- **Multiple files (3)**: 200-900ms
- **Verbose mode**: Same (logging overhead minimal)

Bottleneck: File I/O and table extraction

## Testing

Run test suite:
```bash
cd /Users/jwalinshah/projects/officeqa-arena/nomcp/bottomup
python3 test_search_task.py
```

Tests cover:
- Input validation (missing fields)
- File handling (invalid year, ambiguous months)
- Result structure and types
- Error handling (graceful failures)
- Confidence bounds (0.0-1.0)

## Debugging Tips

### Enable Verbose Mode
```python
result = task.execute(piece, verbose=True)
```

Shows:
- File discovery
- Table location
- Row/column matching
- Subsearch results
- Consensus voting

### Inspect Voting Details
```python
voting = result['voting_details']
print(voting['strict_result'])
print(voting['fuzzy_result'])
print(voting['contextual_result'])
print(voting['consensus'])
```

### Check File Results
```python
for file_result in result['file_results']:
    print(f"{file_result['file']}: {file_result['value']} (conf: {file_result['confidence']:.2f})")
```

### Trace Steps
```python
for step in result['steps']:
    print(step)
```

## Future Enhancements

1. **Parallel file processing**: Process multiple files concurrently
2. **Year ranges**: Support "1995-1997" decomposition pieces
3. **Period types**: Handle calendar vs fiscal year distinction
4. **Caching**: Cache table extractions for repeated access
5. **Retry logic**: Automatic retry with fallback strategies
6. **ML-based voting**: Learn optimal strategy weights from data

## Related Documentation

- [SearchTask README](SEARCH_TASK_README.md) - User guide
- [ConsensusVoter](CONSENSUS_VOTER.md) - Voting logic details
- [TableValidator Guide](TABLE_VALIDATOR_GUIDE.md) - Validation logic
- [ColumnFinder](column_finder.py) - Column matching code
- [RowMatcher](row_matcher.py) - Row matching code
