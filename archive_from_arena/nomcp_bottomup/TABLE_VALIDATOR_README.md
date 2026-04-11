**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# TableValidator: Table Quality Assessment Component

## Quick Start

```python
from table_validator import TableValidator

validator = TableValidator()
result = validator.validate(rows, table_name="FFO-1")

print(f"Valid: {result['is_valid']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Issues: {result['structural_issues']}")
```

## What It Does

The `TableValidator` checks extracted Treasury Bulletin tables for:

1. **Structural Integrity** - All rows have consistent columns, no malformed data
2. **Completeness** - Minimum row count, required columns present, acceptable null rates
3. **Data Quality** - Numeric columns are present, values in reasonable ranges, no suspicious patterns

**Output**: A confidence score (0.0-1.0) and detailed metrics to decide whether to use the table.

## Key Features

- **Comprehensive validation** across three dimensions (structure, completeness, quality)
- **Detailed metrics** including null percentages, numeric column detection, value ranges
- **Confidence scoring** (0.0-1.0) that reflects overall table quality
- **Actionable recommendations** for why a table passed or failed
- **Integration-ready** works seamlessly with TableFinder and DataNormalizer
- **Tested** with 10-test suite covering edge cases and real Treasury data

## Files

| File | Purpose |
|------|---------|
| `table_validator.py` | Core TableValidator class |
| `test_table_validator.py` | Comprehensive test suite (10 tests, all passing) |
| `example_validator_integration.py` | Integration examples and usage patterns |
| `TABLE_VALIDATOR_GUIDE.md` | Detailed documentation and architecture |

## Validation Thresholds

| Check | Threshold | Configurable |
|-------|-----------|--------------|
| Minimum rows | 5 | ✓ |
| Minimum complete rows % | 50% | ✓ |
| Maximum null % overall | 50% | ✓ |
| Maximum null % per column | 70% | ✓ |
| Minimum numeric columns | 1 | ✓ |

## Output Example

```python
{
    'is_valid': True,
    'confidence': 1.00,
    'table_name': 'FFO-1',
    'row_count': 20,
    'column_count': 12,
    'structural_issues': [],
    'completeness_issues': [],
    'quality_metrics': {
        'null_percentage': 0.0,
        'complete_rows': 20,
        'complete_rows_pct': 100.0,
        'numeric_columns': 11,
        'data_range_check': 'PASS',
        'all_zeros': False,
        'value_ranges': {
            'total_receipts': {'min': 760375.0, 'max': 1054260.0, 'mean': 920000.0}
        }
    },
    'recommendations': [
        'Excellent completeness: 100% of rows have all values',
        'Table spans multiple years (20 rows): Good for time-series analysis'
    ]
}
```

## Usage Patterns

### Pattern 1: Basic Validation
```python
result = validator.validate(rows, "FFO-1")
if result['is_valid']:
    use_table(rows)
```

### Pattern 2: Quality-Gated Pipeline
```python
result = validator.validate(rows, "FFO-1")
if result['confidence'] > 0.7:
    use_table(rows)
elif result['confidence'] > 0.5:
    log_warning(f"Low confidence: {result['recommendations']}")
    use_table(rows)  # Use with caution
else:
    skip_table()  # Try alternative source
```

### Pattern 3: Integration with Decomposition
```python
from table_validator import TableValidator
from example_validator_integration import ValidatingTableExtractor

extractor = ValidatingTableExtractor(min_confidence=0.5)
result = extractor.extract_and_validate(filepath, table_id)

if result['success']:
    # Row data already extracted, normalized, and validated
    analyze_table(result['rows'])
```

## Confidence Interpretation

| Confidence | Meaning | Action |
|-----------|---------|--------|
| 0.9-1.0 | Excellent | Use without hesitation |
| 0.7-0.9 | Good | Use normally |
| 0.5-0.7 | Acceptable | Use but log warning |
| 0.2-0.5 | Poor | Use only if necessary, apply extra caution |
| 0.0-0.2 | Invalid | Do not use, try alternative |

## Testing

All 10 test cases pass:

```bash
python3 nomcp/bottomup/test_table_validator.py
```

Tests cover:
- Empty tables
- Single-row tables
- Well-formed tables
- Inconsistent structures
- Missing values
- No numeric columns
- All-zero values
- Required column validation
- Mixed numeric/text quality
- Real Treasury Bulletin data

## Integration Example

See `example_validator_integration.py` for complete working examples:

```bash
python3 nomcp/bottomup/example_validator_integration.py
```

## Performance

- **Time Complexity**: O(n*m) - single pass through rows and columns
- **Space Complexity**: O(m) - metrics stored per column
- **Suitable for**: Tables up to 10,000+ rows

## Customization

```python
validator = TableValidator()
validator.min_rows = 3              # Lower minimum row count
validator.max_null_pct = 40         # Stricter null threshold
validator.min_complete_pct = 60     # More complete rows required
```

## Edge Cases Handled

✓ Empty tables
✓ Single-row tables
✓ Inconsistent column counts
✓ High null percentages
✓ No numeric columns
✓ All-zero values
✓ Empty headers
✓ Mixed numeric/text data
✓ Various numeric formats ($, %, negative, thousands separators)

## Architecture

```
TableValidator
├── check_structural_integrity()  → Validates column consistency
├── check_completeness()          → Validates row count, required cols, nulls
├── check_data_quality()          → Analyzes numeric cols, ranges, patterns
├── score_table_quality()         → Calculates confidence (0.0-1.0)
└── _generate_recommendations()   → Creates actionable feedback
```

## Design Philosophy

- **Not over-optimistic**: Messy data triggers warnings/failures
- **Not over-pessimistic**: Accepts tables with known limitations
- **Transparent**: All issues listed, reasoning clear
- **Actionable**: Recommendations guide next steps
- **Production-ready**: Handles Treasury Bulletin complexity

## Related Components

- **TableFinder** (`table_finder.py`) - Locates tables in raw text
- **DataNormalizer** (`data_normalizer.py`) - Cleans and parses values
- **TableValidator** (`table_validator.py`) - This component
- **ValidatingTableExtractor** (`example_validator_integration.py`) - End-to-end pipeline

## References

- Full documentation: `TABLE_VALIDATOR_GUIDE.md`
- Integration examples: `example_validator_integration.py`
- Test suite: `test_table_validator.py`
