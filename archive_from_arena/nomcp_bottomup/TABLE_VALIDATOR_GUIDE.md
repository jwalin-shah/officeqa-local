**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# Table Validator Guide

## Overview

The `TableValidator` class validates Treasury Bulletin tables for structural integrity, completeness, and data quality. It's designed to work with tables extracted from markdown-formatted Treasury Bulletin documents and normalized by the `DataNormalizer`.

## Purpose

The validator checks whether extracted tables:
- Have consistent structure (same columns per row)
- Have sufficient data (minimum row count, required columns)
- Maintain reasonable data quality (acceptable null rates, sensible values)
- Are suitable for further analysis or arithmetic operations

## Architecture

```
Treasury Bulletin Text
        ↓
    TableFinder (locate & extract raw table)
        ↓
    DataNormalizer (clean & normalize values)
        ↓
    TableValidator (validate structure & quality)
        ↓
    Validation Result (is_valid, confidence, metrics)
```

## Usage

### Basic Validation

```python
from table_validator import TableValidator

validator = TableValidator()

# Assuming you have rows from TableFinder/DataNormalizer
rows = [
    {'year': '2020', 'revenue': '1000', 'expense': '800'},
    {'year': '2021', 'revenue': '1100', 'expense': '850'},
    {'year': '2022', 'revenue': '1200', 'expense': '900'},
    # ... more rows
]

result = validator.validate(rows, "FFO-1")

print(f"Valid: {result['is_valid']}")
print(f"Confidence: {result['confidence']:.2f}")
print(f"Issues: {result['structural_issues']}")
```

### With Required Columns

```python
result = validator.validate(
    rows,
    table_name="RECEIPT-SUMMARY",
    required_columns=["fiscal_year", "total_receipts"]
)
```

### With Expected Row Count

```python
result = validator.validate(
    rows,
    table_name="FFO-1",
    expected_row_count=10
)
```

## Output Format

The validator returns a dictionary with the following structure:

```python
{
    'is_valid': True,              # Passes all critical checks
    'confidence': 0.92,            # Quality score (0.0-1.0)
    'table_name': 'FFO-1',         # Input table name
    'row_count': 20,               # Number of data rows
    'column_count': 12,            # Number of columns

    'structural_issues': [         # Column alignment, segmentation, etc.
        # Empty if structure is valid
    ],

    'completeness_issues': [       # Missing columns, too few rows, etc.
        # Empty if completeness is valid
    ],

    'quality_metrics': {           # Detailed metrics
        'null_percentage': 5.2,           # % of all cells missing
        'null_by_column': {               # % missing per column
            'column_name': 2.5,
            ...
        },
        'complete_rows': 19,              # Rows with all values
        'complete_rows_pct': 95.0,        # % of rows with all values
        'numeric_columns': 11,            # Count of numeric columns
        'numeric_values_count': 209,      # Count of numeric values
        'numeric_values_complete': 209,   # Numeric values that parsed
        'data_range_check': 'PASS',       # 'PASS', 'WARNING', 'FAIL'
        'all_zeros': False,               # All numeric values are 0
        'value_ranges': {                 # Min/max/mean per numeric col
            'column_name': {
                'min': 0.0,
                'max': 1000.0,
                'mean': 500.0
            },
            ...
        }
    },

    'recommendations': [           # Actionable feedback
        'Excellent completeness: 95% of rows have all values',
        'Table spans multiple years (20 rows): Good for time-series analysis'
    ]
}
```

## Validation Rules

### PASS (valid=True)
Table passes validation if:
- ✓ All rows have same column count
- ✓ At least 5 data rows (configurable)
- ✓ < 50% missing values overall (configurable)
- ✓ At least 1 numeric column
- ✓ Column headers are reasonable (non-empty, have alphanumeric chars)
- ✓ No completeness issues (required columns present, etc.)

### WARN (valid=True, lower confidence)
Table passes but with warnings:
- ⚠ 30-50% missing values in some columns
- ⚠ Between 5-10 rows only
- ⚠ < 70% of rows have complete data
- ⚠ Some numeric columns entirely missing values

### FAIL (valid=False)
Table fails validation if:
- ✗ Rows have different column counts
- ✗ Fewer than 5 rows
- ✗ > 50% missing values
- ✗ No numeric columns found
- ✗ All numeric values are zero (likely data issue)
- ✗ Missing required columns

## Confidence Scoring

Confidence is a 0.0-1.0 score reflecting overall table quality:

- **1.0**: Perfect table (all checks pass, high data quality)
- **0.8-0.9**: Good table (minor issues, still reliable)
- **0.5-0.7**: Acceptable table (some concerns, use with caution)
- **0.2-0.4**: Poor table (multiple issues, limited usefulness)
- **0.0-0.1**: Invalid table (critical failures)

Scoring logic:
- Start at 1.0
- Structural issues: -0.3 each (max -1.0)
- Completeness issues: -0.2 each (max -0.5)
- Quality penalties:
  - Null % > 50%: -0.3
  - Null % 30-50%: -0.15
  - Null % 10-30%: -0.05
  - Complete rows < 30%: -0.2
  - Complete rows 30-60%: -0.1
  - Data range FAIL: -0.3
  - Data range WARNING: -0.1
- Quality bonuses:
  - Null % < 5% AND complete rows > 90%: +0.1

## Numeric Column Detection

Columns are classified as numeric if:
- >70% of non-null values can be parsed as numbers
- Includes currency symbols ($1,000), percentages (95%), negatives (-100)
- Properly handles thousand separators and formatting

## Integration with Decomposition Pipeline

The TableValidator is designed to work with the decomposition/search pipeline:

```python
# In decomposition or search step
from table_finder import TableFinder
from data_normalizer import DataNormalizer
from table_validator import TableValidator

finder = TableFinder()
normalizer = DataNormalizer()
validator = TableValidator()

# Search for a table
content, meta = finder.find_table(filepath, table_id)

if content:
    # Extract and normalize rows
    rows = finder.extract_rows(content)
    cleaned_rows, clean_meta = normalizer.clean_rows(rows)

    # Validate quality
    validation_result = validator.validate(cleaned_rows, table_id)

    if validation_result['is_valid']:
        # Table is valid, proceed with analysis
        use_table_for_analysis(cleaned_rows)
    elif validation_result['confidence'] > 0.5:
        # Table is marginal, log warning and proceed with caution
        log_warning(f"Marginal table quality: {validation_result['recommendations']}")
        use_table_for_analysis(cleaned_rows)
    else:
        # Table is invalid, reject
        log_error(f"Invalid table: {validation_result['structural_issues']}")
        # Try alternative source or skip question
```

## Customization

You can adjust validation thresholds:

```python
validator = TableValidator()
validator.min_rows = 3                    # Minimum rows (default: 5)
validator.min_complete_pct = 60           # Min complete rows % (default: 50)
validator.max_null_pct = 40               # Max overall null % (default: 50)
validator.max_null_pct_column = 60        # Max column null % (default: 70)
```

## Examples

### Example 1: High-Quality Treasury Table

```
Input: 20 rows, 12 columns, 0% missing, all numeric values present
Output:
  is_valid: True
  confidence: 1.00
  structural_issues: []
  completeness_issues: []
  complete_rows_pct: 100.0
  Recommendations:
    - Excellent completeness: 100% of rows have all values
    - Table spans multiple years (20 rows): Good for time-series analysis
```

### Example 2: Small Table with Sparse Data

```
Input: 3 rows, 4 columns, 35% missing values
Output:
  is_valid: False
  confidence: 0.65
  structural_issues: []
  completeness_issues:
    - Too few rows: 3 < 5
  complete_rows_pct: 66.7
  quality_metrics:
    null_percentage: 35.0
  Recommendations:
    - High missing data (35.0%): Consider filtering out sparse columns
```

### Example 3: Inconsistent Structure

```
Input: Rows with different column counts
Output:
  is_valid: False
  confidence: 0.0
  structural_issues:
    - Row 2 has 3 columns, expected 4
  completeness_issues: []
```

## Edge Cases Handled

- **Empty tables**: Returns valid=False, confidence=0.0
- **Single row tables**: Too few rows, triggers completeness issue
- **All missing values**: High null%, completeness issues
- **No numeric columns**: Text-only tables marked invalid
- **All zeros**: Data range check fails
- **Inconsistent columns**: Structural issue caught
- **Empty headers**: Structural issue caught
- **Nested/segmented rows**: Not explicitly handled by validator (should be handled by TableFinder/DataNormalizer)

## Performance

- Linear time complexity O(n*m) where n=rows, m=columns
- Single pass through data for most metrics
- Suitable for large tables (1000+ rows)

## Testing

Run the comprehensive test suite:

```bash
python3 nomcp/bottomup/test_table_validator.py
```

Tests cover:
- Empty tables
- Single-row tables (below minimum)
- Well-formed tables
- Inconsistent column structures
- Missing values and sparsity
- No numeric columns (text-only)
- All-zero numeric values
- Required column validation
- Mixed numeric/text quality
- Real Treasury Bulletin data

All tests should pass with the current implementation.
