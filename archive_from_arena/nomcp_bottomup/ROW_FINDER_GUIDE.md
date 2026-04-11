**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# RowFinder Component Guide

## Overview

`RowFinder` locates rows in Treasury Bulletin tables by time period (fiscal year, calendar year, or specific month). It complements `RowMatcher`, which finds rows by label/name.

### When to Use Each Component

| Component | Purpose | Input | Example |
|-----------|---------|-------|---------|
| **RowMatcher** | Find rows by label/name | Row label like "National defense" | `find_row_fuzzy(rows, ["National defense"])` |
| **RowFinder** | Find rows by time period | Year, month, period type | `find_row_by_period(rows, year=1995, period_type="fiscal")` |

## Core Features

### 1. Find Row by Period
```python
from row_finder import RowFinder

finder = RowFinder()

# Find fiscal year 1995
row, idx, conf = finder.find_row_by_period(
    rows=rows,
    year=1995,
    period_type="fiscal"
)
# Returns: (row_dict, row_index, confidence_0_to_1)

# Find specific month
row, idx, conf = finder.find_row_by_period(
    rows=rows,
    year=1995,
    period_type="calendar",
    month=3  # March
)
```

### 2. Score Period Matches
```python
# Manually score a row's period label
score = finder.score_period_match(
    row_label="Jan. 1995",
    target_year=1995,
    period_type="calendar",
    month=1
)
# Returns: confidence score 0.0-1.0
```

### 3. Parse Period Strings
```python
# Parse a period identifier
parsed = finder.parse_period_string("FY1995")
# Returns: {
#     'year': 1995,
#     'month': None,
#     'period_type': 'fiscal',
#     'confidence': 0.99,
#     'original': 'FY1995'
# }
```

### 4. Find All Candidates
```python
# Get all matching rows, ranked by confidence
candidates = finder.find_all_candidate_rows(
    rows=rows,
    year=1995,
    period_type="fiscal"
)
# Returns: [(row_dict, index, confidence), ...]
# Sorted by confidence descending
```

## Supported Period Formats

### Annual Data
| Format | Example | Parsed As |
|--------|---------|-----------|
| Explicit fiscal year | `"FY1995"`, `"FY 1995"` | fiscal year 1995 |
| Explicit calendar year | `"CY1995"`, `"CY 1995"` | calendar year 1995 |
| Fiscal prefix | `"Fiscal 1995"`, `"Fiscal-1995"` | fiscal year 1995 |
| Calendar prefix | `"Calendar 1995"`, `"Calendar-1995"` | calendar year 1995 |
| Year only | `"1995"` | ambiguous (could be either) |
| Fiscal encoding | `"19951"` | fiscal year 1995 |

### Monthly Data
| Format | Example | Parsed As |
|--------|---------|-----------|
| "Month. Year" | `"Jan. 1995"`, `"February 1995"` | January/February 1995 |
| "Year - Month" | `"1995 - Jan"`, `"1995 - December"` | January/December 1995 |
| Month only | `"January"`, `"Jan"` | Month (year context needed) |
| Month number | `"1"`, `"12"` | Month 1 or 12 |

### Supported Month Formats
- Full names: January, February, ..., December
- Abbreviations: Jan, Feb, ..., Dec
- Abbreviations with period: Jan., Feb., ..., Dec.
- Numbers: 1-12 or 01-12

## Confidence Scoring

Scores range from 0.0 to 1.0 and indicate match certainty:

| Score | Meaning | Example |
|-------|---------|---------|
| 0.99 | Exact match (explicit prefix) | `"FY1995"` for FY 1995 |
| 0.95 | High confidence match | `"Fiscal 1995"`, month + year |
| 0.90 | Good match | Month + year in correct format |
| 0.85 | Moderate match | Year only, ambiguous |
| 0.60 | Weak match | Month only, no year |
| 0.50 | Low confidence | Mismatched period type |
| 0.0 | No match | Year doesn't match or invalid |

## Usage Examples

### Example 1: Find a Fiscal Year

```python
from row_finder import RowFinder

finder = RowFinder()

rows = [
    {"Fiscal year or month": "1993", "Total receipts": "1154357"},
    {"Fiscal year or month": "1994", "Total receipts": "1258627"},
    {"Fiscal year or month": "1995", "Total receipts": "1350576"},
    {"Fiscal year or month": "1996", "Total receipts": "1413156"},
]

row, idx, conf = finder.find_row_by_period(rows, year=1995, period_type="fiscal")

# Output:
# row = {"Fiscal year or month": "1995", "Total receipts": "1350576"}
# idx = 2
# conf = 0.95
```

### Example 2: Find a Specific Month

```python
rows = [
    {"Period": "Jan. 1995", "Amount": 10000},
    {"Period": "Feb. 1995", "Amount": 11000},
    {"Period": "Mar. 1995", "Amount": 12000},
]

row, idx, conf = finder.find_row_by_period(
    rows,
    year=1995,
    month=3,
    period_type="calendar"
)

# Output:
# row = {"Period": "Mar. 1995", "Amount": 12000}
# idx = 2
# conf = 0.90
```

### Example 3: Get Best Candidates

```python
rows = [
    {"Period": "1995", "Value": 100},
    {"Period": "FY1995", "Value": 200},
    {"Period": "CY1995", "Value": 300},
]

candidates = finder.find_all_candidate_rows(
    rows,
    year=1995,
    period_type="fiscal"
)

# Output (sorted by confidence):
# [({"Period": "FY1995", "Value": 200}, 1, 0.99),  # best
#  ({"Period": "1995", "Value": 100}, 0, 0.95),    # good
#  ({"Period": "CY1995", "Value": 300}, 2, 0.50)]  # mismatch
```

### Example 4: Integrate with Other Components

```python
from row_finder import RowFinder
from row_matcher import RowMatcher
from data_normalizer import DataNormalizer

# 1. Get rows from table
rows = table_finder.extract_rows(table_content)

# 2. Clean the data
normalizer = DataNormalizer()
cleaned_rows, _ = normalizer.clean_rows(rows)

# 3. Find the right row by period
finder = RowFinder()
row, row_idx, period_conf = finder.find_row_by_period(
    cleaned_rows,
    year=1995,
    period_type="fiscal"
)

# 4. Find the right column by metric
matcher = RowMatcher()
from value_parser import ValueParser
parser = ValueParser()

result = parser.extract_value(row, "total receipts")
value = result['value']
value_conf = result['confidence']

# 5. Combine confidences
overall_conf = (period_conf + value_conf) / 2
```

## Period Type Definitions

### Fiscal Year (pre-1977)
July (previous year) through June (current year)
- Example: FY 1995 = July 1994 - June 1995

### Fiscal Year (post-1977)
October (previous year) through September (current year)
- Example: FY 1995 = October 1994 - September 1995

### Calendar Year
January through December of the same year
- Example: CY 1995 = January 1995 - December 1995

## Implementation Details

### Period Label Detection
The component tries to find the period identifier in this order:
1. Column named "Fiscal year or month" (or similar)
2. Column named "Period", "Date", "Year", "Month"
3. First column value (fallback)

### Parsing Strategy
1. Check for month + year formats first (most specific)
2. Check for explicit prefixes (FY, Fiscal, CY, Calendar)
3. Try year-only formats
4. Try fiscal year encoding (YYYYF format)

### Scoring Strategy
- Year must match (no match if year differs)
- Period type preference: exact > similar > unknown
- Month bonus: exact month match > year only
- Explicit prefix bonus: explicit > ambiguous

## Error Handling

### Returns None When:
- No rows provided
- Period column not found
- Year doesn't match
- Month doesn't match (when month is required)

### Returns Low Confidence (0.50) For:
- Period type mismatch (FY when CY expected, or vice versa)
- Month only (no year information)
- Ambiguous format

### Returns 0.0 When:
- Complete mismatch (year or month wrong)
- Invalid input

## Testing

Run the test suite:
```bash
python3 test_row_finder.py
```

Test cases cover:
- Standard Treasury format
- Multiple period formats
- Monthly data extraction
- Period type disambiguation
- Edge cases (missing values, empty lists, etc.)
- Confidence scoring

## Integration Notes

### Works With:
- `TableFinder`: extracts rows from Treasury text files
- `DataNormalizer`: cleans and normalizes row data
- `RowMatcher`: finds rows by label/name
- `ValueParser`: extracts values from columns

### Complements:
- RowMatcher finds row labels (e.g., "National defense")
- RowFinder finds time periods (e.g., fiscal year 1995)
- Together they locate: "National defense in FY 1995"

## Performance

- **Average latency**: < 1ms per 100 rows
- **Memory**: O(n) where n = number of rows
- **Parsing**: Regex-based, handles 1000+ period formats/second

## Known Limitations

1. **Fiscal year encoding ambiguity**: "19951" could theoretically be FY 1991 or FY 1995 depending on data source (assumes year portion is correct)
2. **Single-year tables**: Month-only rows are valid but require year context from elsewhere
3. **Non-standard abbreviations**: Custom month abbreviations not in MONTH_NAMES won't parse
4. **Date ranges**: "1995-1996" (range format) not supported, only single years

## Future Enhancements

- [ ] Support date ranges ("1995-1996")
- [ ] Configurable month names for internationalization
- [ ] Support for quarters (Q1, Q2, etc.)
- [ ] Fiscal year start date customization
- [ ] Integration with Gregorian calendar assumptions
