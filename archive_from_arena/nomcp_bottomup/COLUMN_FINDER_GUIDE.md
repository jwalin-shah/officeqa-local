**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# ColumnFinder Guide

## Overview

The `ColumnFinder` class locates target columns in Treasury Bulletin tables with hierarchical multi-level headers.

**Problem**: Treasury Bulletin tables have complex column structures like:
```
"Total on-budget and off-budget results > Total receipts (1)"
"Total on-budget and off-budget results > On-budget receipts (2)"
```

**Solution**: Match by normalized keywords and hierarchy flattening to find the right column.

## Quick Start

```python
from column_finder import ColumnFinder

finder = ColumnFinder()

# Extract rows from table (dict format)
rows = [
    {"Fiscal year": "1995", "Total > Receipts": 1350576.0},
    {"Fiscal year": "1996", "Total > Receipts": 1413156.0}
]

# Find column matching "total receipts"
col_name, confidence = finder.find_column(rows, "total receipts")

print(f"Found: {col_name}")  # Output: "Total > Receipts"
print(f"Confidence: {confidence}")  # Output: 0.99
```

## Matching Strategy

The matcher uses a 5-level scoring system:

| Coverage | Score | Condition |
|----------|-------|-----------|
| 100% match | 0.99 | All keywords found (after exact match) |
| 100% keywords present | 0.85 | All identifier keywords found in column |
| >= 50% coverage | 0.65 | At least half the keywords match |
| < 50% coverage | 0.55-0.60 | Some keywords match (less reliable) |
| No match | 0.0 | No keywords found |

## Normalization Rules

Headers are normalized for comparison:

1. **Flatten hierarchy**: "Total > Receipts" → "total receipts"
2. **Remove numbering**: "Receipts (1)" → "receipts"
3. **Handle dashes**: "On-budget" → "on budget"
4. **Clean punctuation**: Remove pipes, quotes, extra spaces
5. **Lowercase**: Convert to lowercase for case-insensitive matching

## Key Methods

### find_column(rows, column_identifier)
Find the best matching column.

**Returns**: `(column_name, confidence)`
- `column_name`: Original column name (preserved casing/structure), or None
- `confidence`: Float 0.0-1.0

**Example**:
```python
col_name, conf = finder.find_column(rows, "receipts")
# Finds column with highest "receipts" keyword match
```

### score_column_match(column_name, identifier)
Score how well a single column matches an identifier.

**Returns**: Float 0.0-1.0

**Example**:
```python
score = finder.score_column_match("Total > Receipts", "total receipts")
# Returns 0.99 (exact match after normalization)
```

### find_all_candidate_columns(rows, identifier, min_confidence=0.50)
Find ALL columns matching the identifier above a confidence threshold.

**Returns**: List of `(column_name, confidence)` tuples, sorted by confidence descending

**Example**:
```python
candidates = finder.find_all_candidate_columns(rows, "receipts", min_confidence=0.60)
# Returns [("Total Receipts", 0.85), ("On-Budget Receipts", 0.85), ...]
```

### flatten_hierarchical_header(header)
Flatten hierarchical header structure.

**Returns**: Flattened string

**Example**:
```python
flattened = finder.flatten_hierarchical_header("A > B > C")
# Returns "a b c"
```

## Common Patterns

### Exact Match
```python
rows = [{"Total > Receipts": 1350576.0}]
col, conf = finder.find_column(rows, "total receipts")
# Returns ("Total > Receipts", 0.99)
```

### Hierarchical with Numbering
```python
rows = [{"Total receipts (1)": 1350576.0}]
col, conf = finder.find_column(rows, "total receipts")
# Returns ("Total receipts (1)", 0.99)
# Numbering is stripped during normalization
```

### Partial Keywords
```python
rows = [{"Total Receipts": 1350576.0, "Total Outlays": 1514389.0}]
col, conf = finder.find_column(rows, "receipts")
# Returns ("Total Receipts", 0.85)
# Single keyword match scores 0.85
```

### Multiple Candidates
```python
rows = [{
    "Total Receipts": 1350576.0,
    "On-Budget Receipts": 1000000.0,
    "Off-Budget Receipts": 350576.0
}]
candidates = finder.find_all_candidate_columns(rows, "receipts")
# All three columns match, same confidence
```

## Design Decisions

### Preserve Original Column Names
The finder returns the original column name as it appears in the data:
```python
col_name, _ = finder.find_column(rows, "total receipts")
# Returns "Total > Receipts" (not normalized)
```

This allows downstream code to use the exact key for dict lookups.

### No Row Modification
The finder is read-only; it never modifies input rows:
```python
finder.find_column(rows, "receipts")
# rows remains unchanged
```

### Graceful Degradation
Missing columns return `None` confidence:
```python
col_name, conf = finder.find_column(rows, "nonexistent")
# Returns (None, 0.0) if no columns match at all
```

## Performance

- **Time complexity**: O(n * m) where n = number of columns, m = average column name length
- **Space complexity**: O(1) (only stores best match)
- **For large tables**: Use `find_all_candidate_columns()` with higher `min_confidence` to filter

## Testing

Run the comprehensive test suite:

```bash
python3 -m pytest test_column_finder.py -v
```

34 tests cover:
- Exact matching with hierarchical headers
- Partial keyword matching
- Numbering/punctuation handling
- Case insensitivity
- Empty/None handling
- Sorting by confidence
- Real Treasury Bulletin data patterns
