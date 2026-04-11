**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# SearchTask: Value Extraction Orchestrator

## Overview

`SearchTask` is the main orchestrator for the entire data extraction pipeline. It takes a single **decomposition piece** (one value to extract) and executes a complete multi-file, multi-strategy search to find the answer.

## Architecture

```
Decomposition Piece
  ↓
File Locator: Find file(s) for year/month
  ↓
For EACH file candidate (parallel):
  1. Table Finder: Locate table by ID
  2. Data Normalizer: Clean and normalize rows
  3. Table Validator: Check table quality
  4. If valid, spawn 3 subsearches (parallel):
     - Strict: Exact matching
     - Fuzzy: Keyword matching
     - Contextual: Structure-aware matching
  5. Consensus Voter: Select best result
  ↓
Compare across files
  ↓
Return winning value + metadata
```

## Usage

```python
from search_task import SearchTask

task = SearchTask()

# Define decomposition piece
piece = {
    'piece_id': 1,
    'year': 1995,
    'period_type': 'fiscal',
    'table_id': 'FFO-1',
    'row_identifier': ['Total', 'Receipts'],
    'column_identifier': 'Total receipts',
    'prefer_months': [12, 6],  # Optional: month priority
    'row_index': None  # Optional: expected position
}

# Execute search
result = task.execute(piece, verbose=True)

# Result contains:
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
        'method': 'strict',  # Which strategy worked
        'agreement_level': 'full'  # How many strategies agreed
    },
    'steps': [
        'Validating inputs...',
        'Found 2 file candidates...',
        '...'
    ],
    'file_results': [
        {'file': '1995_12.txt', 'value': 1350576.0, 'confidence': 0.95, 'success': True},
        {'file': '1995_06.txt', 'value': 1350576.0, 'confidence': 0.92, 'success': True}
    ],
    'voting_details': {
        'strict_result': {...},
        'fuzzy_result': {...},
        'contextual_result': {...},
        'consensus': {...}
    },
    'errors': []
}
```

## Input Format

**decomposition_piece** dictionary:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `piece_id` | int | Yes | Unique identifier for this piece |
| `year` | int | Yes | Fiscal/calendar year to search |
| `period_type` | str | No | 'fiscal' or 'calendar' (informational only) |
| `table_id` | str | Yes | Table identifier (e.g., 'FFO-1', 'receipt') |
| `row_identifier` | list\[str\] or str | Yes | Words to match in row label (e.g., ['Total', 'Receipts']) |
| `column_identifier` | str | Yes | Column name/keywords (e.g., 'Total receipts') |
| `prefer_months` | list\[int\] | No | Month priority order [12, 6, 3] |
| `month` | int | No | Specific month if known |
| `row_index` | int | No | Expected row position for contextual matching |

## Output Format

Result dictionary with these key fields:

### Basic Result Fields
- `piece_id`: Input piece ID
- `value`: Extracted value (float) or None
- `confidence`: Confidence score 0.0-1.0
- `success`: Boolean - whether extraction succeeded

### Source Metadata
- `source`: Dict containing:
  - `file`: Filename where value was found
  - `table`: Table name/ID
  - `row_identifier`: Row labels matched
  - `column`: Actual column name in table
  - `method`: Strategy that succeeded ('strict', 'fuzzy', 'contextual')
  - `agreement_level`: 'full' (3/3), 'partial' (2/3), or 'none'

### Debug Information
- `steps`: List of step descriptions (for debugging)
- `file_results`: Per-file results:
  - `file`: Filename
  - `value`: Value from this file
  - `confidence`: Confidence for this file
  - `success`: Whether successful
- `voting_details`: Full consensus voting output:
  - `strict_result`: Result from strict strategy
  - `fuzzy_result`: Result from fuzzy strategy
  - `contextual_result`: Result from contextual strategy
  - `consensus`: Final voting decision
- `errors`: List of error messages

## Three Subsearch Strategies

### Strategy A: Strict
- **Row matching**: Exact match only
- **Column matching**: Exact or high confidence match
- **Confidence**: High when works, 0 when doesn't
- **Use case**: When identifiers are precise and unambiguous

```
match_row(rows, ["Total", "Receipts"])  # Exact match
find_column(rows, "total receipts")      # Exact or close match
extract_value(row, column)
```

### Strategy B: Fuzzy
- **Row matching**: Contains all keywords (any order)
- **Column matching**: Keyword-based matching
- **Confidence**: Moderate, reduces if fuzzy
- **Use case**: Partial identifiers, typos, spacing variations

```
fuzzy_match_row(rows, ["Total"])         # Find "TOTAL RECEIPTS" etc
find_column(rows, "receipts")            # Find column with "receipts"
extract_value(row, column)
```

### Strategy C: Contextual
- **Row matching**: Structure-aware (position, category context)
- **Column matching**: Same as fuzzy (keyword-based)
- **Confidence**: Moderate, slightly lower than fuzzy
- **Use case**: Complex tables with multiple sections, expected position hints

```
contextual_match_row(rows, ["Total"], row_index=5)
find_column(rows, "receipts")
extract_value(row, column)
```

## Voting Logic (ConsensusVoter)

After running 3 strategies in parallel:

1. **Extract valid values**: Ignore any that returned None
2. **Group by agreement**: Which strategies returned the same value?
3. **Find largest group**: Pick the group with most strategies
4. **Boost confidence**: More strategies agree = higher confidence
   - All 3 agree: confidence += 0.2 (boost)
   - 2 agree: confidence = average of their confidences
   - 1 only: confidence = that strategy's confidence
5. **Return winner**: Best value + highest confidence

## Error Handling

SearchTask gracefully handles all failure modes:

| Scenario | Result |
|----------|--------|
| Missing input fields | `success=False`, descriptive error |
| Year not in corpus | `success=False`, returns None |
| File not found | Tries all candidates, returns None if all fail |
| Table not found | Skips file, tries next |
| Invalid table (bad structure) | Skips file |
| Row not found in any strategy | Tries next file |
| All files fail | `success=False`, returns None |

## Performance Notes

- **File processing**: Sequential (one file at a time)
- **Subsearch strategies**: Parallel (3 strategies in parallel)
- **Multiple files**: All candidates processed, best result returned
- **Typical execution time**: 100-500ms per piece

## Verbose Mode

Pass `verbose=True` to `execute()` to get step-by-step output:

```python
result = task.execute(piece, verbose=True)
```

Output includes:
- File discovery results
- Table found/not found status
- Row count after cleaning
- Table validation results
- Per-strategy subsearch results
- Consensus voting decision
- Final success/failure

## Integration with Other Components

SearchTask orchestrates these components:

1. **FileLocator**: Maps year → file path(s)
2. **TableFinder**: Extracts table from file + parses rows
3. **DataNormalizer**: Cleans row data (numbers, text)
4. **TableValidator**: Checks table quality (enough data, etc)
5. **RowMatcher**: Finds rows by identifier (3 strategies)
6. **ColumnFinder**: Finds columns by name (fuzzy matching)
7. **ValueParser**: Extracts & parses cell value
8. **ConsensusVoter**: Votes on 3 strategies

Each component is called exactly once per strategy per file, so:
- 1 file × 3 strategies = ~24 component calls
- 2 files × 3 strategies = ~48 component calls

## Testing

```bash
cd /Users/jwalinshah/projects/officeqa-arena/nomcp/bottomup
python3 search_task.py
```

Includes 3 test cases:
1. Valid piece (should succeed if corpus/table data is present)
2. Missing year (should fail gracefully)
3. Invalid year 2050 (should fail gracefully)

## Known Issues

1. **TableFinder limitation**: Currently extracts table by finding "TABLE ID" line, but actual data table may be further down in the file. This is a pre-existing issue with the TableFinder component that affects SearchTask.

2. **Row/column normalization**: Different table formats may require tuning of the matching thresholds.

3. **Multi-year pieces**: Currently only searches one year per piece. Future enhancement would support year ranges.
