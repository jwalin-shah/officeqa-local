# How to Read Treasury Bulletin Tables

Treasury Bulletin tables in the TXT files follow a pipe-delimited markdown format.
Understanding the layout is critical for extracting the right values.

## Table Structure

```
Table IFS-2.—International Capital Movements: U.S. Liabilities to Foreigners
(In millions of dollars)

| Type of liability | End of June 1990 | End of June 1991 | End of Sept 1991 |
| --- | --- | --- | --- |
| Total liabilities | 1,234,567 | 1,345,678 | 1,400,000 |
|   Short-term | 800,000 | 850,000 | 875,000 |
|     Bank-reported | 600,000 | 625,000 | 640,000 |
|   Long-term | 434,567 | 495,678 | 525,000 |
```

## Key Layout Patterns

### Indentation = Hierarchy
- No indent: Category total (e.g., "Total liabilities")
- 2 spaces: Subcategory (e.g., "  Short-term")
- 4 spaces: Sub-subcategory (e.g., "    Bank-reported")
- The TOTAL row = sum of immediate children

### Column Headers
- May span multiple rows with ">" separating levels
- Example: "Type of liability > End of June 1990" means the column is "End of June 1990"
- "This month" vs "Fiscal year to date" vs "Comparable prior year" are DIFFERENT columns

### Multi-year Tables
- Some tables show data across many years in rows:
  ```
  | 1938 | 6,765 | 6,840 | -75 |
  | 1939 | 5,261 | 6,267 | -1,006 |
  | 1940 | 5,387 | 9,468 | -4,081 |
  ```
- These annual rows are FISCAL YEAR totals unless labeled otherwise

### Monthly Tables
- Monthly rows appear as:
  ```
  | January | 132 | 216 |
  | February | 129 | 199 |
  ```
  or:
  ```
  | 1940-January | 132 |
  | 1940-February | 129 |
  ```

### Separator Rows
- `| --- | --- | --- |` separates headers from data
- Sometimes used between sections within a table

## Common Pitfalls

1. **Multi-column values**: The value you want might not be in the first numeric column.
   Count columns carefully using the header row.

2. **Merged cells**: Some rows span all columns (section headers within a table).
   These have text in the first cell and empty cells after.

3. **Continuation tables**: A table may be split across pages. The second part
   starts with the same column headers but different row labels.

4. **Footnote rows**: Rows starting with numbers like "1/" or "2/" at the bottom
   are footnotes, not data. But the markers appear IN data cells too:
   "1,580 3/" means value is 1580 with footnote 3.

5. **"Total" vs "Grand total"**: Some tables have subtotals AND grand totals.
   Make sure you're reading the right level.
