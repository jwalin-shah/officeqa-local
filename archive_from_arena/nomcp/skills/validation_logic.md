# Answer Validation Rules

Use these checks to verify your answer before submitting.

## Sanity Checks

1. **Unit consistency**: If the question asks "in millions of dollars" and the table header says "(In millions of dollars)", use values as-is. Do NOT multiply again.

2. **Sign check**: Deficits are negative. If the question asks for "expenditures" (positive) but you get a negative number, you may be looking at a surplus/deficit row instead.

3. **Order of magnitude**: Federal budget numbers are typically:
   - 1930s-1940s: hundreds to low thousands (in millions)
   - 1950s-1960s: thousands to tens of thousands (in millions)
   - 1970s-1990s: tens of thousands to hundreds of thousands (in millions)
   - 2000s-2020s: hundreds of thousands to millions (in millions)

4. **Total = sum of parts**: If you found a total row AND individual components, the components should sum to approximately the total. Use this to verify you read the right columns.

5. **Percent change formula**: `|(new - old) / old| * 100` for absolute percent change. The question specifies "absolute" to mean the unsigned magnitude.

6. **Rounding**: Follow exactly what the question asks:
   - "rounded to the nearest hundredths place" = 2 decimal places (e.g., 1608.80)
   - "rounded to the nearest whole number" = 0 decimal places
   - If unspecified, match the precision shown in the table

## Common Errors to Avoid

1. **Wrong column**: Tables often have multiple numeric columns (e.g., "This month", "This fiscal year to date", "Comparable period prior year"). Read the column header carefully.

2. **Wrong row**: "National defense" vs "National defense and international affairs" are different line items. Match the exact wording from the question.

3. **Footnote contamination**: Strip "3/", "*", "r/", "p/" from numbers before computing. "1,580 3/" means 1580.

4. **Comma handling**: Always strip commas: "1,580" -> 1580 for computation.

5. **Calendar vs fiscal**: This is the #1 error. Re-read the question. Does it say "calendar year" or "fiscal year"?

## CPI Adjustment Rules

When the question asks for inflation-adjusted values:
- Annual CPI-U (not seasonally adjusted) from BLS/Minneapolis Fed
- Formula: `adjusted_value = nominal_value * (target_CPI / source_CPI)`
- Common CPI values: 1940≈14.0, 1950≈24.1, 1960≈29.6, 1970≈38.8, 1980≈82.4, 1990≈130.7, 2000≈172.2

## Answer Format

- Numeric only: no words, no units
- Use commas if the question's expected format uses them (e.g., "2,602")
- Include % sign only if the question asks for a percentage (e.g., "1608.80%")
- Use the decimal precision the question specifies

## Emergency: No Data Found

If solve.py returns nothing and grep finds nothing:
1. Try broader keywords: "defense" instead of "national defense expenditures"
2. Try adjacent years: data for 1940 may be in bulletin from 1941 or 1942
3. Submit your best educated guess — a wrong answer scores higher than no answer
