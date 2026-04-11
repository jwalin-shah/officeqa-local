# Unit Conversion Atlas

## Treasury Bulletin Unit Patterns

Tables declare units in their header. Always check get_table_profile before using any value.

| Table says | Multiply raw value by | Example: raw "1,234" means |
|---|---|---|
| "In thousands" | × 1,000 | $1,234,000 |
| "In millions" | × 1,000,000 | $1,234,000,000 |
| "In billions" | × 1,000,000,000 | $1,234,000,000,000 |
| "In thousands of dollars" | × 1,000 | $1,234,000 |
| No unit stated / "Dollars" | × 1 (use as-is) | $1,234 |

## When value_scaled Is Present

If query_table_rows or extract_values returns `value_scaled`, that value is ALREADY in actual dollars. Do NOT multiply again. Use it directly.

## Mixing Units Across Tables

If Table A is "in millions" and Table B is "in thousands":
1. Convert both to the same base: A_actual = A_raw × 1,000,000; B_actual = B_raw × 1,000
2. Then compute: compute_expression(expression="a - b", variables={"a": A_actual, "b": B_actual})

## Answer Unit

- If the question asks "in millions", divide your actual-dollar result by 1,000,000.
- If the question asks for the raw number, return actual dollars.
- If the question doesn't specify, match the table's unit scale.
