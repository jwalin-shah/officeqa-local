# Computation Patterns

When the question requires a calculation, use these Python templates. ALWAYS use `python3 -c` — never do arithmetic in your head.

## Percent Change
```
python3 -c "old=2602.4; new=39482.03; print(round(((new - old) / old) * 100, 2))"
```

## Difference
```
python3 -c "a=9468; b=6548; print(round(a - b, 2))"
```

## Sum of Monthly Values
```
python3 -c "vals=[120,130,140,150,160,170,180,190,200,210,220,230]; print(sum(vals))"
```

## Geometric Mean
```
python3 -c "
import math
vals = [1.05, 1.03, 1.07, 1.02, 1.04]
geo = math.prod(vals) ** (1/len(vals))
print(round(geo, 6))
"
```

## Compound Annual Growth Rate (CAGR)
```
python3 -c "
begin=1000; end=2500; years=10
cagr = (end/begin)**(1/years) - 1
print(round(cagr * 100, 2))
"
```

## Ratio / Share
```
python3 -c "part=6548; whole=39482; print(round(part / whole * 100, 2))"
```

## Rounding
- "rounded to the nearest whole number": `round(x, 0)` → format as int
- "to 2 decimal places": `round(x, 2)`
- "to the nearest tenth": `round(x, 1)`

## Multi-Step (save values to scratchpad)
When you need data from multiple tables or rows:
```
echo "VAL_1940=2602.4" >> /tmp/plan.txt
echo "VAL_1953=39482.03" >> /tmp/plan.txt
```
Then compute:
```
python3 -c "
val_1940 = 2602.4
val_1953 = 39482.03
change = ((val_1953 - val_1940) / val_1940) * 100
print(round(change, 2))
"
```

## Common Pitfalls
- **Commas in numbers**: Strip commas before computing: `float("39,482.03".replace(",",""))`
- **Negative values in parens**: `(2,920)` means `-2920`
- **Footnote markers**: `6,548r/` — the `r/` is a revision marker, not part of the number
- **Units mismatch**: If table says "(In millions)" and you need thousands, multiply by 1000
