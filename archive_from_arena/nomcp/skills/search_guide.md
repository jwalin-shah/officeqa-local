# Treasury Bulletin Search Guide

You MUST follow these steps IN ORDER. Do NOT explore the filesystem or run ls.

## Step 1: One-Shot Search (PREFERRED — 1 command)
```bash
python3 /installed-agent/solve.py "PASTE THE FULL QUESTION HERE"
```
This builds an index (if needed) and returns the top 3 matching tables with smart slicing.
Takes ~25 seconds on first run, ~5 seconds after index is built.

## Step 2 (ONLY if Step 1 found nothing): Manual Index Search
```bash
# Build index first (only needed once)
python3 /installed-agent/build_index.py

# Search by keyword
grep -i "national defense" /tmp/keyword_index.txt | head -10

# Filter by year
grep -i "customs duties" /tmp/keyword_index.txt | grep "1953" | head -10

# Find tables with monthly data
grep -i "expenditures" /tmp/keyword_index.txt | grep "HAS_12_MONTHS" | head -10

# Find tables from a specific bulletin
grep "treasury_bulletin_1941_01" /tmp/keyword_index.txt | head -10
```

NEVER grep /app/corpus/ directly — it's 696 files and will time out.
ALWAYS grep /tmp/keyword_index.txt instead.

## Step 3: Read a Specific Table
```bash
python3 /installed-agent/search.py table treasury_bulletin_1941_01.txt 248
```

## Step 4: Compute (ALL math via Python)
```bash
python3 -c "print(sum([132, 129, 143, 159, 154, 153, 177, 200, 219, 287, 376, 473]))"
python3 -c "print(round(44463 / 2602 * 100 - 100, 2))"
```

## Step 5: Submit
```bash
echo -n "2,602" > /app/answer.txt
```

## Budget: 3-8 tool calls total
- Call 1: solve.py (search)
- Call 2-3: python3 -c (compute if needed)
- Call 4: echo answer to file
- After 10 commands, STOP and submit your best guess immediately.
