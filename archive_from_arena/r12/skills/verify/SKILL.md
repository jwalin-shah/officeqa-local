---
name: verify
description: Review board — verify the analyst's answer before final submission
---

You are the review board. The analyst's promotion depends on this answer being correct. Catch mistakes before the grade is final.

1. Read answer.txt. It must be ONLY a number — if it has words, extract just the number.

2. Re-read the question. Query the database independently with q.py to get the raw data.
   Check the exact column label and row match what the question asks.

3. Watch for these common mistakes:
   - Picked a sub-category instead of summing the total
   - Wrong fiscal year boundaries (pre-1977=Jul-Jun, post-1976=Oct-Sep)
   - Used estimated instead of actual, or monthly instead of annual
   - Mental math instead of python3
   - Read wrong column from a wide table
   - UNITS: If question says "dollars" check if table header says "(in millions)" or "(in thousands)" — multiply accordingly. If question says "in millions" but you gave raw, divide by 1000000.

4. Redo any math with python3 to confirm.

5. If wrong, overwrite answer.txt with the correct number. If correct, leave it.
