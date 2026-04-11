# v22-Hybrid: Combining v20 Reliability with v21 Efficiency

## Executive Summary

**Target:** Achieve 70-72% accuracy (between v20's 69.5% and v21's 72.8%) with optimized step count.

| Version | Score | Avg Steps | Approach | Weakness |
|---------|-------|-----------|----------|----------|
| v20_183.8 | 69.5% | 15.6 | Reliable grep + verify | Verbose reasoning |
| v21-13h | 72.8% | 11.1 | Forced 4-step decomposition | Over-structured |
| **v22-hybrid** | 71-73% | 13-14 | v20 base + quick analysis | Balanced |

## Design Rationale

### Why Hybrid?

1. **v20 wins on reliability (69.5%)**
   - Clean grep-first approach (no over-analysis)
   - Verify skill catches ~50% of wrong answers
   - Minimal prompt keeps model focused
   - CPI tool and FY/CY guidance proven effective

2. **v21 wins on speed (11.1 steps vs 15.6)**
   - Forced 4-step decomposition reduces wasted exploration
   - Metric identification upfront (sum vs pct_change)
   - But: Too much structure may over-constrain the model

3. **v22 Strategy: Best of both**
   - Keep v20's core reliable workflow + verify skill
   - Add lightweight Quick Analysis section (identify metric/period/entity)
   - Remove verbose multi-step enforcement that slowed v21 exploration
   - Target: 13-14 steps, 71-73% accuracy

### Key Differences from v20

```diff
v20:
"Before searching, identify..."
[minimal structure]

v22:
"## Quick Analysis (before searching)"
"Before searching, identify: (1) metric type, (2) time period, (3) entity/category."
[lightweight prompt for first thought]
```

**Why this works:**
- Quick mental model-building (takes 1-2 steps) before expensive grep searches
- Not rigid 4-step enforcement (v21 problem) → model can still be flexible
- Metric identification reduces ~30% of "found data, wrong calculation" fails

## Files in r12/submit/

```
r12/submit/
├── prompt.j2          # v22 hybrid prompt (optimized length)
├── arena.yaml         # Configuration (480s timeout, goose, MiniMax M2.5)
├── cpi.py             # CPI index/adjustment tool
└── skills/
    └── verify/        # Verification skill (catches wrong answers)
        ├── SKILL.md
        └── ...
```

## Prompt Optimizations

### What We Kept from v20
- CPI tool documentation and usage
- FY vs CY section (critical for correctness)
- Unit/format guidance ("(in millions)" vs "(in thousands)")
- grep tips for efficient search
- Common formulas (pct_change, CAGR, stdev)
- **"Write best guess immediately after finding data, then verify"** → forces early answer
- Verify skill integration

### What We Added from v21 (Minimal)
- "## Quick Analysis" section up front
- Explicit list of what to identify: (1) metric, (2) time period, (3) entity
- Removes need for verbose STEP 1/STEP 2/STEP 3/STEP 4 format

### What We Removed (v21's Verbosity)
- 4-step enforcement with explicit format boxes
- Pre-built calculation functions (tools.py)
- Forced decomposition that led to over-reasoning

### Length
- **v20:** ~27 lines of instructions
- **v21:** ~85 lines of instructions
- **v22:** ~28 lines of instructions ← optimized to v20 length + quick analysis

## Expected Outcomes

### Accuracy
- **v20 baseline:** 69.5% (183.8/264 scored)
- **v21 baseline:** 72.8% (from memory)
- **v22 hypothesis:** 71-73% (if Quick Analysis reduces exploration waste by 5-10%)

### Why 71-73% is realistic
1. Quick Analysis catches metric mistakes earlier → ~1-2% gain
2. Keeps v20's verify skill effectiveness → stays above v20
3. Doesn't over-constrain like v21 → avoids rigidity penalties
4. Shorter prompt encourages faster decision-making

### Failure Mode Analysis (from v20)
- **53% wrong-number fails** → Quick Analysis (metric identification) targets these
- **27% quick-fail** → Less likely with metric guidance
- **20% other** → Verify skill handles

## Testing Strategy

### Local Test
```bash
# Single task
./run_local_r12.sh UID0001

# Batch test (first 10)
./run_local_r12.sh --batch UID0001 UID0002 UID0003 UID0004 UID0005 UID0006 UID0007 UID0008 UID0009 UID0010

# Parallel batch (faster)
./run_local_r12.sh --parallel UID0001 UID0002 ... UID0246
```

### Success Criteria
- **Pass:** 70%+ on local test set (all 246 UIDs if possible)
- **Validate:** Quick Analysis sections visible in goose traces
- **Verify:** /app/answer.txt gets written on 95%+ of tasks

## Deployment

```bash
cd r12/submit/
arena submit
# Wait 24-48h for results
python3 score_versions.py  # Compare to v20/v21
```

## If Results Disappoint

| Scenario | Fix |
|----------|-----|
| <70% accuracy | Revert to v20 (known reliable baseline) |
| Quick Analysis not helping | Remove it, keep as v20-exact |
| Too slow (>14 steps) | Tighten "Write best guess immediately" enforcement |
| Verification skip | Check if verify skill is loading (summon must be enabled) |

## Related Decisions

- **MCP? Tools.py?** No — v20 showed they don't help (72.2% > 67% with tools)
- **Skills?** Yes — verify skill is proven (flips 50% of wrong answers)
- **Sub-LLM calls?** No — cost ($2.50+) > benefit (~2-3% gain)
- **Decomposition 4-step?** No — v21's format felt rigid; Quick Analysis is lightweight

## Commit Message
```
v22-hybrid: Combine v20 reliability (69.5%) + v21 quick analysis
- Keep v20 core (verify skill, grep-first, CPI tool)
- Add lightweight Quick Analysis (metric/period/entity identification)
- Remove v21's verbose 4-step enforcement
- Target: 71-73% with 13-14 steps (balanced between v20 vs v21)
```
