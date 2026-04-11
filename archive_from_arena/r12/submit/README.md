# r12 (v22-Hybrid) — Ready for Arena Submission

## Quick Start

### Local Testing
```bash
cd /Users/jwalinshah/projects/officeqa-arena
./run_local_r12.sh UID0001                    # Test single task
./run_local_r12.sh --batch UID0001 UID0002    # Test multiple
./run_local_r12.sh --parallel UID0001 UID0002 # Parallel test
```

### Arena Submission
```bash
cd r12/submit
arena submit
# Expected: ~71-73% (between v20's 69.5% and v21's 72.8%)
# Wait 24-48h for results
```

## What Changed from v20?

### Prompt Changes (28 lines vs 27)
1. **Added:** "## Quick Analysis" section at top
   - Helps model identify metric type (sum/mean/pct_change/etc.)
   - Identifies time period (CY/FY/specific range)
   - Identifies entity/category being asked about

2. **Kept:** Everything that worked in v20
   - CPI tool usage
   - FY vs CY guidance (critical for correctness)
   - Unit format guidance
   - grep tips
   - Formula reference
   - **Verify skill integration** ("After answering, load("verify")")

3. **Removed:** v21's verbose 4-step enforcement
   - No rigid STEP 1/STEP 2/STEP 3/STEP 4 boxes
   - No forced decomposition blocks
   - Model stays flexible while getting metric guidance

### Why This Works

**v20 Baseline:** 69.5% (solid grep-first approach)
- Problem: Model sometimes found right data but computed wrong metric

**v21 Approach:** 72.8% (forced 4-step decomposition)
- Problem: Over-structured, model got trapped in format boxes

**v22 Hybrid:** 71-73% (quick analysis + flexibility)
- Quick Analysis asks "what metric?" upfront (1-2 steps)
- Model can still explore naturally (not rigid format)
- Verify skill catches wrong answers (50% of mistakes)

## Configuration

### arena.yaml
- **Model:** openrouter/minimax/minimax-m2.5 (proven best)
- **Timeout:** 480 seconds (standard)
- **Harness:** goose 1.29.1 (supports skills)

### Skills
- **verify:** Proof-checking skill (load() for post-answer verification)

### Tools
- **cpi.py:** CPI index/adjustment tool (21KB, proven effective)

## Files

```
r12/submit/
├── prompt.j2           # Hybrid prompt (Quick Analysis + v20 core)
├── arena.yaml          # Arena configuration
├── cpi.py              # CPI tool (copied from r11)
├── skills/
│   └── verify/
│       └── SKILL.md    # Verification skill (reused from r11)
└── README.md           # This file
```

## Expected Results

### Accuracy
- **Optimistic:** 72-73% (if Quick Analysis + verify work together)
- **Conservative:** 70-71% (if Quick Analysis adds 1% over v20)
- **Minimum:** 69%+ (worst case: just like v20)

### Step Count
- **v20:** 15.6 avg steps
- **v21:** 11.1 avg steps
- **v22 target:** 13-14 avg steps (middle ground)

## Troubleshooting

### "Got low score" → Compare to v20 (69.5% baseline)
- If >69.5%: Success! (hybrid worked)
- If <69.5%: Revert to v20 (confirm quick analysis not helping)

### "Verify skill didn't load"
- Ensure summon extension is in goose recipe
- Check: `ln -sf /path/to/skills/verify ~/.config/goose/skills/verify`

### "Steps took too long"
- Tighten the "Write best guess immediately" instruction
- Remove "Quick Analysis" section (fall back to v20)

### "CPI tool not found"
- Verify /installed-agent/cpi.py is copied during arena submit
- Check skill definition includes tool path

## Commit/Deploy Checklist

- [x] prompt.j2 created (28 lines, Quick Analysis added)
- [x] arena.yaml configured (480s, goose, MiniMax M2.5)
- [x] cpi.py included (21KB, unchanged from r11)
- [x] skills/verify/ included (verification skill, unchanged)
- [x] run_local_r12.sh script created (ready for testing)
- [x] V22_HYBRID_STRATEGY.md documented (design rationale)
- [ ] Local testing: `./run_local_r12.sh --parallel UID0001 ... UID0246`
- [ ] Submit: `cd r12/submit && arena submit`
- [ ] Monitor: Pull traces after 24h, compare to v20/v21

## Next Steps After Submission

1. **Pull traces** (~24h after submit):
   ```bash
   python3 pull_latest_traces.py  # Check r12 submission ID
   ```

2. **Analyze results**:
   ```bash
   python3 analyze_traces.py r12 v20 v21  # Compare performance
   ```

3. **If 71%+**: Success! Document findings
4. **If <71%**: Revert to v20 or iterate on prompt

## Related Documentation

- `../V22_HYBRID_STRATEGY.md` — Detailed design + rationale
- `/Users/jwalinshah/.claude/projects/.../memory/project_v20_trace_analysis.md` — v20 failure modes
- `/Users/jwalinshah/.claude/projects/.../memory/project_v21_decomposition.md` — v21 approach

---

**Author:** Claude
**Date:** 2026-04-08
**Based on:** v20_183.8 (69.5%) + v21-13h (72.8%) analysis
