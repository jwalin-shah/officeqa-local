# OfficeQA Arena: Comprehensive Project History

> **Compiled:** April 6, 2026 (final)
> **Sources:** 130+ git commits, 15+ Claude Code sessions, 9 Cursor sessions, 55 memory files, all traces/results/docs
> **Duration:** March 29 -- April 6, 2026 (9 days)
> **Best Score:** 184.5/246 (v5, Goose shell-only) | 12 submissions, 9 architectural generations, ~4,400 evaluations

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Day-by-Day Timeline](#2-day-by-day-timeline)
3. [Architecture Evolution](#3-architecture-evolution)
4. [Every Submission & Score](#4-every-submission--score)
5. [Claude Code Sessions (Chronological)](#5-claude-code-sessions)
6. [Cursor Sessions](#6-cursor-sessions)
7. [Key Technical Discoveries](#7-key-technical-discoveries)
8. [Failure Analysis Deep Dives](#8-failure-analysis-deep-dives)
9. [Prompt Engineering Evolution](#9-prompt-engineering-evolution)
10. [Infrastructure & Tooling](#10-infrastructure--tooling)
11. [What Worked vs What Didn't](#11-what-worked-vs-what-didnt)
12. [Lessons Learned](#12-lessons-learned)
13. [Appendix: Memory File Index](#13-appendix-memory-file-index)

---

## 1. Executive Summary

This project competed in the Sentient Arena "grounded-reasoning" track, answering 246 questions about U.S. Treasury Bulletin financial data spanning 1939–2025. The agent receives raw text/JSON files and must extract, compute, and submit numeric answers.

**Score progression over 8 days:**
```
Day 1 (Mar 30):  5% local  →  25% first MCP  →  55% fixed tools
Day 2 (Mar 31):  135.5 pts (57%) — First real submission
Day 3 (Apr 1):   147–167 pts — Harness experiments
Day 4 (Apr 2):   180.4 pts (69%) — Tool fixes, anti-spin
Day 5 (Apr 3):   166–189 pts — Search improvements, cleanup
Day 6 (Apr 4):   184.5 pts (75%) — Best score (v5, grep-only)
Day 7 (Apr 5):   v7 submitted — Skills + inline CPI
Day 8 (Apr 6):   v7=184, v8=173, v9=175, v10=180, v12=181 — ceiling confirmed
```

**Key insight that emerged:** MiniMax M2.5 performs best when given raw text files and shell tools (grep/cat/sed). Every attempt to add structured tools (MCP, databases, pipelines) either matched or underperformed the simple grep approach. The bottleneck is evidence selection (finding the right table/row/column), not arithmetic or reasoning.

---

## 2. Day-by-Day Timeline

### Day 1: March 29–30 — Bootstrap & First Eval

**Commits:** 16 | **Claude sessions:** 2 (626b1888, 00cb13ab) | **Cursor sessions:** 3

**What happened:**
- Forked the arena starter repo. Initial system: 7 MCP tools, 11GB SQLite database, DigitalOcean droplet hosting MCP SSE server
- **Critical bug found immediately:** `search_tables` crashed — `'sqlite3.Row' object has no attribute 'get'`. Search was completely broken from the start
- First dev eval after fix: **1/20 = 5%** (only UID0010 passed, using model training knowledge for JPY exchange rate)
- Root cause: model ran out of iterations (15 turns), tables too wide for context, `round()` bug, table structure confusion

**The pivotal archaeological discovery:**
- Dug through old repos and found a **previous system using OpenCode harness scored 60%** (12/20) with MiniMax
- That system used `grep`, `bash`, `read` on raw text files — NOT MCP tools
- MCP tools were configured but MiniMax **chose grep over MCP every time**
- OpenHands SDK scored **0% on every multi-question run** — only OpenCode worked

**Cursor sessions (3):**
- Deep codebase audit: found `get_multi_year_series` burns search budget, `extract_values` omits `value_scaled`, `verify_answer` is weak, dead code everywhere
- Found CSV has 246 rows (not 789 — multiline fields inflate count)
- Tested WebFetch availability — not available in local harness, only in actual OpenCode container

**Key decisions:**
- Switch to OpenCode harness with grep+bash as primary, MCP as supplement
- Fix compute_expression (list syntax, `^` operator, `round()`)
- Ported 3 high-value tools from archive: `extract_values`, `get_multi_year_series`, `resolve_agency_alias`

### Day 2: March 31 — Database Rebuild & First Submission

**Commits:** 23 | **Claude sessions:** 2 (4581b00d, 788a6ac2)

**What happened:**
- Discovered MCP tools were NOT connecting inside Arena containers. Root cause: `/installed-agent/run_mcp.sh` **did not exist** — file was at `/opt/officeqa/run_mcp.sh`
- Docker image had Python 3.12 but **NO pip and NO uv** — couldn't install dependencies
- Built a **zero-dependency MCP server** (`mcp_stdio.py`) using only Python stdlib (JSON-RPC 2.0 over stdin/stdout)
- Discovered the baked-in SSE MCP server on port 8080 was accessible via Docker bridge (`172.17.0.1:8080`)
- Built lean DB: 928MB → 434MB compressed, 92K tables, 203K metric aliases
- Found `extract_values` was **broken** — always returned empty because it discarded `table_pk` and filtered by `row_label=metric` when metrics were in COLUMNS
- Built fixed SSE server on port 8081: **3/5 = 60%** with correct tools
- Added cell indexes: **query_table_rows went from 12.3s → 0.008s (1500x speedup)**

**Critical observation:**
> "MCP availability CORRELATED WITH FAILURE. When MCP was available, the model wasted 10-21 steps on bad search results. When MCP was unavailable, it immediately fell back to grep and succeeded faster."

**First real submission:** Score = **135.5** (140/246 = 56.9%, avg latency 189.3s, cost $30.03)

**Key lesson:** `max_iterations: 30` was WORSE than 15 — model did 80 steps going in circles

### Day 3: April 1 — Harness Wars

**Commits:** 26 | **Claude sessions:** 1 (9d3caddc)

**What happened:**
- Tested Goose vs OpenHands harnesses side by side
- `verify_answer` fix was a big win — Goose went from wrong to correct on multiple questions
- Built Meta-Harness architecture: state machine with SEARCH → COMPUTE → SUBMIT phases
- Compact schema returns to prevent context inflation
- `route_question` tool for path-based execution (Ledger vs Table)
- Per-path budgets and deterministic finalizer
- Structured severity-based warnings in `verify_answer`
- Added timing/cost tracking: Goose ~$0.001-0.003 per question (very cheap)

**Submission scores:** 147–167 range

### Day 4: April 2 — The Big Fix Day (+13 points)

**Commits:** 27 | **Claude sessions:** 2 (d7969b6d, 8bc8ffc3)

**What happened — 6 critical bug fixes (+13.2 points):**
1. `query_table_rows` key mismatch (was silently returning wrong data)
2. Silent fallback masking errors
3. Budget mismatch 22→20 (tools exceeded budget)
4. System prompt tool references out of sync
5. Decade parsing bug
6. Tool exposure inconsistencies

**Before:** 167.2 (64.6%) → **After:** 180.4 (68.7%)

**Pivoted to no-MCP pipeline:**
- Built `nomcp/` directory — pure Python pipeline, grep on raw TXT
- Achieved **80% local accuracy** (12/15 passing after timeout fixes)
- 2 real failures, 5 timeouts (MiniMax kept searching instead of computing)
- Root cause of timeouts: "Successful runs: 5-8 commands, ~50s. Timeouts: 30+ commands, 180s+"

**Switched to openhands-sdk harness:**
- OpenCode gave MiniMax native tools alongside MCP; model bypassed MCP 6/7 runs
- Top 4 leaderboard teams all used openhands-sdk
- OpenHands gives Terminal/FileEditor/TaskTracker + MCP

**Tarball challenges:**
- Arena max 200MB. Slim DB without cell_blobs = 118MB gzip
- Tried zstd level 19 and 22 compression
- Confirmed MCP tools already in tarball but curl bootstrap unnecessary

### Day 5: April 3 — Search Improvements & Major Cleanup

**Commits:** 21 | **Claude sessions:** 4 (22603b67, 332ed12e, 84957ade, f1550ae1) | **Cursor sessions:** 4

**What happened:**
- **Shipped broken empty DB** (30MB instead of 118MB) by mistake in one submission
- Added network probing: confirmed `allow_internet = true`, outbound HTTP works
- Built deterministic solve pipeline: Python parses → searches → extracts → computes (no LLM in loop)
- Period-aware table selection, multi-term row scoring, vertical serialization
- Keyword index with filename-based year filtering

**Search recall testing:**
- Before fixes: 47% (14/30) → After: 63% (19/30)
- All 246 questions have source files present in corpus
- 85% of answers in year or year+1 bulletins
- Finding the right file is the #1 bottleneck

**Massive cleanup commit:** Removed ~886K lines (860 results files, 30 archive files, 26 docs)

**Cursor sessions:**
- Pulled 241 arena traces from submission v3.0.0_166
- Built `scripts/audit_traces.py` to scan for agent signals
- Key finding: manifest.json is **pre-selected files**, not entire corpus
- 236/241 traces had manifest signals; 0 had MCP tool-name hits
- Documentation cleanup: found README has dead links, skills_dir set but skills removed, secrets in git

**Multi-pipeline idea explored:**
- Run deterministic + decompose + analyst in parallel, vote on answers
- Year penalty scoring fix: -1 → -5 per year distance, +8 exact year bonus
- Consensus baseline: 2/10 → 3/10 (modest improvement)

### Day 6: April 4 — Bottom-Up Pipeline & MCP v2

**Commits:** 16 | **Claude sessions:** 3 (0ac755dc, 36cc047c, 1afe14d1) | **Cursor sessions:** 1

**Bottom-up component pipeline (Wave 1 + Wave 2):**
- FileLocator → TableFinder → ColumnFinder → RowFinder → ValueParser → DataNormalizer → TableValidator → ConsensusVoter → SearchTask → Orchestrator
- 7 new files, 2,676 lines of code
- Three-strategy voting: Strict/Fuzzy/Contextual with consensus
- Confidence: pessimistic (min of pieces), boosted by agreement (+15% all agree, +5% two agree)
- 103/103 component tests passing
- BUT: E2E results nondeterministic — UID0001 gave 2,602 (correct) in one run, 23,280 in another

**MCP Server v2:**
- Markdown table parser (82 tables from Databricks-transformed TXT)
- FTS5 + difflib fuzzy search
- Smart header flattening (extracts leaf labels from `Parent > Child > 1940` paths)
- Collapsed 9 tools → 5: find_values, read_page, compute, verify, submit_answer
- Full DB: 677K master_ledger + 131K YoY changes + 92K table_index + 204K metric_aliases

**A/B test: grep-primary vs MCP-first:**
- Mixed results — some questions better with grep, others with MCP
- Grep more consistent on simple lookups
- MCP occasionally found data grep missed but less reliable overall

**Cursor session:** Ran bottom-up pipeline against 30-question test set. Created .env from arena.yaml credentials. Pipeline hung mid-execution (slow API calls).

### Day 7: April 5 — v7 Research & Submission (~6 hours)

**Claude sessions:** 1 (massive, documented in docs/v7-research-findings.md)

**Key discoveries:**

1. **Skills format breakthrough:** Previous 485 traces had **0% skill usage** because flat `skills/*.md` files are silently ignored by Goose. Correct format: `skills/skill-name/SKILL.md` with YAML frontmatter + `summon` extension

2. **Inline CPI proven:** Embedded CPI-U annual averages (1929-2024) directly in prompt. UID0005 passed 3x with inline CPI, zero curl calls. Eliminates CPI hallucination failure mode

3. **Arena test ≠ arena submit:** `arena test` doesn't copy skills or project files into Docker container. Skills can ONLY be validated through actual submissions

4. **MCP server abandoned:** stdio MCP failed with "process quit before initialization" when handed to OpenRouter MiniMax

5. **v5→v6 regression explained:** 9-point drop was mostly API flakiness, not prompt changes

6. **max_turns not enforced:** Harbor doesn't generate goose command with `--max-turns` flag. Real limit is the 300s timeout

7. **todo_write waste:** 479 todo_write calls across 244 tasks (~2 per task), consuming turns for no value

8. **JSON parsing loops:** 8 tasks got stuck repeating JSON parse commands 30-70 times on 460KB files

9. **Oracle resources:** Arena provides minimal files: manifest.json + 1-5 page TXT files (~8KB each) + full corpus as fallback

10. **When MiniMax answers, it's usually right:** 8/9 confident answers were correct. The problem is getting it to the right data

**v7 submission:**
- 5 SKILL.md skills: cpi-reference, computation-patterns, fiscal-calendar, data-files, table-extraction
- Inline CPI in 340-word prompt
- No MCP server, no DB
- 28KB tarball
- Submission ID: `7d9fca03-f947-4750-9d17-93902cf6a09c`

### Day 8: April 6 — Final Submissions & Ceiling Confirmed

**5 submissions in one day**, all confirming the 64-69% band:

| Version | Score | Pass Rate | Key Change |
|---------|-------|-----------|------------|
| v7 | 184.3 | 69.4% | Skills + inline CPI (skills dead in arena) |
| v8 | 172.7 | 64.2% | CPI inline + negative instructions (regression) |
| v9 | 174.6 | 65.4% | Stripped prompt, no MCP |
| v10 | 180.1 | 68.5% | Ultra-minimal 3-line prompt |
| v12 | 181.0 | ~68% | Minimal + file-drop backdoor |

**Key discoveries:**
- MCP tools never connected across any arena submission (~4,400 evaluations)
- Skills confirmed dead (summon extension not in Harbor recipe)
- Negative instructions backfire (MiniMax curls MORE when told not to)
- File-drop backdoor works (MCP args write files to /app/resources/ before goose starts)
- 70% is MiniMax's hard ceiling; score band is noise, not signal
- Local A/B testing was biased by MAX_TURNS=25 (arena runs uncapped)

---

## 3. Architecture Evolution

### Generation 1: MCP + SQLite DB (Days 1-2)
```
arena.yaml → OpenCode → run_mcp.sh → MCP SSE Server → 11GB SQLite
                                          ↓
                                    7 tools: search_tables, query_table_rows,
                                    get_file_structure, get_table_profile,
                                    compute_expression, get_cpi_index,
                                    get_fiscal_year_bounds
```
**Result:** 5% → 55% after fixes. Infrastructure fragile, model bypassed MCP for grep.

### Generation 2: Meta-Harness + Anti-Spin (Day 3)
```
arena.yaml → Goose/OpenHands → State Machine (SEARCH→COMPUTE→SUBMIT)
                                     ↓
                              route_question → Ledger path / Table path
                              Per-path budgets, deterministic finalizer
                              Severity-based warnings in verify_answer
```
**Result:** 147–167 pts. State machine partially worked but MiniMax ignored phase blocks.

### Generation 3: No-MCP Shell Pipeline (Day 4)
```
arena.yaml → Goose → solve.py → grep raw TXT → Python compute → answer.txt
                      ↓
                build_index.py (keyword index, 86K tables, 26s build)
                Deterministic route vs decompose route with fallback
```
**Result:** 80% local, 180.4 pts submitted. Simple and reliable.

### Generation 4: Bottom-Up Component Pipeline (Day 6)
```
Question → Decompose → [Piece 1, Piece 2, ...]
                             ↓
Each piece → FileLocator → TableFinder → RowMatcher → ColumnFinder
                                                          ↓
                                              3× parallel: Strict/Fuzzy/Contextual
                                                          ↓
                                                   ConsensusVoter → Value
                                                          ↓
                                              Orchestrator combines pieces
```
**Result:** 103/103 component tests pass but E2E nondeterministic. Not submitted.

### Generation 5: MCP Server v2 + FTS5 (Day 6)
```
arena.yaml → Goose/OpenHands → MCP stdio → 5 tools over FTS5+difflib
                                    ↓
                          find_values, read_page, compute, verify, submit_answer
                          677K master_ledger, 131K YoY, 92K table_index
```
**Result:** Mixed A/B results vs grep. Not reliably better.

### Generation 6: MCP Server v3 + find_evidence (Day 6-7)
```
arena.yaml → OpenHands → MCP stdio → find_evidence (LLM sub-calls)
                                         ↓
                              Decomposes question → sub-queries
                              list_tables → get_table → compute
```
**Result:** Abandoned after stdio MCP failed with MiniMax ("process quit before initialization").

### Generation 7: Skills + Inline CPI (Day 7) — Current
```
arena.yaml → Goose → 340-word prompt (inline CPI) → shell grep/cat/sed
                ↓
         5 SKILL.md skills (fiscal-calendar, cpi-reference,
         computation-patterns, data-files, table-extraction)
         loaded via Goose summon extension on demand
```
**Result:** v7 submitted, awaiting results. Expected 185-195.

---

## 4. Every Submission & Score

| Version | Date | Harness | Architecture | Score | Pass Rate | Avg Latency | Notes |
|---------|------|---------|-------------|-------|-----------|-------------|-------|
| arena-v0.6 | Mar 31 | OpenCode+SSE MCP | 7 MCP tools, 11GB DB | ~151.8 | 63.0% | 189s | First submission |
| arena-v0.7 | Apr 1 | Goose+SSE MCP | Meta-Harness, state machine | ~147.1 | 62.5% | — | Regression |
| arena-v0.8 | Apr 1 | OpenHands+MCP | Same tools, different harness | ~148.9 | 56.5% | — | Worst score |
| arena-v0.9 | Apr 2 | Goose | Tool fixes, anti-spin | ~167.2 | 64.6% | — | Recovery |
| arena-v1 | Apr 2 | Goose | 6 bug fixes, nomcp pipeline | ~180.4 | 66.9% | — | Best (at time) |
| nomcp-v2 | Apr 3 | Goose | grep-primary, keyword index | ~189.2 | 65.5% | — | Competitive |
| v3 MCP | Apr 3 | Goose+MCP | First MCP test | ~166 | 61.9% | 59.5s | Broken empty DB |
| v4 | Apr 4 | OpenHands+MCP | find_evidence + LLM sub-calls | — | — | — | Unknown result |
| v5 | Apr 4 | Goose | Shell grep, no MCP | **184.5** | **75%** | 185s | **Best ever** |
| v6 | Apr 5 | Goose | Prompt tweaks | ~175 | — | — | Regression (API flaky) |
| v7 | Apr 6 | Goose | Skills + inline CPI | TBD | TBD | TBD | Submitted, awaiting |

**15-run stability analysis (246 tasks):**
- ALWAYS_PASS: 56 tasks (22.8%) — reliable wins
- USUALLY_PASS (≥80%): 75 tasks (30.5%) — mostly stable
- FLIP_FLOP (30-70%): 38 tasks (15.4%) — nondeterministic
- USUALLY_FAIL (<30%): 40 tasks (16.3%) — hard problems
- ALWAYS_FAIL (0%): 37 tasks (15.0%) — systematic gaps

---

## 5. Claude Code Sessions (Chronological)

### Session 1: `626b1888` — Mar 30, 08:41 (5.0MB)
**Phase 1 Review & First Eval**
- Found and fixed search_tables crash bug
- Ran 789-question search audit: table_term_index had 8.7M rows, searches took 1-2s each
- Simplified search scoring from ~500 lines to transparent fast scoring
- First dev eval: 1/20 = 5%
- Archaeological discovery: old OpenCode system got 60% using grep, not MCP
- OpenHands SDK scored 0% on every run
- Decision: switch to OpenCode + grep primary

### Session 2: `00cb13ab` — Mar 30, 12:32 (4.3MB)
**MCP Connection & Tool Fixes**
- Ported extract_values, get_multi_year_series, resolve_agency_alias
- Discovered MCP not connecting: wrong path in container
- Docker had Python 3.12 but no pip/uv
- Built zero-dependency MCP server (stdlib only)
- Found SSE server accessible via Docker bridge
- First MCP test: uid0199 PASSED (previously always failed)
- Full test: 5/20 = 25%
- Built fixed SSE server: 3/5 = 60%
- Key finding: MCP availability correlated with FAILURE

### Session 3: `661a33c9` — Mar 30, 14:04 (3.5MB)
**DB Optimization & First Submission Push**
- Built lean DB (4.8GB → 581MB compressed)
- Cell indexes: query time 12.3s → 0.008s (1500x speedup)
- 4/4 = 100% on test, 11/20 = 55% on full set
- First submission: 135.5 pts (57%)
- max_iterations: 30 worse than 15

### Session 4: `4581b00d` — Mar 31, 07:00 (3.8MB)
**New DB Build**
- Re-ingested 697 Treasury Bulletin JSONs
- 92,425 tables, 203,384 metric aliases
- Built enriched evidence payloads with period_basis

### Session 5: `788a6ac2` — Mar 31, 13:54 (4.1MB)
**Submission Struggles**
- Tarball 1.4GB — had to exclude DB (615MB)
- Fought .arenaignore issues (arena CLI ignores it)
- Submitted successfully

### Session 6: `9d3caddc` — Apr 1, 18:36 (4.1MB)
**Dual Harness Testing**
- Goose vs OpenHands side by side
- verify_answer fix was big win
- get_multi_year_series prompt tuning
- Cost tracking: $0.001-0.003/question

### Session 7: `d7969b6d` — Apr 2, 15:58 (4.1MB)
**DB Compression**
- zstd level 19 and 22 for tarball budget
- Confirmed 200MB arena limit
- 2 submissions/day, resets midnight UTC

### Session 8: `8bc8ffc3` — Apr 2, 19:45 (3.7MB)
**No-MCP Pivot**
- Built nomcp/ directory, pure Python
- 80% local accuracy (12/15 after timeout fixes)
- Root cause: MiniMax keeps searching instead of computing
- Added auto-answer fallback

### Session 9: `22603b67` — Apr 3, 04:04 (5.3MB)
**Broken DB Submission & Network Probing**
- Shipped broken empty 30MB DB by mistake
- Added network probes: HTTP access, filesystem layout, tool availability
- Confirmed allow_internet = true
- Planning external DB hosting

### Session 10: `332ed12e` — Apr 3, 13:01 (8.8MB) — LARGEST
**Search Improvements & Vertical Serialization**
- Running at 153/242 (63.2%), below best 189.2
- Vertical serialization of tables
- Wider grep context, better prompts
- Keyword index with year filtering
- Header parsing issues (numbers as column names)
- Search ranking still wrong sometimes

### Session 11: `84957ade` — Apr 3, 15:26 (7.2MB)
**Deterministic Solve Pipeline**
- Python parses → searches → extracts → computes
- Submitted: ID 3c6c395b
- 2 remaining submissions that day

### Session 12: `f1550ae1` — Apr 3, 20:12 (3.8MB)
**Multi-Pipeline & Consensus**
- LLM search planner: one cheap call → {pub_years, keywords}
- 3-way parallel pipeline with voting
- Year penalty: -1 → -5 per year, +8 exact match bonus

### Session 13: `e654f952` — Apr 3, 23:09 (4.1MB)
**Flakiness Analysis**
- DEC_01 flaky, DEC_08 different wrong numbers each run
- Merged scorer fix PR
- Pushed continuation prompt

### Session 14: `0ac755dc` — Apr 4, 11:36 (4.7MB)
**Component Fixes & E2E Testing**
- 7 bug fixes: CY fallback, column word-boundary, consensus improvements
- 103/103 component tests passing
- E2E nondeterministic: same question → different answers across runs

### Session 15: `36cc047c` — Apr 4, 12:50 (4.5MB)
**MCP Server v2 Build**
- Markdown table parser (82 tables from Databricks TXT)
- FTS5 + difflib fuzzy search
- Smart header flattening
- Key insight: corpus TXT files ARE the Databricks-transformed files

### Session 16: `1afe14d1` — Apr 4, 23:34 (5.2MB)
**A/B Testing grep vs MCP-first**
- 20-question comparison
- Mixed results: grep more consistent, MCP sometimes finds missed data
- Both on 10-turn limit

### Session 17: v7 Session — Apr 5 (~6 hours)
**Skills Discovery & v7 Submission**
- Skills format breakthrough (subdirectory/SKILL.md)
- Inline CPI proven (uid0005 passed 3x)
- Arena test ≠ submit confirmed
- MCP server abandoned
- v7 submitted with 5 skills + inline CPI

---

## 6. Cursor Sessions

### Cursor 1: `5d6098c3` — Mar 30 (6 lines)
**Deep Codebase Audit**
- Architecture mapping: arena.yaml → OpenCode → run_mcp.sh → MCP → OfficeQATools
- Found: extract_values omits value_scaled, verify_answer weak, dead code
- CSV confirmed 246 rows (multiline fields inflated count)
- WebSearch not available in repo

### Cursor 2: `840ef7cd` — Mar 30 (2 lines)
**Shorter Audit (duplicate)**
- Architecture map: 10 layers identified
- src/agent.py only registers 7 function tools vs full MCP surface
- Local eval path differs from submission path

### Cursor 3: `a8958127` — Mar 30 (6 lines)
**WebFetch Testing**
- OpenCode has webfetch and websearch tools
- Can't validate locally — need actual Arena container
- No .env file exists; secrets from env vars or agent.env

### Cursor 4: `62e1a49c` — Apr 3 (30 lines)
**Documentation Cleanup**
- ARCHITECTURE.md mostly current
- README has dead links to docs/running.md and docs/runner-pool.md
- README claims skills removed but arena.yaml sets skills_dir
- Root arena.yaml describes different harness than analyst/nomcp configs

### Cursor 5: `a2b3e6a4` — Apr 3 (4 lines)
**ARCHITECTURE.md Review**
- Strengths: end-to-end story, actionable diagnosis
- Gaps: README mismatch, fragile line citations, **secrets in git** (plaintext API keys)
- Plan: unify onboarding, remove secrets, add .env pattern

### Cursor 6: `ea940185` — Apr 3 (4 lines)
**ARCHITECTURE.md Review (duplicate)**
- Flagged secrets in git as highest priority
- Suggested routing diagram: question type → recommended pipeline
- Recommended function-name citations over line numbers

### Cursor 7: `c23ee795` — Apr 3 (1 line)
**Empty/greeting** — No substantive work

### Cursor 8: `285c68ce` — Apr 3 (42 lines)
**Bottom-Up Pipeline Test Run**
- Modified test_pipeline.py for CSV/limit support
- First run: 0/30 (no API key)
- Created .env from arena.yaml credentials
- Second run got real answers but appeared to hang
- Clarified Goose overhead: system prompt stable, conversation grows per turn

### Cursor 9: `247083b3` — Apr 4 (83 lines) — LARGEST
**Trace Recovery & MCP Plan**
- Fixed pull_arena_traces.py, pulled 241 traces from v3.0.0_166
- Built audit_traces.py for trace scanning
- Updated all 6 arena.yaml files for consistency
- Created Dedalus MCP SSE Docker deployment
- Key finding: manifest.json = pre-selected files, not full corpus
- 236/241 traces had manifest signals; 0 had MCP tool mentions

---

## 7. Key Technical Discoveries

### Discovery 1: MiniMax Prefers Grep Over MCP
**When:** Day 1 (Mar 30)
**Impact:** Fundamental architecture shift

MiniMax M2.5, when given both MCP tools and shell access, chose grep/cat/sed **every time**. In the old OpenCode system, MCP was configured but unused — the model scored 60% purely through shell commands on raw text files.

When MCP was made the only option (by blocking shell), accuracy dropped. When both were available, MCP caused the model to waste 10-21 steps on bad search results instead of falling back to the reliable grep approach.

### Discovery 2: More Iterations = Worse Results
**When:** Day 2 (Mar 31)
**Impact:** Changed iteration limits permanently

Setting `max_iterations: 30` caused MiniMax to do 80 steps going in circles instead of committing to an answer. Successful runs used 5-8 commands in ~50 seconds. Timeout failures used 30+ commands over 180+ seconds. The model's over-searching behavior couldn't be fixed by prompting — it was structural.

### Discovery 3: The Terminal Override
**When:** Day 4 (Apr 2)
**Impact:** Forced MCP usage

`sitecustomize.py` via PYTHONPATH patches OpenHands' TerminalTool to block data retrieval. Without it, MiniMax used Terminal exclusively (50+ commands, 0 MCP calls, 0% accuracy on MCP-dependent questions). Allow-list: writing answer.txt, pip/apt, basic fs ops. Everything else returns an error suggesting MCP tools.

### Discovery 4: Skills Format Must Be Subdirectory/SKILL.md
**When:** Day 7 (Apr 5)
**Impact:** Explained why skills never loaded in 485+ traces

Goose's `summon` extension only discovers skills in `skills/skill-name/SKILL.md` format with YAML frontmatter. Flat `skills/*.md` files are **silently ignored**. This explains why v5-v6 had 0% skill usage despite configuration. Previous 485 traces across all versions showed zero load() calls.

### Discovery 5: Arena Test ≠ Arena Submit
**When:** Day 7 (Apr 5)
**Impact:** Changed validation workflow

`arena test` only puts `install.sh` in `/installed-agent/`. Skills copy fails silently. `/app/resources/` is empty. `arena submit` extracts the full tarball, copies skills, populates resources with oracle files. Skills can ONLY be validated through actual submissions and trace analysis.

### Discovery 6: Manifest.json Contains Pre-Selected Files
**When:** Day 5 (Apr 3, Cursor session)
**Impact:** Changed understanding of task structure

The Arena provides each task with a manifest.json pointing to **pre-selected files** (oracle pages), not the entire corpus. Tasks get: manifest.json + 1-5 page TXT files (~8KB each) + full corpus as fallback. The page files are the highest-signal data source (76.3% pass rate when used first).

### Discovery 7: CPI Hallucination Is Avoidable
**When:** Day 7 (Apr 5)
**Impact:** Eliminated a failure category

When MiniMax needs CPI data and doesn't have it inline, it attempts to curl BLS/FRED APIs, fails, then hallucinates values (e.g., 255.7/259.2 for May/Jun 1979 — wrong). Embedding CPI-U annual averages (1929-2024, ~50 values) directly in the prompt eliminates this failure mode entirely. UID0005 passed 3/3 times with inline CPI, zero curl calls.

### Discovery 8: 41% of Tables Have NULL Year Metadata
**When:** Day 3 (Apr 1)
**Impact:** Required fallback search strategy

The keyword index missed 41% of tables because year metadata was NULL. Added `_direct_label_search` (LIKE on column_label, 452K rows) as fallback. This improved overall search recall but widened candidate pools, causing scorer flakiness on some questions.

### Discovery 9: Corpus TXT Files = Databricks Transformed
**When:** Day 6 (Apr 4)
**Impact:** Simplified parsing

The TXT files Sentient provides in /app/corpus/ are the Databricks-transformed versions of the source JSONs. They already have flattened markdown headers with `>` separators and pipe-delimited tables. No need to parse JSON — the TXT files ARE the canonical structured format.

### Discovery 10: todo_write Wastes Turns
**When:** Day 7 (Apr 5)
**Impact:** Identified ~2 wasted turns per task

479 `todo_write` calls across 244 tasks (~2 per task) in v6 traces. Each call consumes a turn for no analytical value. MiniMax uses it as a "thinking" mechanism but it produces no useful output.

---

## 8. Failure Analysis Deep Dives

### 8.1 Stability Report (246 tasks × 15 runs)

**Version evolution:**
| Version | Pass Rate | Runs |
|---------|-----------|------|
| arena-v0.6 | 63.0% | 2 |
| arena-v0.7 | 62.5% | 3 |
| arena-v0.8 | 56.5% | 1 |
| arena-v0.9 | 64.6% | 1 |
| arena-v1 | 66.9% | 4 |
| nomcp-v2 | 65.5% | 4 |

**Difficulty is destiny:**
- Hard questions: 41% success rate, 89% of always-fail cohort
- Easy questions: 89% success rate, only 3.5% always-fail
- Hard questions are **7x more likely** to always-fail (25% vs 3.5%)

### 8.2 v5 Failure Breakdown (31 UIDs analyzed)

| Category | Count | % | Fixable? |
|----------|-------|---|----------|
| Wrong data extraction | 15 | 43% | Partially — better search |
| Ambiguous interpretation | 7 | 20% | Partially — FY/CY/sign rules |
| Wrong formula/method | 5 | 14% | Yes — formula cheat sheet |
| Insufficient data | 3 | 9% | No — data doesn't exist |
| Format errors | 3 | 9% | Yes — output formatting |

**Specific failure archetypes:**
- CPI hallucination: 4-6 cases (fixed by inline CPI)
- FY/CY confusion: 5-8 cases (addressed by fiscal-calendar skill)
- Wrong row/column: 15-20 cases (evidence selection bottleneck)
- JSON parsing loops: 8 cases (addressed by data-files skill saying "avoid JSON")
- No answer (timeout): 23 cases (model over-searching)

### 8.3 Always-Fail Analysis (37 tasks, 15 sampled)

**Root causes:**
1. **Data source/table confusion (33%):** Wide tables (10+ columns) cause misreads. Yield spread, T-bill offerings, bond types easily confused
2. **Missing sources (27%):** NBER paper dates not in corpus, Treasury 2025 forecasts unpublished, external lookups required
3. **Calculation errors (20%):** Percent difference vs change, Theil index, polynomial regression
4. **Wrong time period (13%):** Proposal month vs FY, bond yield misclassification

**Critical finding from detailed trace analysis:**
> ALL 15 sampled always-failing traces made **ZERO tool calls**. The agent hallucinated answers without searching data. This is a fundamental LLM behavior issue, not a data retrieval problem.

### 8.4 Flipped Traces (12 FAIL→PASS)

All 12 traces that flipped from FAIL to PASS showed:
- MCP tools called: **0/12** (none)
- All used shell commands (cat, grep, sed) + direct file access
- v4 had broken fallback detection; latest uses filesystem fallback correctly
- Execution ranged from 15 events (simple lookups) to 6,791 events (complex multi-search)

### 8.5 Decomposition Quality

- 246 questions decomposed, 242 successful (98.4%)
- Quality: 82.5% correct decomposition
- Critical issues:
  - 16 CY annual total mismatches (Treasury only has FY totals, need monthly sum)
  - 66 sum/total computation cases
  - 7 period type mismatches (CY vs FY)
- Key insight: parser can't decide operations without seeing table data. 46% correct, 25% partial, 29% wrong on operations field

---

## 9. Prompt Engineering Evolution

### v1: Generic Agent (Day 1)
```
You are a helpful assistant with access to MCP tools for querying
Treasury Bulletin data...
```
**Result:** 5%. Model ignored tools, hallucinated.

### v2: Imperative Rules (Day 2-3)
```
HARD RULE: NEVER use more than 2 search_tables calls.
MANDATORY SEQUENCE: search → extract → compute → submit.
WARNING: You have {budget} calls remaining. USE THEM WISELY.
```
**Result:** 55-63%. Model followed rules mechanically or ignored them entirely. Loud warnings made things WORSE (UID0127 regressed).

### v3: Situational Framing (Day 3-4)
```
You are a senior Treasury analyst. You've learned from experience
that the most common mistake is extracting from the wrong table.
Your typical workflow: identify the bulletin year, search for the
specific table, verify the column headers match...
```
**Result:** 65-69%. Model role-played effectively, natural workflow emergence.

### v4: Mentor Pattern (Day 4-5, Analyst pipeline)
```
You're reviewing your intern's work on a Treasury question.
Run solve_briefing.py to get their initial analysis, then verify
their data sources and calculations. If you find conflicts,
investigate further...
```
**Result:** 180.4 pts. Best structured approach — gave model a reason to verify.

### v5: Minimal Prompt + Skills (Day 7)
```
Answer this Treasury Bulletin question. Search /app/resources/ for data.
Use python3 -c for calculations. Write answer to /app/answer.txt.

CPI-U Annual Averages: 1929:17.2, 1930:16.7, ... 2024:314.2
Fiscal Year: Pre-1977 Jul-Jun, Post-1977 Oct-Sep.
```
**Result:** v7 submission, 340 words. Skills loaded on demand via Goose summon.

**Key prompt lessons:**
1. Situational framing > imperative rules (always)
2. Inline reference data > API calls (eliminates hallucination)
3. Shorter prompts > longer prompts (model follows better)
4. "Review intern's work" > "Do this yourself" (verification built in)
5. Budget text should be calm, not threatening ("You have a budget of 20 tool calls" not "WARNING: ONLY 20 CALLS LEFT")

---

## 10. Infrastructure & Tooling

### Servers & Droplets
- **DB/Telemetry droplet:** 147.182.206.223 (sfo3) — hosted SSE MCP server + SQLite DB
- **Runner droplet:** 143.198.129.204 — arena CLI execution
- **Daytona snapshot:** Pre-built with slim DB + Python deps for sandbox testing
- **Docker bridge:** MCP SSE accessible at 172.17.0.1:8080 from containers
- **Telemetry:** Fire-and-forget to webhook.site

### Database Versions
| Version | Size (raw) | Size (gz) | Tables | Records | Notes |
|---------|-----------|-----------|--------|---------|-------|
| v1 | 11GB | ~2GB | ~90K | millions | Full with cell_blobs |
| v2 (lean) | 4.8GB | 581MB | 92K | — | Without heavy tables |
| v3 (slim) | ~1GB | 118MB | 92K | — | No cell_blobs |
| v4 (enriched) | ~150MB | 68MB | 92K | 677K ledger | With precomputed YoY |
| v5 (FTS5) | ~200MB | — | 92K | 677K + FTS5 | Full-text search |
| None | 0 | 0 | — | — | **v7: no DB, grep only** |

### Key Scripts
| Script | Purpose |
|--------|---------|
| `scripts/poll_submission.py` | Poll Arena API for submission status |
| `scripts/pull_arena_traces.py` | Download agent trajectories (purged after ~24h) |
| `scripts/triage_traces_vs_stability.py` | Cross-reference traces with stability buckets |
| `scripts/audit_traces.py` | Scan traces for harness signals |
| `scripts/build_install.sh` | Bootstrap dependencies in container |
| `scripts/pre_submit.sh` | Pre-submission validation |
| `run_local_v7.sh` | Local harness replicating arena submit |
| `nomcp/run_local.sh` | No-MCP local runner |

### Harness Comparison
| Harness | Tools Given | MCP Support | Skills | Result |
|---------|------------|-------------|--------|--------|
| OpenCode | webfetch, websearch, shell, read, edit + MCP | Yes | No | 60% baseline (grep only) |
| OpenHands SDK | Terminal, FileEditor, TaskTracker + MCP | Yes | Subdirectory SKILL.md | 0% initially, improved later |
| Goose | Shell + MCP (optional) | Yes (stdio/SSE) | Subdirectory SKILL.md | **Best: 184.5 pts** |
| Codex | Unknown | Yes | Unknown | Not tested |

---

## 11. What Worked vs What Didn't

### What Worked

| Approach | Evidence | Impact |
|----------|----------|--------|
| **Shell grep on raw TXT** | 60% from Day 1, 184.5 pts at best | Core reliable approach |
| **Situational framing** | 65-69% vs 55% with imperative rules | +10% accuracy |
| **Mentor/review pattern** | 180.4 pts (analyst pipeline) | Built-in verification |
| **Inline CPI data** | uid0005: 3/3 pass, 0 curl calls | Eliminated hallucination category |
| **Vertical serialization** | Wide tables became readable | Solved table confusion |
| **Bug fixes (Day 4)** | 167→180.4 pts (+13.2) | Single biggest score jump |
| **Lean iteration limits** | 8-15 turns optimal | Prevented over-searching |
| **Period-aware search** | 47%→63% recall | Better file selection |
| **Page files first** | 76.3% vs 70.9% pass rate | Best data source strategy |
| **Keyword index (26s build)** | 86K tables searchable | Fast file location |

### What Didn't Work

| Approach | Evidence | Why It Failed |
|----------|----------|---------------|
| **MCP tools as primary** | Correlated with failure | Model wasted steps on bad results |
| **11GB database** | Too large for tarball, slow queries | Infrastructure overhead |
| **Imperative prompt rules** | "MAX 2 calls" → model stops at exactly 2 or ignores | MiniMax interprets mechanically |
| **Anti-spin warnings** | Made results WORSE (UID0127 regressed) | Model treats warnings as error signals |
| **max_iterations: 30** | 80 steps going in circles | More rope = more hanging |
| **Bottom-up pipeline** | Nondeterministic E2E, correct→wrong across runs | Complexity without reliability |
| **Consensus voting** | 2/10→3/10 baseline improvement | Modest gain, high cost |
| **OpenHands SDK** | 0% initially, unreliable | System prompt conflict, tool issues |
| **Terminal blocking** | 50+ commands, 0 MCP, 0% accuracy | Model worked around restrictions |
| **Flat skills/*.md** | 0/485 traces loaded | Wrong format (silently ignored) |
| **todo_write** | 479 calls, 0 value | Wasted ~2 turns per task |
| **JSON file access** | 48% wasted first turn on 460KB JSON | Too large, parsing loops |
| **FTS5 + difflib** | Mixed A/B results | Not reliably better than grep |
| **MCP server v3 (stdio)** | "process quit before initialization" | MiniMax can't handle stdio MCP |

### What Was Promising But Unfinished

| Approach | Status | Potential |
|----------|--------|-----------|
| **Proper SKILL.md skills** | Submitted in v7, awaiting results | Could add +10-15 pts |
| **Oracle page-first routing** | Identified but not enforced | 76.3% vs 70.9% pass rate |
| **Deterministic solve pipeline** | Built but search ranking unreliable | Could eliminate flaky category |
| **Formula cheat sheet (skills)** | In v7 skills, untested in arena | Fixes 3-5 UIDs directly |
| **External DB hosting** | Network confirmed available | Could bring back DB advantages |

---

## 12. Lessons Learned

### On Model Behavior
1. **MiniMax follows the path of least resistance.** If grep works, it will grep. If MCP is available alongside grep, it will still grep. Don't fight this — design for it.
2. **More tools = worse results.** 4 excellent tools > 9 mediocre ones. 1-5 tool calls = 82% pass, 11+ calls = 60% pass.
3. **When MiniMax answers confidently, it's usually right (8/9).** The problem is getting it to the right data, not reasoning about the data.
4. **MiniMax can't do controlled tool loops.** Sends empty args, uses all budget on grep, never calls submit. Don't give it a loop — do search/extraction deterministically, give one focused call.
5. **Prompt-based phase transitions don't work.** MiniMax ignores "stop searching" even with escalating urgency. Must enforce structurally (hard-block tools at server level).

### On Architecture
6. **The bottleneck is evidence selection, not arithmetic.** 100% of pipeline failures are wrong table/row/column selection. When given correct data, MiniMax computes correctly.
7. **Simpler always wins.** Shell grep consistently outperformed MCP, databases, multi-stage pipelines, and component architectures.
8. **Inline reference data eliminates hallucination categories.** CPI, fiscal year rules, formula references — embed them, don't make the model look them up.
9. **Deterministic > interactive for data retrieval.** Python searching is more reliable than LLM-guided exploration. Use LLM for reasoning, Python for retrieval.

### On Infrastructure
10. **Always commit before submit.** When submissions regress, can't tell what changed without git trail.
11. **Arena test ≠ arena submit.** Never trust local validation — always verify through actual submission traces.
12. **Traces are purged after ~24h.** Pull immediately when submission completes.
13. **200MB tarball limit is real.** Design for lean submissions. v7 was 28KB.

### On Development Process
14. **Bug fixes had the highest ROI.** Day 4's 6 bug fixes added 13.2 points — more than any architectural change.
15. **Flaky tasks are better ROI than always-fail.** 57 flaky tasks (30-70% pass) respond to improvements. 37 always-fail tasks have systematic gaps that may be unfixable.
16. **Trace analysis is essential.** Without traces, you're guessing. Every major insight came from reading actual agent behavior in arena.

---

## 13. Appendix: Memory File Index

### Feedback (12 files)
| File | Key Rule |
|------|----------|
| feedback_prompt_philosophy.md | Situational framing > imperative rules |
| feedback_evidence_selection.md | Bottleneck is evidence selection, not arithmetic |
| feedback_minimax_ignores_prompts.md | Hard-block tools at server level, not via prompts |
| feedback_minimax_tool_loop.md | Single focused LLM call, not tool loop |
| feedback_no_token_limits.md | Don't cap max_tokens (was truncating at 4000) |
| feedback_chatgpt_review.md | 4 tools > 9; stabilize flaky before fixing always-fail |
| feedback_openhands_sysprompt.md | OpenHands injects 12K char default that conflicts |
| feedback_commit_before_submit.md | Always git commit before arena submit |
| feedback_search_all_months.md | Search ALL months, not just planner-suggested |
| feedback_minimax_exploration.md | Don't prescribe file access; MiniMax finds pages naturally |
| feedback_skills_format.md | Must use subdirectory/SKILL.md with frontmatter |
| feedback_arena_test_vs_submit.md | Test doesn't copy skills; submit does |

### Project (27 files)
Covering: parse findings, terminal override, tool fixes, status updates, anti-spin results, openhands switch/migration, orchestrator architecture, nomcp results, grep-primary v2, search improvements, submissions (apr3, v3, v4, v5, v7), consensus pipeline, scorer flakiness, search recall, test case findings, decompose pipeline, decomposition v3 issues, e2e results, component fixes, wave2 wiring, MCP server v3, trace analysis, goose MCP findings, v7 findings.

### Reference (11 files)
Covering: infrastructure (droplets, Daytona), arena limitations (.arenaignore), corpus contents (raw TXT only), tarball limits (200MB), question types (55% sums, 39% CY), officeqa repo (databricks/officeqa), harness options (4 harnesses), task structure (container layout), pull traces (arena-cli async API), goose skills format, local harness (run_local_v7.sh).

---

## Appendix: Git Statistics

- **Total commits:** 129
- **Date range:** March 29 – April 4, 2026
- **Branches:** main + 2 merged PRs (setup-arena-challenge, fix-table-scorer)
- **Stashes:** 2 (Day 2 comparison fixes, Day 2 tool improvements)
- **Largest commit:** 69,157 insertions (MCP server + trace analysis)
- **Largest cleanup:** 886K lines removed (old results, archive, docs)
- **PR #1:** "Add standalone OfficeQA runner with prompt experimentation framework" (Apr 2)
- **PR #2:** "Improve bulletin year matching scoring and update corpus paths" (Apr 3)

**Commits by day:**
| Day | Date | Commits | Focus |
|-----|------|---------|-------|
| 1 | Mar 29 | 1 | Initial setup |
| 2 | Mar 30 | 15 | Parsing, first eval, OpenCode discovery |
| 3 | Mar 31 | 23 | DB rebuild, first real submission |
| 4 | Apr 1 | 26 | Harness testing, Meta-Harness |
| 5 | Apr 2 | 27 | Bug fixes (+13 pts), no-MCP pivot |
| 6 | Apr 3 | 21 | Search improvements, major cleanup |
| 7 | Apr 4 | 16 | Bottom-up pipeline, MCP v2, A/B test |
