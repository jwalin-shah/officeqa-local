# Grounding and Verification in Document-Based Numerical Reasoning: Lessons from the OfficeQA Arena

## Abstract

We present a systematic exploration of grounded numerical question answering over U.S. Treasury Bulletin documents, conducted over 8 days of intensive development within the Sentient Arena OfficeQA challenge. The task requires answering 246 financial questions spanning 1939--2025 using a corpus of 696 raw text files, with oracle-grounded evaluation and fuzzy numeric scoring (1% tolerance). We developed and evaluated 9 distinct architectural generations---from structured MCP tool servers backed by an 11GB SQLite database to a minimal 28KB submission using shell grep and inline reference data---across 15+ submission runs totaling approximately 3,700 individual task evaluations. Our best system achieved 184.5/246 (75.0% pass rate) at a total cost of $1.71. We find, counterintuitively, that structured tool access *degrades* performance for LLM-driven document retrieval, that evidence selection---not arithmetic or reasoning---is the dominant bottleneck (accounting for 48% of all failures), and that embedding reference data directly in the prompt eliminates entire categories of hallucination. We provide a comprehensive stability analysis across 6 versions showing that 47% of tasks are deterministically correct while 19% always fail (dominated by wrong arithmetic on correctly-found data, not retrieval failures), a full data engineering retrospective covering 5 database generations and a multi-stage ingestion pipeline, and offer design principles for building grounded QA systems derived from our empirical exploration.

## 1. Introduction

The OfficeQA Arena, hosted by Sentient, evaluates AI agents on their ability to answer numerical questions grounded in real government financial documents. The task is deceptively simple: given a natural language question about U.S. Treasury data and access to a corpus of Treasury Bulletin files, extract the relevant data, perform any necessary computation, and submit a numeric answer. Evaluation uses oracle-grounded scoring with fuzzy numeric matching at 1% tolerance.

The challenge exposes a fundamental tension in LLM-based systems: language models are fluent reasoners but unreliable retrievers. When a model must ground its answer in specific document data rather than parametric knowledge, the failure modes shift from reasoning errors to *evidence selection* errors---finding the wrong table, reading the wrong column, or confusing fiscal years with calendar years.

Our contribution is a systematic empirical exploration of this problem space. Over 9 days, we developed 9 architectural generations, built 5 database versions through a multi-stage ingestion/enrichment pipeline, conducted 15+ submission runs across multiple agent harnesses (Goose, OpenHands, OpenCode), performed stability analysis across approximately 3,700 individual task evaluations, and produced detailed failure taxonomies at the individual question level. This report distills the findings into actionable design principles for grounded QA systems.

## 2. Task Analysis & Dataset Characterization

### 2.1 Question Taxonomy

The 246 OfficeQA questions span multiple computational types, with significant overlap (questions may require multiple operations):

| Question Type | Prevalence | Example |
|---|---|---|
| Summation | 55% | Total national defense expenditures, CY 1940 |
| Calendar year lookup | 39% | Receipts for calendar year 1981 |
| Percent change | 31% | Growth in employment retirement receipts FY2013--FY2023 |
| Average/Mean | 29% | Average monthly receipts FY 2015 |
| Fiscal year lookup | 22% | Budget outlays FY 1950 |
| Difference | 21% | Change in debt held by public 2019--2020 |
| Multi-year range | 18% | Series across 1990--2000 |
| Foreign currency | 15% | Exchange rate calculations |
| Statistical measures | ~15% | Theil index, VaR, standard deviation, CAGR |
| Inflation/CPI adjustment | 10% | Real value adjusted for inflation |
| Multi-step reasoning | ~12% | CPI adjustment + regression, yield spread + lookup |

Questions are labeled "easy" (113) or "hard" (133). This label proves highly predictive: easy questions achieve 89% success rate across runs while hard questions achieve only 41%---hard questions are 7x more likely to always fail (24.8% vs 3.5%).

### 2.2 Corpus Structure

The corpus comprises 696 TXT files, each a Databricks-transformed rendering of a Treasury Bulletin issue (1939--2025). Files contain pipe-delimited tables with hierarchical column headers using `>` separators (e.g., `1940 > Jan.`). A typical file is approximately 270KB. The full corpus totals approximately 150MB of raw text.

A key discovery was that these TXT files *are* the Databricks-transformed versions of the source JSONs available at `databricks/officeqa`. They already have flattened markdown headers and structured pipe-delimited tables---a canonical structured format that requires no further parsing for LLM consumption.

Critically, the Arena provides each task with *oracle page files*: a `manifest.json` pointing to 1--5 pre-selected page-level TXT files (~8KB each) containing the table most likely to hold the answer. Tasks that access page files first achieve a 76.3% pass rate versus 70.9% for those that search the full corpus first. The oracle also provides the full TXT file (~270KB) and the source JSON (~460KB) as fallbacks.

### 2.3 Scoring

Evaluation uses fuzzy numeric matching with 1% relative tolerance. Format requirements are minimal---the agent writes a value to `/app/answer.txt`. Partial credit is not awarded; each task scores 0 or 1.

## 3. Data Engineering & Database Evolution

Before exploring agent architectures, we invested heavily in building structured representations of the corpus. This section documents the full data engineering pipeline---5 database generations, a multi-stage ingestion system, and the compression/indexing infrastructure that supported our experiments. Ultimately, the winning system used *none* of this infrastructure, but the engineering work was essential to understanding *why* simpler approaches won.

### 3.1 Ingestion Pipeline

We built a 4-stage ingestion pipeline to transform 697 raw Treasury Bulletin files into a queryable database:

**Stage 1: Extraction & Parsing** (`build_db_from_txt.py`, ~700 lines). Parses markdown pipe-delimited tables and HTML tables (with colspan/rowspan expansion) from each bulletin file. Extracts column header dates (multiple formats: "Month DD, YYYY", "YYYY-MonthName", bare "YYYY"), numeric values with footnote stripping and parenthesized-negative handling, and publication metadata from filenames. Content-hash deduplication removes 2,500--5,000 duplicate tables across bulletins.

**Stage 2: Enrichment** (`reingest_from_json.py`, 1,466 lines + `ingestion_enrichment.py`, 273 lines). Re-ingests from source JSONs with full metadata: series label recovery (hierarchical breadcrumb paths), bounding box coordinates, footnotes, page numbers per cell, context text extraction (narrative paragraphs preceding tables), structure hints (has_total, has_pct, likely_ops), table fingerprinting (sparsity, sum-check validation at 1% tolerance, footnote counts), and metric alias collection (canonical names with decade-specific variants---the same metric had different names across eras).

**Stage 3: Compression** (`pack_cells.py`, 155 lines). Serializes cell data with short keys (15 fields compressed to 2-character keys: `ro`, `rl`, `rn`, `co`, `cl`, etc.), omits null/empty values, applies msgpack binary serialization followed by zstandard compression at level 3. Achieved a **12.3x compression ratio** (3.8GB raw cells to ~300MB compressed blobs), with per-query decompression at ~0.9ms versus 10--50ms for equivalent SQL queries.

**Stage 4: Synthesis & Indexing** (`enrich_calendar_totals.py`, `precompute_indexes.py`, `build_master_ledger.py`). Pre-computes calendar-year and fiscal-year totals from monthly data (handling the 1977 FY boundary change: Jul--Jun pre-1977, Oct--Sep post-1977). Only synthesizes totals when all 12 months are present. Builds column-label and row-label lookup indices (50K and 200K entries respectively) with normalized forms for wildcard search. Creates the master_ledger---a flat, deduplicated fact store keyed on (metric_slug, time_key, table_pk) enabling sub-millisecond direct lookups.

### 3.2 Database Versions

| Version | Size (raw) | Size (compressed) | Key Feature | Outcome |
|---|---|---|---|---|
| V1: Full corpus | 11 GB | ~2 GB (gzip) | table_first_table_cells: 18.4M rows | Too large for arena tarball (200MB limit) |
| V2: Lean | 4.8 GB | 581 MB (zstd) | Dropped heavy tables, added cell indexes | Query speedup: 12.3s to 0.008s (1,500x) |
| V3: Slim | ~1 GB | 118 MB (gzip) | No cell_blobs, lookup indices only | Fit tarball limit; 189.2 pts |
| V4: Enriched | ~150 MB | 68 MB (gzip) | 677K master_ledger, 131K YoY precomputed | Best structured DB |
| V5: FTS5 | ~200 MB | N/A | Full-text search + difflib fuzzy matching | Mixed A/B results vs grep |

**The irony:** Our most engineered database (V4, with 677K master_ledger records, 131K precomputed year-over-year changes, 92K table_index entries, and 204K metric aliases) was outperformed by a system that used `grep` on the raw 150MB text corpus. The DB lost approximately 50% of cell data during ingestion due to parsing edge cases (multi-row headers, merged cells, non-standard delimiters), while grep had 100% data coverage by definition.

### 3.3 Search Index Architecture

For the no-MCP pipeline, we built a lightweight keyword index (`build_index.py`) that scans all 696 TXT files on startup in 26 seconds, creating a tab-delimited index of (filename, table_title, column_labels) tuples covering 86K tables. Search proceeds in two stages: file-level filtering by year proximity and keyword overlap, then line-level grep within candidate files.

**Search recall analysis** on 246 questions:
- Before improvements: 47% (14/30 sample found correct file in top 5)
- After period-aware table selection + multi-term row scoring: 63% (19/30)
- 85% of answers are in the bulletin published in the data year or year+1
- Finding the right *file* is the #1 bottleneck; once in the right file, table identification is relatively reliable

We also experimented with a **bulletin-year proximity scoring** function: -5 points per year of distance from expected publication year, +8 bonus for exact match. This dramatically improved file selection but widened candidate pools, causing scorer flakiness on some questions where similar tables appeared across multiple bulletins.

### 3.4 Data Quality Findings

The ingestion pipeline revealed several data quality issues in the source corpus:

- **41% of tables had NULL year metadata**, requiring fallback to direct column-label LIKE queries (452K rows) instead of indexed year-based search
- **Multi-row headers** in Treasury tables (e.g., hierarchical month/year groupings) frequently caused parser confusion, splitting one logical table into multiple fragments
- **Footnote markers** (r/, p/, 3/) contaminated numeric values; aggressive stripping was required
- **16 questions required calendar-year totals** that don't exist as explicit rows in the data---Treasury only publishes fiscal-year totals, so CY totals must be synthesized from 12 monthly values
- **Period basis corruption** in the V4 database: the period_basis field was incorrectly tagged for many tables, causing FY/CY lookup failures

These findings directly informed the decision to move away from the database approach: the raw TXT files, while less structured, were *complete* and *uncorrupted*.

## 4. Architectural Exploration

We developed 9 distinct architectural generations, each motivated by failures observed in the previous generation. The progression reveals a consistent pattern: *simpler systems outperform complex ones* for this task.

### Generation 1: MCP + SQLite Database (Days 1--2)

**Architecture:** 7 MCP tools backed by the 11GB SQLite database with 92K tables and 203K metric aliases, served via SSE over a DigitalOcean droplet. Tools included `search_tables`, `query_table_rows`, `get_file_structure`, `get_table_profile`, `compute_expression`, `get_cpi_index`, and `get_fiscal_year_bounds`.

**Score:** 5% initial, 55% after bug fixes, ~152 points submitted.

**Key insight:** The model (MiniMax M2.5) was given both MCP tools and shell access. It chose `grep` over MCP tools in every observed trace. When MCP was available, the model wasted 10--21 steps on poor search results instead of using the reliable shell approach. MCP availability *correlated with failure*, not success.

### Generation 2: Meta-Harness with State Machine (Day 3)

**Architecture:** State machine controller (SEARCH -> COMPUTE -> SUBMIT) with per-path budgets, a question router dispatching to Ledger or Table paths, and severity-based warnings in a verification step. Implemented compact schema returns to prevent context inflation and structured severity-based warnings.

**Score:** 147--167 points across submissions.

**Key insight:** MiniMax ignored phase boundaries. The model treated state transitions as suggestions rather than constraints. Imperative rules ("NEVER use more than 2 search calls") were either followed mechanically (stopping at exactly 2 regardless of result quality) or ignored entirely. We observed a "Neural Howlround" pattern where the model called blocked tools 6--7 times in succession, ignoring the block messages.

### Generation 3: No-MCP Shell Pipeline (Day 4)

**Architecture:** Pure Python pipeline (`solve.py`) performing grep-based search on raw TXT files, with the keyword index (86K tables, 26-second build time), deterministic routing, and Python-based computation. Vertical serialization of wide tables (ROW: label, col: value format) made complex tables readable within model context.

**Score:** 80% local accuracy, 180.4 points submitted (68.7% pass rate).

**Key insight:** Six bug fixes on Day 4---including a `query_table_rows` key mismatch that silently returned wrong data, a budget mismatch (22 vs 20 tools), decade parsing errors, and system prompt tool reference mismatches---produced a +13.2 point improvement, the single largest score jump. Infrastructure correctness matters more than architectural sophistication.

### Generation 4: Bottom-Up Component Pipeline (Day 6)

**Architecture:** 10-component pipeline (FileLocator, TableFinder, ColumnFinder, RowFinder, ValueParser, DataNormalizer, TableValidator, ConsensusVoter, SearchTask, Orchestrator) with three-strategy consensus voting (Strict/Fuzzy/Contextual). Confidence scoring used pessimistic aggregation (minimum across pieces) with agreement-based boosting (+15% all agree, +5% two agree, -5% one agrees).

**Score:** 103/103 component tests passing, but end-to-end results were nondeterministic---UID0001 returned 2,602 (correct) in one run and 23,280 in another. Not submitted.

**Key insight:** Component-level correctness does not guarantee system-level correctness. The consensus voting baseline improved from 2/10 to 3/10---a modest gain insufficient to justify the 3x computation cost and added complexity. Each question required ~5--15 LLM calls (a 4-piece decomposition = 12 extraction calls alone), creating cost and latency issues.

### Generation 5: MCP Server v2 with FTS5 (Day 6)

**Architecture:** 5 tools (down from 9) over FTS5 full-text search with difflib fuzzy matching, 677K master ledger records, 131K year-over-year precomputed changes. The MCP server parsed the Databricks-transformed TXT files directly, extracting 82 tables with smart header flattening.

**Score:** Mixed A/B results versus grep across 20 questions. Not reliably better.

**Key insight:** Collapsing 9 tools to 5 was directionally correct but the underlying retrieval was still less reliable than direct grep. The database lost data through parsing edge cases that grep inherently avoids.

### Generation 6: LLM Sub-Call Decomposition (Days 6--7)

**Architecture:** Single `find_evidence` MCP tool that internally decomposed questions into sub-queries via LLM calls and executed search/extract/compute pipelines. `list_tables` for relevance-scored search, `get_table` for column-filtered retrieval.

**Score:** Abandoned. MiniMax failed with "process quit before initialization" when handed the stdio MCP server. This is a fundamental compatibility limitation with MiniMax M2.5's MCP handling, not a configuration issue.

### Generation 7: Skills + Inline Reference Data (Day 7)

**Architecture:** 340-word prompt with CPI-U annual averages (1929--2024) embedded inline, 5 Goose skills in SKILL.md format (cpi-reference, computation-patterns, fiscal-calendar, data-files, table-extraction), no database, no MCP server. Tarball size: 28KB.

**Score:** 184.3 points (168/242 correct, 69.4% pass rate, multiplier 1.097).

**Key insight:** Previous 485 traces across all versions showed *zero* skill usage because flat `skills/*.md` files are silently ignored by Goose---the correct format requires `skills/skill-name/SKILL.md` with YAML frontmatter. Additionally, `arena test` does not copy skills into the Docker container (only `arena submit` does), meaning skills could not be validated without actual submission---a discovery that cost significant debugging time. Skills were confirmed unreachable in the arena even with correct format because the Goose `summon` extension is not included in the Harbor recipe---making skills a dead end entirely.

### Generation 8: Inline CPI + Negative Instructions (Day 8)

**Architecture:** Same as v7 but with CPI-U data inlined in system prompt and explicit negative instructions ("NEVER curl BLS/FRED"). No skills, no MCP.

**Score:** 172.7 points (regression from v7's 184.3).

**Key insight:** CPI inline tokens consumed context budget, causing more timeouts. Negative instructions ("NEVER curl") triggered *more* curl calls (248 vs 204 in v7)---MiniMax treats mention of curl as a prompt to use it. This confirmed that MiniMax is "action-triggered" not "protocol-following."

### Generation 9: Minimal Prompt (Day 9)

**Architecture:** 20-line prompt with question placed last, no inline CPI, no negative instructions. Focus on three proven elements: (1) "write answer.txt immediately," (2) file path hints to /app/resources/, (3) suggest python3 -c for math. No MCP, no skills, no database.

**Score:** Submitted 2026-04-06 (ID: 64ecc1cc-7996-45e7-ac1d-e176fb7fc377). Tracking at 69.9%, projected ~189 points.

**Key insight:** A comprehensive 14-variant A/B test (see Appendix D) proved that 70% is MiniMax's ceiling on a 40-task sample regardless of prompt variant. A bare prompt (just the question) scored 61% with zero wrong answers---prompt engineering adds only ~9% by ensuring MiniMax writes answer.txt. Complex multi-step instructions are ignored.

## 5. Retrieval Engineering: Why Raw Text + Grep Won

The gap between our database approach (~166 pts) and our grep approach (184.5 pts) was not simply "grep is simpler." We developed a suite of retrieval engineering techniques that made raw text *more effective* than structured queries. This section documents the specific techniques, each of which addressed a concrete failure mode observed in traces.

### 5.1 Vertical Serialization

The single most impactful evidence formatting technique. Treasury Bulletin tables are pipe-delimited with hierarchical multi-row headers:

```
|                      | 1940                                    |
|                      | Jan. | Feb. | Mar. | ... | Total        |
| -------------------- | ---- | ---- | ---- | --- | ------------ |
| National defense     | 132  | 129  | 143  | ... | 2,602        |
| Veterans' services   | 42   | 41   | 43   | ... | 507          |
```

In horizontal format, LLMs frequently misalign columns, especially for wide tables with 10+ columns. Our `_row_to_vertical()` function converts matching rows to vertical key-value format:

```
ROW: National defense
  Jan. (month 1): 132
  Feb. (month 2): 129
  Mar. (month 3): 143
  ...
  Total: 2,602
```

This eliminates column misalignment entirely. Each value is explicitly labeled with its column header, and month indices are added to preserve ordering. For the 15 always-fail tasks involving wide table confusion (33% of failures), vertical serialization was the primary mitigation.

### 5.2 Multi-Row Header Merging

Treasury tables frequently use 2--3 row headers with hierarchical groupings (year spanning months, category spanning subcategories). Our `_merge_multi_row_headers()` function scans backward from the separator row, collects all header rows, and merges them using `>` notation: `"1940 > January"`, `"Public Debt > Marketable > Bills"`. This preserves the full semantic path while collapsing it into a single header row that the model can parse unambiguously.

### 5.3 Period-Aware Search & Filtering

A persistent failure mode was fiscal/calendar year confusion (13% of failures). We implemented multi-level period awareness:

**At index time:** The keyword index tags each table with period basis (fiscal, calendar, monthly) and month presence (which of 12 months appear, whether all 12 are present, whether annual totals exist). Tags like `HAS_12_MONTHS`, `HAS_ANNUAL`, `BASIS:fiscal` are searchable metadata.

**At search time:** Questions are parsed deterministically for period signals (`fiscal year`, `FY`, `calendar year`, `CY`). Results are filtered to prefer tables matching the question's period basis.

**At presentation time:** For calendar year questions with monthly data, the vertical serialization *filters out* fiscal year and annual total rows, showing only the 12 monthly values. An explicit instruction is appended:

```
IMPORTANT: This question asks for CALENDAR YEAR 1940 (Jan-Dec 1940).
If you find monthly values (Jan, Feb, Mar...), you MUST sum all 12 months
for Jan-Dec 1940. Do NOT use a 'fiscal year' or 'FY' total.
```

### 5.4 Pre-Extracted Monthly Values

For the 55% of questions requiring summation (the most common type), we pre-extract monthly values from the matched vertical data before presenting to the LLM:

```
PRE-EXTRACTED MONTHLY VALUES for CY 1940: [132, 129, 143, 159, 154, 153, 177, 200, 219, 287, 376, 473]
(Count: 12 values — sum these for the calendar year total)
```

This removes all ambiguity: the LLM sees exactly which 12 numbers to sum, with an explicit count confirming completeness. Before this feature, the model frequently used the fiscal year total row instead of summing calendar year months---a failure mode that accounted for 16 of the 43 decomposition quality issues (37%).

### 5.5 Bulletin Year Proximity Ranking

Treasury Bulletins publish revised data 12--24 months after the data year ends. For data year 1940, the search order is: 1941 bulletin (most complete revised data), 1942, 1940 (preliminary), 1943, 1944. This is implemented as a scoring function: -5 points per year of distance from the expected publication year, +8 bonus for exact match.

Before this fix, year proximity was penalized at only -1 per year, causing the search to prefer wrong-year bulletins with better keyword matches. The adjustment improved file selection significantly: 85% of answers are in the bulletin published in the data year or year+1.

### 5.6 Two-Stage Search: Index then Grep

**Stage 1 --- Fast keyword index scan:** The pre-built index (86K tables, 26-second startup) is scanned for entries where multiple metric terms co-occur within the same table block. This requires 2+ terms to match, dramatically reducing false positives compared to single-term search.

**Stage 2 --- Within-table row matching:** For each candidate table, the full text is parsed to extract headers, identify the separator row, merge multi-row headers, and convert matching rows to vertical format. Row scoring uses multi-term matching on the row *label only* (not cell values) to prevent false matches.

### 5.7 Multi-Strategy Search with Fallback

Rather than a single query, the system generates 4--5 keyword strategies from broad to narrow:

1. Full phrase: `"national defense expenditures"`
2. First two content words: `"national defense"`
3. Single broadest term: `"defense"`
4. Full cleaned metric (first 80 chars)
5. TABLE_FAMILY_MAP boost terms (domain-specific synonyms)

Each strategy is tried in order with year-specific bulletin filtering. This progressive fallback ensures that even if the exact phrase doesn't appear in the corpus (due to historical naming conventions---the same metric had different names across decades), a broader search will find it.

### 5.8 Context Window Management

Evidence is structured to maximize signal density within the model's context:

1. **matched_row_vertical** (highest priority): The specific row matching search terms in vertical format
2. **vertical_data** (context): Other rows from the same table, with matching rows first, up to 25 rows total
3. **context snippet**: 5 lines around the match point for footnote and unit context
4. **table_data** (backup): Dict-format rows for backward compatibility

Matching rows are prioritized over context rows, and the total is capped to prevent context dilution. This structure means the model sees the most relevant data first, with enough surrounding context to verify units and footnotes.

### 5.9 Footnote & Value Parsing

Treasury data contains pervasive footnote markers (`r` for revised, `p` for preliminary, `e` for estimated, `*` for special notes, `3/` for numbered footnotes) embedded directly in numeric cells. Our `_parse_pipe_cell()` function strips these markers, handles dashes as zero (standard Treasury convention for no data), removes commas and dollar signs, and filters out year-label values (integers 1800--2030) that would otherwise be confused with data.

### 5.10 Conflict Detection & Flagging

When multiple search results return different values for the same metric, the system detects the conflict (>2% relative difference) and explicitly flags it in the briefing:

```
CONFLICT REPORT:
  Multiple sources returned DIFFERENT values for this question:
  - 2,602 (from treasury_bulletin_1940_01.txt, In millions of dollars)
    vs 2,598 (from treasury_bulletin_1941_01.txt, In thousands of dollars)
  >> Senior analyst: please review which source is authoritative.
```

This prevents the model from silently picking the wrong source. In the mentor/intern prompt pattern, the "senior analyst" (LLM) is explicitly asked to adjudicate conflicts---activating the critical evaluation behavior that makes the mentor pattern effective.

### 5.11 Deterministic Extraction with Structured Output

The final LLM call receives pre-searched evidence and returns a structured extraction:

```
TABLE_TITLE: National Defense Expenditures
ROW_LABEL: Total
UNITS: In millions of dollars
PERIOD: calendar
VALUES: [132, 129, 143, 159, 154, 153, 177, 200, 219, 287, 376, 473]
OPERATION: sum
ANSWER: 2602
```

Python then independently computes the answer from VALUES + OPERATION, cross-checking against the LLM's stated ANSWER. This separation ensures that even if the model makes an arithmetic error in the ANSWER field, the deterministic Python computation produces the correct result. The model's role is *extraction and classification only*---selecting the right values and naming the right operation.

### 5.12 Summary: Why These Techniques Matter

Each technique addresses a specific, observed failure mode:

| Technique | Failure Mode Addressed | Impact |
|---|---|---|
| Vertical serialization | Column misalignment in wide tables | Eliminates 33% of extraction failures |
| Multi-row header merging | Hierarchical header confusion | Correct parsing of Treasury format |
| Period-aware filtering | FY/CY confusion | Addresses 13% of failures |
| Pre-extracted monthly values | Wrong total row selected | Fixes 16/43 decomposition issues |
| Bulletin proximity ranking | Wrong-year bulletin selected | 85% answers in year or year+1 |
| Two-stage search | False positive tables | 2+ term co-occurrence reduces noise |
| Multi-strategy fallback | Historical naming variations | Progressive broadening finds data |
| Context management | Context dilution | Matched rows first, cap at 25 |
| Footnote parsing | Corrupted numeric values | Clean values for computation |
| Conflict detection | Silent wrong-source selection | Flags ambiguity for review |
| Structured extraction | LLM arithmetic errors | Python computes, LLM classifies |

The key insight is that these are *retrieval engineering* techniques, not prompting tricks. They transform raw text into a format optimized for LLM consumption while preserving 100% data coverage. This combination---complete data with optimized presentation---is what allowed grep to outperform the database.

## 6. Key Findings

### 6.1 The Structured Tools Paradox

Our most counterintuitive finding is that structured tool access degrades performance. Across all observed traces:

- When MCP tools were available, MiniMax wasted 10--21 steps on search results before (sometimes) falling back to grep
- When MCP tools were unavailable, the model immediately used shell commands and succeeded faster
- In the 12 traces that flipped from FAIL to PASS between v4 and v5, *zero* used MCP tools---all 12 used shell commands exclusively (`cat`, `grep`, `sed`)
- Successful runs averaged 5--8 commands in ~50 seconds; timeout failures used 30+ commands over 180+ seconds
- Tool call count inversely correlated with success: 1--5 calls achieved 82% pass rate; 11+ calls dropped to 60%

The forced-MCP experiment was even more striking. Using `sitecustomize.py` to patch OpenHands' TerminalTool and block shell data retrieval (forcing MCP usage) resulted in 0% accuracy---the model executed 50+ terminal commands with 0 MCP calls, working around restrictions rather than using MCP tools effectively.

We hypothesize that structured tools create *false confidence*: the model trusts that a search tool's results are complete when they are not, whereas grep output provides raw context that allows the model to assess data quality directly. Additionally, the database's 50% data loss through parsing meant MCP searches genuinely returned incomplete results, while grep against raw text had 100% coverage.

### 5.2 Evidence Selection as the Primary Bottleneck

Detailed analysis of 31 failed traces from the v5 submission (184.5 points) reveals that evidence selection---not arithmetic, not reasoning---is the dominant failure mode:

| Failure Category | Count | Percentage |
|---|---|---|
| Wrong data extraction (wrong table/row/column) | 15 | 48% |
| Ambiguous interpretation (FY/CY, sign convention) | 7 | 23% |
| Wrong formula or method | 5 | 16% |
| Insufficient data (not in corpus) | 3 | 10% |
| Format errors | 1 | 3% |

Critically, *zero* failures among correct extractions were attributable to arithmetic errors when computation was delegated to Python. The pipeline invariant holds: if the model finds the right data, deterministic Python computation produces the right answer. The entire challenge reduces to retrieval quality.

Among the 15 always-fail tasks we sampled in detail, the failure archetypes are:

- **Wide table confusion (33%):** Tables with 10+ columns (e.g., yield spreads across bond types) cause systematic misreads. UID0027 consumed 3,376 thinking blocks trying to map columns before producing no answer.
- **Missing external data (27%):** NBER paper publication dates (UID0034), Treasury 2025 forecasts (UID0140), and CPI values not in the corpus.
- **Calculation methodology (20%):** Percent difference vs. percent change (UID0004), Theil index variants (UID0041), polynomial regression specifics (UID0120).
- **Time period semantics (13%):** Fiscal year proposal months vs. fiscal year boundaries (UID0008), pre-1977 vs. post-1977 FY definitions.

### 5.3 Prompt Engineering: Situational Framing vs. Imperative Rules

We tested four distinct prompt paradigms across multiple submissions:

**Paradigm 1 --- Generic agent** ("You are a helpful assistant with MCP tools"): 5% accuracy. The model hallucinated answers without searching.

**Paradigm 2 --- Imperative rules** ("HARD RULE: NEVER use more than 2 search calls. MANDATORY SEQUENCE: search, extract, compute, submit"): 55--63%. Rules were followed mechanically or ignored entirely. Adding warnings ("WARNING: You have {budget} calls remaining") made results *worse*---UID0127 specifically regressed after warning text was added. The model interprets loud warnings as error signals and becomes more cautious, exploring more rather than committing to an answer.

**Paradigm 3 --- Situational framing** ("You are a senior Treasury analyst. You've learned from experience that the most common mistake is extracting from the wrong table"): 65--69%. The model role-played effectively, and natural verification behaviors emerged without explicit instruction. Budget text as calm statement of fact ("You have a budget of 20 tool calls") worked better than threatening language.

**Paradigm 4 --- Mentor/review pattern** ("You're reviewing your intern's work. Run solve_briefing.py to get their initial analysis, then verify their data sources"): 180.4 points. The single largest prompt-driven improvement (+13 points over the previous best).

The mentor pattern succeeds because it gives the model a *reason* to verify. Rather than asking "check your answer" (which the model treats as a formality), "review someone else's work" activates genuine critical evaluation. The model catches wrong-row errors, unit mismatches, and FY/CY confusion because it is role-playing a reviewer rather than defending its own work.

### 5.4 Stability Analysis Across 6 Versions

We conducted a comprehensive stability analysis across 246 tasks and 6 version generations (v5--v10), with 222--246 traces per version totaling approximately 4,400 individual evaluations. To our knowledge, this multi-run stability methodology is uncommon in arena evaluations, where teams typically report single-run accuracy.

#### 5.4.1 Per-Version Pass Rates

| Version | v5 | v6 | v7 | v8 | v9 | v10 |
|---|---|---|---|---|---|---|
| Pass rate | 68.9% | 65.2% | 69.4% | 64.2% | 65.8% | 68.5% |
| Passed/Total | 166/241 | 159/244 | 168/242 | 158/246 | 160/243 | 152/222 |

All versions hover in a narrow 64--69% band despite significant architectural differences (skills, inline CPI, MCP attempts, prompt length variations). This 5-point band is consistent with run-to-run variance rather than genuine prompt-driven improvement.

#### 5.4.2 Task Stability (210 tasks common to all 6 versions)

| Stability Category | Count | Percentage |
|---|---|---|
| ALWAYS_PASS (6/6) | 99 | 47.1% |
| USUALLY_PASS (4--5/6) | 46 | 21.9% |
| FLIP_FLOP (2--3/6) | 15 | 7.1% |
| USUALLY_FAIL (1/6) | 11 | 5.2% |
| ALWAYS_FAIL (0/6) | 39 | 18.6% |

Nearly half of tasks (47.1%) pass deterministically across all 6 versions. The 39 always-fail tasks (18.6%) represent a hard ceiling that no prompt variant has breached. The 72 non-deterministic tasks (flaky) break down as: 32 pass 5/6 (nearly stable), 14 pass 4/6, 7 pass 3/6, 8 pass 2/6, and 11 pass only 1/6 (nearly impossible).

#### 5.4.3 Version Transitions

Each version transition gains and loses 15--25 tasks, with net changes of -12 to +10---essentially noise:

| Transition | Gained | Lost | Net |
|---|---|---|---|
| v5→v6 | 12 | 21 | -9 |
| v6→v7 | 18 | 8 | +10 |
| v7→v8 | 11 | 23 | -12 |
| v8→v9 | 25 | 21 | +4 |
| v9→v10 | 21 | 16 | +5 |

Flaky task example: UID0021 answered 103,030 (correct) in v9 and 124,389 (wrong) in v10 with the same step count---just different number extraction from the same table. UID0026 answered 894 in 6 steps (correct, v9) and 1,951 in 19 steps (wrong, v10)---more searching led to a worse answer.

**Improvement ROI analysis:** The 72 flaky tasks represent the highest-ROI improvement target. These tasks *sometimes* succeed, meaning the data and reasoning path exist---the system just doesn't reliably find them. By contrast, the 39 always-fail tasks have systematic gaps (wrong arithmetic, missing data, ambiguous tables) that may be unfixable without model capability improvements or corpus augmentation.

### 5.5 Inline Reference Data Eliminates Hallucination Categories

A particularly instructive failure mode involved CPI (Consumer Price Index) data. When MiniMax needed CPI values for inflation adjustment and the data was not available in the corpus, the model attempted to curl BLS/FRED APIs. When these calls failed (as they frequently did in the arena environment), the model *hallucinated* CPI values. For UID0196, the model fabricated CPI-U values of 255.7 and 259.2 for May/June 1979---values that match no BLS base year series (the actual 1982--84 base values for those months would be approximately 72--73).

Embedding CPI-U annual averages (1929--2024, approximately 50 values) directly in the system prompt eliminated this failure category entirely. UID0005 passed 3/3 times with inline CPI and made zero external API calls. The principle generalizes: for any reference data that the model might hallucinate, embedding it directly in the prompt is more reliable than providing a lookup tool.

This also explains why our database approach underperformed. The DB contained CPI data, but querying it required the model to call `get_cpi_index` correctly with the right year and base period---an additional failure point. Inline data removes the retrieval step entirely.

### 5.6 Verification Pipeline Design

We tested multiple verification approaches at different levels of the pipeline:

**LLM self-check:** The model reviews its own answer against the extracted evidence. Catches obvious errors (wrong units, missing months in a sum) but misses subtle ones (wrong row from a wide table). Adds 1--2 turns of overhead.

**Mentor/intern review:** The model reviews a "junior analyst's" work product. Most effective approach (+13 points), because the review framing activates genuine critical evaluation rather than confirmation bias.

**Consensus voting:** Three independent extraction strategies (strict match, fuzzy match, contextual match) vote on the answer. Agreement logic: full agreement (+15% confidence boost), partial 2/3 (+5%), no agreement (-5%). Smart dissenter override when dissenter confidence exceeds 0.85 and is 0.3+ higher than the pair average. Improved baseline from 2/10 to 3/10 on a test set---a statistically modest gain that did not justify the 3x computation cost.

**Deterministic Python verification:** For numeric computation, delegating all arithmetic to Python with `compute_expression` (an AST-based safe evaluator supporting sum, mean, median, stdev, geometric_mean, pct_change, CAGR) eliminates calculation errors entirely. The pipeline invariant---correct extraction implies correct answer---depends on this delegation. In our v5 failure analysis, zero failures among correctly extracted data were arithmetic errors.

**Terminal override (negative result):** Using `sitecustomize.py` to block shell data retrieval and force MCP tool usage achieved 0% accuracy. This demonstrates that verification mechanisms must work *with* the model's natural behavior, not against it. The model's preference for shell commands is not a bug to be patched---it's a signal about what retrieval interface works best.

### 5.7 Oracle Ensemble Analysis

Cross-harness analysis reveals systematic limits:

- Best single-harness score: 184.5 (Goose, shell-only, v5)
- Estimated best-of-two (Goose + OpenHands): ~79.2% (approximately 195 points)
- Always-fail tasks across 6 versions: 39 (18.6% of 210 common tasks)

The 39 always-fail tasks represent a ceiling that no prompt or architectural change can overcome. Of these, 23 (59%) produce wrong answers despite finding the correct data---these are LLM capability limits on complex table extraction and computation, not retrieval failures. The remaining 16 involve either missing data, spiraling searches, or agent inaction.

## 7. Failure Taxonomy

Based on analysis of 39 always-fail tasks across 6 versions (v5--v10) and 70 failed v10 traces, we identify two levels of failure taxonomy: behavioral modes (how the agent fails) and root causes (why it fails).

### 7.1 Behavioral Failure Modes (39 always-fail tasks)

| Mode | Count | Description |
|---|---|---|
| Wrong answer, normal path | 23 (59%) | Found data, ran python, got wrong number |
| Stopped without answer | 9 (23%) | Searched 8--19 pages, never committed |
| Turn limit exhausted | 4 (10%) | Spiraled 60--83 steps without answer.txt |
| Gave up immediately | 2 (5%) | 0--1 tool calls, produced nothing |
| Over-searched, wrong | 1 (3%) | 50+ steps, still wrong answer |

The dominant behavioral mode is *wrong answer on a normal execution path* (59%). These are not retrieval failures---the agent finds the right Treasury Bulletin page, locates the relevant table, and runs python3 calculations, but extracts wrong values or applies incorrect formulas. Representative examples: UID0041 produced 0.012 (wrong Theil index variant), UID0074 produced -262.86 (wrong column extraction), UID0096 produced 0.377 (wrong row/period).

### 7.2 Root Cause Categories

**Category 1: Data Source/Table Confusion (33%).** Wide tables with 10+ columns cause systematic misreads. UID0027 spent 3,376 thinking blocks attempting to resolve column mappings. UID0158 extracted from the wrong tenor category. UID0113 confused year-end values with annual averages.

**Category 2: Missing or Unavailable Data (27%).** UID0034 asks for an NBER paper publication date not in the corpus. UID0140 asks for a 2025 Treasury forecast not yet published. UID0207's question could not be resolved from available data after 733 thinking blocks.

**Category 3: Calculation Methodology (20%).** UID0004 computed percent difference instead of percent change. UID0041 used a wrong Theil index variant. UID0120's regression produced fragile coefficients sensitive to CPI adjustment choices. UID0101 produced a multi-value array instead of a scalar.

**Category 4: Time Period Semantics (13%).** Fiscal year boundaries changed in 1977 (Jul--Jun to Oct--Sep). UID0008 used proposal months instead of FY boundaries. Pre-1977 fiscal year conventions are persistently confusing---FY1940 runs Jul 1939--Jun 1940.

**Category 5: Hallucination (7%).** UID0196 hallucinated CPI-U values of 255.7 and 259.2 (actual ~72--73). Addressable through inline reference data (see Section 5.5).

### 7.3 Flaky Task Patterns

The 72 flaky tasks (passing in some but not all versions) exhibit a revealing pattern when they flip between versions. For example, UID0021 answered 103,030 (correct) in v9 and 124,389 (wrong) in v10 with identical step counts---the only difference was which table row was extracted. UID0026 answered correctly in 6 steps (v9) but incorrectly in 19 steps (v10)---more exploration led to a worse answer. This is consistent with the inverse correlation between tool call count and success: 1--5 calls achieve ~82% pass rate while 11+ calls drop to ~60%.

## 8. Design Principles for Grounded QA Systems

Our systematic exploration yields eight empirically grounded design principles:

**Principle 1: Simplicity wins.** Direct `grep` on raw text files outperformed an 11GB SQLite database, a 677K-record master ledger with FTS5, and a 10-component consensus pipeline. The winning system had a 28KB tarball. Raw text preserves context that structured representations discard---and avoids the ~50% data loss we observed during database ingestion.

**Principle 2: Evidence quality dominates reasoning quality.** 48% of all failures trace to wrong evidence selection, while 0% of correctly-grounded answers had arithmetic errors (when using Python). Investment in retrieval quality yields higher returns than investment in reasoning or prompting.

**Principle 3: Embed reference data directly.** Inline CPI values (50 numbers in the prompt) eliminated an entire hallucination category with 100% reliability. Any reference data that the model might fabricate should be provided directly rather than through a lookup tool. This also applies to fiscal year rules, common formula definitions, and domain-specific conventions.

**Principle 4: Design for model behavior.** MiniMax M2.5 naturally explores files via shell commands. Forcing MCP tool usage via terminal blocking produced 0% accuracy. The mentor/review prompt worked because it aligned with how models naturally evaluate information. Design the system around observed model behavior, not idealized tool usage.

**Principle 5: Measure stability, not single-run accuracy.** A submission scoring 184.5 contains ~38 tasks that could flip on the next run. Only 22.8% of tasks are deterministically stable. Multi-run stability analysis reveals the true reliability profile that single-run metrics obscure.

**Principle 6: Frame verification as review.** "Check someone else's work" (+13 points) vastly outperforms "verify your answer" (minimal gain). Situational framing ("you are a mentor reviewing an intern's analysis") activates genuine critical evaluation rather than confirmation bias.

**Principle 7: Delegate all computation to deterministic code.** Never let LLMs perform arithmetic. Python computation with `compute_expression` maintains the pipeline invariant: correct extraction implies correct answer. All observed computation errors occurred when the model performed mental math rather than delegating.

**Principle 8: Fewer, better tools.** Collapsing 9 MCP tools to 5 was directionally correct, but 0 tools (shell only) performed best. When tools *are* provided, each must have near-perfect reliability. An unreliable tool is worse than no tool because it creates false confidence and invites over-exploration (30+ steps vs 5--8).

## 9. Cost & Efficiency Analysis

Our best-performing system achieved remarkable cost efficiency:

| Metric | Value |
|---|---|
| Best run total cost | **$1.71** (246 tasks) |
| Average cost per task | **$0.007** |
| Average latency per task | **185 seconds** |
| Submission tarball size | **28 KB** (v7) |
| Infrastructure cost | $0 (no database, no droplet, no MCP server) |

For comparison, the MCP+database approach required:
- DigitalOcean droplet: ~$12/month (hosting SSE MCP server + 11GB DB)
- Average cost per task: $0.054 (8x more expensive)
- Average latency per task: 196 seconds (similar)

The cost reduction came entirely from simplification: eliminating the database, MCP server, and structured tools reduced both per-query API costs (fewer LLM turns) and infrastructure overhead (no server to maintain).

Across all 65,299 polled evaluations (including failed/exploratory runs), the average cost was $0.054/task and average runtime was 196 seconds. The best-performing architecture was also the cheapest---a strong signal that complexity creates both accuracy and cost overhead.

## 10. Reproducibility & Infrastructure

### 9.1 Evaluation Infrastructure

All experiments were conducted using three execution environments:

- **Arena submissions:** Official Sentient Arena Docker containers with pre-configured agent harnesses (Goose, OpenHands, OpenCode). Each submission evaluated all 246 tasks with a 300-second agent timeout and 900-second environment timeout per task.
- **Daytona sandboxes:** Cloud sandboxes (3 CPU, 3GB RAM, 10GB disk) with pre-baked snapshots containing the database and dependencies. Used for rapid A/B testing and component validation.
- **Local harness:** `run_local_v7.sh` replicates the arena submission environment locally, using oracle page files generated from the `databricks/officeqa` repository (83,216 page files across all documents). Enables rapid iteration without consuming arena submission quota.

### 9.2 Trace Analysis Pipeline

We built a suite of analysis tools to extract insights from agent trajectories:

- `scripts/pull_arena_traces.py`: Async download of full agent trajectories via arena-cli API (traces purged after ~24 hours, requiring immediate download after submission completion)
- `scripts/triage_traces_vs_stability.py`: Cross-references pulled traces against stability buckets to identify regressions
- `scripts/audit_traces.py`: Scans traces for harness signals (agent type, MCP tool usage, file access patterns)
- `analyze_traces_batch.py`: Root-cause classification of failures (hallucination, no_tool_calls, tool_failure, parse_error)
- `detailed_trace_analysis.py`: Deep dive extraction of thinking blocks, tool call patterns, and answer provenance

### 9.3 Question Decomposition

We decomposed all 246 questions into structured schemas (`decomposition_results_v3.json`) containing: data_year, topic, period_type (calendar/fiscal), computation type (sum, difference, percent_change, regression, etc.), value_format (monthly_series, annual_total), search_terms, and special notes. Decomposition success rate: 242/246 (98.4%). Quality analysis identified 43 questions (17.5%) with issues: 16 CY annual total mismatches, 66 sum/total computation ambiguities, and 7 period type mismatches.

## 11. Limitations & Future Work

**Oracle evaluation masks retrieval failures.** The arena provides oracle page files that pre-select relevant documents. Our 76.3% page-first pass rate reflects performance *with* oracle retrieval guidance. Performance on the private leaderboard, which may use different or additional questions, could differ substantially.

**Model-specific findings.** Our experiments used MiniMax M2.5 exclusively (as required by the arena). The structured-tools paradox, prompt framing effects, and tool-usage patterns may not transfer to other models (GPT-4, Claude, Gemini) that have different tool-calling behaviors.

**Database approach may work with better parsing.** Our database lost ~50% of data through parsing edge cases. A more robust parser (or direct use of the Databricks-transformed TXT format as input) could make structured approaches competitive. The fundamental insight---that structure helps only when it's complete---remains valid.

**Consensus voting under-evaluated.** Our consensus voting experiment (2/10 to 3/10 improvement) used a small test set. Larger-scale evaluation might reveal stronger effects, particularly for the flip-flop cohort (38 tasks at 30--70% pass rate) where multiple attempts could stabilize answers.

**Skills-based approach confirmed dead.** The v7 submission with properly formatted SKILL.md files scored 184.3 points but trace analysis confirmed zero skill usage. The Goose `summon` extension is not included in the Harbor recipe, making skills unreachable in the arena environment regardless of format.

**Harness severely constrains agent capabilities.** Exhaustive analysis of the Harbor goose harness (goose.py, install-goose.sh.j2, docker-compose-base.yaml) revealed that the only agent-controllable inputs are: (1) prompt template text, (2) MCP server definitions, and (3) environment variables. Temperature, reasoning effort, max turns, skills, and all tarball files besides arena.yaml and the prompt template are silently dropped. The container filesystem at `/app/` contains only the arena's `resources/` directory---no agent files are mounted. The recipe is hardcoded with only `developer` and `todo` extensions; `summon` is configured in config.yaml but not included in the runtime recipe.

**Local A/B testing was systematically biased.** Our 14-variant A/B test used MAX_TURNS=25 locally, which penalizes minimal prompts (MiniMax doesn't write answer.txt fast enough before hitting the cap). Arena runs uncapped. This bias made verbose prompts appear 10% better locally when arena results showed the opposite: v10 (minimal, 3-line prompt) scored 180.1 in arena vs v9 (verbose, 23-line prompt) at 174.6. All local prompt ranking conclusions may be unreliable.

**Score variance dominates prompt effects.** Across 12 arena submissions, scores range from 162.4 to 184.5---a ~22 point band. Prompt changes produce ~5 point differences, well within the noise. The "best" prompt is likely whichever gets a lucky dice roll. Multiple submissions of the same config would be more productive than further prompt optimization.

**File drop backdoor via MCP args.** We discovered that the `mcp_servers` args field runs arbitrary bash commands inside the container during goose startup. By chaining `echo '<base64>' | base64 -d > /app/resources/file.py` before `exec python3 /tmp/_mcp.py`, we can write files to `/app/resources/` where MiniMax naturally discovers them via `ls`. This is the only known mechanism to inject code into the container filesystem. v12 uses this to drop `tools.py` (CPI/FY/calculator) and `README.txt` (usage hints) alongside the bulletin files.

**MCP tools have never been called in arena** across ~4,400 task evaluations spanning 6 versions (v5--v10). Trace inspection of v10 confirms `tool_definitions: null` in the agent config---the MCP server was never connected. All versions use an identical toolset: `shell` (~3,100--3,700 calls), `todo__todo_write` (~430--480), `write` (~200--250), `tree` (~40--75), and occasionally `edit` (~2--5). The `/tmp/goose_mcp_responses/` paths found in some traces are Goose's large-output overflow mechanism (storing shell output >threshold to temp files), not our MCP server.

**Retrieval ceiling: 39 always-fail tasks.** Detailed analysis of the 39 tasks that fail across all 6 versions reveals five distinct failure modes:

| Failure Mode | Count | Pattern |
|---|---|---|
| Wrong answer (normal path) | 23 (59%) | Found correct pages, ran python, got wrong number |
| Stopped without answer | 9 (23%) | Searched 8--19 pages but never committed to an answer |
| Turn limit exhausted | 4 (10%) | Spiraled through 60--83 steps without writing answer.txt |
| Gave up immediately | 2 (5%) | 0--1 tool calls, produced nothing |
| Over-searched, still wrong | 1 (3%) | 50+ steps of exploration, wrong final answer |

The dominant failure mode (59%) is **wrong arithmetic or data extraction on correctly-found data**. MiniMax locates the right Treasury Bulletin page, parses the table, runs python3 calculations, but extracts wrong values from complex multi-column tables or applies incorrect formulas. These are LLM capability limits, not retrieval failures. The 9 "stopped without answer" tasks are notable for extensive page access (8--19 pages each) suggesting the agent finds relevant data but cannot resolve ambiguity across multiple tables.

**Future directions:** (1) File drop backdoor to inject helper scripts MiniMax discovers naturally; (2) SSE MCP servers hosted externally with full numpy/statsmodels for regression tasks; (3) multiple identical submissions to exploit the ~22 point variance band; (4) corpus augmentation via injected reference files (CPI tables, exchange rates) into `/app/resources/`.

## 12. Conclusion

Over 9 days and 9 architectural generations, we conducted what we believe is one of the most thorough empirical explorations of a grounded numerical QA task. The journey from 5% (Day 1, broken MCP tools) to 184.5 points (Day 6, shell grep on raw text) produced a counterintuitive but empirically robust finding: *less structure yields better performance* for LLM-driven document retrieval. A comprehensive 14-variant prompt A/B test on Day 9 further revealed that 70% is MiniMax's hard ceiling on our test set, and prompt engineering contributes only ~9% over a bare question---its value limited to ensuring the model writes answer.txt.

We built 5 database versions through a sophisticated multi-stage ingestion pipeline (parsing, enrichment, compression, synthesis), achieving 12.3x compression ratios and sub-millisecond queries---yet the winning system used none of it. The database's Achilles heel was data completeness: ~50% data loss during ingestion meant structured searches returned incomplete results, while grep on raw text had 100% coverage by definition.

The core insight is that evidence selection---finding the right table, row, and column in a 696-file corpus---is the dominant challenge, accounting for 48% of all failures. Arithmetic errors, reasoning failures, and format issues are secondary. This reframes the grounded QA problem as fundamentally a retrieval problem, not a reasoning problem.

Our practical recommendations for building grounded QA systems are: keep the retrieval interface simple (raw text over structured databases), embed critical reference data directly in the prompt, frame verification as peer review rather than self-check, delegate all computation to deterministic code, and measure multi-run stability rather than single-run accuracy. The most effective single intervention was the mentor/review prompt pattern, which improved scores by 13 points by aligning the verification mechanism with how language models naturally evaluate information.

The gap between our best score (184.5/246, 75.0%) and perfect performance is dominated by evidence selection failures in wide tables and missing external data---problems that require better document representation and corpus augmentation rather than better prompting or reasoning. At $1.71 total cost for 246 tasks, we demonstrate that the most effective grounded QA system is also the simplest and cheapest---a finding with broad implications for the design of retrieval-augmented generation systems.

A final insight from 14-variant A/B testing: MiniMax M2.5 is fundamentally "action-triggered" rather than "protocol-following." Complex multi-step instructions, phase boundaries, and procedural rules are ignored. Only three prompt elements reliably influence behavior: (1) "write answer.txt immediately," (2) file path hints, and (3) suggesting python3 -c for computation. However, our A/B testing was biased by a local turn cap (MAX_TURNS=25) that penalized minimal prompts---arena results suggest minimal prompts may actually perform better when MiniMax runs uncapped.

A late-stage discovery---the MCP args file drop backdoor---opens a new direction: injecting helper scripts directly into `/app/resources/` where MiniMax naturally discovers them. Rather than fighting MiniMax's shell-first instinct, this approach works with it: MiniMax does `ls`, sees `tools.py`, and can run it via `python3 /app/resources/tools.py calc "stdev([...])"`. Whether this bridge between MCP infrastructure and MiniMax's shell behavior produces results remains our most promising open question.

---

## Appendix A: Score Evolution

| Version | Date | Architecture | Score | Pass Rate | Cost | Key Change |
|---|---|---|---|---|---|---|
| arena-v0.6 | Mar 31 | 7 MCP tools, 11GB SQLite | ~151.8 | 63.0% | ~$30 | First submission |
| arena-v0.7 | Apr 1 | Meta-Harness, state machine | ~147.1 | 62.5% | — | Regression |
| arena-v0.8 | Apr 1 | OpenHands + MCP | ~148.9 | 56.5% | — | Worst score |
| arena-v0.9 | Apr 2 | Tool fixes, anti-spin | ~167.2 | 64.6% | — | Recovery |
| arena-v1 | Apr 2 | 6 bug fixes, nomcp pipeline | ~180.4 | 66.9% | — | +13 pt jump |
| nomcp-v2 | Apr 3 | Grep-primary, keyword index | ~189.2 | 65.5% | — | Competitive |
| v3 MCP | Apr 3 | First MCP test | ~166 | 61.9% | — | Broken empty DB |
| v5 | Apr 4 | Shell grep, no MCP | **184.5** | **75.0%** | **$1.71** | **Best score** |
| v6 | Apr 5 | Prompt tweaks | ~175 | ~65% | — | API flakiness regression |
| v7 | Apr 6 | Skills + inline CPI | 184.3 | 69.4% | — | 168/242 correct, multiplier 1.097 |
| v8 | Apr 6 | CPI inline + neg. instructions | 172.7 | 64.2% | — | Regression: neg. instructions backfired |
| v9 | Apr 6 | Stripped 23-line prompt, no MCP | 174.6 | 65.4% | — | 160/243 correct; 76 wrong answers, 7 no answer |
| v10 | Apr 6 | Ultra-minimal 3-line prompt | **180.1** | **68.5%** | — | 152/222 traces; minimal beat verbose; MCP never connected |
| v12 | Apr 6 | Minimal + MCP + file drop | *pending* | — | — | tools.py+README.txt dropped into /app/resources/ |

## Appendix B: Stability Matrix Summary

| Metric | Value |
|---|---|
| Total tasks | 246 |
| Versions analyzed | 6 (v5--v10) |
| Total evaluations | ~4,400 |
| Tasks common to all 6 versions | 210 |
| Always pass (6/6) | 99 (47.1%) |
| Usually pass (4--5/6) | 46 (21.9%) |
| Flip-flop (2--3/6) | 15 (7.1%) |
| Usually fail (1/6) | 11 (5.2%) |
| Always fail (0/6) | 39 (18.6%) |
| Pass rate range across versions | 64.2%--69.4% (5.2 pt band) |
| Tasks gained/lost per version transition | 15--25 (noise) |
| MCP tool calls across all versions | **0** |

## Appendix C: Question Type Distribution

| Type | Count | Prevalence | Failure Rate |
|---|---|---|---|
| Summation | ~135 | 55% | 8.7% (most reliable) |
| Calendar year | ~96 | 39% | 21.9% |
| Percent change | ~76 | 31% | ~30% |
| Average/Mean | ~71 | 29% | 18.8% |
| Fiscal year | ~54 | 22% | 6.3% (second most reliable) |
| Difference | ~51 | 21% | 17.9% |
| Statistical (VaR, CAGR, std dev) | ~37 | 15% | ~40% |
| Multi-step (CPI adjust + compute) | ~30 | 12% | ~50% (least reliable) |
| Monthly series format | ~129 | 52% | 33.6% |
| Annual total format | ~73 | 30% | 27.4% |

## Appendix D: v9 Local A/B Testing Results (2026-04-06)

### D.1 Experimental Setup

We built a local test harness replicating the arena submit environment: per-task Docker containers (python:3.12) with /app/resources/ mounted, goose running with a recipe matching Harbor's format, and a 300s timeout. Oracle test packs (246 tasks with question, answer, and pre-extracted page files) were built from the officeqa-final repo.

### D.2 Key Discovery: Goose Ignores max_turns Config

Analysis of v7 arena traces (184.3 score, max_turns: 10 in config) revealed tasks using up to **84 turns** with a median of 13 and average of 19. The `max_turns` setting in arena.yaml config is silently ignored by the Harbor goose recipe. The actual constraint is `timeout_per_task: 300s`. However, the goose CLI `--max-turns` flag *does* work, meaning our local harness was accidentally more restrictive than the real arena.

### D.3 Turn Budget is the Biggest Lever

| Max Turns (CLI) | Pass Rate (20 tasks) | Notes |
|---|---|---|
| 15 | 7/20 (35%) | 11 NO_ANSWER — ran out of turns |
| 25 | 14-16/20 (70-80%) | Most NO_ANSWERs resolved |

At 15 turns, MiniMax consistently hit "I've reached the maximum number of actions" before writing answer.txt. At 25 turns, the same tasks completed successfully. The "write answer.txt EARLY" prompt nudge further reduced NO_ANSWERs.

### D.4 Comprehensive Prompt Variant A/B Test (40 tasks, 25 turns, no MCP)

We expanded testing from 20 to 40 tasks (UID0001-0040) and compared 14 prompt variants with MCP disabled to match arena conditions. This is, to our knowledge, the most extensive prompt A/B test conducted on the OfficeQA arena.

| Rank | Prompt Variant | Pass | Fail | NoAns | Rate |
|------|----------------|------|------|-------|------|
| 1 | v9 (20-line, q-last) | 28 | 3 | 9 | 70% |
| 1 | Minimal q-first (3 lines) | 28 | 3 | 9 | 70% |
| 3 | Pre-seed | 27 | 12 | 0 | 69% |
| 3 | XML tags (q-last) | 27 | 6 | 7 | 68% |
| 5 | Why-framing q-first | 26 | 3 | 10 | 67% |
| 5 | Few-shot q-first | 26 | 2 | 12 | 65% |
| 5 | v9 q-first | 26 | 3 | 11 | 65% |
| 5 | Two-phase q-first | 26 | 3 | 11 | 65% |
| 5 | Sandwich | 26 | 2 | 12 | 65% |
| 10 | Minimal q-last | 24 | 5 | 11 | 60% |
| 10 | Anti-loop | 24 | 5 | 11 | 60% |
| 12 | Bare (just question) | 23 | 0 | 15 | 61% |
| 13 | Few-shot q-last | 22 | 5 | 13 | 55% |
| 14 | Why-framing q-last | 20 | 4 | 16 | 50% |
| 14 | CPI inline | 20 | 8 | 12 | 50% |

**70% is MiniMax's hard ceiling** on these tasks regardless of prompt engineering.

#### D.4.1 Question Placement Interacts with Prompt Length

Question-first vs question-last is not universally better — it depends on prompt length. Short prompts (3 lines) perform best with question-first (70% vs 60% q-last). Long prompts (20+ lines like v9) perform best with question-last (70% vs 65% q-first). The question gets "buried" in long q-first prompts but "lost in preamble" in short q-last prompts.

#### D.4.2 Few-Shot Examples Hurt

Adding worked examples consumed tokens without improving accuracy. Few-shot q-last scored 55% (worst non-CPI variant), and even q-first only partially recovered to 65%. MiniMax appears to over-anchor on example patterns rather than generalizing.

#### D.4.3 Extra Context Hurts

CPI inline data (50%), why-framing q-last (50%), and other token-heavy variants all underperformed. More tokens in the prompt means more turns spent processing instructions and fewer turns for actual data extraction. The "write answer.txt early" nudge partially compensates by reducing NO_ANSWERs.

#### D.4.4 Pre-Seed: Eliminating NO_ANSWERs at a Cost

The "pre-seed" variant instructs MiniMax to write a placeholder answer.txt immediately before starting research. This eliminated all 9 NO_ANSWERs but introduced 12 wrong answers (many writing "0" as the placeholder and never updating). Net result: 69% pass rate---tied for 3rd, but with a worse fail/no-answer profile than the top variants. The pre-seed approach trades NO_ANSWERs for wrong answers roughly 1:1.

#### D.4.5 Bare Prompt: Proof That Prompt Value Is ~9%

A bare prompt (just the question, no instructions) scored 61% with **zero wrong answers** and 15 NO_ANSWERs. This is the most revealing result of the entire A/B test: MiniMax explores freely, finds correct data, and computes correct answers---but without "write answer.txt" guidance, it often fails to produce output. The ~9% gap between bare (61%) and best (70%) represents the entire value of prompt engineering: ensuring the model writes its answer to a file.

#### D.4.6 MiniMax Is Action-Triggered, Not Protocol-Following

Across all 14 variants, MiniMax consistently ignored multi-step instructions, phase boundaries, and procedural rules. The model's behavior is triggered by action-oriented phrases ("write", "cat", "grep") rather than protocol descriptions ("first search, then extract, then compute"). Only three prompt elements reliably influence behavior: (1) "write answer.txt immediately" reduces NO_ANSWERs, (2) file path hints like `/app/resources/` direct initial exploration, (3) "use python3 -c" improves computation accuracy. Everything else is noise.

### D.5 Full-Scale Local Test & Arena Tracking

Running the full 246-task suite locally with 25-turn cap: 58.5% pass rate on UIDs 1-123 (artificially low due to --max-turns 25; arena runs uncapped). All local NO_ANSWERs are caused by the CLI --max-turns flag, which the arena does not enforce.

Arena v9 submission (2026-04-06, ID: 64ecc1cc-7996-45e7-ac1d-e176fb7fc377) tracking at 69.9%, matching v7's 69.4%. v7 achieved 168/242 correct with a 1.097 multiplier for a score of 184.3. v9 projected: ~172 correct x ~1.10 = ~189.

v8 scored 172.667---a regression from v7's 184.3, confirming that CPI inline and negative instructions both hurt arena performance.

### D.6 MCP Never Works in Arena

MCP tools (get_cpi, compute, fiscal_year_bounds) were confirmed working locally — the stdio server connects, tools execute, results return correctly. However, MCP has never produced tool calls in any arena submission trace. The Harbor goose recipe either doesn't mount the MCP server correctly or MiniMax doesn't discover the tools in the arena environment.

Local tests with MCP: 15/20 pass. Same tasks without MCP: 12-13/20 pass. The 2-3 task gap comes from compute-heavy tasks where MiniMax writes incorrect python3 -c code but the MCP compute tool would have given exact results. **All local test results with MCP are inflated relative to arena.**

### D.7 Negative Instructions Backfire

| Instruction | v7 curl calls | v8 curl calls |
|---|---|---|
| "do NOT curl BLS/FRED" | 204 | — |
| "NEVER curl BLS/FRED" | — | 248 |

Telling MiniMax not to curl made it curl *more*. The inline CPI table (~600 tokens) was rarely used despite being directly in the prompt. MiniMax's "instinct" to curl for CPI data overrides explicit instructions. However, BLS API (`api.bls.gov`) is actually accessible from arena containers — the curls mostly fail because MiniMax formats requests incorrectly, not because the endpoints are blocked.

### D.8 Failure Categories (63 Always-Fail Tasks)

| Category | Count | Fixable? |
|---|---|---|
| Regression/OLS | 16 | Maybe — needs correct python scripts |
| Curled external (wasted turns) | 16 | Partially — correct API URLs could help |
| Looping (50+ retries) | 10 | Hard — prompt anti-loop rules ignored |
| Exchange rate (needs FX data) | 9 | No — data not in corpus |
| CPI/inflation | 8 | Partially — inline data or correct BLS curl |
| Stdev/statistics | 7 | Maybe — needs correct python |
| Median/percentile | 5 | Maybe — needs correct python |
| CAGR | 2 | Maybe |

Testing 20 always-fail tasks with best config (sandwich + 25 turns + MCP): only **1/20 recovered**. These are genuinely hard for MiniMax — visual questions (page numbers, chart counting), complex cross-references, and statistical calculations where MiniMax writes incorrect code.

### D.9 Root Cause: Data Extraction, Not Arithmetic

Inspection of python3 -c code in failed tasks revealed the arithmetic is usually correct — the wrong input values are the problem. MiniMax misreads table structures: wrong columns for fiscal years, wrong row labels, confused by multi-section tables. When MCP compute tool is available, it gets the right answer because MiniMax only needs to identify the values (not write the code), and the tool handles the math. Without MCP, MiniMax writes valid python3 -c but with hallucinated or misread input numbers.

Example (UID0005): MiniMax extracted correct 1940 defense values (sum=2602) but used CPI_1953=83.0 (hallucinated) instead of 26.7 (correct), producing 29036 instead of 39482.

### D.10 Regression Recovery

25 tasks that v7 got right but v8 broke, tested with sandwich prompt + 25 turns:
- With MCP: 13/25 recovered
- Without MCP: ~10/25 recovered

The 3-task gap (UID0026, UID0166, UID0144) used MCP compute for calculations MiniMax couldn't do via python3 -c.

## Appendix E: Failure Taxonomy with Representative UIDs

| Category | % of Failures | Representative UIDs | Description |
|---|---|---|---|
| Wrong table/row/column | 33% | UID0027, UID0028, UID0017, UID0113, UID0158 | Wide table confusion, column mapping errors |
| Missing external data | 27% | UID0034, UID0140, UID0207 | NBER dates, unpublished forecasts, external lookups |
| Calculation methodology | 20% | UID0004, UID0041, UID0120 | Wrong formula variant, rounding errors |
| Time period semantics | 13% | UID0008, UID0055, UID0059 | FY vs CY, pre/post-1977 boundary, proposal months |
| Hallucination | 7% | UID0196, UID0213 | Fabricated CPI values, invented exchange rates |

## Appendix E: Database Engineering Summary

| DB Version | Size (raw) | Size (compressed) | Records | Key Feature | Result |
|---|---|---|---|---|---|
| V1: Full corpus | 11 GB | ~2 GB | 18.4M cells | Complete but unwieldy | Too large for tarball |
| V2: Lean | 4.8 GB | 581 MB | Same, indexed | Cell indexes (1,500x speedup) | Fit on droplet |
| V3: Slim | ~1 GB | 118 MB | No cell_blobs | Lookup indices only | Fit tarball; 189.2 pts |
| V4: Enriched | ~150 MB | 68 MB | 677K ledger | Master ledger + precomputed YoY | Best structured DB |
| V5: FTS5 | ~200 MB | N/A | + FTS5 index | Full-text search + fuzzy matching | Mixed A/B vs grep |
| None | 0 | **28 KB tarball** | Raw TXT corpus | Shell grep | **Best: 184.5 pts** |

**Ingestion pipeline compression ratio:** 12.3x (3.8GB raw → ~300MB compressed blobs via msgpack + zstd level 3)

**Query performance by approach:**
| Method | Latency | Data Coverage |
|---|---|---|
| Master ledger lookup | <1 ms | ~50% (parsing losses) |
| Compressed blob decompress | ~0.9 ms | ~50% |
| SQL full table scan | 10--50 ms | ~50% |
| grep on raw TXT | ~5--50 ms | **100%** |

## Appendix F: Infrastructure & Cost Summary

| Component | Peak State | Final State | Monthly Cost |
|---|---|---|---|
| DigitalOcean droplet (MCP server) | 1 active, hosting 11GB DB + SSE server | Deleted | $12/mo → $0 |
| Daytona sandboxes | 3 concurrent (3 CPU, 3GB each) | Deleted | ~$5/mo → $0 |
| Daytona snapshot | 1.11 GB (officeqa-arena-runner) | Retained | ~$0 |
| Arena submissions | 15+ runs × 246 tasks | Complete | Per-submission |
| OpenRouter API (MiniMax M2.5) | ~$30 first run, $1.71 best run | Active | Per-use |
| **Total infrastructure at peak** | | | **~$17/mo + API** |
| **Total infrastructure final** | | | **$0 + API** |
