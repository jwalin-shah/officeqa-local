# Lossless Ledger Plan

The goal: rebuild `ledger.sqlite` so that **every element of every type from `corpus_json/` is recorded, indexed, and addressable by `(file, element_seq)` with parent context attached.** Nothing the JSON contains should be invisible to retrieval.

## Why this matters: the arena's cautionary tale

The original officeqa-arena built five database generations and ran a four-stage ingestion pipeline (`build_db_from_txt.py`, `reingest_from_json.py`, `pack_cells.py`, `build_master_ledger.py`). Their best db (V4) had 677K master_ledger records, 131K precomputed year-over-year changes, 92K table_index entries, 204K metric aliases, and 12.3× msgpack+zstd compression.

**It still lost ~50% of cell data.** From the research report: *"The DB lost approximately 50% of cell data during ingestion due to parsing edge cases (multi-row headers, merged cells, non-standard delimiters), while grep had 100% data coverage by definition."* The arena's winning system used `grep` over the raw text and ignored the database entirely.

The lesson is not "don't build a database." The lesson is **a structured database is only worth building if it actually retains everything**. Half-coverage is worse than no coverage because it gives a false sense of completeness. Our north star is the half they lost.

The arena also documented the data quality issues we'd hit: *41% of tables had NULL year metadata, multi-row headers caused parser confusion splitting one logical table into multiple fragments, footnote markers contaminated numeric values, period_basis was mis-tagged.* These are the same edge cases our `build_ledger.py` will encounter.

## Source-of-truth chain

```
~/archive/officeqa/treasury_bulletins_parsed/jsons/  ← upstream parsed corpus
                  │
                  ▼
        corpus_json/*.json   ← canonical, what we ingest
                  │
                  ├──▶ corpus/*.txt (derived markdown rendering, lossy, legacy)
                  └──▶ ledger.sqlite (what we're building)
```

JSON has typed elements (`text`, `table`, `section_header`, `title`, `caption`, `footnote`, `figure`, `page_header`, `page_footer`, `page_number`), each with `bbox` coordinates and `page_id`. Tables are stored as raw HTML inside the `content` field. The `corpus/*.txt` files were generated FROM the JSONs by the upstream `transform_parsed_files.py` script — they are strictly less than the JSONs and exist only as a legacy fallback. **Never cross-check txt against JSON; the JSON has everything they have plus more.**

## What we know about the arena's `master_ledger`

Inferred from `archive_from_arena/analyst/solve.py` queries (the actual db file is gone). The arena's central table was `master_ledger`, with these columns observed in SELECTs:

```
master_ledger(
    table_pk,        -- foreign key into table_index
    metric_slug,     -- normalized row label
    time_key,        -- e.g. "1940", "1940-01"
    period_basis,    -- "fiscal" | "calendar"
    value,           -- numeric
    value_raw,       -- original cell text
    source_file,     -- treasury_bulletin_YYYY_MM.txt
    table_title,
    row_type         -- presumably "data" / "subtotal" / "total"
)
```

**`metric_slug` is just a normalized row label** — not row × col, just the row. From `_normalize_metric_slug()` at line 604:
```python
def _normalize_metric_slug(metric):
    s = metric.lower().strip()
    s = re.sub(r'\s*\d+/', '', s)        # strip footnote markers like 1/, 2/
    s = re.sub(r'[^\w\s\-]', '', s)      # strip punctuation except hyphens
    s = re.sub(r'\s+', ' ', s)           # collapse whitespace
    return s
```

So a row labeled `"National defense 1/"` became `metric_slug = "national defense"`. A row labeled `"Veterans' Administration"` became `metric_slug = "veterans administration"`. The slug is short, lowercase, and substring-searchable.

The arena's retrieval primitive was `query_table_rows(table_pk, row_label, column_label, year)`. It searched `metric_slug` first as exact match, then as `LIKE '%term%'`. There was no BM25, no semantic similarity, no rerank — just substring lookup over a normalized index. The matching `(table_pk, time_key, value)` rows came back as direct fetches.

The supporting tables built around `master_ledger`:

| Table | Purpose |
|---|---|
| `table_index` | `(table_pk, source_file, table_title, units_line, period_basis, min_year, max_year)` — table-level metadata, used to rank candidates by year range and look up units |
| `table_cell_blobs` | `(table_pk, data)` — full cell payload msgpack+zstd compressed (12.3× ratio, 0.9ms decompression). Used when the `master_ledger` row was insufficient and the agent needed the full table |
| `row_label_lookup` | `(table_pk, row_label_norm)` — pre-built substring search index over normalized row labels (200K entries in V4) |
| `col_label_lookup` | `(table_pk, column_label, col_norm)` — same for column labels (50K entries), built from `master_ledger.metric_slug` |
| `canonical_facts` | The deduplicated fact view (we already have something similar) |

**Architectural difference vs our current ledger:** the arena was metric-oriented (one row in `master_ledger` per metric per time). We are cell-oriented (one row in `cells` per (table, row, col) position). Cell-oriented preserves more raw flags (footnote, revised, preliminary, parse_status); metric-oriented gives faster row-label lookup but requires duplicating those flags per metric or losing them.

## Decision: we go hybrid

**Cell-oriented as the source of truth, metric layer derived on top.**

The cell store stays exactly as it is (lossless on per-cell metadata). On top of it, we build a `metrics` materialized table that mirrors the arena's `master_ledger` shape, populated by a query against `cells + table_rows + table_columns`. Every metric carries forward the flags (`has_footnote`, `is_revised`, `is_preliminary`) from its underlying cell so we don't lose anything.

This gives us both retrieval primitives:
- **FTS-based retrieval** (our current `tables_fts` over title/section/caption/columns/rows) for fuzzy topic search
- **Substring lookup** over `metric_slug` for exact "find the row called X" queries — the arena's primary retrieval mode

Both come from the same underlying cells. No duplication of truth, just two indices over it.

## The plan in phases

Each phase ends with a rebuild + eval. If a phase doesn't move both `eval_ledger.py` (structural reachability) and `retrieve_v2.py --test-ledger` (table-level recall) numbers, we stop and reassess before the next phase.

### Phase 0 — Decisions (settled)

- ✅ Schema shape: hybrid. Cell-oriented source of truth + derived metric layer.
- ⏳ `corpus_index.pkl` retriever: keep as parallel baseline for one head-to-head, then retire.

### Phase 1 — Cheap wins to the existing schema

Patches to `build_ledger.py` that ride in one rebuild. No new tables.

1. **Pandas-only parse rescue.** When `lxml` fails but `pd.read_html` succeeds, convert the DataFrame back into the `(grid, is_origin, is_header)` shape that `build_column_paths` and `build_row_entries` expect, instead of dropping the table. Stops silent loss of hundreds of tables.
2. **`*` is footnote, not missing.** Strip `*` from `MISSING_TOKENS`. When a cell is `*`, set `has_footnote=1` and continue parsing whatever else is there.
3. **Add `is_estimated` flag.** Mirrors `is_revised` and `is_preliminary`. The `e/` suffix is currently stripped silently with no flag set.
4. **Tighten `detect_table_kind`.** Require the literal phrase `"table of contents"`, not `\bcontents\b`. Recovers ~2,400 data tables currently mis-flagged as TOC and excluded from FTS.
5. **Section header heuristic fix.** Require the row label to be **non-numeric** before flagging as section header. A row labeled `"1940"` with empty data cells should not be a section header.
6. **Year inheritance** ✅ already done — row leaf → row_path fallback for `extract_year_month`.
7. **Don't drop captions at 5 elements.** Drop the magic-number heuristic; use page boundary as the cutoff instead (a caption applies until the next table on the same page or until the page changes).

Time: ~1 hour. Rebuild: ~5 minutes. Validates the year-inheritance fix in the same shot.

### Phase 2 — Ingest the dropped element types

The biggest leak. Across 8 sample files we counted ~217 `text`, ~144 `footnote`, and ~130 `page_header` elements per file being completely dropped. Across 697 files that's roughly:

| Element type | Per-file avg | Estimated total | Currently | After Phase 2 |
|---|---|---|---|---|
| `text` | 217 | ~151,000 | dropped | `prose` table + `prose_fts` |
| `footnote` | 144 | ~100,000 | dropped | `footnotes` table + `footnotes_fts` |
| `page_header` | 130 | ~90,000 | dropped | `page_metadata` table |

#### New table: `prose`

```sql
CREATE TABLE prose (
    id              INTEGER PRIMARY KEY,
    file            TEXT NOT NULL,
    element_seq     INTEGER NOT NULL,
    page_id         INTEGER,
    file_year       INTEGER,
    file_month      INTEGER,
    section         TEXT,           -- latched current_section at time of element
    title           TEXT,           -- latched current_title
    content         TEXT NOT NULL,
    near_table_id   INTEGER REFERENCES tables(id)  -- nearest table on same page
);
CREATE VIRTUAL TABLE prose_fts USING fts5(
    content, section, title,
    content='prose', content_rowid='id'
);
```

Populated from `text` elements during `process_file`. `near_table_id` is set to the nearest table id on the same page (within ±5 elements either side).

#### New table: `footnotes`

```sql
CREATE TABLE footnotes (
    id              INTEGER PRIMARY KEY,
    file            TEXT NOT NULL,
    element_seq     INTEGER NOT NULL,
    page_id         INTEGER,
    marker          TEXT,           -- '1/', '2/', '*', etc. (parsed from start)
    content         TEXT NOT NULL,
    attached_to_table_id INTEGER REFERENCES tables(id)
);
CREATE VIRTUAL TABLE footnotes_fts USING fts5(
    content, content='footnotes', content_rowid='id'
);
```

Populated from `footnote` elements. `marker` parsed by regex from the start of `content` (Treasury convention: a leading `\d+/` or `\*`). Attachment is the most recent table on the same page above the footnote (footnotes typically follow their tables).

This finally gives meaning to the orphaned `cells.has_footnote` flags — at query time we can join `cells` to `footnotes` via `(file, page_id)` proximity.

#### New table: `page_metadata`

```sql
CREATE TABLE page_metadata (
    file            TEXT NOT NULL,
    page_id         INTEGER NOT NULL,
    header_text     TEXT,
    footer_text     TEXT,
    image_uri       TEXT,           -- from pages[].image_uri (currently None for our corpus)
    PRIMARY KEY (file, page_id)
);
```

Populated from `page_header` and `page_footer` elements, plus the `pages[]` array's `image_uri` field. Cheap. Page headers carry section context across pages within a logical section.

Time: ~3 hours. Rebuild: ~7 minutes. Validates that recall@10 moves on questions whose gold answer was in a footnote or chart caption.

### Phase 3 — Metric layer (the arena pattern)

Build the materialized `metrics` table from the cell store.

```sql
CREATE TABLE metrics (
    id              INTEGER PRIMARY KEY,
    table_id        INTEGER NOT NULL REFERENCES tables(id),
    metric_slug     TEXT NOT NULL,    -- normalized row label
    row_path        TEXT,              -- full hierarchical path
    col_path        TEXT,
    time_key        TEXT,              -- '1940', '1940-01', 'FY1940'
    year            INTEGER,
    month           INTEGER,
    period_basis    TEXT,              -- 'fiscal' | 'calendar' | 'unknown'
    value           REAL,
    value_raw       TEXT,
    unit            TEXT,
    file            TEXT NOT NULL,
    page_id         INTEGER,
    has_footnote    INTEGER DEFAULT 0,
    is_revised      INTEGER DEFAULT 0,
    is_preliminary  INTEGER DEFAULT 0,
    is_estimated    INTEGER DEFAULT 0
);
CREATE INDEX idx_metrics_slug   ON metrics(metric_slug);
CREATE INDEX idx_metrics_year   ON metrics(year);
CREATE INDEX idx_metrics_table  ON metrics(table_id);
```

Populated by a single pass over `cells JOIN table_rows JOIN table_columns JOIN tables` after the cell ingest finishes. `metric_slug` is computed from `row_path` using the arena's `_normalize_metric_slug` regex. Each cell that has a data value becomes one metric row. Cell flags propagate forward.

Then build the lookup indices the arena had:

```sql
CREATE TABLE row_label_lookup AS
    SELECT DISTINCT table_id, metric_slug, row_path FROM metrics;
CREATE INDEX idx_rll_slug ON row_label_lookup(metric_slug);

CREATE TABLE col_label_lookup AS
    SELECT DISTINCT table_id, col_path,
           LOWER(REPLACE(REPLACE(col_path, '/', ''), '.', '')) AS col_norm
    FROM metrics WHERE col_path IS NOT NULL;
CREATE INDEX idx_cll_norm ON col_label_lookup(col_norm);
```

Add a new retrieval primitive in `retrieve_v2.py`:

```python
def retrieve_by_metric(question: str, top_k: int = 10):
    """Substring lookup over metric_slug + col_norm. Arena-style fast path."""
```

The retrieval funnel becomes a two-channel union:

1. **FTS channel** (current): topical/fuzzy search over titles, sections, captions, columns, rows
2. **Metric channel** (new): exact/substring lookup over `metric_slug` and `col_norm`

Each channel returns top-N candidates; we union and re-rank by file_year proximity. Most questions probably match better through one channel than the other; the union captures both.

Time: ~2 hours. Rebuild: +2 minutes for the materialization.

### Phase 4 — Restore lost structure

Deeper parser work. Lower confidence, but each item closes a real edge case.

1. **Real `indent_level` from raw HTML.** Walk `<th>` / `<td>` `style` attributes for `padding-left`, or count nested `<table>` depth. Restores hierarchical depth currently hardcoded to 0.
2. **Section header push semantics.** When a row is detected as a section header at indent level N, push it onto ancestry at position N rather than wiping the whole stack.
3. **Multi-page table continuation detection.** Use `bbox` y-coordinates and `page_id` to detect tables that span pages. Currently each page becomes a separate table, missing continuation context.
4. **Re-run numeric extraction on text-classified cells.** Cells like `"$2.5 million (revised)"` are currently classified as text and the numeric value is never extracted. Try the numeric path first; only fall back to text if it fails.
5. **CY total synthesis from monthly rows.** The arena's `enrich_calendar_totals.py` synthesized calendar-year totals from 12 monthly rows. The research report says 16 benchmark questions need this. Cheap to do at build time.
6. **Replace `'*'` MISSING_TOKEN check entirely** — already in Phase 1, but verify cell-counting invariants still pass.

Time: ~4 hours. Rebuild: ~10 minutes.

### Phase 5 — Validation and head-to-head

After Phases 1–4, run the full eval suite and decide whether to retire `corpus_index.pkl`:

- `eval_ledger.py` — structural reachability (file, page, cell-present)
- `retrieve_v2.py --test-ledger` — table-level recall@1/5/10/30
- One-shot rebuild of `corpus_index.pkl` and run `retrieve_v2.py --test` for direct head-to-head
- Compare on 246 questions, full table

If ledger-backed retrieval ≥ pkl on table-level recall@10, delete `build_index.py`, `corpus_index.pkl`, the `.cache/` directory, and the BM25/semantic code paths in `retrieve_v2.py`. The ledger becomes the only retrieval substrate.

## Validation gates after each phase

After each phase, the rule from `feedback_lossless_principle.md`: if neither `eval_ledger.py` nor `retrieve_v2.py --test-ledger` numbers move, **stop** and figure out why before continuing. Don't compound complexity on a phase that didn't pay.

Concrete targets:
- After Phase 1: structural reachable@page should hold or improve from 93.5%; table recall@10 should improve by at least a few points (year inheritance alone should help)
- After Phase 2: a measurable improvement on questions whose gold answer is in footnote/chart context (the 6% gap from earlier)
- After Phase 3: substring metric lookup should hit on questions where the row label is a clean noun phrase (e.g. "national defense") — these should jump to recall@1
- After Phase 4: cleanup phase, expect smaller deltas

## What I need from you to start

1. ✅ Phase 0 schema decision: hybrid (cell-oriented source of truth + metric layer)
2. ⏳ Greenlight Phase 1
3. ⏳ Decision on retiring `corpus_index.pkl` after Phase 5

Once Phases 1+2+3 land, the ledger should be honestly close to "loses nothing" — every element type is in a queryable table, every cell carries its flags forward into a metric, every footnote and prose element is searchable, and both retrieval primitives (FTS topical and metric substring) are wired up.

## Open questions to revisit later

- **Compression of the cell store.** The arena got a 12.3× ratio with msgpack+zstd at level 3, with 0.9ms decompression. Our `cells` table is currently ~2GB uncompressed. Worth doing once everything else is correct.
- **Pre-computed fiscal/calendar year totals.** The arena's `enrich_calendar_totals.py` synthesized these. Listed in Phase 4 but might want to break out as its own phase if the impact is large.
- **Metric aliases across decades.** The arena's V4 had 204K metric aliases — the same metric had different names across eras (e.g. "National defense" vs "Defense expenditures" vs "Military expenditures"). They built a canonical-name mapping. We don't have this and may need it for cross-decade questions.
- **Bbox-based spatial layout.** We extract `page_id` from `bbox` and throw away the coordinates. They could disambiguate co-located tables and help footnote attachment. Currently unused.
