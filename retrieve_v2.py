#!/usr/bin/env python3
"""Table-level retrieval against ledger.sqlite.

Two-channel funnel, both backed by the same lossless cell store:

  FTS channel    — FTS5 `tables_fts` over title/section/caption/columns/rows
                   with year filtering. Good for topical/fuzzy questions.
  metric channel — substring lookup over `metrics.metric_slug` (normalized
                   row label) + optional `col_label_lookup.col_norm`. Good
                   when the question's noun phrase matches a row label
                   cleanly ("national defense" → exact slug hit). Mirrors
                   the arena's master_ledger primary retrieval primitive.

Each channel returns top-N candidates; the union is reranked by file_year
proximity and returned as top-k dicts shape-compatible with extract.py.

The old corpus_index.pkl + BM25 + sentence-transformer retriever was retired
once the ledger-backed funnel pulled ahead on head-to-head recall.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from ledger_paths import get_ledger_sqlite_path

_reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(_reconfigure_stdout):
    _reconfigure_stdout(line_buffering=True)

LEDGER_PATH = get_ledger_sqlite_path()

STOPWORDS = {
    "the",
    "a",
    "an",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "by",
    "with",
    "from",
    "and",
    "or",
    "but",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "what",
    "which",
    "who",
    "whom",
    "when",
    "where",
    "why",
    "how",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "their",
    "there",
    "we",
    "you",
    "total",
    "value",
    "values",
    "number",
    "amount",
    "report",
    "reported",
    "using",
    "specifically",
    "only",
    "all",
    "individual",
    "calendar",
    "fiscal",
    "year",
    "years",
    "month",
    "months",
    "dollars",
    "millions",
    "billions",
    "thousands",
    "nominal",
    "real",
    "inflation",
    "adjusted",
    "us",
    "united",
    "states",
    "federal",
    "government",
    "data",
    "according",
    "between",
    "during",
    "per",
    "about",
    "into",
    "over",
    "under",
    "than",
    "then",
    "also",
    "both",
    "any",
    "each",
    "some",
    "do",
    "does",
    "did",
    "have",
    "has",
    "had",
    "can",
    "could",
    "should",
    "would",
    "will",
    "shall",
    "may",
    "might",
    "must",
    "just",
    "more",
    "most",
    "very",
    "million",
    "billion",
    "thousand",
    "dollar",
    "percent",
    "percentage",
    "rate",
    "sum",
    "average",
    "mean",
    "median",
    "count",
    "list",
    "treasury",
    "bulletin",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-3]\d)\b")
_MONTH_NAME_RE = re.compile(
    r"(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{4})",
    re.I,
)
_MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_MONTHLY_RE = re.compile(
    r"\b(month|monthly|months|january|february|march|april|may|june|"
    r"july|august|september|october|november|december)\b",
    re.IGNORECASE,
)

# Arena-compatible metric slug normalizer (matches build_ledger.normalize_metric_slug)
_METRIC_SLUG_FOOTNOTE_RE = re.compile(r"\s*\d+/")
_METRIC_SLUG_PUNCT_RE = re.compile(r"[^\w\s\-]")
_METRIC_SLUG_WS_RE = re.compile(r"\s+")

_SYNONYM_GROUPS = [
    ("assets", "liabilities", "condition", "balance sheet"),
    ("debt", "public debt", "obligations", "securities", "bonds", "notes", "bills", "certificates"),
    ("estimates", "projections", "budget", "forecasts"),
    ("currency", "money", "circulation", "coins", "notes", "cash"),
    ("expenditures", "outlays", "spending", "disbursements"),
    ("receipts", "revenue", "income", "collections"),
    ("defense", "national defense", "military"),
    ("veterans", "veterans affairs", "veterans administration", "va"),
    ("social security", "social insurance", "trust funds"),
    ("deficit", "shortfall"),
    ("grants", "grants-in-aid", "federal grants"),
    ("debt", "public debt", "obligations"),
    ("interest", "net interest", "interest on debt"),
    ("medicare", "health", "health insurance"),
    ("surplus", "excess", "balance"),
    ("tax", "taxation", "taxes", "revenue"),
]

FTS_SYNONYM_THRESHOLD = 100
METRIC_SYNONYM_THRESHOLD = 100

# Progressive-stage bonuses: trace metadata already records which stage first
# retrieved each table; nudging the reranker rewards higher-confidence paths.
_FTS_STRATEGY_BONUS: dict[str, float] = {
    "exact_year_filter": 0.60,
    "exact_year_window": 0.40,
    "synonym_year_window": 0.30,
    "synonym_unrestricted": 0.20,
    "year_shifted": 0.10,
    "exact_unrestricted": 0.20,
}
_METRIC_STRATEGY_BONUS: dict[str, float] = {
    "primary_exact": 0.80,
    "synonym_exact": 0.50,
    "partial_match": 0.20,
}


def _strategy_score_nudge(
    fts_strategy: str | None,
    metric_strategy: str | None,
) -> float:
    """Tiebreaker from first-hit retrieval stage (FTS ∪ metric progressive funnel)."""
    fb = _FTS_STRATEGY_BONUS.get(fts_strategy or "", 0.0)
    ms = metric_strategy or ""
    if ms.startswith("alt:"):
        ms = ms[4:]
    mb = _METRIC_STRATEGY_BONUS.get(ms, 0.0)
    # Use max so dual-channel hits are not double-counted into an oversized bump.
    return max(fb, mb)


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def content_tokens(text: str) -> list[str]:
    return [t for t in _tokenize(text) if t not in STOPWORDS and len(t) >= 3]


def normalize_metric_slug(metric: str | None) -> str:
    if not metric:
        return ""
    s = metric.lower().strip()
    s = _METRIC_SLUG_FOOTNOTE_RE.sub("", s)
    s = _METRIC_SLUG_PUNCT_RE.sub("", s)
    s = _METRIC_SLUG_WS_RE.sub(" ", s).strip()
    return s


def _build_synonym_map(groups: list[tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    merged: dict[str, set[str]] = {}
    for group in groups:
        normalized = [normalize_metric_slug(term) for term in group if normalize_metric_slug(term)]
        for term in normalized:
            bucket = merged.setdefault(term, set())
            for alt in normalized:
                if alt != term:
                    bucket.add(alt)
    return {
        term: tuple(sorted(alts, key=lambda alt: (alt.count(" "), len(alt), alt)))
        for term, alts in merged.items()
    }


SYNONYMS = _build_synonym_map(_SYNONYM_GROUPS)


def _dedupe_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        norm = normalize_metric_slug(value)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        ordered.append(value)
    return ordered


def _matched_synonym_keys(text: str) -> list[str]:
    norm = normalize_metric_slug(text)
    if not norm:
        return []

    matches: list[tuple[int, int, str]] = []
    for key in sorted(SYNONYMS, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(key)}\b")
        for match in pattern.finditer(norm):
            start, end = match.span()
            if any(not (end <= ms or start >= me) for ms, me, _ in matches):
                continue
            matches.append((start, end, key))

    matches.sort(key=lambda item: item[0])
    return [key for _, _, key in matches]


def _expand_synonyms(terms: list[str], source_text: str = "") -> list[str]:
    expanded = _dedupe_keep_order(terms)
    seen = {normalize_metric_slug(term) for term in expanded}

    def add_term(term: str) -> None:
        norm = normalize_metric_slug(term)
        if not norm or norm in seen:
            return
        seen.add(norm)
        expanded.append(term)

    for term in list(expanded):
        for synonym in SYNONYMS.get(normalize_metric_slug(term), ()):
            add_term(synonym)

    for key in _matched_synonym_keys(source_text):
        add_term(key)
        for synonym in SYNONYMS.get(key, ()):
            add_term(synonym)

    return expanded


def _row_hint_variants(row_hint: str) -> list[str]:
    norm = normalize_metric_slug(row_hint)
    if not norm:
        return []

    variants = [norm]
    seen = {norm}

    no_hyphen = normalize_metric_slug(norm.replace("-", " "))
    if no_hyphen and no_hyphen not in seen:
        seen.add(no_hyphen)
        variants.append(no_hyphen)

    keys_found = list(_matched_synonym_keys(norm))
    if keys_found:
        if len(keys_found) <= 3:
            current_combos = {norm}
            for key in keys_found:
                opts = [key] + list(SYNONYMS.get(key, ()))
                next_combos = set()
                for combo in current_combos:
                    for opt in opts:
                        cand = normalize_metric_slug(
                            re.sub(rf"\b{re.escape(key)}\b", opt, combo, count=1)
                        )
                        if cand:
                            next_combos.add(cand)
                current_combos = next_combos
            for c in current_combos:
                if c not in seen:
                    seen.add(c)
                    variants.append(c)
        else:
            for key in keys_found:
                for synonym in SYNONYMS.get(key, ()):
                    candidate = normalize_metric_slug(
                        re.sub(rf"\b{re.escape(key)}\b", synonym, norm, count=1)
                    )
                    if candidate and candidate not in seen:
                        seen.add(candidate)
                        variants.append(candidate)

    stripped = re.sub(r"[-\s]+", "", norm)
    if stripped and stripped not in seen:
        seen.add(stripped)
        variants.append(stripped)

    return variants


def _bulk_cell_probe(
    conn: sqlite3.Connection,
    table_ids: set[int] | list[int],
    row_hints: list[str],
) -> dict[int, tuple[int, int]]:
    """Return {table_id: (matching_row_count, best_row_non_null_cells)}.

    Ground-truth row probe: for each candidate table, find rows whose
    `metric_slug` matches one of the row_hint variants, and for each
    matching row count its non-null, non-missing cells. Report:

      - matching_row_count: how many distinct rows in this table match
        (1 is ideal; >1 means the DR is ambiguous here)
      - best_row_non_null_cells: the *largest* non-null cell count among
        matching rows (this is the "can this row actually supply values"
        signal — a row with 12 monthlies beats a row with 1 annual)

    Total cells was a bad signal: big tables with many detailed-breakdown
    rows (e.g. "National defense", "National defense limitation", "…
    reserves") accumulate cells regardless of whether any single row can
    answer the DR.

    Zero matching rows → drop candidate. Deliberately does NOT year-
    filter; file-year scoping is handled upstream by the FTS window.

    One SQL round-trip for the whole candidate set.
    """
    if not table_ids or not row_hints:
        return {}

    slug_patterns: list[str] = []
    seen: set[str] = set()
    for rh in row_hints:
        for variant in _row_hint_variants(rh) or [normalize_metric_slug(rh)]:
            if not variant or variant in seen:
                continue
            seen.add(variant)
            slug_patterns.append(f"%{variant}%")
    if not slug_patterns:
        return {}

    tid_list = list(table_ids)
    tid_placeholders = ",".join("?" * len(tid_list))
    row_clause = " OR ".join(["r.metric_slug LIKE ?"] * len(slug_patterns))

    # Per (table, row) non-null cell count, then aggregate up to table.
    sql = f"""
        WITH row_cells AS (
            SELECT r.table_id AS table_id,
                   r.id       AS row_id,
                   SUM(CASE WHEN c.numeric_value IS NOT NULL
                              AND COALESCE(c.is_missing, 0) = 0
                             THEN 1 ELSE 0 END) AS non_null_count
              FROM table_rows r
              LEFT JOIN cells c ON c.row_id = r.id AND c.table_id = r.table_id
             WHERE r.table_id IN ({tid_placeholders})
               AND ({row_clause})
             GROUP BY r.table_id, r.id
        )
        SELECT table_id,
               COUNT(*)            AS matching_row_count,
               MAX(non_null_count) AS best_row_non_null_cells
          FROM row_cells
         GROUP BY table_id
    """
    params: tuple = (*tid_list, *slug_patterns)
    rows = conn.execute(sql, params).fetchall()

    res = {int(r[0]): (int(r[1] or 0), int(r[2] or 0)) for r in rows}
    if res:
        return res

    # Fallback: permissive token-set match on row_path
    fallback_sql = f"""
        WITH row_cells AS (
            SELECT r.table_id AS table_id,
                   r.id       AS row_id,
                   r.row_path AS row_path,
                   SUM(CASE WHEN c.numeric_value IS NOT NULL
                              AND COALESCE(c.is_missing, 0) = 0
                             THEN 1 ELSE 0 END) AS non_null_count
              FROM table_rows r
              LEFT JOIN cells c ON c.row_id = r.id AND c.table_id = r.table_id
             WHERE r.table_id IN ({tid_placeholders})
               AND r.row_path IS NOT NULL
             GROUP BY r.table_id, r.id
        )
        SELECT table_id, row_id, row_path, non_null_count
          FROM row_cells
    """
    all_rows = conn.execute(fallback_sql, tid_list).fetchall()

    hint_tokens_list = []
    for rh in row_hints:
        tokens = set(re.findall(r"[a-z0-9]+", rh.lower()))
        if tokens:
            hint_tokens_list.append(tokens)

    if not hint_tokens_list:
        return {}

    matching_rows_by_table = {}
    for r in all_rows:
        t_id = int(r[0])
        r_id = int(r[1])
        path = str(r[2] or "").lower()
        path_tokens = set(re.findall(r"[a-z0-9]+", path))

        match = False
        for hint_tokens in hint_tokens_list:
            if hint_tokens.issubset(path_tokens):
                match = True
                break
            # Or if "most" tokens match (e.g., all but one, and at least 3 tokens)
            if (
                len(hint_tokens) >= 3
                and len(hint_tokens.intersection(path_tokens)) >= len(hint_tokens) - 1
            ):
                match = True
                break

        if match:
            if t_id not in matching_rows_by_table:
                matching_rows_by_table[t_id] = []
            matching_rows_by_table[t_id].append((r_id, int(r[3] or 0)))

    for t_id, t_rows in matching_rows_by_table.items():
        matching_row_count = len(t_rows)
        best_row_non_null = max(non_null for _, non_null in t_rows) if t_rows else 0
        res[t_id] = (matching_row_count, best_row_non_null)

    return res


def _candidate_compatibility(
    conn: sqlite3.Connection,
    table_ids: set[int] | list[int],
    row_hints: list[str],
    col_hints: list[str],
    target_years: list[int],
    requested_granularities: set[str],
) -> dict[int, CandidateCompatibility]:
    if not table_ids:
        return {}

    tid_list = list(table_ids)
    compat = {tid: CandidateCompatibility() for tid in tid_list}
    tid_placeholders = ",".join("?" * len(tid_list))

    if target_years:
        year_placeholders = ",".join("?" * len(target_years))
        year_rows = conn.execute(
            f"""
            SELECT table_id, year
              FROM (
                    SELECT table_id, year_extracted AS year
                      FROM table_columns
                     WHERE table_id IN ({tid_placeholders})
                    UNION ALL
                    SELECT table_id, year_extracted AS year
                      FROM table_rows
                     WHERE table_id IN ({tid_placeholders})
                   )
             WHERE year IN ({year_placeholders})
            """,
            (*tid_list, *tid_list, *target_years),
        ).fetchall()

        table_years: dict[int, set[int]] = {}
        for row in year_rows:
            tid = int(row[0])
            year = int(row[1])
            if tid not in table_years:
                table_years[tid] = set()
            table_years[tid].add(year)

        for tid, years in table_years.items():
            compat[tid].has_target_year = True
            compat[tid].matching_target_years_count = len(years)

    month_rows = conn.execute(
        f"""
        SELECT DISTINCT table_id
          FROM (
                SELECT table_id, month_extracted AS month
                  FROM table_columns
                 WHERE table_id IN ({tid_placeholders})
                UNION ALL
                SELECT table_id, month_extracted AS month
                  FROM table_rows
                 WHERE table_id IN ({tid_placeholders})
               )
         WHERE month IS NOT NULL
        """,
        (*tid_list, *tid_list),
    ).fetchall()
    for row in month_rows:
        compat[int(row[0])].has_month_data = True

    if requested_granularities:
        monthly_like = {"monthly_all", "specific_month"}
        quarterly_like = {"quarterly"}
        annual_like = {"annual"}
        for state in compat.values():
            if requested_granularities & monthly_like or requested_granularities & quarterly_like:
                state.has_required_granularity = state.has_month_data
            elif requested_granularities & annual_like:
                state.has_required_granularity = True
            else:
                state.has_required_granularity = True
    else:
        for state in compat.values():
            state.has_required_granularity = True

    probe_stats = _bulk_cell_probe(conn, tid_list, row_hints)
    for tid, (row_match_count, best_row_non_null_cells) in probe_stats.items():
        state = compat.setdefault(tid, CandidateCompatibility())
        state.row_match_count = row_match_count
        state.best_row_non_null_cells = best_row_non_null_cells
        state.has_row_hint_match = row_match_count > 0 and best_row_non_null_cells > 0

    if col_hints:
        normalized_hints = []
        for hint in col_hints:
            tokens = set(re.findall(r"[a-z0-9]+", hint.lower()))
            if tokens:
                normalized_hints.append(tokens)
        if normalized_hints:
            col_rows = conn.execute(
                f"""
                SELECT table_id, col_path, col_leaf
                  FROM table_columns
                 WHERE table_id IN ({tid_placeholders})
                """,
                tid_list,
            ).fetchall()
            table_col_matches: dict[int, int] = {}
            for row in col_rows:
                table_id = int(row[0])
                col_text = " ".join((str(row[1] or ""), str(row[2] or ""))).lower()
                col_tokens = set(re.findall(r"[a-z0-9]+", col_text))
                if not col_tokens:
                    continue
                best_match = 0
                for hint_tokens in normalized_hints:
                    if hint_tokens.issubset(col_tokens):
                        best_match = max(best_match, len(hint_tokens))
                        continue
                    overlap = len(hint_tokens.intersection(col_tokens))
                    if len(hint_tokens) >= 2 and overlap >= max(1, len(hint_tokens) - 1):
                        best_match = max(best_match, overlap)
                if best_match > 0:
                    table_col_matches[table_id] = max(
                        table_col_matches.get(table_id, 0), best_match
                    )
            for tid, count in table_col_matches.items():
                state = compat.setdefault(tid, CandidateCompatibility())
                state.col_match_count = count
                state.has_col_hint_match = True

    return compat


def _apply_year_compatibility_filter(
    candidate_ids: set[int],
    compat: dict[int, CandidateCompatibility],
    target_years: list[int],
    top_k: int,
) -> set[int]:
    if not candidate_ids or not target_years:
        return candidate_ids

    year_compatible = {
        tid for tid in candidate_ids if compat.get(tid, CandidateCompatibility()).has_target_year
    }
    min_pool = max(min(top_k, 50), 10)
    if len(year_compatible) >= min_pool:
        return year_compatible
    return candidate_ids


def _apply_structure_compatibility_filter(
    candidate_ids: set[int],
    compat: dict[int, CandidateCompatibility],
    requested_granularities: set[str],
    col_hints: list[str],
    top_k: int,
) -> set[int]:
    if not candidate_ids:
        return candidate_ids

    min_pool = max(min(top_k, 50), 10)
    filtered = set(candidate_ids)

    if requested_granularities & {"monthly_all", "specific_month", "quarterly"}:
        granularity_compatible = {
            tid
            for tid in filtered
            if compat.get(tid, CandidateCompatibility()).has_required_granularity
        }
        if len(granularity_compatible) >= min_pool:
            filtered = granularity_compatible

    if col_hints:
        col_compatible = {
            tid for tid in filtered if compat.get(tid, CandidateCompatibility()).has_col_hint_match
        }
        if len(col_compatible) >= min_pool:
            filtered = col_compatible

    return filtered


def _merge_ranked_rows(*groups: list[sqlite3.Row], top_n: int) -> list[sqlite3.Row]:
    merged: list[sqlite3.Row] = []
    seen_ids: set[int] = set()
    for group in groups:
        for row in group:
            row_id = int(row["id"])
            if row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            merged.append(row)
            if len(merged) >= top_n:
                return merged
    return merged


@dataclass
class ChannelTrace:
    rows: list[sqlite3.Row]
    strategy_by_id: dict[int, str]
    attempts: list[tuple[str, int]]


@dataclass
class CandidateCompatibility:
    has_target_year: bool = False
    matching_target_years_count: int = 0
    has_month_data: bool = False
    has_required_granularity: bool = False
    has_row_hint_match: bool = False
    has_col_hint_match: bool = False
    row_match_count: int = 0
    col_match_count: int = 0
    best_row_non_null_cells: int = 0


def _append_unique_rows(
    merged: list[sqlite3.Row],
    seen_ids: set[int],
    strategy_by_id: dict[int, str],
    rows: list[sqlite3.Row],
    strategy: str,
    top_n: int,
) -> None:
    for row in rows:
        row_id = int(row["id"])
        if row_id in seen_ids:
            continue
        seen_ids.add(row_id)
        strategy_by_id[row_id] = strategy
        merged.append(row)
        if len(merged) >= top_n:
            return


def _collect_progressive_rows(
    stages: list[tuple[str, Callable[[], list[sqlite3.Row]]]],
    threshold: int,
    top_n: int,
) -> ChannelTrace:
    merged: list[sqlite3.Row] = []
    seen_ids: set[int] = set()
    strategy_by_id: dict[int, str] = {}
    attempts: list[tuple[str, int]] = []

    for strategy, runner in stages:
        rows = runner()
        attempts.append((strategy, len(rows)))
        _append_unique_rows(merged, seen_ids, strategy_by_id, rows, strategy, top_n)
        if len(merged) >= top_n or len(rows) >= threshold:
            break

    return ChannelTrace(rows=merged, strategy_by_id=strategy_by_id, attempts=attempts)


# ── Question parsing helpers ────────────────────────────────────────────────


def parse_years_from_question(question: str) -> list[int]:
    return sorted({int(y) for y in _YEAR_RE.findall(question)})


def parse_direct_bulletin_refs(question: str) -> list[tuple[int, int]]:
    """Pull (year, month) pairs out of phrases like 'the July 1946 bulletin'."""
    return [
        (int(m.group(2)), _MONTH_MAP[m.group(1).lower()]) for m in _MONTH_NAME_RE.finditer(question)
    ]


def detect_year_mode(question: str) -> str:
    q = question.lower()
    if "calendar year" in q or "calendar month" in q or "calendar quarter" in q:
        return "calendar"
    if "fiscal year" in q or re.search(r"\bfy\s*\d", q):
        return "fiscal"
    return "unknown"


def _detect_wants_monthly(question: str) -> bool:
    return bool(_MONTHLY_RE.search(question))


def _file_year_bonus(file_year, file_month, target_years, year_mode) -> float:
    """Asymmetric bonus: gold files cluster at offset +1 for calendar questions
    and offset 0 for fiscal, with strong month tiebreakers."""
    if not target_years:
        return 0.0
    try:
        fy = int(file_year) if file_year is not None else None
    except (TypeError, ValueError):
        return 0.0
    if fy is None:
        return 0.0
    try:
        fm = int(file_month) if file_month is not None else None
    except (TypeError, ValueError):
        fm = None

    max_ty = max(target_years)
    offset = fy - max_ty
    # Treasury-specific publication cadence (this is harness-level domain
    # knowledge, intentional): "calendar year N" questions want the annual
    # summary published in the following Jan-Mar (offset +1). Fiscal-year
    # questions want the bulletin covering FY end (offset 0, months 7-10).
    # Snapshot-style questions ("as of 2016", "in mid-March 2016") land on
    # offset 0 — we treat those as year_mode=="unknown".
    if year_mode == "calendar":
        if offset == 1:
            base = 6.0
        elif offset == 2:
            base = 3.5
        elif offset == 3:
            base = 2.5
        elif 4 <= offset <= 6:
            base = 1.5
        elif offset == 0:
            base = 2.0
        elif offset < 0 and min(target_years) <= fy:
            base = 1.5
        else:
            base = 0.0
    elif year_mode == "fiscal":
        if offset == 0:
            base = 6.0
        elif offset == 1:
            base = 3.5
        elif offset == 2:
            base = 2.5
        elif offset == 3:
            base = 2.0
        elif 4 <= offset <= 6:
            base = 1.0
        elif offset < 0 and min(target_years) <= fy:
            base = 1.5
        else:
            base = 0.0
    else:
        if offset == 0:
            base = 5.0
        elif offset == 1:
            base = 4.5
        elif offset == 2:
            base = 3.0
        elif offset == 3:
            base = 2.5
        elif 4 <= offset <= 6:
            base = 1.5
        elif offset < 0 and min(target_years) <= fy:
            base = 2.0
        else:
            base = 0.0

    month_bonus = 0.0
    if fm is not None and base > 0:
        # Calendar-year questions: the annual summary lands in the
        # January-March issue of the following year (offset +1, Q1 months).
        # Fiscal-year questions: the annual summary is in the bulletin
        # covering FY end (offset 0, months >=7).
        if year_mode == "calendar" and offset == 1:
            if fm <= 3:
                month_bonus = 2.0
            elif fm <= 6:
                month_bonus = 1.0
        elif year_mode == "fiscal" and offset == 0:
            if 7 <= fm <= 10:
                month_bonus = 2.0
            elif fm >= 7:
                month_bonus = 1.0
        elif offset == 0 and fm >= 6:
            month_bonus = 0.5
    return base + month_bonus


def _period_aware_score_delta(year_mode: str, table_period: str | None) -> float:
    """Score delta for period-aware reranking.

    When the question targets calendar/fiscal year, tables whose period
    column matches get a boost (>=0.5) and mismatched tables get a
    penalty (<=-0.3). Unknown mode or empty/None period → 0.0.
    """
    if year_mode not in ("calendar", "fiscal"):
        return 0.0
    if not table_period:
        return 0.0
    tp = table_period.lower().strip()
    if not tp:
        return 0.0
    # Map table period values to standard forms
    # Common period values in ledger: "fiscal", "calendar", "monthly", "annual", etc.
    if "fiscal" in tp or tp.startswith("fy"):
        normalized = "fiscal"
    elif "calendar" in tp or tp.startswith("cy"):
        normalized = "calendar"
    else:
        # Unknown period type — neither boost nor penalty
        return 0.0
    if year_mode == normalized:
        return 0.5
    else:
        return -0.3


_LEDGER_TLS = threading.local()
_JSON_ELEMENT_CACHE: dict[str, dict[int, dict]] = {}
_JSON_ELEMENT_CACHE_LOCK = threading.Lock()


def _ledger_conn() -> sqlite3.Connection:
    conn = getattr(_LEDGER_TLS, "conn", None)
    if conn is None:
        if not LEDGER_PATH.exists():
            raise FileNotFoundError(f"{LEDGER_PATH} not found — run build_ledger.py first")
        conn = sqlite3.connect(str(LEDGER_PATH))
        conn.row_factory = sqlite3.Row
        _LEDGER_TLS.conn = conn
    return conn


def _load_element_html(file: str, element_seq: int) -> str:
    """Fetch a table element's HTML from corpus_json on demand.

    Cached per file because a single query's top-k is likely to hit the
    same bulletin repeatedly for year-comparison questions. Thread-safe:
    uses a lock around cache writes so parallel eval workers don't corrupt
    the shared dict."""
    cache = _JSON_ELEMENT_CACHE.get(file)
    if cache is None:
        path = Path("corpus_json") / file
        if not path.exists():
            return ""
        doc = json.loads(path.read_text())
        cache = {}
        for idx, el in enumerate(doc.get("document", {}).get("elements", [])):
            if el.get("type") == "table":
                cache[idx] = el
        with _JSON_ELEMENT_CACHE_LOCK:
            # Double-check: another thread may have populated it
            if file not in _JSON_ELEMENT_CACHE:
                _JSON_ELEMENT_CACHE[file] = cache
            else:
                cache = _JSON_ELEMENT_CACHE[file]
    el = cache.get(element_seq)
    return (el.get("content") or "") if el else ""


def _fts_escape(tokens: list[str], mode: str = "or") -> str:
    """Build a permissive FTS5 MATCH query. `mode` is 'or' or 'and'."""
    safe = [f'"{t}"' for t in tokens if t and "'" not in t and '"' not in t]
    safe = safe[:24]
    if not safe:
        return ""
    joiner = " AND " if mode == "and" else " OR "
    return joiner.join(safe)


# ── Hint extraction from decompose plans ────────────────────────────────────


def _extract_hints(plan: dict | None, question: str) -> list[dict]:
    """Extract retrieval hints from ALL data_requests in a plan.

    Returns a list of dicts (one per data_request), each with keys:
      metric, col_hint, row_hint, row_hint_alternatives, years, granularity, source
    If plan is None or has no data_requests, returns [].
    """
    reqs = (plan or {}).get("data_requests") or []
    if not reqs:
        return []
    question_years = parse_years_from_question(question)
    hints: list[dict] = []
    for req in reqs:
        metric = (req.get("label") or (plan or {}).get("metric") or "").strip()
        col_hint = (req.get("column_hint") or "").strip()
        row_hint = (req.get("row_hint") or "").strip()
        row_hint_alts = req.get("row_hint_alternatives") or []
        if isinstance(row_hint_alts, str):
            row_hint_alts = [row_hint_alts]
        tys: list[int] = []
        for y in req.get("years") or []:
            with suppress(TypeError, ValueError):
                tys.append(int(y))
        if not tys:
            tys = question_years
        hints.append(
            {
                "metric": metric,
                "col_hint": col_hint,
                "row_hint": row_hint,
                "row_hint_alternatives": list(row_hint_alts),
                "years": tys,
                "granularity": (req.get("granularity") or "").strip(),
                "source": (req.get("source") or "").strip(),
            }
        )
    return hints


# ── FTS channel ─────────────────────────────────────────────────────────────


def _fts_channel(
    conn: sqlite3.Connection,
    q_tokens: list[str],
    target_years: list[int],
    wants_monthly: bool,
    top_n: int = 400,
    source_text: str = "",
) -> list[sqlite3.Row]:
    return _fts_channel_trace(
        conn,
        q_tokens,
        target_years,
        wants_monthly,
        top_n=top_n,
        source_text=source_text,
    ).rows


def _fts_channel_trace(
    conn: sqlite3.Connection,
    q_tokens: list[str],
    target_years: list[int],
    wants_monthly: bool,
    top_n: int = 1000,
    source_text: str = "",
    year_mode: str = "unknown",
    limit_ids: set[int] | None = None,
) -> ChannelTrace:
    """Run the FTS5 channel with progressive broadening and trace metadata."""
    if not q_tokens:
        return ChannelTrace(rows=[], strategy_by_id={}, attempts=[])

    # Fast path: candidate set already narrowed by family+year funnel — just fetch
    # them directly by ID. No FTS needed; reranker handles scoring downstream.
    if limit_ids is not None:
        if not limit_ids:
            return ChannelTrace(rows=[], strategy_by_id={}, attempts=[])
        id_ph = ",".join("?" * len(limit_ids))
        rows = conn.execute(
            f"SELECT id, file, element_seq, element_id, page_id, file_year, file_month,"
            f" title, section, caption, unit, period, n_rows, n_cols, signature,"
            f" 0.0 AS score"
            f" FROM tables WHERE id IN ({id_ph}) AND table_kind = 'data'",
            list(limit_ids),
        ).fetchall()
        return ChannelTrace(
            rows=rows,
            strategy_by_id={r["id"]: "direct_candidate" for r in rows},
            attempts=[("direct_candidate", len(rows))],
        )

    def _build_year_filter_clause() -> tuple[str, tuple]:
        if not target_years:
            return "", ()
        ty_placeholders = ",".join("?" * len(target_years))
        month_restrict = "AND month_extracted IS NOT NULL" if wants_monthly else ""
        clause = (
            " AND t.id IN ("  # nosec B608
            f" SELECT table_id FROM table_columns"
            f" WHERE year_extracted IN ({ty_placeholders}) {month_restrict}"
            " UNION"
            f" SELECT table_id FROM table_rows"
            f" WHERE year_extracted IN ({ty_placeholders}) {month_restrict}"
            " )"
        )
        return clause, (*target_years, *target_years)

    year_filter_clause, year_filter_params = _build_year_filter_clause()

    def _run(
        fts_query: str,
        strategy: str,
        *,
        with_year_filter: bool = False,
        file_year_range: tuple[int, int] | None = None,
    ) -> tuple[str, list[sqlite3.Row]]:
        file_year_clause = ""
        file_year_params: tuple[int, int] | None = None
        if file_year_range is not None:
            file_year_clause = "AND t.file_year BETWEEN ? AND ?"
            file_year_params = file_year_range

        wf_clause = year_filter_clause if with_year_filter else ""

        # When limit_ids is set: run FTS unrestricted (fast — no JOINs on FTS5),
        # fetch a large pool, then Python-filter to limit_ids. Adding any JOIN or
        # subquery to FTS5 forces a full-index scan and makes it 100x slower.
        if limit_ids is not None:
            if not limit_ids:
                return strategy, []
            # Run FTS unrestricted with a large pool, Python-filter to candidates.
            # Any JOIN/subquery on FTS5 forces full-index scan; plain FTS + Python filter is faster.
            params_fts: list[object] = [fts_query]
            pool_size = max(5000, len(limit_ids) * 10)
            sql = (
                " SELECT"  # nosec B608
                " t.id, t.file, t.element_seq, t.element_id, t.page_id,"
                " t.file_year, t.file_month,"
                " t.title, t.section, t.caption, t.unit, t.period,"
                " t.n_rows, t.n_cols, t.signature,"
                " bm25(tables_fts, 2.0, 1.5, 1.5, 2.0, 2.5) AS score"
                " FROM tables_fts"
                " JOIN tables t ON t.id = tables_fts.rowid"
                " WHERE tables_fts MATCH ?"
                " AND t.table_kind = 'data'"
                " ORDER BY score"
                f" LIMIT {pool_size}"
            )
            all_rows = conn.execute(sql, params_fts).fetchall()
            filtered = [r for r in all_rows if r["id"] in limit_ids]
            return strategy, filtered[:top_n]

        params: list[object] = [fts_query]
        if file_year_params is not None:
            params.extend(file_year_params)
        if wf_clause and year_filter_clause:
            params.extend(year_filter_params)

        sql = (
            " SELECT"  # nosec B608
            " t.id, t.file, t.element_seq, t.element_id, t.page_id,"
            " t.file_year, t.file_month,"
            " t.title, t.section, t.caption, t.unit, t.period,"
            " t.n_rows, t.n_cols, t.signature,"
            " bm25(tables_fts, 2.0, 1.5, 1.5, 2.0, 2.5) AS score"
            " FROM tables_fts"
            " JOIN tables t ON t.id = tables_fts.rowid"
            " WHERE tables_fts MATCH ?"
            f" {file_year_clause}"
            f" {wf_clause}"
            " AND t.table_kind = 'data'"
            " ORDER BY score"
            f" LIMIT {top_n}"
        )
        return strategy, list(conn.execute(sql, params))

    synonym_tokens = _expand_synonyms(q_tokens, source_text)
    exact_query = _fts_escape(q_tokens, mode="and")
    synonym_query = _fts_escape(synonym_tokens, mode="or")

    stages: list[tuple[str, Callable[[], list[sqlite3.Row]]]] = []

    # Priority Stage: Year-Strict FTS. Only search tables that
    # definitely contain the target years. This is the "Human Funnel"
    # approach: narrow to 1940-compatible tables first, then search.
    if target_years and exact_query:
        stages.append(
            (
                "exact_year_strict",
                lambda query=exact_query: _run(
                    query,
                    "exact_year_strict",
                    with_year_filter=True,
                )[1],
            )
        )
        if synonym_query and synonym_query != exact_query:
            stages.append(
                (
                    "synonym_year_strict",
                    lambda query=synonym_query: _run(
                        query,
                        "synonym_year_strict",
                        with_year_filter=True,
                    )[1],
                )
            )

    # Secondary Stage: Broad Window (allow later bulletins to cover the years)
    if target_years and exact_query:
        file_year_window = (min(target_years), max(target_years) + 5)
        stages.append(
            (
                "exact_year_window",
                lambda query=exact_query, window=file_year_window: _run(
                    query,
                    "exact_year_window",
                    file_year_range=window,
                )[1],
            )
        )
        stages.append(
            (
                "exact_unrestricted",
                lambda query=exact_query: _run(query, "exact_unrestricted")[1],
            )
        )
        if synonym_query and synonym_query != exact_query:
            stages.append(
                (
                    "synonym_year_window",
                    lambda query=synonym_query, window=file_year_window: _run(
                        query,
                        "synonym_year_window",
                        file_year_range=window,
                    )[1],
                )
            )
            stages.append(
                (
                    "synonym_unrestricted",
                    lambda query=synonym_query: _run(query, "synonym_unrestricted")[1],
                )
            )

        shift_window = (min(target_years) + 5, max(target_years) + 10)
        shifted_query = synonym_query or exact_query
        if shifted_query:
            stages.append(
                (
                    "year_shifted",
                    lambda query=shifted_query, window=shift_window: _run(
                        query,
                        "year_shifted",
                        with_year_filter=True,
                        file_year_range=window,
                    )[1],
                )
            )
    elif exact_query:
        stages.append(
            (
                "exact_unrestricted",
                lambda query=exact_query: _run(query, "exact_unrestricted")[1],
            )
        )
        if synonym_query and synonym_query != exact_query:
            stages.append(
                (
                    "synonym_unrestricted",
                    lambda query=synonym_query: _run(query, "synonym_unrestricted")[1],
                )
            )

    return _collect_progressive_rows(stages, threshold=FTS_SYNONYM_THRESHOLD, top_n=top_n)


# ── Metric channel ──────────────────────────────────────────────────────────


def _metric_channel(
    conn: sqlite3.Connection,
    metric: str,
    row_hint: str,
    col_hint: str,
    target_years: list[int],
    wants_monthly: bool,
    top_n: int = 400,
) -> list[sqlite3.Row]:
    return _metric_channel_trace(
        conn,
        metric,
        row_hint,
        col_hint,
        target_years,
        wants_monthly,
        top_n=top_n,
    ).rows


def _metric_channel_trace(
    conn: sqlite3.Connection,
    metric: str,
    row_hint: str,
    col_hint: str,
    target_years: list[int],
    wants_monthly: bool,
    top_n: int = 1000,
    limit_ids: set[int] | None = None,
) -> ChannelTrace:
    """Arena-style substring lookup over metrics.metric_slug + col_norm.

    Searches for any of the content tokens from metric/row_hint/col_hint as
    substrings in metric_slug. Score = number of matched tokens / total tokens.
    Filters by target year on the metric row (the cell's own year) and, for
    monthly questions, requires month IS NOT NULL. Returns table-level rows
    in the same shape as the FTS channel so the union can merge them.
    """
    # Assemble the query terms — strip stopwords, favor distinctive tokens.
    # When the caller supplied a concrete row_hint (the decompose pipeline),
    # we trust it; otherwise (oracle test mode) we fall back to the metric
    # label only and skip the full question blob, which was poisoning the
    # substring match with too many noisy OR terms.
    source = f"{row_hint} {metric}" if row_hint else metric
    terms = content_tokens(source)
    seen = set()
    uniq_terms = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            uniq_terms.append(t)
    terms = uniq_terms[:10]
    if not terms:
        return ChannelTrace(rows=[], strategy_by_id={}, attempts=[])

    # Guard: the metric channel is designed for clean literal row labels
    # like "national defense" or "individual income taxes, net". It is
    # actively harmful on long or descriptive row_hints (the LONG_ROW_HINT
    # pattern from validate_decompose.py) and on full-sentence blobs from
    # oracle mode. Skip in all those cases and let FTS carry the query.
    if not row_hint and len(terms) > 3:
        return ChannelTrace(rows=[], strategy_by_id={}, attempts=[])
    if row_hint and len(row_hint.split()) > 15:
        return ChannelTrace(rows=[], strategy_by_id={}, attempts=[])
    # "all <X> rows", "excluding...", "e.g....", "aggregates" — these are
    # instructions about which rows to pick, not literal row labels.
    descriptive = re.compile(
        r"\b(excluding|all\s+\w+\s+rows|aggregates?|including|e\.g\.|such\s+as)\b",
        re.I,
    )
    if row_hint and descriptive.search(row_hint):
        return ChannelTrace(rows=[], strategy_by_id={}, attempts=[])

    exact_slug = normalize_metric_slug(f"{row_hint} {metric}".strip())
    col_norm_hint = normalize_metric_slug(col_hint)

    def _run_lookup(
        source_text: str,
        exact_text: str,
        *,
        require_all_terms: bool,
        with_year_filter: bool = False,
    ) -> list[sqlite3.Row]:
        source_terms = content_tokens(source_text)
        seen_terms = set()
        uniq_source_terms = []
        for term in source_terms:
            if term not in seen_terms:
                seen_terms.add(term)
                uniq_source_terms.append(term)
        limited_terms = uniq_source_terms[:10]
        if not limited_terms:
            return []

        join = " AND " if require_all_terms and len(limited_terms) > 1 else " OR "
        like_clauses = join.join(["metric_slug LIKE ?"] * len(limited_terms))
        like_params = [f"%{term}%" for term in limited_terms]

        year_filter_clause = ""
        year_filter_params: list[int] = []
        if with_year_filter and target_years:
            ty_placeholders = ",".join("?" * len(target_years))
            month_restrict = "AND month_extracted IS NOT NULL" if wants_monthly else ""
            year_filter_clause = (
                " AND table_id IN ("
                f" SELECT table_id FROM table_columns"
                f" WHERE year_extracted IN ({ty_placeholders}) {month_restrict}"
                " UNION"
                f" SELECT table_id FROM table_rows"
                f" WHERE year_extracted IN ({ty_placeholders}) {month_restrict}"
                " )"
            )
            year_filter_params = target_years + target_years

        limit_clause = ""

        # When limit_ids is set, scan only rows for those tables using the table_id index,
        # then apply the LIKE filter in Python. This avoids the query planner choosing
        # the LIKE index (millions of rows) over the table_id index (N rows).
        if limit_ids:
            id_ph = ",".join("?" * len(limit_ids))
            sql = (
                " WITH candidate_rows AS ("
                f"  SELECT table_id, metric_slug FROM row_label_lookup WHERE table_id IN ({id_ph})"
                " ),"
                " matched AS ("
                " SELECT table_id, metric_slug,"
                " CASE WHEN metric_slug = ? THEN 1 ELSE 0 END AS is_exact"
                " FROM candidate_rows"
                f" WHERE ({like_clauses})"
                " ),"
                " per_table AS ("
                " SELECT table_id, COUNT(*) AS slug_hits, MAX(is_exact) AS had_exact"
                " FROM matched GROUP BY table_id"
                f" ORDER BY had_exact DESC, slug_hits DESC LIMIT {top_n}"
                " )"
                " SELECT t.id, t.file, t.element_seq, t.element_id, t.page_id,"
                " t.file_year, t.file_month, t.title, t.section, t.caption,"
                " t.unit, t.period, t.n_rows, t.n_cols, t.signature,"
                " -(p.slug_hits + 5.0 * p.had_exact) AS score"
                " FROM per_table p JOIN tables t ON t.id = p.table_id"
                " WHERE t.table_kind = 'data' ORDER BY score"
            )
            return conn.execute(sql, [*limit_ids, exact_text, *like_params]).fetchall()

        rll_filter = ""
        rll_params: list = []

        sql = (
            " WITH matched AS ("  # nosec B608
            " SELECT"
            " rll.table_id,"
            " rll.metric_slug,"
            " CASE WHEN rll.metric_slug = ? THEN 1 ELSE 0 END AS is_exact"
            " FROM row_label_lookup rll"
            f" WHERE ({like_clauses}) {year_filter_clause} {rll_filter}"
            " LIMIT 5000"
            " ),"
            " per_table AS ("
            " SELECT"
            " table_id,"
            " COUNT(*) AS slug_hits,"
            " MAX(is_exact) AS had_exact"
            " FROM matched"
            " GROUP BY table_id"
            " ORDER BY had_exact DESC, slug_hits DESC"
            f" LIMIT {top_n}"
            " )"
            " SELECT"
            " t.id, t.file, t.element_seq, t.element_id, t.page_id,"
            " t.file_year, t.file_month,"
            " t.title, t.section, t.caption, t.unit, t.period,"
            " t.n_rows, t.n_cols, t.signature,"
            " -(p.slug_hits + 5.0 * p.had_exact) AS score"
            " FROM per_table p"
            " JOIN tables t ON t.id = p.table_id"
            " WHERE t.table_kind = 'data'"
            f" {limit_clause}"
            " ORDER BY score"
        )
        return conn.execute(
            sql, [exact_text, *like_params, *year_filter_params, *rll_params]
        ).fetchall()

    row_variants = _row_hint_variants(row_hint)
    synonym_sources: list[str] = []
    seen_variants = {exact_slug}
    for row_variant in row_variants[1:]:
        alt_source = normalize_metric_slug(f"{row_variant} {metric}".strip())
        if not alt_source or alt_source in seen_variants:
            continue
        seen_variants.add(alt_source)
        synonym_sources.append(alt_source)

    stages: list[tuple[str, Callable[[], list[sqlite3.Row]]]] = []
    if exact_slug:
        # Priority: Year-Strict Metric match
        if target_years:
            stages.append(
                (
                    "primary_exact_year",
                    lambda src=source, slug=exact_slug: _run_lookup(
                        src,
                        slug,
                        require_all_terms=True,
                        with_year_filter=True,
                    ),
                )
            )
        stages.append(
            (
                "primary_exact",
                lambda src=source, slug=exact_slug: _run_lookup(
                    src,
                    slug,
                    require_all_terms=True,
                ),
            )
        )

    if synonym_sources:
        if target_years:
            stages.append(
                (
                    "synonym_exact_year",
                    lambda alts=synonym_sources: _merge_ranked_rows(
                        *[
                            _run_lookup(
                                alt_source,
                                alt_source,
                                require_all_terms=True,
                                with_year_filter=True,
                            )
                            for alt_source in alts
                        ],
                        top_n=top_n,
                    ),
                )
            )
        stages.append(
            (
                "synonym_exact",
                lambda alts=synonym_sources: _merge_ranked_rows(
                    *[
                        _run_lookup(alt_source, alt_source, require_all_terms=True)
                        for alt_source in alts
                    ],
                    top_n=top_n,
                ),
            )
        )

    partial_sources = [exact_slug, *synonym_sources]
    if partial_sources:
        if target_years:
            stages.append(
                (
                    "partial_match_year",
                    lambda alts=partial_sources: _merge_ranked_rows(
                        *[
                            _run_lookup(
                                partial_source,
                                partial_source,
                                require_all_terms=False,
                                with_year_filter=True,
                            )
                            for partial_source in alts
                            if partial_source
                        ],
                        top_n=top_n,
                    ),
                )
            )
        stages.append(
            (
                "partial_match",
                lambda alts=partial_sources: _merge_ranked_rows(
                    *[
                        _run_lookup(partial_source, partial_source, require_all_terms=False)
                        for partial_source in alts
                        if partial_source
                    ],
                    top_n=top_n,
                ),
            )
        )

    trace = _collect_progressive_rows(
        stages,
        threshold=METRIC_SYNONYM_THRESHOLD,
        top_n=top_n,
    )
    rows = trace.rows

    # Optional column-hint filter: if the col_hint has distinctive tokens,
    # intersect with col_label_lookup so we don't return tables that have
    # the right row but the wrong column. Keep as a soft prune — drop the
    # filter if it wipes everything.
    if col_norm_hint and rows:
        table_ids = [r["id"] for r in rows]
        ph = ",".join("?" * len(table_ids))
        col_sql = (
            " SELECT DISTINCT table_id FROM col_label_lookup"  # nosec B608
            f" WHERE table_id IN ({ph}) AND col_norm LIKE ?"
        )
        col_hits = conn.execute(col_sql, (*table_ids, f"%{col_norm_hint}%")).fetchall()
        col_ok = {r[0] for r in col_hits}
        if col_ok:
            rows = [r for r in rows if r["id"] in col_ok]

    return ChannelTrace(
        rows=rows,
        strategy_by_id={
            rid: strategy
            for rid, strategy in trace.strategy_by_id.items()
            if rid in {int(r["id"]) for r in rows}
        },
        attempts=trace.attempts,
    )


# ── Prose/footnote supplementary channel ────────────────────────────────────


def _prose_footnote_channel(
    conn: sqlite3.Connection,
    q_tokens: list[str],
    target_years: list[int],
    top_n: int = 50,
) -> list[dict]:
    """Search prose_fts and footnotes_fts for supplementary context.

    Returns hits with file, page_id, element_seq, content, near_table_id
    metadata. Prose/footnote hits are weighted lower than direct table hits
    — they are supplementary context for ~6% of questions that require
    footnote/prose data not captured in the table cells.
    """
    if not q_tokens:
        return []

    # Try AND first (more precise), fall back to OR query
    and_query = _fts_escape(q_tokens, mode="and")
    or_query = _fts_escape(q_tokens, mode="or")
    fts_query = and_query or or_query
    if not fts_query:
        return []

    # Build optional year range filter
    year_clause = ""
    year_params: list[int] = []
    if target_years:
        window_min = min(target_years)
        window_max = max(target_years) + 4
        year_clause = "AND p.file_year BETWEEN ? AND ?"
        year_params = [window_min, window_max]

    fn_year_clause = ""
    fn_year_params: list[int] = []
    if target_years:
        window_min = min(target_years)
        window_max = max(target_years) + 4
        fn_year_clause = "AND f.file_year BETWEEN ? AND ?"
        fn_year_params = [window_min, window_max]

    results: list[dict] = []

    # ── prose_fts channel ──
    try:
        prose_sql = (
            " SELECT p.id, p.file, p.element_seq, p.page_id,"  # nosec B608
            " p.file_year, p.file_month, p.section, p.content,"
            " p.near_table_id,"
            " bm25(prose_fts, 2.0, 1.5, 1.5) AS score"
            " FROM prose_fts"
            " JOIN prose p ON p.id = prose_fts.rowid"
            f" WHERE prose_fts MATCH ? {year_clause}"
            " ORDER BY score"
            f" LIMIT {top_n}"
        )
        rows = conn.execute(prose_sql, [fts_query, *year_params]).fetchall()
        for r in rows:
            results.append(
                {
                    "file": r["file"],
                    "element_seq": int(r["element_seq"]),
                    "page_id": int(r["page_id"]) if r["page_id"] is not None else None,
                    "file_year": r["file_year"],
                    "file_month": r["file_month"],
                    "section": r["section"] or "",
                    "content": r["content"] or "",
                    "near_table_id": r["near_table_id"],
                    "score": float(r["score"]),
                    "source": "prose",
                }
            )
    except sqlite3.OperationalError:
        pass

    # ── footnotes_fts channel ──
    try:
        fn_sql = (
            " SELECT f.id, f.file, f.element_seq, f.page_id,"  # nosec B608
            " f.file_year, f.file_month, f.content,"
            " f.attached_to_table_id AS near_table_id,"
            " bm25(footnotes_fts, 2.0) AS score"
            " FROM footnotes_fts"
            " JOIN footnotes f ON f.id = footnotes_fts.rowid"
            f" WHERE footnotes_fts MATCH ? {fn_year_clause}"
            " ORDER BY score"
            f" LIMIT {top_n}"
        )
        fn_rows = conn.execute(fn_sql, [fts_query, *fn_year_params]).fetchall()
        for r in fn_rows:
            results.append(
                {
                    "file": r["file"],
                    "element_seq": int(r["element_seq"]),
                    "page_id": int(r["page_id"]) if r["page_id"] is not None else None,
                    "file_year": r["file_year"],
                    "file_month": r["file_month"],
                    "section": "",
                    "content": r["content"] or "",
                    "near_table_id": r["near_table_id"],
                    "score": float(r["score"]),
                    "source": "footnote",
                }
            )
    except sqlite3.OperationalError:
        pass

    # Sort ascending (lower bm25 score = better match)
    results.sort(key=lambda x: x["score"])
    return results[:top_n]


# ── Top-level retrieve (two-channel union) ──────────────────────────────────


class _RetrievalContext:
    def __init__(self, plan: dict | None, question: str):
        self.plan = plan
        self.question = question
        self.hints = _extract_hints(plan, question)
        self.year_mode = detect_year_mode(question)
        self.wants_monthly = _detect_wants_monthly(question)
        self.direct_refs = parse_direct_bulletin_refs(question)
        self.direct_files = {f"treasury_bulletin_{yr}_{mo:02d}.json" for yr, mo in self.direct_refs}

        self.all_metrics: list[str] = []
        self.all_row_hints: list[str] = []
        self.all_col_hints: list[str] = []
        self.all_target_years: set[int] = set()
        self.corpus_row_hints: list[str] = []
        self.requested_granularities: set[str] = set()

        for h in self.hints:
            if h["metric"]:
                self.all_metrics.append(h["metric"])
            if h["row_hint"]:
                self.all_row_hints.append(h["row_hint"])
            if h["col_hint"]:
                self.all_col_hints.append(h["col_hint"])
            self.all_target_years.update(h["years"])

            if h.get("source") == "corpus" and h["row_hint"]:
                self.corpus_row_hints.append(h["row_hint"])
                for alt in h.get("row_hint_alternatives") or []:
                    alt = str(alt).strip()
                    if alt:
                        self.corpus_row_hints.append(alt)
            if h.get("source") == "corpus" and h.get("granularity"):
                self.requested_granularities.add(h["granularity"])

        self.target_years = sorted(self.all_target_years)
        self.query_text = f"{question} {' '.join(self.all_metrics)} {' '.join(self.all_row_hints)} {' '.join(self.all_col_hints)}".strip()
        self.q_tokens = content_tokens(self.query_text)

        self.title_patterns: list[str] = []
        for phrase in self.all_row_hints + self.all_col_hints + self.all_metrics:
            p = (phrase or "").strip().lower()
            if p and len(p) >= 3 and p not in self.title_patterns:
                self.title_patterns.append(p)


def _find_candidate_families(conn: sqlite3.Connection, ctx: _RetrievalContext) -> set[str]:
    # Narrow by year first (aggressive gate)
    if not ctx.target_years:
        rows = conn.execute("SELECT family_id FROM table_families").fetchall()
        return {r[0] for r in rows if r[0]}

    ph = ",".join("?" * len(ctx.target_years))
    year_rows = conn.execute(
        f"SELECT DISTINCT family_id FROM table_family_years WHERE year IN ({ph})", ctx.target_years
    ).fetchall()
    candidate_fids = {r[0] for r in year_rows if r[0]}

    # Narrow by column tokens if we have many families left
    if len(candidate_fids) > 100 and ctx.all_col_hints:
        col_tokens = []
        for hint in ctx.all_col_hints:
            col_tokens.extend(re.findall(r"[a-z0-9]{3,}", hint.lower()))

        if col_tokens:
            col_tokens = list(set(col_tokens))[:3]  # Very selective
            fid_list = list(candidate_fids)
            ph_fid = ",".join("?" * len(fid_list))
            like_clauses = " OR ".join(["col_leaf_lower LIKE ?" for _ in col_tokens])
            like_params = [f"%{t}%" for t in col_tokens]

            sql = f"SELECT DISTINCT family_id FROM table_family_columns WHERE family_id IN ({ph_fid}) AND ({like_clauses})"
            col_rows = conn.execute(sql, (*fid_list, *like_params)).fetchall()
            col_fids = {r[0] for r in col_rows if r[0]}
            if col_fids:
                candidate_fids &= col_fids

    return candidate_fids


def _get_best_instances_for_families(
    conn: sqlite3.Connection, families: set[str], target_years: list[int]
) -> set[int]:
    if not families:
        return set()

    fid_ph = ",".join("?" * len(families))
    fid_list = list(families)

    if target_years:
        # Per (family, year): pick the latest bulletin instance that actually contains
        # that year in its columns/rows. Precomputed as best_table_id in table_family_years.
        # Falls back to latest_table_id if best_table_id column not yet built.
        has_best = conn.execute(
            "SELECT COUNT(*) FROM pragma_table_info('table_family_years') WHERE name='best_table_id'"
        ).fetchone()[0]
        if has_best:
            year_ph = ",".join("?" * len(target_years))
            rows = conn.execute(
                f"""SELECT DISTINCT best_table_id FROM table_family_years
                    WHERE family_id IN ({fid_ph}) AND year IN ({year_ph})
                    AND best_table_id IS NOT NULL""",
                (*fid_list, *target_years),
            ).fetchall()
        else:
            rows = conn.execute(
                f"SELECT latest_table_id FROM table_families WHERE family_id IN ({fid_ph}) AND latest_table_id IS NOT NULL",
                fid_list,
            ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT latest_table_id FROM table_families WHERE family_id IN ({fid_ph}) AND latest_table_id IS NOT NULL",
            fid_list,
        ).fetchall()
    return {int(r[0]) for r in rows}


def _run_search_channels(conn: sqlite3.Connection, ctx: _RetrievalContext, limit_ids: set[int]):
    top_n = 200
    # Load limit_ids into a temp table once so FTS and metric channels can join
    # against it efficiently instead of using IN (N ids) which is slow on FTS5.
    conn.execute("CREATE TEMP TABLE IF NOT EXISTS _fts_limit_ids (id INTEGER PRIMARY KEY)")
    conn.execute("DELETE FROM _fts_limit_ids")
    if limit_ids:
        conn.executemany(
            "INSERT OR IGNORE INTO _fts_limit_ids VALUES (?)", [(i,) for i in limit_ids]
        )

    fts_trace = _fts_channel_trace(
        conn,
        ctx.q_tokens,
        ctx.target_years,
        ctx.wants_monthly,
        source_text=ctx.query_text,
        year_mode=ctx.year_mode,
        limit_ids=limit_ids,
        top_n=top_n,
    )

    all_metric_rows: list[sqlite3.Row] = []
    all_metric_strategy_by_id: dict[int, str] = {}
    seen_ids = set()

    for h in ctx.hints:
        trace = _metric_channel_trace(
            conn,
            h["metric"],
            h["row_hint"],
            h["col_hint"],
            h["years"],
            ctx.wants_monthly,
            limit_ids=limit_ids,
            top_n=top_n,
        )
        for r in trace.rows:
            rid = int(r["id"])
            if rid not in seen_ids:
                seen_ids.add(rid)
                all_metric_rows.append(r)
                if rid in trace.strategy_by_id:
                    all_metric_strategy_by_id[rid] = trace.strategy_by_id[rid]

    class _MergedTrace:
        rows = all_metric_rows
        strategy_by_id = all_metric_strategy_by_id
        attempts = []

    pf_hits = _prose_footnote_channel(conn, ctx.q_tokens, ctx.target_years, top_n=50)
    return fts_trace, _MergedTrace(), pf_hits


def _rank_candidates(
    conn: sqlite3.Connection,
    ctx: _RetrievalContext,
    fts_trace,
    metric_trace,
    limit_ids: set[int],
    top_k: int,
):
    def _normalize(rows) -> dict[int, float]:
        if not rows:
            return {}
        raw = [-r["score"] for r in rows]
        smax = max(raw) or 1.0
        return {r["id"]: (rr / smax) for r, rr in zip(rows, raw, strict=True)}

    fts_norm = _normalize(fts_trace.rows)
    metric_norm = _normalize(metric_trace.rows)

    rows_by_id = {r["id"]: r for r in fts_trace.rows}
    for r in metric_trace.rows:
        rows_by_id.setdefault(r["id"], r)

    all_ids = set(fts_norm) | set(metric_norm)
    if not all_ids:
        return []

    compat = _candidate_compatibility(
        conn,
        all_ids,
        ctx.corpus_row_hints,
        ctx.all_col_hints,
        ctx.target_years,
        ctx.requested_granularities,
    )
    all_ids = _apply_year_compatibility_filter(all_ids, compat, ctx.target_years, top_k)
    all_ids = _apply_structure_compatibility_filter(
        all_ids, compat, ctx.requested_granularities, ctx.all_col_hints, top_k
    )

    probe_stats = {
        tid: (state.row_match_count, state.best_row_non_null_cells)
        for tid, state in compat.items()
        if tid in all_ids
    }
    max_best_row_cells = max((v[1] for v in probe_stats.values()), default=0)

    superseded_ids = set()
    try:
        rows = conn.execute("SELECT table_id FROM supersessions").fetchall()
        superseded_ids = {int(r[0]) for r in rows}
    except Exception:
        pass

    final_ranked = []
    for tid in all_ids:
        row = rows_by_id[tid]
        state = compat.get(tid, CandidateCompatibility())
        match_rows, best_cells = probe_stats.get(tid, (0, 0))

        score = 0.75 * fts_norm.get(tid, 0.0) + 0.25 * metric_norm.get(tid, 0.0)
        score += _strategy_score_nudge(
            fts_trace.strategy_by_id.get(tid), metric_trace.strategy_by_id.get(tid)
        )

        if max_best_row_cells > 0 and match_rows > 0:
            score += 5.0 * (best_cells / max_best_row_cells)
            if match_rows > 1:
                score -= 0.05 * min(match_rows - 1, 8)

        if ctx.target_years and state.matching_target_years_count > 0:
            score += 2.0 * (state.matching_target_years_count / len(ctx.target_years))

        if state.has_col_hint_match:
            score += 0.45 + 0.10 * min(state.col_match_count, 3)
        if ctx.wants_monthly and state.has_month_data:
            score += 0.35
        elif ctx.wants_monthly and ctx.requested_granularities and not state.has_month_data:
            score -= 0.25
        if ctx.requested_granularities and state.has_required_granularity:
            score += 0.20

        title_text = f"{row['title'] or ''} {row['caption'] or ''}".lower()
        if title_text and ctx.title_patterns:
            hits = sum(1 for p in ctx.title_patterns if p in title_text)
            if hits:
                score += 0.25 * min(hits, 3)

        score += 0.12 * _file_year_bonus(
            row["file_year"], row["file_month"], ctx.target_years, ctx.year_mode
        )
        score += 0.2 * _period_aware_score_delta(ctx.year_mode, row["period"])
        if tid in superseded_ids:
            score -= 0.30
        if ctx.direct_files and row["file"] in ctx.direct_files:
            score += 5.0

        final_ranked.append(
            (
                score,
                row,
                state,
                (match_rows, best_cells),
                fts_norm.get(tid, 0.0),
                metric_norm.get(tid, 0.0),
                fts_trace,
                metric_trace,
            )
        )

    return sorted(final_ranked, key=lambda x: -x[0])


def _build_entries(conn, ctx, ranked, pf_hits, top_k, dedupe_by_file, max_per_file, load_html):
    file_cap = max_per_file if max_per_file is not None else 1 if dedupe_by_file else None
    entries = []
    file_counts = {}

    for _score, r, state, probe_res, fts_norm_v, metric_norm_v, fts_trace, metric_trace in ranked:
        row_id, file = int(r["id"]), r["file"]
        if file_cap is not None and file_counts.get(file, 0) >= file_cap:
            continue
        file_counts[file] = file_counts.get(file, 0) + 1

        cols = conn.execute(
            "SELECT col_path FROM table_columns WHERE table_id = ? ORDER BY col_index", (row_id,)
        ).fetchall()
        labels = conn.execute(
            "SELECT row_path FROM table_rows WHERE table_id = ? ORDER BY row_index LIMIT 80",
            (row_id,),
        ).fetchall()
        years = sorted(
            [
                int(y[0])
                for y in conn.execute(
                    """
            SELECT DISTINCT year FROM (
                SELECT year_extracted AS year FROM table_columns WHERE table_id = ?
                UNION
                SELECT year_extracted AS year FROM table_rows    WHERE table_id = ?
            ) WHERE year IS NOT NULL""",
                    (row_id, row_id),
                ).fetchall()
            ]
        )

        strategy = []
        if row_id in fts_trace.strategy_by_id:
            strategy.append(f"fts:{fts_trace.strategy_by_id[row_id]}")
        if row_id in metric_trace.strategy_by_id:
            strategy.append(f"metric:{metric_trace.strategy_by_id[row_id]}")

        entries.append(
            {
                "probe_matched_rows": probe_res[0],
                "probe_best_cells": probe_res[1],
                "probe_col_matches": state.col_match_count,
                "has_target_year": state.has_target_year,
                "has_month_data": state.has_month_data,
                "has_required_granularity": state.has_required_granularity,
                "file": file,
                "element_id": r["element_id"],
                "element_seq": r["element_seq"],
                "page_id": r["page_id"],
                "signature": r["signature"],
                "file_year": r["file_year"],
                "file_month": r["file_month"],
                "section": r["section"] or "",
                "title": r["title"] or "",
                "caption": r["caption"] or "",
                "near_content": [
                    p[0]
                    for p in conn.execute(
                        "SELECT content FROM prose WHERE file = ? AND page_id = ?",
                        (file, r["page_id"]),
                    ).fetchall()
                ]
                + [
                    f[0]
                    for f in conn.execute(
                        "SELECT content FROM footnotes WHERE file = ? AND page_id = ?",
                        (file, r["page_id"]),
                    ).fetchall()
                ],
                "column_headers": [c[0] or "" for c in cols],
                "row_labels": [lab[0] or "" for lab in labels],
                "years": years,
                "unit": r["unit"],
                "period": r["period"],
                "n_rows": r["n_rows"],
                "n_cols": r["n_cols"],
                "retrieval_strategy": " | ".join(strategy),
                "retrieval_channel": "fts" if fts_norm_v >= metric_norm_v else "metric",
                "html": _load_element_html(file, r["element_seq"]) if load_html else "",
            }
        )
        if len(entries) >= top_k:
            break

    if pf_hits:
        seen = {(e["file"], e["element_seq"]) for e in entries}
        added = 0
        for hit in sorted(pf_hits, key=lambda x: x["score"]):
            if added >= 3:
                break
            if (hit["file"], hit["element_seq"]) in seen:
                continue
            if (
                ctx.target_years
                and _file_year_bonus(
                    hit.get("file_year"), hit.get("file_month"), ctx.target_years, ctx.year_mode
                )
                == 0
            ):
                continue
            seen.add((hit["file"], hit["element_seq"]))
            entries.append(
                {
                    "file": hit["file"],
                    "element_id": None,
                    "element_seq": hit["element_seq"],
                    "page_id": hit["page_id"],
                    "file_year": hit.get("file_year"),
                    "file_month": hit.get("file_month"),
                    "section": hit.get("section") or "",
                    "title": "",
                    "caption": "",
                    "column_headers": [],
                    "row_labels": [],
                    "years": [],
                    "unit": "",
                    "period": "",
                    "n_rows": 0,
                    "n_cols": 0,
                    "retrieval_strategy": f"prose_footnote:{hit['source']}",
                    "retrieval_channel": "prose_footnote",
                    "html": "",
                    "content": hit["content"],
                    "near_table_id": hit.get("near_table_id"),
                }
            )
            added += 1
    return entries


def retrieve(
    plan: dict | None,
    question: str,
    top_k: int = 10,
    verbose: bool = False,
    dedupe_by_file: bool = True,
    load_html: bool = True,
    vintage: str = "latest",
    max_per_file: int | None = None,
    debug_meta: dict | None = None,
) -> list[dict]:
    conn = _ledger_conn()
    ctx = _RetrievalContext(plan, question)

    fids = _find_candidate_families(conn, ctx)
    if verbose:
        print(f"  family funnel: {len(fids)} families", flush=True)

    limit_ids = _get_best_instances_for_families(conn, fids, ctx.target_years)
    if verbose:
        print(f"  narrowed to {len(limit_ids)} best instances", flush=True)

    if not limit_ids:
        return []

    fts_trace, metric_trace, pf_hits = _run_search_channels(conn, ctx, limit_ids)
    ranked = _rank_candidates(conn, ctx, fts_trace, metric_trace, limit_ids, top_k)

    return _build_entries(
        conn, ctx, ranked, pf_hits, top_k, dedupe_by_file, max_per_file, load_html
    )


def retrieve_from_question(question: str, top_k: int = 10) -> list[dict]:
    """Oracle-mode retrieval: question itself is the metric hint.

    Used by scout.py and the intrinsic recall test to measure retrieval in
    isolation, bypassing decompose."""
    plan = {
        "data_requests": [
            {
                "label": question,
                "row_hint": "",
                "column_hint": "",
                "years": parse_years_from_question(question),
            }
        ],
    }
    return retrieve(plan, question, top_k=top_k)


# Back-compat: the old ledger-only entrypoint. Now identical to retrieve().
def retrieve_from_ledger(
    question: str,
    top_k: int = 10,
    year_window: int = 4,
    dedupe_by_file: bool = True,
    load_html: bool = True,
) -> list[dict]:
    plan = {
        "data_requests": [
            {
                "label": question,
                "row_hint": "",
                "column_hint": "",
                "years": parse_years_from_question(question),
            }
        ],
    }
    return retrieve(
        plan,
        question,
        top_k=top_k,
        dedupe_by_file=dedupe_by_file,
        load_html=load_html,
    )


# ── Intrinsic recall test (table-level, ledger-backed) ──────────────────────


def _winning_hit_details(
    results: list[dict],
    gold_locs: set[tuple[str, int]],
) -> tuple[int | None, str]:
    for rank, entry in enumerate(results, 1):
        if (entry["file"], entry["page_id"]) in gold_locs:
            strategy = (
                entry.get("retrieval_strategy") or entry.get("retrieval_channel") or "unknown"
            )
            return rank, strategy
    return None, "none"


def _test_recall_ledger(n: int = 0) -> None:
    """Table-level recall using the benchmark's (file, page) gold anchors."""
    import csv

    url_page_re = re.compile(r"[?&]page=(\d+)")

    with open("officeqa_full.csv") as f:
        rows = list(csv.DictReader(f))
    if n:
        rows = rows[:n]

    print(f"Testing ledger retriever (table-level) on {len(rows)} questions...\n", flush=True)

    hits_at_1 = hits_at_5 = hits_at_10 = hits_at_20 = hits_at_30 = total = 0
    file_hits_at_10 = 0
    pf_contrib = 0  # questions where prose/footnote entries appeared in results
    per_q_times: list[float] = []
    t0 = time.time()

    for i, row in enumerate(rows, 1):
        doc_lines = [
            line.strip() for line in (row["source_docs"] or "").split("\n") if line.strip()
        ]
        file_lines = [
            line.strip() for line in (row["source_files"] or "").split("\n") if line.strip()
        ]
        gold_locs: set[tuple[str, int]] = set()
        for url, fname in zip(doc_lines, file_lines, strict=False):
            m = url_page_re.search(url)
            if not m:
                continue
            page = int(m.group(1))
            stem = Path(fname).stem
            gold_locs.add((f"{stem}.json", page))
        gold_files = {f for f, _ in gold_locs}
        if not gold_locs:
            continue

        qt = time.time()
        results = retrieve_from_question(row["question"], top_k=30)
        per_q_times.append(time.time() - qt)

        retrieved_locs = [(e["file"], e["page_id"]) for e in results]
        winning_rank, winning_strategy = _winning_hit_details(results, gold_locs)

        top1 = any(loc in gold_locs for loc in retrieved_locs[:1])
        top5 = any(loc in gold_locs for loc in retrieved_locs[:5])
        top10 = any(loc in gold_locs for loc in retrieved_locs[:10])
        top20 = any(loc in gold_locs for loc in retrieved_locs[:20])
        top30 = any(loc in gold_locs for loc in retrieved_locs[:30])

        total += 1
        if top1:
            hits_at_1 += 1
        if top5:
            hits_at_5 += 1
        if top10:
            hits_at_10 += 1
        if top20:
            hits_at_20 += 1
        if top30:
            hits_at_30 += 1
        if any(e["file"] in gold_files for e in results[:10]):
            file_hits_at_10 += 1
        if any(e.get("retrieval_channel") == "prose_footnote" for e in results):
            pf_contrib += 1

        mark = (
            "HIT@1"
            if top1
            else (
                "HIT@5"
                if top5
                else (
                    "HIT@10" if top10 else ("HIT@20" if top20 else ("HIT@30" if top30 else "MISS "))
                )
            )
        )
        dt_ms = per_q_times[-1] * 1000
        strategy_msg = (
            f" rank={winning_rank:<2d} via={winning_strategy}"
            if winning_rank is not None
            else " rank=-- via=none"
        )
        running = (
            f"@1={hits_at_1}/{total} @5={hits_at_5}/{total} "
            f"@10={hits_at_10}/{total} @30={hits_at_30}/{total}"
        )
        print(
            f"  [{i:3d}/{len(rows)}] {mark} {dt_ms:5.0f}ms  {strategy_msg}  "
            f"{running}  uid={row.get('uid', '?')}",
            flush=True,
        )

    elapsed = time.time() - t0
    print(f"\n=== Ledger retriever — TABLE-level recall ({total} q, {elapsed:.1f}s) ===")
    print(f"  recall@1  = {hits_at_1}/{total} = {hits_at_1 / total * 100:.1f}%")
    print(f"  recall@5  = {hits_at_5}/{total} = {hits_at_5 / total * 100:.1f}%")
    print(f"  recall@10 = {hits_at_10}/{total} = {hits_at_10 / total * 100:.1f}%")
    print(f"  recall@20 = {hits_at_20}/{total} = {hits_at_20 / total * 100:.1f}%")
    print(f"  recall@30 = {hits_at_30}/{total} = {hits_at_30 / total * 100:.1f}%")
    print(
        f"  (ref) file-level recall@10 = {file_hits_at_10}/{total} "
        f"= {file_hits_at_10 / total * 100:.1f}%"
    )
    print(
        f"  prose/footnote contributions = {pf_contrib}/{total} "
        f"= {pf_contrib / total * 100:.1f}% of questions"
    )
    if per_q_times:
        per_q_times.sort()
        p50 = per_q_times[len(per_q_times) // 2]
        p95 = per_q_times[int(len(per_q_times) * 0.95)]
        print(f"  latency: p50={p50 * 1000:.0f}ms  p95={p95 * 1000:.0f}ms")


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--test-ledger" in args or "--test" in args:
        flag = "--test-ledger" if "--test-ledger" in args else "--test"
        n = 0
        for i, a in enumerate(args):
            if a == flag and i + 1 < len(args):
                with suppress(ValueError):
                    n = int(args[i + 1])
        _test_recall_ledger(n)
    else:
        print("Usage: uv run python retrieve_v2.py --test [N] | --test-ledger [N]")
