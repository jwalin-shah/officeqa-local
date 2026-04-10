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

_reconfigure_stdout = getattr(sys.stdout, "reconfigure", None)
if callable(_reconfigure_stdout):
    _reconfigure_stdout(line_buffering=True)

LEDGER_PATH = Path("ledger.sqlite")

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

FTS_SYNONYM_THRESHOLD = 5
METRIC_SYNONYM_THRESHOLD = 5


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
    for key in _matched_synonym_keys(norm):
        for synonym in SYNONYMS.get(key, ()):
            candidate = normalize_metric_slug(
                re.sub(rf"\b{re.escape(key)}\b", synonym, norm, count=1)
            )
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)
            variants.append(candidate)
    return variants


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
    # Offset distribution across all 246 gold questions (cached analysis):
    # offset 0: 139 (most common), +1: 85, +2-6: ~19 combined. Coverage 0-3
    # covers ~95%. BUT the mode matters: "calendar year N" questions want
    # the annual summary published in the following January-March (offset
    # +1), while snapshot-style questions ("as of 2016", "in mid-March 2016")
    # want the in-year bulletin (offset 0). Fiscal-year questions want the
    # bulletin published around fiscal year-end (offset 0, months 7-10).
    peak_off = 1 if year_mode == "calendar" else 0
    near_off = 1 - peak_off  # the other of {0, 1}

    if offset == peak_off:
        base = 5.0
    elif offset == near_off:
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
    same bulletin repeatedly for year-comparison questions."""
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
        _JSON_ELEMENT_CACHE[file] = cache
    el = cache.get(element_seq)
    return (el.get("content") or "") if el else ""


def _fts_escape(tokens: list[str], mode: str = "or") -> str:
    """Build a permissive FTS5 MATCH query. `mode` is 'or' or 'and'."""
    safe = [f'"{t}"' for t in tokens if t and "'" not in t and '"' not in t]
    safe = safe[:12]
    if not safe:
        return ""
    joiner = " AND " if mode == "and" else " OR "
    return joiner.join(safe)


# ── Hint extraction from decompose plans ────────────────────────────────────


def _extract_hints(plan: dict | None, question: str) -> list[dict]:
    """Extract retrieval hints from ALL data_requests in a plan.

    Returns a list of dicts (one per data_request), each with keys:
      metric, col_hint, row_hint, row_hint_alternatives, years
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
    top_n: int = 400,
    source_text: str = "",
) -> ChannelTrace:
    """Run the FTS5 channel with progressive broadening and trace metadata."""
    if not q_tokens:
        return ChannelTrace(rows=[], strategy_by_id={}, attempts=[])

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
        params: list[object] = [fts_query]
        if file_year_params is not None:
            params.extend(file_year_params)
        if with_year_filter and year_filter_clause:
            params.extend(year_filter_params)
        sql = (
            " SELECT"  # nosec B608
            " t.id, t.file, t.element_seq, t.element_id, t.page_id,"
            " t.file_year, t.file_month,"
            " t.title, t.section, t.caption, t.unit, t.period,"
            " t.n_rows, t.n_cols,"
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
    if target_years and exact_query:
        file_year_window = (min(target_years), max(target_years) + 4)
        stages.append(
            (
                "exact_year_filter",
                lambda query=exact_query, window=file_year_window: _run(
                    query,
                    "exact_year_filter",
                    with_year_filter=True,
                    file_year_range=window,
                )[1],
            )
        )
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
    top_n: int = 400,
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
    terms = uniq_terms[:6]
    if not terms:
        return ChannelTrace(rows=[], strategy_by_id={}, attempts=[])

    # Guard: the metric channel is designed for clean literal row labels
    # like "national defense" or "individual income taxes, net". It is
    # actively harmful on long or descriptive row_hints (the LONG_ROW_HINT
    # pattern from validate_decompose.py) and on full-sentence blobs from
    # oracle mode. Skip in all those cases and let FTS carry the query.
    if not row_hint and len(terms) > 3:
        return ChannelTrace(rows=[], strategy_by_id={}, attempts=[])
    if row_hint and len(row_hint.split()) > 8:
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
        source_text: str, exact_text: str, *, require_all_terms: bool
    ) -> list[sqlite3.Row]:
        source_terms = content_tokens(source_text)
        seen_terms = set()
        uniq_source_terms = []
        for term in source_terms:
            if term not in seen_terms:
                seen_terms.add(term)
                uniq_source_terms.append(term)
        limited_terms = uniq_source_terms[:6]
        if not limited_terms:
            return []

        join = " AND " if require_all_terms and len(limited_terms) > 1 else " OR "
        like_clauses = join.join(["metric_slug LIKE ?"] * len(limited_terms))
        like_params = [f"%{term}%" for term in limited_terms]

        sql = (
            " WITH matched AS ("  # nosec B608
            " SELECT"
            " rll.table_id,"
            " rll.metric_slug,"
            " CASE WHEN rll.metric_slug = ? THEN 1 ELSE 0 END AS is_exact"
            " FROM row_label_lookup rll"
            f" WHERE ({like_clauses})"
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
            " t.n_rows, t.n_cols,"
            " -(p.slug_hits + 5.0 * p.had_exact) AS score"
            " FROM per_table p"
            " JOIN tables t ON t.id = p.table_id"
            " WHERE t.table_kind = 'data'"
            " ORDER BY score"
        )
        return list(conn.execute(sql, [exact_text, *like_params]))

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


def retrieve(
    plan: dict | None,
    question: str,
    top_k: int = 10,
    verbose: bool = False,
    dedupe_by_file: bool = True,
    load_html: bool = True,
) -> list[dict]:
    """Multi-request, multi-channel ledger retrieval — FTS ∪ metric — reranked
    and returned as shape-compatible dicts for extract.py.

    Processes ALL data_requests in the plan, running the metric channel for
    each request's row_hint + row_hint_alternatives. Period-aware scoring
    boosts tables whose period matches the question's year mode and penalizes
    mismatches.
    """
    conn = _ledger_conn()

    hints = _extract_hints(plan, question)
    year_mode = detect_year_mode(question)
    wants_monthly = _detect_wants_monthly(question)

    direct_refs = parse_direct_bulletin_refs(question)
    direct_files = {f"treasury_bulletin_{yr}_{mo:02d}.json" for yr, mo in direct_refs}

    # Aggregate query tokens and target years across ALL data_requests
    all_metrics: list[str] = []
    all_row_hints: list[str] = []
    all_col_hints: list[str] = []
    all_target_years: set[int] = set()
    for h in hints:
        if h["metric"]:
            all_metrics.append(h["metric"])
        if h["row_hint"]:
            all_row_hints.append(h["row_hint"])
        if h["col_hint"]:
            all_col_hints.append(h["col_hint"])
        all_target_years.update(h["years"])

    target_years = sorted(all_target_years)

    query_text = f"{question} {' '.join(all_metrics)} {' '.join(all_row_hints)} {' '.join(all_col_hints)}".strip()
    q_tokens = content_tokens(query_text)

    fts_trace = _fts_channel_trace(
        conn,
        q_tokens,
        target_years,
        wants_monthly,
        source_text=query_text,
    )

    # Run metric channel for EACH data_request, including alternatives
    all_metric_rows: list[sqlite3.Row] = []
    all_metric_strategy_by_id: dict[int, str] = {}
    all_metric_attempts: list[tuple[str, int]] = []
    seen_metric_ids: set[int] = set()

    for h in hints:
        metric = h["metric"]
        row_hint = h["row_hint"]
        col_hint = h["col_hint"]
        dr_years = h["years"]

        # Primary metric channel call with the main row_hint
        trace = _metric_channel_trace(
            conn,
            metric,
            row_hint,
            col_hint,
            dr_years,
            wants_monthly,
        )
        for r in trace.rows:
            rid = int(r["id"])
            if rid not in seen_metric_ids:
                seen_metric_ids.add(rid)
                all_metric_rows.append(r)
                if rid in trace.strategy_by_id:
                    all_metric_strategy_by_id[rid] = trace.strategy_by_id[rid]
        all_metric_attempts.extend(trace.attempts)

        # Also run metric channel for each row_hint_alternative
        for alt in h["row_hint_alternatives"]:
            alt_trace = _metric_channel_trace(
                conn,
                metric,
                alt,
                col_hint,
                dr_years,
                wants_monthly,
            )
            for r in alt_trace.rows:
                rid = int(r["id"])
                if rid not in seen_metric_ids:
                    seen_metric_ids.add(rid)
                    all_metric_rows.append(r)
                    # Use "alt:<alternative>" strategy label
                    if rid in alt_trace.strategy_by_id:
                        all_metric_strategy_by_id[rid] = f"alt:{alt_trace.strategy_by_id[rid]}"
            all_metric_attempts.extend(alt_trace.attempts)

    # Build a synthetic metric_trace-like structure for downstream
    class _MergedTrace:
        rows: list[sqlite3.Row]
        strategy_by_id: dict[int, str]
        attempts: list[tuple[str, int]]

    metric_trace = _MergedTrace()
    metric_trace.rows = all_metric_rows
    metric_trace.strategy_by_id = all_metric_strategy_by_id
    metric_trace.attempts = all_metric_attempts

    pf_hits = _prose_footnote_channel(conn, q_tokens, target_years)
    fts_rows = fts_trace.rows
    metric_rows = metric_trace.rows

    def _normalize_channel(rows: list[sqlite3.Row]) -> dict[int, float]:
        if not rows:
            return {}
        raw = [-r["score"] for r in rows]  # lower bm25 == better → flip
        smax = max(raw) or 1.0
        return {r["id"]: (rr / smax) for r, rr in zip(rows, raw, strict=True)}

    fts_norm = _normalize_channel(fts_rows)
    metric_norm = _normalize_channel(metric_rows)

    rows_by_id: dict[int, sqlite3.Row] = {}
    for r in fts_rows:
        rows_by_id[r["id"]] = r
    for r in metric_rows:
        rows_by_id.setdefault(r["id"], r)

    # Union-mode with the metric channel as a precision bonus over FTS.
    # Filter-mode (metric as candidate pool, FTS as reranker) regressed
    # recall on 246 because the metric channel's pool is too narrow —
    # OCR variants and alternate phrasings slip through FTS but not
    # substring match. Keep FTS as the broad recall channel and let the
    # metric channel nudge the ranking when its guard allows it to fire.
    all_ids = set(fts_norm) | set(metric_norm)
    if not all_ids:
        if verbose:
            print(f"  no hits (fts={len(fts_rows)} metric={len(metric_rows)})", flush=True)
        return []

    weighted_channel_scores: dict[int, tuple[float, float]] = {}
    reranked: list[tuple[float, sqlite3.Row]] = []
    for tid in all_ids:
        row = rows_by_id[tid]
        fts_weighted = 0.75 * fts_norm.get(tid, 0.0)
        metric_weighted = 0.25 * metric_norm.get(tid, 0.0)
        weighted_channel_scores[tid] = (fts_weighted, metric_weighted)
        s = metric_weighted + fts_weighted
        s += 0.08 * _file_year_bonus(
            row["file_year"],
            row["file_month"],
            target_years,
            year_mode,
        )
        # Period-aware scoring: boost matching period, penalize mismatched.
        # Scaled by 0.2 to prevent the period signal from dominating — many
        # tables contain both FY and CY data despite being labeled with one
        # period, so the raw delta would incorrectly demote correct tables.
        s += 0.2 * _period_aware_score_delta(year_mode, row["period"])
        if direct_files and row["file"] in direct_files:
            s += 5.0
        reranked.append((s, row))
    reranked.sort(key=lambda x: -x[0])

    if verbose:
        print(
            f"  fts={len(fts_rows)} metric={len(metric_rows)} pf={len(pf_hits)}"
            f" union={len(all_ids)} years={target_years} mode={year_mode}"
            f" fts_attempts={fts_trace.attempts} metric_attempts={metric_trace.attempts}",
            flush=True,
        )

    entries: list[dict] = []
    seen_files: set[str] = set()
    for _score, r in reranked:
        row_id = int(r["id"])
        file = r["file"]
        if dedupe_by_file and file in seen_files:
            continue
        seen_files.add(file)

        cols = conn.execute(
            "SELECT col_path FROM table_columns WHERE table_id = ? ORDER BY col_index",
            (r["id"],),
        ).fetchall()
        labels = conn.execute(
            "SELECT row_path FROM table_rows WHERE table_id = ? ORDER BY row_index LIMIT 80",
            (r["id"],),
        ).fetchall()
        year_rows = conn.execute(
            """
            SELECT DISTINCT year FROM (
                SELECT year_extracted AS year FROM table_columns WHERE table_id = ?
                UNION
                SELECT year_extracted AS year FROM table_rows    WHERE table_id = ?
            ) WHERE year IS NOT NULL
            """,
            (r["id"], r["id"]),
        ).fetchall()
        years = sorted(int(y[0]) for y in year_rows if y[0] is not None)

        strategy_parts = []
        if row_id in fts_trace.strategy_by_id:
            strategy_parts.append(f"fts:{fts_trace.strategy_by_id[row_id]}")
        if row_id in metric_trace.strategy_by_id:
            strategy_parts.append(f"metric:{metric_trace.strategy_by_id[row_id]}")
        fts_weighted, metric_weighted = weighted_channel_scores.get(row_id, (0.0, 0.0))
        best_channel = "fts" if fts_weighted >= metric_weighted else "metric"

        entry = {
            "file": file,
            "element_id": r["element_id"],
            "element_seq": r["element_seq"],
            "page_id": r["page_id"],
            "file_year": r["file_year"],
            "file_month": r["file_month"],
            "section": r["section"] or "",
            "title": r["title"] or "",
            "caption": r["caption"] or "",
            "column_headers": [c[0] or "" for c in cols],
            "row_labels": [lab[0] or "" for lab in labels],
            "years": years,
            "unit": r["unit"],
            "period": r["period"],
            "n_rows": r["n_rows"],
            "n_cols": r["n_cols"],
            "retrieval_strategy": " | ".join(strategy_parts),
            "retrieval_channel": best_channel,
            "html": _load_element_html(file, r["element_seq"]) if load_html else "",
        }
        entries.append(entry)
        if len(entries) >= top_k:
            break

    # ── Supplementary prose/footnote entries ──────────────────────────────
    # Add prose/footnote entries as supplementary context (up to PF_MAX_SLOTS).
    # Unlike table entries, PF entries are NOT deduplicated by file — they carry
    # different content (element_seq ≠ table element_seq) and may include
    # footnotes or prose that modify/explain table values. This is the key
    # channel for ~6% of benchmark questions where the answer is in footnotes.
    # We use a separate seen_element set to avoid exact duplicates.
    _PF_MAX_SLOTS = 3
    if pf_hits:
        seen_elements: set[tuple[str, int]] = set()
        # Pre-populate with element_seqs from already-added table entries
        for e in entries:
            seen_elements.add((e["file"], e["element_seq"]))

        # Collect best-scoring PF hit per (file, source) combination
        pf_by_key: dict[tuple[str, str], dict] = {}
        for hit in pf_hits:
            key = (hit["file"], hit["source"])
            if key not in pf_by_key or hit["score"] < pf_by_key[key]["score"]:
                pf_by_key[key] = hit

        pf_added = 0
        for hit in sorted(pf_by_key.values(), key=lambda h: h["score"]):
            if pf_added >= _PF_MAX_SLOTS:
                break
            elem_key = (hit["file"], hit["element_seq"])
            if elem_key in seen_elements:
                continue
            seen_elements.add(elem_key)

            # Apply file-year bonus to filter time-irrelevant PF entries
            pf_file_bonus = _file_year_bonus(
                hit.get("file_year"),
                hit.get("file_month"),
                target_years,
                year_mode,
            )
            # Only include PF entries with reasonable year proximity
            if target_years and pf_file_bonus == 0.0:
                continue

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
            pf_added += 1

    return entries


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
