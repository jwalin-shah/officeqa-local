#!/usr/bin/env python3
"""Search recall validator for the Treasury Bulletin QA system.

Tests whether search_raw_corpus() can find the correct source file(s)
for each of the 246 questions in officeqa_full.csv.

Usage:
    python3 nomcp/test_search_recall.py --corpus /path/to/corpus
    python3 nomcp/test_search_recall.py --corpus /path/to/corpus --quick
    python3 nomcp/test_search_recall.py --corpus /path/to/corpus --uid UID0042
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Setup: ensure CORPUS_DIR is set before importing solve.py functions
# ---------------------------------------------------------------------------


def setup_env(corpus_dir):
    """Set environment variables needed by solve.py and build_index.py."""
    os.environ["CORPUS_DIR"] = corpus_dir
    os.environ.setdefault("KEYWORD_INDEX_PATH", "/tmp/keyword_index.txt")
    os.environ.setdefault("INDEX_PATH", "/tmp/table_index.jsonl")


# ---------------------------------------------------------------------------
# Synonym / alternative-phrasing maps
# ---------------------------------------------------------------------------

SYNONYM_MAP = {
    "expenditures": ["outlays", "spending", "expenses"],
    "outlays": ["expenditures", "spending", "expenses"],
    "spending": ["expenditures", "outlays", "expenses"],
    "expenses": ["expenditures", "outlays", "spending"],
    "receipts": ["revenue", "income", "collections"],
    "revenue": ["receipts", "income", "collections"],
    "income": ["receipts", "revenue", "collections"],
    "collections": ["receipts", "revenue", "income"],
    "debt": ["obligations", "liabilities", "borrowing"],
    "obligations": ["debt", "liabilities"],
    "deficit": ["shortfall", "excess of expenditures"],
    "surplus": ["excess of receipts"],
    "customs": ["tariff", "import duties"],
    "tariff": ["customs", "import duties"],
    "duties": ["customs duties", "tariffs"],
    "defense": ["military", "national defense", "war"],
    "military": ["defense", "national defense"],
    "war": ["defense", "military"],
    "claims": ["assets", "holdings"],
    "assets": ["claims", "holdings"],
    "investments": ["securities", "holdings"],
    "securities": ["investments", "bonds", "notes"],
    "bonds": ["securities", "notes", "obligations"],
    "interest": ["interest cost", "interest paid"],
    "circulation": ["outstanding", "in circulation"],
    "currency": ["money", "coin and currency"],
    "grants": ["aid", "grants-in-aid"],
    "aid": ["grants", "grants-in-aid", "assistance"],
    "taxes": ["tax receipts", "tax collections", "tax revenue"],
    "employment": ["payroll", "jobs", "workforce"],
    "payroll": ["employment"],
    "imports": ["merchandise imports"],
    "exports": ["merchandise exports"],
}


# ---------------------------------------------------------------------------
# Keyword extraction
# ---------------------------------------------------------------------------

_STOPWORDS = {
    "the",
    "and",
    "for",
    "this",
    "that",
    "with",
    "from",
    "these",
    "those",
    "which",
    "where",
    "when",
    "what",
    "how",
    "has",
    "had",
    "have",
    "been",
    "were",
    "was",
    "are",
    "not",
    "but",
    "all",
    "its",
    "only",
    "also",
    "each",
    "into",
    "than",
    "then",
    "they",
    "their",
    "will",
    "can",
    "just",
    "about",
    "did",
    "does",
    "any",
    "both",
    "should",
    "would",
    "could",
    "using",
    "specifically",
    "reported",
    "values",
    "value",
    "individual",
    "corresponding",
    "million",
    "millions",
    "billion",
    "billions",
    "dollars",
    "nominal",
    "fiscal",
    "calendar",
    "year",
    "total",
    "amount",
    "sum",
    "figure",
    "rounded",
    "nearest",
    "according",
    "place",
    "places",
    "percent",
    "percentage",
    "calculate",
    "determine",
    "report",
    "your",
    "answer",
    "decimal",
    "enter",
    "number",
    "found",
    "include",
    "included",
    "contain",
    "containing",
}


def _extract_metric_phrases(question):
    """Extract the core metric phrase(s) from a question using pattern matching.
    Returns a list of candidate metric phrases."""
    q = question
    phrases = []

    # Direct pattern matching for known metric categories
    metric_patterns = [
        (
            r"(?:expenditures?|outlays?|spending)\s+(?:for\s+)?(?:the\s+)?([\w\s]+?)(?:\s+in\s+|\s+for\s+|\s+during\s+|,|\?|$)",
            None,
        ),
        (r"(national defense\s*(?:and associated activities)?(?:\s*expenditures?)?)", None),
        (r"(customs?\s+duties)", None),
        (r"(individual income tax\s*(?:receipts?|revenue|collections?)?)", None),
        (r"(corporation income tax\s*(?:receipts?|revenue|collections?)?)", None),
        (r"(income tax\s*(?:receipts?|revenue)?)", None),
        (r"(interest\s+(?:cost|paid|on the (?:public\s+)?debt|outlays?))", None),
        (r"(budget\s+(?:receipts?|expenditures?|outlays?))", None),
        (r"(public debt)", None),
        (r"(veterans?\s+(?:administration|affairs))", None),
        (r"(grants?[\s-]+in[\s-]+aid)", None),
        (r"(employment\s+(?:and\s+)?(?:general\s+)?retirement)", None),
        (r"(social\s+(?:security|insurance))", None),
        (r"(savings?\s+bonds?)", None),
        (r"(coin\s+and\s+currency)", None),
        (r"(treasury\s+(?:notes?|bonds?|bills?))", None),
        (r"(foreign\s+(?:exchange|investments?|currencies?))", None),
        (r"(claims?\s+(?:owed|reported))", None),
        (r"(merchandise\s+(?:imports?|exports?))", None),
        (r"(excise\s+taxes?)", None),
        (r"(estate\s+(?:and\s+gift\s+)?taxes?)", None),
        (r"(internal\s+revenue)", None),
        (r"(trust\s+fund)", None),
        (r"(net\s+(?:budget\s+)?receipts?)", None),
        (r"(gross\s+(?:debt|receipts?))", None),
        (r"(postal\s+(?:service|revenue|deficit))", None),
        (r"(highway\s+trust\s+fund)", None),
        (r"(unemployment\s+(?:trust|insurance|compensation))", None),
        (r"((?:federal\s+)?old[\s-]+age)", None),
        (r"((?:net\s+)?interest\s+(?:on\s+)?(?:the\s+)?(?:public\s+)?debt)", None),
        (r"(international\s+(?:affairs?|assistance|cooperation))", None),
        (r"(agriculture\s+(?:department|expenditures?|outlays?))", None),
        (r"(education\s+(?:department|expenditures?|outlays?))", None),
        (r"(health\s+(?:and\s+human\s+services|education|department))", None),
        (r"(treasury\s+(?:department|securities|bonds?|notes?))", None),
    ]

    for pat, _ in metric_patterns:
        m = re.search(pat, q, re.I)
        if m:
            phrase = m.group(1).strip()
            # Clean up
            phrase = re.sub(r"\s+", " ", phrase).strip(" ,.")
            if len(phrase) > 3:
                phrases.append(phrase)

    return phrases


def extract_search_params(question):
    """Extract metric, year, period_basis, and multiple keyword strategies from a question."""
    # Years
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)
    year = int(years[0]) if years else None
    all_years = sorted(set(int(y) for y in years))

    # Period basis
    period = "calendar"
    if re.search(r"fiscal\s+year|FY\s*\d{4}", question, re.I):
        period = "fiscal"

    # --- Approach A: Pattern-based metric extraction ---
    metric_phrases = _extract_metric_phrases(question)

    # --- Approach B: Boilerplate-removal extraction ---
    q = question
    # Strip leading "Using specifically only the reported values..."
    q = re.sub(
        r"^Using\s+specifically\s+only\s+.*?(?:,\s*what\s+is|,\s*what\s+was|,\s*what\s+were)",
        "What were",
        q,
        flags=re.I | re.DOTALL,
    )
    # Strip leading "According to the US Treasury's breakdown..."
    q = re.sub(
        r"^According\s+to\s+.*?(?:,\s*what\s+is|,\s*what\s+was)",
        "What was",
        q,
        flags=re.I | re.DOTALL,
    )
    # Remove common question prefixes
    q = re.sub(
        r"^(?:What|How\s+much|Determine|Using)\s+(?:were?|is|are|was|did)\s+(?:the\s+)?",
        "",
        q,
        flags=re.I,
    )
    q = re.sub(r"^(?:What|How\s+much)\s+(?:were?|is|are|was|did)\s+", "", q, flags=re.I)
    # Remove total/sum phrasing
    q = re.sub(
        r"total\s+(?:sum|value|amount|dollar\s+value)\s+of\s+(?:these\s+values\s+of\s+)?",
        "",
        q,
        flags=re.I,
    )
    # Remove unit specifiers
    q = re.sub(
        r"(?:in\s+(?:millions?|billions?|thousands?)\s+of\s+(?:nominal\s+)?(?:dollars?|USD))",
        "",
        q,
        flags=re.I,
    )
    q = re.sub(r"(?:rounded\s+to\s+.*?)(?:\.|$)", "", q, flags=re.I)
    q = re.sub(r"(?:expressed\s+in\s+.*?)(?:\.|,)", "", q, flags=re.I)
    # Remove year/period references
    q = re.sub(
        r"\b(?:for|in|of|the|during)\s+(?:the\s+)?(?:calendar|fiscal)\s+year\s+(?:of\s+)?\d{4}\??",
        "",
        q,
        flags=re.I,
    )
    q = re.sub(r"\bFY\s*\d{4}\b", "", q, flags=re.I)
    q = re.sub(r"\b(?:calendar|fiscal)\s+year\s*\d{4}\b", "", q, flags=re.I)
    q = re.sub(r"\b\d{4}\s*[-–]\s*\d{4}\b", "", q)
    # Remove U.S. government references
    q = re.sub(r"\bU\.?S\.?\s*(?:federal\s+)?(?:government\'?s?\s*)?", "", q, flags=re.I)
    q = re.sub(r"\b(?:the\s+)?(?:federal\s+)?government\'?s?\s+", "", q, flags=re.I)
    # Remove year numbers
    q = re.sub(r"\b(?:19|20)\d{2}\b", "", q)
    # Remove trailing punctuation and whitespace
    q = re.sub(r"\s+", " ", q).strip(" ?,.\n\r\t")

    # Filter to meaningful words
    words = q.split()
    meaningful = [w for w in words if len(w) > 2 and w.lower() not in _STOPWORDS]

    # Build strategies list
    strategies = []

    # Strategy 1: Pattern-extracted phrases (best quality)
    for i, mp in enumerate(metric_phrases[:2]):
        strategies.append(("pattern_match", mp.lower()))

    # Strategy 2: Cleaned boilerplate - full phrase (first 6 words)
    full_phrase = " ".join(meaningful[:6]).lower()
    if full_phrase and len(full_phrase) > 5:
        strategies.append(("full_phrase", full_phrase))

    # Strategy 3: Shortened (first 3 meaningful words)
    short_phrase = " ".join(meaningful[:3]).lower()
    if short_phrase and short_phrase != full_phrase and len(short_phrase) > 5:
        strategies.append(("shortened", short_phrase))

    # Strategy 4: Key terms only (first 2 meaningful words)
    key_terms = " ".join(meaningful[:2]).lower()
    if key_terms and key_terms != short_phrase and len(key_terms) > 5:
        strategies.append(("key_terms", key_terms))

    # Strategy 5: Synonym-based alternatives on the best phrase
    base_phrase = metric_phrases[0].lower() if metric_phrases else full_phrase
    if base_phrase:
        synonym_phrases = set()
        for word in base_phrase.split():
            if word in SYNONYM_MAP:
                for syn in SYNONYM_MAP[word][:2]:
                    alt = base_phrase.replace(word, syn)
                    if alt != base_phrase:
                        synonym_phrases.add(alt)
        for sp in list(synonym_phrases)[:3]:
            strategies.append(("synonym", sp))

    # Fallback: extract distinctive terms from the raw question
    if not strategies or all(len(s[1].split()) < 2 for s in strategies):
        raw_terms = re.findall(r"\b[a-zA-Z]{4,}\b", question)
        raw_terms = [t.lower() for t in raw_terms if t.lower() not in _STOPWORDS]
        # Deduplicate preserving order
        seen = set()
        unique_terms = []
        for t in raw_terms:
            if t not in seen:
                seen.add(t)
                unique_terms.append(t)
        if unique_terms:
            strategies.append(("raw_terms", " ".join(unique_terms[:4])))

    # Deduplicate strategies (same keywords)
    seen_kw = set()
    deduped = []
    for name, kw in strategies:
        if kw not in seen_kw:
            seen_kw.add(kw)
            deduped.append((name, kw))
    strategies = deduped

    metric = metric_phrases[0] if metric_phrases else full_phrase
    return {
        "year": year,
        "all_years": all_years,
        "period": period,
        "metric": metric[:100] if metric else q[:100],
        "strategies": strategies,
    }


# ---------------------------------------------------------------------------
# Load questions from CSV
# ---------------------------------------------------------------------------


def load_questions(csv_path, uid_filter=None, quick=False):
    """Load questions from officeqa_full.csv. Handles multi-line fields."""
    questions = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            uid = row["uid"].strip()
            if uid_filter and uid != uid_filter:
                continue
            source_files_raw = row.get("source_files", "").strip()
            # source_files may be newline-separated
            source_files = [sf.strip() for sf in source_files_raw.split("\n") if sf.strip()]
            questions.append(
                {
                    "uid": uid,
                    "question": row["question"].strip(),
                    "answer": row.get("answer", "").strip(),
                    "source_files": source_files,
                    "difficulty": row.get("difficulty", "").strip(),
                }
            )
            if quick and len(questions) >= 20:
                break
    return questions


# ---------------------------------------------------------------------------
# Phase 2: Search recall test via search_raw_corpus
# ---------------------------------------------------------------------------


def test_search_recall(search_fn, questions, verbose=False):
    """Test each question x strategy, return detailed results."""
    results = []
    strategy_hits = defaultdict(int)
    strategy_totals = defaultdict(int)
    any_hit = 0
    total = len(questions)

    for idx, q in enumerate(questions):
        uid = q["uid"]
        expected_files = set(q["source_files"])
        params = extract_search_params(q["question"])
        year = params["year"]
        strategies_tried = []
        found_any = False

        if verbose or (idx % 25 == 0):
            print(
                f"  [{idx + 1}/{total}] {uid}: year={year}, strategies={len(params['strategies'])}"
            )

        for strat_name, keywords in params["strategies"]:
            strategy_totals[strat_name] += 1
            try:
                resp = search_fn(keywords=keywords, year=year, limit=5)
                result_files = set()
                for r in resp.get("results", []):
                    fname = r.get("file", "")
                    result_files.add(fname)

                hit = bool(result_files & expected_files)
                if hit:
                    strategy_hits[strat_name] += 1
                    found_any = True

                strategies_tried.append(
                    {
                        "strategy": strat_name,
                        "keywords": keywords,
                        "year": year,
                        "hit": hit,
                        "result_files": sorted(result_files),
                        "result_count": resp.get("count", 0),
                    }
                )
            except Exception as e:
                strategies_tried.append(
                    {
                        "strategy": strat_name,
                        "keywords": keywords,
                        "year": year,
                        "hit": False,
                        "error": str(e),
                        "result_files": [],
                    }
                )

        # Also try with no year filter if all strategies failed
        if not found_any and params["strategies"]:
            strat_name_noyear = "no_year_fallback"
            best_kw = params["strategies"][0][1]
            strategy_totals[strat_name_noyear] += 1
            try:
                resp = search_fn(keywords=best_kw, year=None, limit=10)
                result_files = set()
                for r in resp.get("results", []):
                    fname = r.get("file", "")
                    result_files.add(fname)
                hit = bool(result_files & expected_files)
                if hit:
                    strategy_hits[strat_name_noyear] += 1
                    found_any = True
                strategies_tried.append(
                    {
                        "strategy": strat_name_noyear,
                        "keywords": best_kw,
                        "year": None,
                        "hit": hit,
                        "result_files": sorted(result_files),
                        "result_count": resp.get("count", 0),
                    }
                )
            except Exception as e:
                strategies_tried.append(
                    {
                        "strategy": strat_name_noyear,
                        "keywords": best_kw,
                        "year": None,
                        "hit": False,
                        "error": str(e),
                        "result_files": [],
                    }
                )

        if found_any:
            any_hit += 1

        results.append(
            {
                "uid": uid,
                "question": q["question"][:200],
                "expected_files": sorted(expected_files),
                "difficulty": q["difficulty"],
                "year": year,
                "metric": params["metric"][:100],
                "found_any": found_any,
                "strategies": strategies_tried,
            }
        )

    return {
        "results": results,
        "strategy_hits": dict(strategy_hits),
        "strategy_totals": dict(strategy_totals),
        "any_hit": any_hit,
        "total": total,
    }


# ---------------------------------------------------------------------------
# Phase 3: Raw grep baseline
# ---------------------------------------------------------------------------


def test_grep_baseline(questions, corpus_dir, verbose=False):
    """For each question, do a raw grep to see if we can find the right file."""
    total = len(questions)
    hits = 0
    results = []

    for idx, q in enumerate(questions):
        uid = q["uid"]
        expected_files = set(q["source_files"])

        if verbose or (idx % 25 == 0):
            print(f"  [grep {idx + 1}/{total}] {uid}")

        # Extract 2-3 key search terms from the question
        terms = _extract_grep_terms(q["question"])
        found_files = set()

        if terms:
            found_files = _grep_corpus(
                terms, corpus_dir, expected_year=extract_search_params(q["question"])["year"]
            )

        hit = bool(found_files & expected_files)
        if hit:
            hits += 1

        results.append(
            {
                "uid": uid,
                "terms": terms,
                "hit": hit,
                "found_files": sorted(found_files)[:10],
                "expected_files": sorted(expected_files),
            }
        )

    return {
        "hits": hits,
        "total": total,
        "results": results,
    }


def _extract_grep_terms(question):
    """Extract 2-3 distinctive terms for grep search."""
    # Remove common words and extract distinctive terms
    q = question.lower()
    # Remove very common words
    stopwords = {
        "what",
        "were",
        "was",
        "the",
        "total",
        "for",
        "and",
        "this",
        "that",
        "with",
        "from",
        "these",
        "which",
        "where",
        "when",
        "how",
        "much",
        "did",
        "has",
        "had",
        "have",
        "been",
        "are",
        "not",
        "but",
        "all",
        "using",
        "specifically",
        "only",
        "reported",
        "values",
        "value",
        "million",
        "millions",
        "billion",
        "billions",
        "dollars",
        "nominal",
        "rounded",
        "nearest",
        "according",
        "figure",
        "should",
        "include",
        "number",
        "found",
        "amount",
        "each",
        "individual",
        "corresponding",
        "enter",
        "percentage",
        "calculate",
        "determine",
        "report",
        "your",
        "answer",
        "decimal",
        "place",
        "places",
    }

    # Find multi-word phrases that are distinctive
    phrases = []

    # Try to find key metric phrases
    metric_patterns = [
        r"(national defense\b[\w\s]*(?:expenditures|outlays|spending)?)",
        r"(customs duties)",
        r"(individual income tax\b[\w\s]*(?:receipts|revenue)?)",
        r"(corporation income tax\b[\w\s]*(?:receipts|revenue)?)",
        r"(interest (?:cost|paid|on the (?:public )?debt))",
        r"(budget (?:receipts|expenditures|outlays))",
        r"(public debt)",
        r"(veterans (?:administration|affairs))",
        r"(grants[\- ]in[\- ]aid)",
        r"(employment\s+(?:and\s+)?(?:general\s+)?retirement)",
        r"(social\s+(?:security|insurance))",
        r"(savings?\s+bonds?)",
        r"(coin\s+and\s+currency)",
        r"(treasury\s+(?:notes?|bonds?|bills?))",
        r"(foreign\s+(?:exchange|investments?|currencies?))",
        r"(claims\s+(?:owed|reported))",
        r"(merchandise\s+(?:imports?|exports?))",
        r"(excise\s+taxes?)",
        r"(estate\s+(?:and\s+gift\s+)?taxes?)",
        r"(internal\s+revenue)",
        r"(trust\s+fund)",
    ]

    for pat in metric_patterns:
        m = re.search(pat, q)
        if m:
            phrase = m.group(1).strip()
            # Take first 3 meaningful words of the phrase
            pwords = [w for w in phrase.split() if w not in stopwords and len(w) > 2]
            phrases = pwords[:3]
            break

    if not phrases:
        # Fall back: extract longest non-stopword terms
        words = re.findall(r"\b[a-z]{3,}\b", q)
        words = [w for w in words if w not in stopwords]
        # Prefer longer words (more distinctive)
        words.sort(key=len, reverse=True)
        phrases = words[:3]

    return phrases


def _grep_corpus(terms, corpus_dir, expected_year=None):
    """Find corpus files where all terms appear (within a reasonable window).
    Uses Python re for portability."""
    corpus = Path(corpus_dir)
    found = set()

    # Narrow file list by year if available
    files = sorted(corpus.glob("treasury_bulletin_*.txt"))
    if expected_year:
        yr = expected_year
        year_files = [f for f in files if any(f"_{y}_" in f.name for y in range(yr, yr + 5))]
        if year_files:
            files = year_files
        else:
            # Wider range
            files = [f for f in files if any(f"_{y}_" in f.name for y in range(yr - 1, yr + 6))]

    for fpath in files[:30]:  # Cap at 30 files for speed
        try:
            text = fpath.read_text(errors="replace").lower()
        except Exception:
            continue

        # Check if ALL terms appear in the file
        if all(t in text for t in terms):
            # Bonus: check if terms appear within 50-line window
            lines = text.split("\n")
            for i in range(len(lines)):
                window = "\n".join(lines[max(0, i) : min(len(lines), i + 50)])
                if all(t in window for t in terms):
                    found.add(fpath.name)
                    break

    return found


# ---------------------------------------------------------------------------
# Phase 4: Reporting
# ---------------------------------------------------------------------------


def generate_report(search_results, grep_results, output_dir):
    """Generate human-readable summary and JSON detail files."""
    os.makedirs(output_dir, exist_ok=True)

    sr = search_results
    gr = grep_results
    total = sr["total"]

    lines = []
    lines.append("=" * 60)
    lines.append("SEARCH RECALL REPORT")
    lines.append("=" * 60)
    lines.append(f"Total questions: {total}")
    lines.append("")

    # Strategy breakdown
    lines.append("--- Strategy Results ---")
    strat_order = ["full_phrase", "shortened", "key_terms", "synonym", "no_year_fallback"]
    for s in strat_order:
        h = sr["strategy_hits"].get(s, 0)
        t = sr["strategy_totals"].get(s, 0)
        pct = (h / t * 100) if t else 0
        lines.append(f"  {s:25s}: {h:3d}/{t:3d} found correct file ({pct:5.1f}%)")

    lines.append(
        f"  {'ANY strategy':25s}: {sr['any_hit']:3d}/{total:3d} found correct file ({sr['any_hit'] / total * 100:5.1f}%)"
    )
    lines.append(
        f"  {'Raw grep baseline':25s}: {gr['hits']:3d}/{gr['total']:3d} found correct file ({gr['hits'] / gr['total'] * 100:5.1f}%)"
    )
    lines.append("")

    # Failures
    failures = [r for r in sr["results"] if not r["found_any"]]
    lines.append(
        f"=== FAILURES ({len(failures)} questions where NO strategy found the right file) ==="
    )
    for f in failures:
        lines.append("")
        lines.append(f"{f['uid']} ({f['difficulty']}): expected {', '.join(f['expected_files'])}")
        lines.append(f"  Question: {f['question'][:150]}...")
        for st in f["strategies"]:
            kw = st["keywords"][:60]
            files_found = ", ".join(st["result_files"][:3]) if st["result_files"] else "NONE"
            lines.append(f"  [{st['strategy']:15s}] '{kw}' -> {files_found}")

    lines.append("")

    # Per-file analysis
    file_stats = defaultdict(lambda: {"total": 0, "found": 0})
    for r in sr["results"]:
        for ef in r["expected_files"]:
            file_stats[ef]["total"] += 1
            if r["found_any"]:
                # Check if this specific file was found
                for st in r["strategies"]:
                    if st["hit"] and ef in st.get("result_files", []):
                        file_stats[ef]["found"] += 1
                        break

    lines.append("=== PER-FILE ANALYSIS (files with questions) ===")
    # Sort by number of questions descending
    for fname, stats in sorted(file_stats.items(), key=lambda x: x[1]["total"], reverse=True)[:30]:
        t = stats["total"]
        h = stats["found"]
        pct = h / t * 100 if t else 0
        lines.append(f"  {fname}: {t} questions, {h} found ({pct:.0f}%)")

    lines.append("")

    # Difficulty breakdown
    diff_stats = defaultdict(lambda: {"total": 0, "found": 0})
    for r in sr["results"]:
        d = r["difficulty"] or "unknown"
        diff_stats[d]["total"] += 1
        if r["found_any"]:
            diff_stats[d]["found"] += 1

    lines.append("=== DIFFICULTY BREAKDOWN ===")
    for d, stats in sorted(diff_stats.items()):
        t = stats["total"]
        h = stats["found"]
        pct = h / t * 100 if t else 0
        lines.append(f"  {d:10s}: {h}/{t} ({pct:.1f}%)")

    report_text = "\n".join(lines)

    # Write summary
    summary_path = os.path.join(output_dir, "search_recall_summary.txt")
    with open(summary_path, "w") as f:
        f.write(report_text)

    # Write detailed JSON
    json_path = os.path.join(output_dir, "search_recall.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "search_results": sr,
                "grep_results": gr,
                "summary": {
                    "total": total,
                    "any_hit": sr["any_hit"],
                    "any_hit_pct": round(sr["any_hit"] / total * 100, 1),
                    "grep_hit": gr["hits"],
                    "grep_hit_pct": round(gr["hits"] / gr["total"] * 100, 1),
                    "failures": len(failures),
                    "strategy_hits": sr["strategy_hits"],
                    "strategy_totals": sr["strategy_totals"],
                },
            },
            f,
            indent=2,
            default=str,
        )

    return report_text, summary_path, json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Search recall validator for Treasury Bulletin QA")
    parser.add_argument("--corpus", required=True, help="Path to corpus directory with TXT files")
    parser.add_argument("--csv", default=None, help="Path to officeqa_full.csv")
    parser.add_argument("--quick", action="store_true", help="Test only first 20 questions")
    parser.add_argument("--uid", default=None, help="Test a specific question UID (e.g. UID0042)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output per question")
    parser.add_argument(
        "--skip-grep", action="store_true", help="Skip the raw grep baseline (faster)"
    )
    parser.add_argument("--output", default=None, help="Output directory for results")
    args = parser.parse_args()

    # Resolve paths
    corpus_dir = os.path.abspath(args.corpus)
    if not os.path.isdir(corpus_dir):
        print(f"ERROR: Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = args.csv or os.path.join(script_dir, "..", "data", "officeqa_full.csv")
    csv_path = os.path.abspath(csv_path)
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output or os.path.join(script_dir, "test_results")

    # Setup environment
    setup_env(corpus_dir)

    # Build keyword index if needed
    print("Checking keyword index...")
    idx_path = os.environ.get("KEYWORD_INDEX_PATH", "/tmp/keyword_index.txt")
    if not os.path.exists(idx_path) or os.path.getsize(idx_path) < 1000:
        print("Building keyword index (this may take ~30s)...")
        build_script = os.path.join(script_dir, "build_index.py")
        if os.path.exists(build_script):
            subprocess.run(["python3", build_script, corpus_dir], capture_output=False, timeout=120)
        else:
            print(f"WARNING: build_index.py not found at {build_script}", file=sys.stderr)

    # Import search function AFTER setting env
    sys.path.insert(0, script_dir)
    from solve import search_raw_corpus

    # Load questions
    print(f"Loading questions from {csv_path}...")
    questions = load_questions(csv_path, uid_filter=args.uid, quick=args.quick)
    print(f"Loaded {len(questions)} questions")
    if not questions:
        print("No questions matched the filter.", file=sys.stderr)
        sys.exit(1)

    # Phase 2: Search recall
    print(f"\n{'=' * 60}")
    print(f"Phase 1+2: Search recall test ({len(questions)} questions)")
    print(f"{'=' * 60}")
    t0 = time.time()
    search_results = test_search_recall(search_raw_corpus, questions, verbose=args.verbose)
    t1 = time.time()
    print(
        f"Search recall: {search_results['any_hit']}/{search_results['total']} "
        f"({search_results['any_hit'] / search_results['total'] * 100:.1f}%) in {t1 - t0:.1f}s"
    )

    # Phase 3: Grep baseline
    if not args.skip_grep:
        print(f"\n{'=' * 60}")
        print(f"Phase 3: Raw grep baseline ({len(questions)} questions)")
        print(f"{'=' * 60}")
        t2 = time.time()
        grep_results = test_grep_baseline(questions, corpus_dir, verbose=args.verbose)
        t3 = time.time()
        print(
            f"Grep baseline: {grep_results['hits']}/{grep_results['total']} "
            f"({grep_results['hits'] / grep_results['total'] * 100:.1f}%) in {t3 - t2:.1f}s"
        )
    else:
        grep_results = {"hits": 0, "total": len(questions), "results": []}

    # Phase 4: Report
    print(f"\n{'=' * 60}")
    print("Phase 4: Generating report")
    print(f"{'=' * 60}")
    report_text, summary_path, json_path = generate_report(search_results, grep_results, output_dir)
    print(report_text)
    print("\nDetailed results saved to:")
    print(f"  Summary: {summary_path}")
    print(f"  JSON:    {json_path}")


if __name__ == "__main__":
    main()
