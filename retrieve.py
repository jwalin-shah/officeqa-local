#!/usr/bin/env python3
"""Retrieval: year-window candidate files + decompose-hint grep + co-occurrence filter."""

import concurrent.futures
import contextlib
import re
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

sys.stdout.reconfigure(line_buffering=True)
load_dotenv()

CORPUS_DIR = Path("corpus")

SYNONYMS = {
    "expenditures": ["outlays", "spending", "disbursements"],
    "receipts": ["revenue", "income", "collections"],
    "defense": ["national defense", "military"],
    "debt": ["public debt", "obligations"],
    "veterans": ["veterans' administration", "veterans administration", "VA"],
    "international": ["foreign", "claims", "liabilities"],
    "currency": ["coin", "circulation", "money"],
    "savings": ["savings bonds", "series E", "series H"],
}


# ── Year / bulletin parsing ──────────────────────────────────────────────────


def parse_years(question: str) -> list[int]:
    return sorted(set(int(y) for y in re.findall(r"\b(1[89]\d{2}|20[0-2]\d)\b", question)))


def parse_bulletin_ref(question: str) -> list[str]:
    months = {
        "january": "01",
        "february": "02",
        "march": "03",
        "april": "04",
        "may": "05",
        "june": "06",
        "july": "07",
        "august": "08",
        "september": "09",
        "october": "10",
        "november": "11",
        "december": "12",
    }
    matches = []
    for m in re.finditer(r"bulletin\s+(?:published\s+)?(?:in\s+)?(\w+)\s+(\d{4})", question, re.I):
        month_name = m.group(1).lower()
        year = m.group(2)
        if month_name in months:
            matches.append(f"treasury_bulletin_{year}_{months[month_name]}.txt")
    return matches


def candidate_files(years: list[int]) -> list[str]:
    """For each year Y, candidate files are Y-2 through Y+5."""
    candidate_years = set()
    for y in years:
        for offset in range(-2, 6):
            candidate_years.add(y + offset)
    all_corpus = {f.name: str(f) for f in CORPUS_DIR.glob("treasury_bulletin_*.txt")}
    result = []
    for cy in sorted(candidate_years):
        for month in range(1, 13):
            fname = f"treasury_bulletin_{cy}_{month:02d}.txt"
            if fname in all_corpus:
                result.append(all_corpus[fname])
    return result


def detect_year_type(question: str) -> str:
    q = question.lower()
    if "calendar year" in q or "calendar month" in q or "individual calendar" in q:
        return "calendar"
    if "fiscal year" in q or "fy " in q:
        return "fiscal"
    return "fiscal"


def phase_a(question: str) -> list[str]:
    """Candidate files from question years / direct bulletin refs."""
    direct = parse_bulletin_ref(question)
    if direct:
        all_corpus = {f.name: str(f) for f in CORPUS_DIR.glob("treasury_bulletin_*.txt")}
        found = [all_corpus[d] for d in direct if d in all_corpus]
        if found:
            return found

    years = parse_years(question)
    if not years:
        return sorted(str(f) for f in CORPUS_DIR.glob("treasury_bulletin_*.txt"))
    return candidate_files(years)


# ── Grep helpers ─────────────────────────────────────────────────────────────


def expand_terms(terms: list[str]) -> list[str]:
    expanded = list(terms)
    for t in terms:
        for key, syns in SYNONYMS.items():
            if key in t.lower():
                expanded.extend(syns)
    return list(dict.fromkeys(expanded))


def grep_files(term: str, files: list[str]) -> list[str]:
    if not files:
        return []
    try:
        r = subprocess.run(
            ["grep", "-i", "-l", re.escape(term)] + files,
            capture_output=True,
            text=True,
            timeout=15,
        )
        return [f for f in r.stdout.strip().split("\n") if f]
    except Exception:
        return []


# ── Plan-driven retrieval ────────────────────────────────────────────────────


def _plan_grep_terms(plan: dict) -> list[str]:
    """Specific grep terms from a decompose plan: column/row hints, metric, multi-word synonyms."""
    terms: list[str] = []
    col = plan.get("column_hint", "")
    row = plan.get("row_hint", "")
    if col:
        terms.append(col)
    if row and row != col:
        terms.append(row)
    metric = plan.get("metric", "")
    if metric and metric not in terms:
        terms.append(metric)
    for syn in plan.get("metric_synonyms", []):
        if syn not in terms and (len(syn.split()) >= 2 or len(syn) > 10):
            terms.append(syn)
    return [t for t in terms if t and len(t) > 1]


def _year_markers(plan: dict) -> list[str]:
    year = plan.get("year")
    if not year:
        return []
    calc_type = plan.get("calc_type", "lookup")
    if calc_type in ("sum_monthly", "range_monthly"):
        return [f"{year}-January", f"{year}-Jan.", f"{year}-Jan "]
    return [f"| {year} |", f"{year}-January", f"{year}-Jan.", f"{year}-Jan "]


def _grep_filter(markers: list[str], candidates: list[str]) -> set[str]:
    hits: set[str] = set()
    for marker in markers:
        try:
            r = subprocess.run(
                ["grep", "-l", marker] + candidates,
                capture_output=True,
                text=True,
                timeout=15,
            )
            hits.update(f for f in r.stdout.strip().split("\n") if f)
        except Exception:
            continue
    return hits


def _cooccurrence_filter(
    files: list[str], terms: list[str], year: int | None, year_markers: list[str]
) -> list[str]:
    """Keep files where a metric term is within 30 lines of a year marker."""
    if not year or not year_markers:
        return files
    survivors: list[str] = []
    for fpath in files:
        try:
            with open(fpath) as _f:
                lines = _f.readlines()
        except Exception:
            continue
        year_lines = {i for i, line in enumerate(lines) if any(m in line for m in year_markers)}
        if not year_lines:
            continue
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(t.lower() in line_lower for t in terms[:5]) and any(
                abs(i - yl) <= 30 for yl in year_lines
            ):
                survivors.append(fpath)
                break
    return survivors


def retrieve(plan: dict, question: str, verbose: bool = False) -> tuple[list[str], list[str]]:
    """Plan-driven retrieval. Returns (file_paths, grep_terms)."""
    grep_terms = _plan_grep_terms(plan)
    expanded = expand_terms(grep_terms)
    if verbose:
        print(f"  Retrieve terms: {grep_terms}")

    candidates = phase_a(question)
    if verbose:
        print(f"  Retrieve: {len(candidates)} candidate files from year window")

    year_markers = _year_markers(plan)
    if year_markers:
        year_files = _grep_filter(year_markers, candidates)
        if year_files:
            candidates = [f for f in candidates if f in year_files]
            if verbose:
                print(f"  Retrieve: {len(candidates)} files have year data")

    matching: set[str] = set()
    if expanded:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(expanded), 6)) as pool:
            futures = {pool.submit(grep_files, kw, candidates): kw for kw in expanded}
            for future in concurrent.futures.as_completed(futures):
                matching.update(future.result())

    if verbose:
        print(f"  Retrieve: {len(matching)} files match metric terms")

    if not matching:
        return candidates[:10], grep_terms

    year = plan.get("year")
    survivors = _cooccurrence_filter(list(matching), grep_terms, year, year_markers)
    if verbose:
        print(f"  Retrieve: {len(survivors)} files pass co-occurrence filter")

    if not survivors:
        survivors = sorted(matching)[:10]
    return survivors, grep_terms


# ── CLI ──────────────────────────────────────────────────────────────────────


def _load_benchmark():
    import csv

    with open("officeqa_full.csv") as f:
        return list(csv.DictReader(f))


def _test_phase_a():
    rows = _load_benchmark()
    hits, total = 0, 0
    for row in rows:
        gold_files = [f.strip() for f in row["source_files"].split("\n") if f.strip()]
        candidates = phase_a(row["question"])
        cand_names = {Path(f).name for f in candidates}
        for gf in gold_files:
            total += 1
            if gf in cand_names:
                hits += 1
            else:
                print(
                    f"  MISS {row['uid']}: {gf} not in candidates (years: {parse_years(row['question'])})"
                )
    print(f"\nPhase A recall: {hits}/{total} = {hits / total * 100:.1f}%")


def _test_full(n: int = 0):
    """Full retrieval test: decompose the question, then retrieve per data_request."""
    from solve import decompose, retrieve_for_spec  # lazy import to avoid cycle

    rows = _load_benchmark()
    if n:
        rows = rows[:n]
    hits, total = 0, 0
    for row in rows:
        gold_files = [f.strip() for f in row["source_files"].split("\n") if f.strip()]
        spec = decompose(row["question"])
        if spec is None:
            print(f"  DECOMPOSE_FAILED {row['uid']}")
            for _ in gold_files:
                total += 1
            continue
        top_files = retrieve_for_spec(spec, row["question"])
        match_names = {Path(f).name for f in top_files}
        for gf in gold_files:
            total += 1
            if gf in match_names:
                hits += 1
            else:
                print(f"  MISS {row['uid']}: {gf} not in retrieved files")
    print(f"\nRetrieval recall: {hits}/{total} = {hits / total * 100:.1f}%")


if __name__ == "__main__":
    import sys

    args = sys.argv[1:]
    if "--test-phase-a" in args:
        _test_phase_a()
    elif "--test" in args:
        n = 0
        for i, a in enumerate(args):
            if a == "--test" and i + 1 < len(args):
                with contextlib.suppress(ValueError):
                    n = int(args[i + 1])
        _test_full(n)
    else:
        print("Usage: uv run python retrieve.py [--test-phase-a | --test [N]]")
