#!/usr/bin/env python3
"""Build a deduplicated, enriched table index from Treasury Bulletin TXT files.

Encodes master_ledger knowledge: monthly detection, fiscal/calendar, bulletin vintage.
Deduplicates by content hash, keeping the NEWEST bulletin version.

Usage:
    python3 /installed-agent/build_index.py
"""

import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

CORPUS_DIR = os.environ.get("CORPUS_DIR", "/app/corpus")
INDEX_PATH = os.environ.get("INDEX_PATH", "/tmp/table_index.jsonl")
KEYWORD_INDEX_PATH = os.environ.get("KEYWORD_INDEX_PATH", "/tmp/keyword_index.txt")

_MONTH_NAMES = {
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
}


def _pub_year_month(filename):
    """Extract publication year and month from filename."""
    m = re.search(r"treasury_bulletin_(\d{4})_(\d{2})", filename)
    if m:
        return int(m.group(1)), int(m.group(2))
    return 0, 0


def _detect_months(raw_text):
    """Detect which months appear as row labels in the table."""
    found = set()
    for line in raw_text.lower().split("\n"):
        if not line.strip().startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 2:
            continue
        label = cells[1].strip()
        # Check for month names in row label
        for month in _MONTH_NAMES:
            if month in label:
                # Map to month number
                month_map = {
                    "jan": 1,
                    "january": 1,
                    "feb": 2,
                    "february": 2,
                    "mar": 3,
                    "march": 3,
                    "apr": 4,
                    "april": 4,
                    "may": 5,
                    "jun": 6,
                    "june": 6,
                    "jul": 7,
                    "july": 7,
                    "aug": 8,
                    "august": 8,
                    "sep": 9,
                    "sept": 9,
                    "september": 9,
                    "oct": 10,
                    "october": 10,
                    "nov": 11,
                    "november": 11,
                    "dec": 12,
                    "december": 12,
                }
                if month in month_map:
                    found.add(month_map[month])
    return sorted(found)


def _detect_period_basis(title, raw_text):
    """Detect if table is fiscal year, calendar year, or has monthly data."""
    combined = (title + " " + raw_text).lower()
    has_fiscal = bool(re.search(r"fiscal\s+year", combined))
    has_calendar = bool(re.search(r"calendar\s+year", combined))
    has_monthly = bool(
        re.search(
            r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan\.|feb\.|mar\.|apr\.)",
            combined,
        )
    )

    if has_calendar:
        return "calendar"
    if has_fiscal:
        return "fiscal"
    if has_monthly:
        return "monthly"
    return "unknown"


def _detect_has_annual_row(raw_text, years):
    """Check if table has annual total rows (e.g., '| 1940 | 8736 |')."""
    for year in years:
        pattern = rf"^\|\s*{year}\s*\|"
        if re.search(pattern, raw_text, re.MULTILINE):
            return True
    return False


def extract_tables(filepath):
    """Extract markdown tables with enriched metadata."""
    lines = Path(filepath).read_text(errors="replace").splitlines()
    fname = Path(filepath).name
    pub_year, pub_month = _pub_year_month(fname)

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|"):
            i += 1
            continue

        # Scan backward for title and units
        title = ""
        units = ""
        for j in range(max(0, i - 10), i):
            ctx = lines[j].strip()
            if re.search(r"\(.*(?:millions|thousands|dollars|percent|billions).*\)", ctx, re.I):
                units = ctx
            elif ctx and not ctx.startswith("|") and not ctx.startswith("---") and len(ctx) > 8:
                if not re.match(r"^\d+$", ctx.strip()) and not ctx.startswith("Source:"):
                    title = ctx

        # Collect table rows
        table_start = i
        table_lines = []
        while i < len(lines):
            row = lines[i].strip()
            if row.startswith("|") or (row.startswith("---") and i == table_start + 1):
                table_lines.append(row)
                i += 1
            elif not row:
                if i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                    i += 1
                    continue
                break
            else:
                break

        if len(table_lines) < 2:
            continue

        # Parse headers
        header_cells = [c.strip() for c in table_lines[0].split("|") if c.strip()]
        clean_headers = []
        for h in header_cells:
            parts = [p.strip() for p in h.split(">")]
            clean = h
            for p in reversed(parts):
                if "unnamed" not in p.lower() and p.strip():
                    clean = p.strip()
                    break
            clean_headers.append(clean)

        data_rows = [l for l in table_lines if not l.strip().startswith("| ---")]
        row_count = len(data_rows) - 1

        # Extract years
        years = set()
        for row in data_rows[1:]:
            for m in re.finditer(r"\b(19\d{2}|20\d{2})\b", row):
                years.add(int(m.group()))

        raw_text = "\n".join(table_lines)

        # Enriched metadata
        months_present = _detect_months(raw_text)
        period_basis = _detect_period_basis(title, raw_text)
        has_all_12 = len(months_present) >= 11  # Allow 11 for slight OCR issues
        has_annual = _detect_has_annual_row(raw_text, years)

        yield {
            "file": fname,
            "line": table_start + 1,
            "title": title[:150],
            "units": units[:80],
            "headers": clean_headers[:20],
            "years": sorted(years)[:30],
            "rows": row_count,
            "pub_year": pub_year,
            "pub_month": pub_month,
            "months": months_present,
            "has_all_12": has_all_12,
            "has_annual": has_annual,
            "period_basis": period_basis,
            "raw": raw_text,
        }


def build_keywords(table):
    """Build keyword string including enriched metadata tags."""
    text = f"{table['title']} {table['units']} {' '.join(table['headers'])}"

    # Add row labels
    for line in table["raw"].split("\n"):
        if line.startswith("|") and "---" not in line:
            cells = line.split("|")
            if len(cells) > 1:
                label = cells[1].strip().lower()
                if label and not re.match(r"^[\d.,\-+\s*naninf]+$", label):
                    text += " " + label

    # Normalize
    text = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    text = re.sub(r"\s+", " ", text).strip()

    # Deduplicate words
    seen = set()
    words = []
    for w in text.split():
        if w not in seen and len(w) > 2:
            seen.add(w)
            words.append(w)

    return " ".join(words[:100])


def main():
    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else CORPUS_DIR
    index_path = sys.argv[2] if len(sys.argv) > 2 else INDEX_PATH

    corpus = Path(corpus_dir)
    if not corpus.exists():
        print(f"Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    # Process files in REVERSE order (newest first) so dedup keeps newest
    files = sorted(corpus.glob("treasury_bulletin_*.txt"), reverse=True)
    print(f"Scanning {len(files)} files (newest first for dedup)...", file=sys.stderr)

    start = time.time()
    seen_hashes = {}  # hash -> table dict (keep newest)
    total_tables = 0
    dupes = 0

    for fpath in files:
        for table in extract_tables(fpath):
            total_tables += 1

            # Hash for dedup (normalized content)
            normalized = re.sub(r"\s+", "", table["raw"].lower())
            content_hash = hashlib.md5(normalized.encode()).hexdigest()[:12]

            if content_hash in seen_hashes:
                # Keep the one from the NEWER bulletin
                existing = seen_hashes[content_hash]
                if (table["pub_year"], table["pub_month"]) > (
                    existing["pub_year"],
                    existing["pub_month"],
                ):
                    seen_hashes[content_hash] = table  # Replace with newer
                dupes += 1
                continue

            seen_hashes[content_hash] = table

    tables = list(seen_hashes.values())
    elapsed = time.time() - start

    # Write JSONL index (without raw text to save space)
    with open(index_path, "w") as f:
        for t in tables:
            entry = {k: v for k, v in t.items() if k != "raw"}
            entry["keywords"] = build_keywords(t)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Write flat keyword index with enriched tags
    with open(KEYWORD_INDEX_PATH, "w") as f:
        for t in tables:
            kw = build_keywords(t)
            # Add metadata tags for searching
            tags = []
            if t["has_all_12"]:
                tags.append("HAS_12_MONTHS")
            if t["has_annual"]:
                tags.append("HAS_ANNUAL")
            tags.append(f"BASIS:{t['period_basis']}")
            tags.append(f"PUB:{t['pub_year']}")
            tag_str = " ".join(tags)

            f.write(f"{t['file']}:{t['line']}\t{t['title']}\t{t['units']}\t{tag_str} {kw}\n")

    idx_size = os.path.getsize(index_path) // 1024
    kw_size = os.path.getsize(KEYWORD_INDEX_PATH) // 1024

    print(f"Done in {elapsed:.1f}s", file=sys.stderr)
    print(
        f"  Files: {len(files)}, Tables: {total_tables}, Unique: {len(tables)} (deduped {dupes})",
        file=sys.stderr,
    )
    print(
        f"  Index: {index_path} ({idx_size}KB), Keywords: {KEYWORD_INDEX_PATH} ({kw_size}KB)",
        file=sys.stderr,
    )

    # Count enrichment stats
    with_12 = sum(1 for t in tables if t["has_all_12"])
    with_annual = sum(1 for t in tables if t["has_annual"])
    fiscal = sum(1 for t in tables if t["period_basis"] == "fiscal")
    calendar = sum(1 for t in tables if t["period_basis"] == "calendar")
    monthly = sum(1 for t in tables if t["period_basis"] == "monthly")

    print(f"  12-month tables: {with_12}, Annual totals: {with_annual}", file=sys.stderr)
    print(f"  Fiscal: {fiscal}, Calendar: {calendar}, Monthly: {monthly}", file=sys.stderr)

    # Agent instructions
    print("")
    print(f"INDEX READY — {len(tables)} unique tables from {len(files)} files.")
    print("")
    print("Search with: grep -i 'keyword' /tmp/keyword_index.txt | head -10")
    print("")
    print("Tags in each line: HAS_12_MONTHS, HAS_ANNUAL, BASIS:fiscal/calendar/monthly, PUB:YYYY")
    print(
        "  To find tables with monthly breakdowns: grep -i 'keyword' /tmp/keyword_index.txt | grep HAS_12_MONTHS"
    )
    print(
        "  To find newest bulletin: grep -i 'keyword' /tmp/keyword_index.txt | sort -t'PUB:' -k2 -rn"
    )
    print("")
    print("Read table: python3 /installed-agent/search.py table <FILE> <LINE>")


if __name__ == "__main__":
    main()
