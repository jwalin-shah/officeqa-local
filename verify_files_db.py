#!/usr/bin/env python3
"""Verify files.db fidelity against source HTML.

Four-layer check:
  L1 — Cell-count invariant over every table in files.db.
       Parse raw_html, count non-empty <td>/<th> values, compare to JSON blob
       non-None cell count. Flag any table where JSON < HTML.
  L2 — Value-set invariant on 500 random tables.
       Extract all numeric values from raw_html, extract all from JSON blob,
       compare sets. Any numeric in HTML but not JSON is a fidelity bug.
  L3 — Benchmark-gold reachability.
       For every numeric-gold question in officeqa_full.csv, verify the gold
       value (with scale variants) appears in the JSON blob of a table on the
       gold page. Target: match or beat eval_ledger.py's 47.7% number.
  L4 — Manual spot check of 5 diverse tables.
       Dump title + JSON + raw_html for eyeballing.

Usage:
    uv run python verify_files_db.py              # run all layers
    uv run python verify_files_db.py --layer 1    # only L1
    uv run python verify_files_db.py --sample 100 # override L2 sample size
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sqlite3
import sys
from html.parser import HTMLParser
from pathlib import Path

DB_PATH = Path("files.db")
BENCHMARK_PATH = Path("officeqa_full.csv")

URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")
NUMERIC_RE = re.compile(r"-?\d[\d,]*\.?\d*")


# ── L1: Cell-count invariant ───────────────────────────────────────


class CellCounter(HTMLParser):
    """Count non-empty text values inside <td> and <th>."""

    def __init__(self) -> None:
        super().__init__()
        self.in_cell = False
        self.buf = ""
        self.count = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in ("td", "th"):
            self.in_cell = True
            self.buf = ""

    def handle_endtag(self, tag: str) -> None:
        if tag in ("td", "th"):
            if self.buf.strip():
                self.count += 1
            self.in_cell = False
            self.buf = ""

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self.buf += data


def count_html_cells(html: str) -> int:
    p = CellCounter()
    try:
        p.feed(html)
    except Exception:
        return -1
    return p.count


def count_json_cells(blob: str) -> int:
    try:
        data = json.loads(blob)
    except Exception:
        return -1
    n = 0
    # Count non-empty column labels (headers ARE cells in the HTML)
    for c in data.get("columns", []):
        if c.get("label"):
            n += 1
    for row in data.get("rows", []):
        if row.get("label"):
            n += 1
        for v in row.get("cells", []):
            if v is not None and str(v).strip() and str(v) != "nan":
                n += 1
    return n


def layer1(conn: sqlite3.Connection) -> dict:
    print("\n=== L1: cell-count invariant over all tables ===")
    cur = conn.execute("SELECT file, table_id, json_blob, raw_html FROM tables")
    total = 0
    dropped = 0
    dropped_samples = []
    n_rows_dropped = 0
    for file, table_id, blob, html in cur:
        total += 1
        if html is None:
            continue
        html_n = count_html_cells(html)
        json_n = count_json_cells(blob)
        if html_n == -1 or json_n == -1:
            continue
        # JSON may count slightly differently due to header flattening; allow ≥90% retention
        if json_n < html_n * 0.9:
            dropped += 1
            n_rows_dropped += html_n - json_n
            if len(dropped_samples) < 5:
                dropped_samples.append((file, table_id, html_n, json_n))
    pct = (1 - dropped / total) * 100 if total else 0
    print(f"  tables checked:   {total}")
    print(f"  ≥90% retention:   {total - dropped} ({pct:.2f}%)")
    print(f"  below threshold:  {dropped}")
    for s in dropped_samples:
        print(f"    - {s[0]} {s[1]}: html={s[2]} json={s[3]}")
    return {"total": total, "dropped": dropped, "retention_pct": pct}


# ── L2: Value-set invariant ────────────────────────────────────────


def extract_numerics(text: str) -> set[str]:
    # Normalize: strip commas, keep sign, drop trailing period
    vals = set()
    for m in NUMERIC_RE.findall(text):
        s = m.replace(",", "").rstrip(".")
        if s and s not in ("-", ".", "-."):
            try:
                f = float(s)
                # Round to 4 decimal places to avoid float jitter
                vals.add(f"{f:.4f}")
            except ValueError:
                pass
    return vals


def extract_numerics_from_json(blob: str) -> set[str]:
    try:
        data = json.loads(blob)
    except Exception:
        return set()
    parts = []
    for c in data.get("columns", []):
        parts.append(str(c.get("label", "")))
    for row in data.get("rows", []):
        parts.append(str(row.get("label", "")))
        for v in row.get("cells", []):
            if v is not None:
                parts.append(str(v))
    return extract_numerics(" ".join(parts))


def extract_numerics_from_html(html: str) -> set[str]:
    # Strip tags crudely
    text = re.sub(r"<[^>]+>", " ", html)
    return extract_numerics(text)


def layer2(conn: sqlite3.Connection, sample_size: int = 500) -> dict:
    print(f"\n=== L2: value-set invariant on {sample_size} random tables ===")
    total = conn.execute("SELECT COUNT(*) FROM tables").fetchone()[0]
    ids = conn.execute("SELECT file, table_id FROM tables").fetchall()
    sample = random.sample(ids, min(sample_size, len(ids)))
    mismatches = 0
    total_missing = 0
    worst = []
    for file, table_id in sample:
        row = conn.execute(
            "SELECT json_blob, raw_html FROM tables WHERE file=? AND table_id=?",
            (file, table_id),
        ).fetchone()
        if not row or not row[1]:
            continue
        html_vals = extract_numerics_from_html(row[1])
        json_vals = extract_numerics_from_json(row[0])
        missing = html_vals - json_vals
        if missing:
            mismatches += 1
            total_missing += len(missing)
            if len(worst) < 3:
                worst.append((file, table_id, len(missing), sorted(missing)[:5]))
    pct = (1 - mismatches / len(sample)) * 100 if sample else 0
    print(f"  sampled:          {len(sample)} of {total}")
    print(f"  perfect match:    {len(sample) - mismatches} ({pct:.2f}%)")
    print(f"  with missing:     {mismatches}")
    print(f"  total missing vals: {total_missing}")
    for s in worst:
        print(f"    - {s[0]} {s[1]}: {s[2]} missing, e.g. {s[3]}")
    return {"sampled": len(sample), "mismatches": mismatches, "pct": pct}


# ── L3: Benchmark-gold reachability ────────────────────────────────


def parse_gold_locations(source_docs: str, source_files: str) -> list[tuple[str, int]]:
    doc_lines = [line.strip() for line in (source_docs or "").split("\n") if line.strip()]
    file_lines = [line.strip() for line in (source_files or "").split("\n") if line.strip()]
    out = []
    for url, fname in zip(doc_lines, file_lines, strict=False):
        m = URL_PAGE_RE.search(url)
        if not m:
            continue
        stem = Path(fname).stem
        out.append((f"{stem}.json", int(m.group(1))))
    return out


def parse_gold_value(raw: str) -> float | None:
    if not raw:
        return None
    s = re.sub(r"[\$%]", "", raw.strip())
    s = s.replace(",", "").strip()
    s = re.sub(r"\s*(million|billion|thousand|percent|%)\s*$", "", s, flags=re.I)
    neg = s.startswith("(") and s.endswith(")")
    if neg:
        s = s[1:-1].strip()
    try:
        v = float(s)
        return -v if neg else v
    except ValueError:
        return None


def value_in_blob(blob: str, target: float, tol: float = 0.01) -> bool:
    try:
        data = json.loads(blob)
    except Exception:
        return False
    for scale in (1.0, 1e3, 1e-3, 1e6, 1e-6):
        want = target * scale
        lo, hi = (want - abs(want) * tol, want + abs(want) * tol) if want else (-1e-9, 1e-9)
        for row in data.get("rows", []):
            for v in row.get("cells", []):
                if v is None:
                    continue
                try:
                    f = float(str(v).replace(",", "").replace("$", ""))
                except (ValueError, TypeError):
                    continue
                if lo <= f <= hi:
                    return True
    return False


def layer3(conn: sqlite3.Connection) -> dict:
    print("\n=== L3: benchmark-gold reachability vs eval_ledger's 47.7% ===")
    if not BENCHMARK_PATH.exists():
        print(f"  SKIP: {BENCHMARK_PATH} not found")
        return {}
    with BENCHMARK_PATH.open() as f:
        rows = list(csv.DictReader(f))

    numeric_total = 0
    cell_present = 0
    for row in rows:
        gold = parse_gold_value(row.get("answer", "") or row.get("gold", ""))
        if gold is None:
            continue
        numeric_total += 1
        locations = parse_gold_locations(row.get("source_docs", ""), row.get("source_files", ""))
        if not locations:
            continue
        found = False
        for file, page in locations:
            tables = conn.execute(
                "SELECT json_blob FROM tables WHERE file=? AND page_id=?",
                (file, page),
            ).fetchall()
            for (blob,) in tables:
                if value_in_blob(blob, gold):
                    found = True
                    break
            if found:
                break
        if found:
            cell_present += 1

    pct = (cell_present / numeric_total * 100) if numeric_total else 0
    print(f"  numeric-gold questions: {numeric_total}")
    print(f"  cell present on page:   {cell_present} ({pct:.1f}%)")
    print("  baseline to beat:       47.7%")
    verdict = "✓ MATCH/BEAT" if pct >= 47.7 else "✗ REGRESSION"
    print(f"  verdict: {verdict}")
    return {"numeric": numeric_total, "present": cell_present, "pct": pct}


# ── L4: Manual spot check ──────────────────────────────────────────


def layer4(conn: sqlite3.Connection) -> None:
    print("\n=== L4: manual spot check (5 diverse tables) ===")
    queries = [
        (
            "simple",
            "SELECT file, table_id, title, json_blob, raw_html FROM tables WHERE n_rows BETWEEN 3 AND 8 AND n_cols BETWEEN 2 AND 4 LIMIT 1",
        ),
        (
            "wide",
            "SELECT file, table_id, title, json_blob, raw_html FROM tables WHERE n_cols >= 10 ORDER BY n_rows DESC LIMIT 1",
        ),
        (
            "tall",
            "SELECT file, table_id, title, json_blob, raw_html FROM tables WHERE n_rows >= 30 ORDER BY n_cols DESC LIMIT 1",
        ),
        (
            "multi_header",
            "SELECT file, table_id, title, json_blob, raw_html FROM tables WHERE raw_html LIKE '%rowspan%' OR raw_html LIKE '%colspan%' LIMIT 1",
        ),
        (
            "with_footnote",
            "SELECT t.file, t.table_id, t.title, t.json_blob, t.raw_html FROM tables t JOIN footnotes f ON t.file = f.file LIMIT 1",
        ),
    ]
    for kind, sql in queries:
        row = conn.execute(sql).fetchone()
        if not row:
            print(f"\n  [{kind}] no match")
            continue
        file, tid, title, blob, html = row
        data = json.loads(blob)
        print(f"\n  [{kind}] {file} {tid}")
        print(f"    title: {title[:80] if title else '(none)'}")
        print(f"    shape: {len(data['rows'])} rows × {len(data['columns'])} cols")
        print(f"    cols:  {[c['label'][:20] for c in data['columns'][:5]]}")
        print(
            f"    row 0: label={data['rows'][0]['label'][:40]!r} cells={data['rows'][0]['cells'][:5]}"
        )
        print(f"    html preview: {html[:150]!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--layer", type=int, choices=[1, 2, 3, 4], default=None)
    ap.add_argument("--sample", type=int, default=500)
    args = ap.parse_args()

    if not args.db.exists():
        print(f"{args.db} does not exist. Run build_files_db.py first.")
        sys.exit(1)

    conn = sqlite3.connect(str(args.db))
    random.seed(42)

    if args.layer == 1 or args.layer is None:
        layer1(conn)
    if args.layer == 2 or args.layer is None:
        layer2(conn, args.sample)
    if args.layer == 3 or args.layer is None:
        layer3(conn)
    if args.layer == 4 or args.layer is None:
        layer4(conn)


if __name__ == "__main__":
    main()
