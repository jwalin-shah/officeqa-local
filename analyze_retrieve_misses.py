#!/usr/bin/env python3
"""Classify retrieval misses by root cause.

For every UID where the gold file doesn't appear in top-5, report:
  - is the gold *table* (file + page_id) in the top-50 at all?
  - if yes, what rank? what beat it?
  - what do the probe/title signals say about the gold?

Buckets:
  not_in_pool        → FTS+metric never retrieved any gold table. Recall
                       ceiling problem, not a reranking problem.
  ingest_title_bad   → gold table is in pool but its title/caption is
                       garbled (empty, "Table of Contents", etc.).
  decompose_row_miss → gold table in pool, but its row_leaf doesn't
                       substring-match the decompose row_hint. Decompose
                       named the row wrong.
  ranker_mis_scored  → gold in pool, row matches, title fine — it just
                       got outranked. Fixable by better reranking.

Usage:
  uv run python analyze_retrieve_misses.py
"""

from __future__ import annotations

import csv
import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from retrieve_v2 import (  # noqa: E402
    _bulk_cell_probe,
    _extract_hints,
    _ledger_conn,
    normalize_metric_slug,
    retrieve,
)

CSV = HERE / "officeqa_full.csv"
SPECS = HERE / "decompose_eval.full.jsonl"
LEDGER = HERE / "ledger.sqlite"
URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")
TOP_K = 50
MISS_K = 5  # classify anything not in top-K as a miss


def parse_gold_locs(row: dict) -> list[tuple[str, int]]:
    doc_lines = [l.strip() for l in (row.get("source_docs") or "").split("\n") if l.strip()]
    file_lines = [l.strip() for l in (row.get("source_files") or "").split("\n") if l.strip()]
    locs: list[tuple[str, int]] = []
    for url, fname in zip(doc_lines, file_lines, strict=False):
        m = URL_PAGE_RE.search(url)
        if m:
            locs.append((f"{Path(fname).stem}.json", int(m.group(1))))
    return locs


def gold_table_ids(conn: sqlite3.Connection, gold_locs: list[tuple[str, int]]) -> set[int]:
    ids: set[int] = set()
    for f, p in gold_locs:
        rows = conn.execute(
            "SELECT id FROM tables WHERE file=? AND page_id=? AND parse_ok=1 AND table_kind='data'",
            (f, p),
        ).fetchall()
        ids.update(int(r[0]) for r in rows)
    return ids


def table_title_ok(title: str | None, caption: str | None) -> bool:
    t = (title or "").strip().lower()
    c = (caption or "").strip().lower()
    if not t and not c:
        return False
    bad_titles = {"table of contents", "contents", "index"}
    if t in bad_titles:
        return False
    return not (len(t) < 4 and len(c) < 4)


def row_hint_matches(conn: sqlite3.Connection, table_id: int, row_hints: list[str]) -> bool:
    if not row_hints:
        return False
    patterns = []
    for rh in row_hints:
        norm = normalize_metric_slug(rh)
        if norm:
            patterns.append(f"%{norm}%")
    if not patterns:
        return False
    clause = " OR ".join(["metric_slug LIKE ?"] * len(patterns))
    row = conn.execute(
        f"SELECT 1 FROM table_rows WHERE table_id=? AND ({clause}) LIMIT 1",
        (table_id, *patterns),
    ).fetchone()
    return row is not None


def main():
    with open(CSV) as f:
        csv_rows = {r["uid"]: r for r in csv.DictReader(f)}

    specs: dict[str, dict] = {}
    for line in open(SPECS):
        d = json.loads(line)
        if d.get("spec") and d.get("gold_files"):
            specs[d["uid"]] = d

    conn = _ledger_conn()

    buckets: Counter[str] = Counter()
    examples: dict[str, list[str]] = {
        k: []
        for k in ("not_in_pool", "ingest_title_bad", "decompose_row_miss", "ranker_mis_scored")
    }
    total = 0
    n_hit_at_5 = 0

    for uid, d in sorted(specs.items()):
        csv_row = csv_rows.get(uid)
        if not csv_row:
            continue
        gold_locs = parse_gold_locs(csv_row)
        if not gold_locs:
            continue
        gold_tids = gold_table_ids(conn, gold_locs)
        if not gold_tids:
            continue
        total += 1

        spec, q = d["spec"], d["question"]
        try:
            results = retrieve(spec, q, top_k=TOP_K, load_html=False)
        except Exception:
            buckets["error"] += 1
            continue

        {Path(r["file"]).stem for r in results}
        gold_stems = {Path(f).stem for f in d.get("gold_files", [])}
        first_file_rank = next(
            (i + 1 for i, r in enumerate(results) if Path(r["file"]).stem in gold_stems),
            None,
        )
        if first_file_rank and first_file_rank <= MISS_K:
            n_hit_at_5 += 1
            continue

        # Miss at @5. Classify.
        hints = _extract_hints(spec, q)
        row_hints = [h["row_hint"] for h in hints if h["row_hint"]]

        # Is any gold table in the top-50 result set?
        result_tids: list[int] = []
        for r in results:
            r_row = conn.execute(
                "SELECT id FROM tables WHERE file=? AND element_seq=? LIMIT 1",
                (r["file"], r.get("element_seq")),
            ).fetchone()
            if r_row:
                result_tids.append(int(r_row[0]))
        gold_in_result = gold_tids & set(result_tids)

        if not gold_in_result:
            # Not even in top-50 after rerank. Check if FTS channel or metric
            # channel ever saw any gold table at all.
            probe = _bulk_cell_probe(conn, gold_tids, row_hints)
            any_row_match = any(v[0] > 0 for v in probe.values())
            bucket = "not_in_pool"
            note = (
                f"gold={list(gold_tids)[:3]}  row_match={any_row_match}"
                f"  top5={[Path(r['file']).stem for r in results[:5]]}"
            )
        else:
            # In pool. Check the best gold entry's rank and diagnose why.
            best_rank = min(i + 1 for i, tid in enumerate(result_tids) if tid in gold_tids)
            best_gold_tid = next(tid for tid in result_tids if tid in gold_tids)
            t_row = conn.execute(
                "SELECT file, page_id, title, caption FROM tables WHERE id=?",
                (best_gold_tid,),
            ).fetchone()
            title_ok = table_title_ok(t_row["title"], t_row["caption"])
            row_ok = row_hint_matches(conn, best_gold_tid, row_hints)
            if not title_ok:
                bucket = "ingest_title_bad"
            elif not row_ok:
                bucket = "decompose_row_miss"
            else:
                bucket = "ranker_mis_scored"
            note = (
                f"rank={best_rank}  gold_tid={best_gold_tid}  "
                f"title={t_row['title']!r:35}  row_ok={row_ok}  title_ok={title_ok}"
            )

        buckets[bucket] += 1
        if len(examples[bucket]) < 5:
            examples[bucket].append(f"{uid}  {note}")

    print()
    print(f"Total classified: {total}")
    print(f"@{MISS_K} hits:        {n_hit_at_5}  ({n_hit_at_5 / total * 100:.1f}%)")
    print(
        f"@{MISS_K} misses:      {total - n_hit_at_5}  ({(total - n_hit_at_5) / total * 100:.1f}%)"
    )
    print()
    print("Miss buckets:")
    for bucket, count in buckets.most_common():
        print(f"  {bucket:22s}  {count:3d}")
    print()
    print("Examples per bucket:")
    for bucket, exs in examples.items():
        if not exs:
            continue
        print(f"\n  [{bucket}]")
        for ex in exs:
            print(f"    {ex}")


if __name__ == "__main__":
    main()
