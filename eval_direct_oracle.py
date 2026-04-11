#!/usr/bin/env python3
"""Direct-answer oracle eval.

Gives the LLM the question + gold tables rendered as text, and asks it to
reason and answer directly -- no decompose, no structured extraction, no
compute phase.  Tests the ceiling: can the LLM read these tables at all?

Usage:
  uv run python eval_direct_oracle.py                       # first 28 uids
  uv run python eval_direct_oracle.py --n 50
  uv run python eval_direct_oracle.py --uids UID0001,UID0042
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

sys.stdout.reconfigure(line_buffering=True)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from extract import build_context_from_entries  # noqa: E402
from retrieve_v2 import _load_element_html  # noqa: E402
from reward import score_answer  # noqa: E402

LEDGER = HERE / "ledger.sqlite"
URL_PAGE_RE = re.compile(r"[?&]page=(\d+)")

MODEL = os.getenv("OFFICEQA_MODEL", "deepseek-chat")
client = OpenAI(
    api_key=os.getenv("DEDALUS_API_KEY"),
    base_url=os.getenv("DEDALUS_API_BASE"),
)

SYSTEM_PROMPT = """You are an expert analyst answering questions about U.S. Treasury Bulletin data.
You are given one or more tables from Treasury Bulletins and a question.
Read the tables carefully and answer the question.

RULES:
- Show your reasoning step by step.
- When reading pipe-delimited tables, be very careful about which column a value belongs to.
  Count columns by their header position. Wide tables with repeated column names may represent
  different years -- use surrounding context (title, section, row patterns) to determine which
  column group corresponds to which year.
- Pay attention to units stated in the table caption/title (millions, thousands, billions, etc).
- Return the RAW numbers as printed in the table -- do not rescale unless the question explicitly
  asks for a different unit.
- For fiscal year questions: pre-1977 FY = Jul-Jun, post-1976 FY = Oct-Sep.
- For calendar year questions: sum Jan-Dec monthly rows (not the annual FY row).
- Exclude "Total" or aggregate rows when the question asks for individual items.
- When computing statistics (geometric mean, linear regression, etc), show the formula and
  intermediate values.
- CPI-U annual averages (if needed): look them up from Federal Reserve Bank of Minneapolis data.
  Common values: 1940=14.0, 1953=26.7.

After your reasoning, output your final answer on a line by itself starting with "ANSWER: "
followed by just the number(s) in the format the question requests.
Do not include units or words after ANSWER: unless the question format requires them (like [a, b, c])."""


def parse_gold_locs(row: dict) -> list[tuple[str, int]]:
    doc_lines = [ln.strip() for ln in (row.get("source_docs") or "").split("\n") if ln.strip()]
    file_lines = [ln.strip() for ln in (row.get("source_files") or "").split("\n") if ln.strip()]
    locs: list[tuple[str, int]] = []
    for url, fname in zip(doc_lines, file_lines, strict=False):
        m = URL_PAGE_RE.search(url)
        if not m:
            continue
        stem = Path(fname).stem
        locs.append((f"{stem}.json", int(m.group(1))))
    return locs


def build_oracle_entries(conn: sqlite3.Connection, gold_locs: list[tuple[str, int]]) -> list[dict]:
    entries: list[dict] = []
    for file, page_id in gold_locs:
        rows = conn.execute(
            """
            SELECT t.id, t.file, t.element_id, t.element_seq, t.page_id,
                   t.file_year, t.file_month, t.section, t.title, t.caption,
                   t.unit, t.period, t.n_rows, t.n_cols
            FROM tables t
            WHERE t.file = ? AND t.page_id = ?
              AND t.parse_ok = 1 AND t.table_kind = 'data'
            ORDER BY t.element_seq
            """,
            (file, page_id),
        ).fetchall()
        for r in rows:
            cols = conn.execute(
                "SELECT col_path FROM table_columns WHERE table_id = ? ORDER BY col_index",
                (r["id"],),
            ).fetchall()
            labels = conn.execute(
                "SELECT row_path FROM table_rows WHERE table_id = ? ORDER BY row_index LIMIT 200",
                (r["id"],),
            ).fetchall()
            yrs = conn.execute(
                """
                SELECT DISTINCT year FROM (
                    SELECT year_extracted AS year FROM table_columns WHERE table_id = ?
                    UNION
                    SELECT year_extracted AS year FROM table_rows    WHERE table_id = ?
                ) WHERE year IS NOT NULL
                """,
                (r["id"], r["id"]),
            ).fetchall()
            entries.append(
                {
                    "file": r["file"],
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
                    "years": sorted(int(y[0]) for y in yrs if y[0] is not None),
                    "unit": r["unit"],
                    "period": r["period"],
                    "n_rows": r["n_rows"],
                    "n_cols": r["n_cols"],
                    "retrieval_strategy": "oracle",
                    "retrieval_channel": "oracle",
                    "html": _load_element_html(r["file"], r["element_seq"]),
                }
            )
    return entries


def extract_answer(raw: str) -> str:
    """Pull the final answer from LLM response."""
    for line in reversed(raw.strip().split("\n")):
        line = line.strip()
        if line.upper().startswith("ANSWER:"):
            return line[7:].strip()
    return raw.strip().split("\n")[-1].strip()


_TLS = threading.local()


def _conn() -> sqlite3.Connection:
    c = getattr(_TLS, "conn", None)
    if c is None:
        c = sqlite3.connect(f"file:{LEDGER}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        _TLS.conn = c
    return c


def _process_uid(uid: str, row: dict) -> dict:
    gold_locs = parse_gold_locs(row)
    if not gold_locs:
        return {"uid": uid, "outcome": "skip_no_gold"}

    t0 = time.time()
    oracle = build_oracle_entries(_conn(), gold_locs)
    if not oracle:
        return {"uid": uid, "outcome": "skip_no_tables", "gold_locs": gold_locs}

    # Render all gold tables with a generous budget
    context = build_context_from_entries(
        oracle, char_budget=40000, vertical_threshold=999, max_rows=200
    )

    question = row["question"]
    user_msg = f"Question: {question}\n\nTables:\n{context}"

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=4000,
            temperature=0.0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
        )
        raw = resp.choices[0].message.content or ""
    except Exception as e:
        return {"uid": uid, "outcome": "llm_error", "error": str(e)}

    predicted = extract_answer(raw)
    gold = row.get("answer", "")
    score = score_answer(gold, predicted)

    return {
        "uid": uid,
        "outcome": "ok",
        "question": question,
        "gold": gold,
        "predicted": predicted,
        "score": score,
        "gold_locs": gold_locs,
        "n_oracle_tables": len(oracle),
        "elapsed_s": time.time() - t0,
        "context_chars": len(context),
        "reasoning": raw[:500],
    }


def run() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=28)
    ap.add_argument("--uids", type=str, default="")
    ap.add_argument("--csv", type=Path, default=HERE / "officeqa_full.csv")
    ap.add_argument("--out", type=Path, default=HERE / "eval_direct_oracle.jsonl")
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    with open(args.csv) as f:
        rows = {r["uid"]: r for r in csv.DictReader(f)}

    if args.uids:
        uids = [u.strip() for u in args.uids.split(",") if u.strip()]
    else:
        uids = sorted(rows.keys())[: args.n]

    print(
        f"Evaluating {len(uids)} questions in DIRECT oracle mode ({args.workers} workers)",
        flush=True,
    )
    print(f"Model: {MODEL}")
    print(f"Ledger: {LEDGER.name}\n", flush=True)

    results: list[dict] = []
    n_correct = 0
    n_wrong = 0
    n_skip = 0
    n_error = 0
    n_total = 0
    print_lock = threading.Lock()

    def _print_outcome(r: dict) -> None:
        uid = r["uid"]
        out = r["outcome"]
        if out == "ok":
            score = r["score"]
            mark = "+" if score >= 1.0 else ("-" if score == 0 else "~")
            line = (
                f"  {uid}  {mark}  gold={r['gold']!r:30}  "
                f"pred={r['predicted'][:40]!r:42}  score={score:.2f}  "
                f"{r['elapsed_s']:.1f}s  ({r['context_chars']} chars)"
            )
        elif out == "llm_error":
            line = f"  {uid}  ERR: {r.get('error', '?')[:80]}"
        elif out == "skip_no_gold":
            line = f"  {uid}  SKIP (no gold locs)"
        elif out == "skip_no_tables":
            line = f"  {uid}  SKIP (no tables in ledger)"
        else:
            line = f"  {uid}  {out}"
        with print_lock:
            print(line, flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_process_uid, uid, rows[uid]): uid for uid in uids if uid in rows}
        for fut in as_completed(futures):
            r = fut.result()
            _print_outcome(r)
            outcome = r["outcome"]
            if outcome.startswith("skip"):
                n_skip += 1
                continue
            if outcome == "llm_error":
                n_error += 1
                results.append(r)
                continue
            n_total += 1
            if r.get("score", 0) >= 1.0:
                n_correct += 1
            else:
                n_wrong += 1
            results.append(r)

    print()
    print("-" * 60)
    print(f"Direct oracle results (n={n_total})")
    print("-" * 60)
    if n_total:
        print(f"  correct:  {n_correct:3d}  ({n_correct / n_total * 100:4.1f}%)")
        print(f"  wrong:    {n_wrong:3d}  ({n_wrong / n_total * 100:4.1f}%)")
    if n_skip:
        print(f"  skipped:  {n_skip}")
    if n_error:
        print(f"  errors:   {n_error}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} rows to {args.out}")


if __name__ == "__main__":
    run()
