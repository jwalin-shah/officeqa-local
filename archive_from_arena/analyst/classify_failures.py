#!/usr/bin/env python3
"""Classify failures from a JSONL results file into actionable categories.

Usage:
    python scripts/classify_failures.py [results_file.jsonl]

If no file is provided, defaults to the most recently modified .jsonl in results/.
"""

import argparse
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
OUTPUT_PATH = RESULTS_DIR / "failure_analysis.json"

TIMEOUT_SEC = 550

# Search-type tool names (same set as analyze_telemetry.py)
SEARCH_TOOLS = {
    "search_canonical",
    "search_tables",
    "extract_values",
    "search_ledger",
    "grep_corpus",
    "get_time_series",
    "get_multi_year_series",
}

# Phrases that indicate the agent gave up / had no answer
NO_ANSWER_PHRASES = {
    "",
    "none",
    "null",
    "n/a",
    "na",
    "unknown",
    "unable",
    "unable to determine",
    "unable to find",
    "not found",
    "no data",
    "no answer",
}

# Unit-scale ratios that indicate a scaling error
UNIT_SCALE_RATIOS = {1e3, 1e-3, 1e6, 1e-6}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def find_latest_jsonl() -> Path:
    """Return the most recently modified .jsonl file in RESULTS_DIR."""
    candidates = [
        p
        for p in RESULTS_DIR.glob("*.jsonl")
        # Skip leaderboard/telemetry files that don't have uid/gold/predicted rows
        if not any(kw in p.name for kw in ("telemetry", "leaderboard", "poll"))
    ]
    if not candidates:
        raise FileNotFoundError(f"No .jsonl result files found in {RESULTS_DIR}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                rows.append(obj)
            except json.JSONDecodeError as exc:
                print(f"  [warn] line {lineno}: JSON parse error – {exc}", file=sys.stderr)
    return rows


def is_task_row(row: dict) -> bool:
    """True if the row looks like a per-task result (has uid + gold + predicted)."""
    return "uid" in row and "gold" in row and "predicted" in row


def normalise_predicted(raw) -> str:
    """Coerce predicted to a stripped string."""
    if raw is None:
        return ""
    return str(raw).strip()


def is_no_answer(predicted: str) -> bool:
    low = predicted.lower().strip()
    if low in NO_ANSWER_PHRASES:
        return True
    # Partial prefix matches
    for phrase in (
        "unable",
        "no data",
        "not found",
        "cannot determine",
        "i don't",
        "i cannot",
        "n/a",
    ):
        if low.startswith(phrase):
            return True
    return False


def extract_float(s: str):
    """Try to parse a numeric value from a string, stripping commas and % signs."""
    if not s:
        return None
    cleaned = s.replace(",", "").replace("%", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return None


def is_unit_error(predicted: str, gold: str) -> bool:
    """Return True if predicted is exactly a known unit-scale multiple of gold."""
    p = extract_float(predicted)
    g = extract_float(gold)
    if p is None or g is None or g == 0:
        return False
    ratio = p / g
    for scale in UNIT_SCALE_RATIOS:
        if math.isclose(ratio, scale, rel_tol=0.01):
            return True
    return False


def get_elapsed(row: dict):
    """Return elapsed seconds, trying multiple field names."""
    for key in ("elapsed_sec", "elapsed_s", "elapsed"):
        val = row.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


def get_tool_calls(row: dict):
    """Return total tool call count from results row."""
    for key in ("tool_calls", "iterations", "total_tool_calls"):
        val = row.get(key)
        if val is not None:
            try:
                return int(val)
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Trajectory analysis
# ---------------------------------------------------------------------------


def load_trajectory(path: str) -> list[dict]:
    """Load step list from a trajectory JSON file. Returns [] on any error."""
    if not path or not os.path.isfile(path):
        return []
    try:
        with open(path) as f:
            data = json.load(f)
        return data.get("steps", [])
    except Exception:
        return []


def analyse_trajectory(steps: list[dict]) -> dict:
    """Extract signal from trajectory steps for classification."""
    info = {
        "tool_calls": [],  # list of function names called
        "error_contents": [],  # tool result content that contained errors
        "search_call_count": 0,
        "has_db_error": False,
        "intermediate_values": [],  # numeric values seen in tool results
    }

    db_error_pattern = re.compile(
        r"(error|not found|no results|exception|traceback|sqlite|db error|"
        r"database error|no matching|zero rows|empty result)",
        re.IGNORECASE,
    )

    for step in steps:
        if step.get("source") != "agent":
            continue

        tool_calls_in_step = step.get("tool_calls", [])
        obs = step.get("observation", {})
        results = obs.get("results", []) if obs else []

        for tc in tool_calls_in_step:
            fn = tc.get("function_name", tc.get("name", ""))
            info["tool_calls"].append(fn)
            if fn in SEARCH_TOOLS:
                info["search_call_count"] += 1

        for r in results:
            content = r.get("content", "")
            if db_error_pattern.search(content):
                info["has_db_error"] = True
                info["error_contents"].append(content[:200])
            # Extract any numbers that might be intermediate values
            for m in re.findall(r"\b\d[\d,]*\.?\d*\b", content):
                val = extract_float(m)
                if val is not None and val != 0:
                    info["intermediate_values"].append(val)

    return info


def intermediate_close_to_gold(
    intermediates: list[float], gold: str, rel_tol: float = 0.05
) -> bool:
    """True if any intermediate value is within rel_tol of gold (computation error hint)."""
    g = extract_float(gold)
    if g is None or g == 0:
        return False
    for v in intermediates:
        if math.isclose(v, g, rel_tol=rel_tol):
            return True
    return False


# ---------------------------------------------------------------------------
# Main classification
# ---------------------------------------------------------------------------


def classify_task(row: dict, traj_steps: list[dict]) -> str:
    """
    Assign one failure category to a task row.
    Returns category string.
    """
    predicted = normalise_predicted(row.get("predicted"))
    gold = str(row.get("gold", "")).strip()
    elapsed = get_elapsed(row)
    tool_call_count = get_tool_calls(row)

    # --- timeout ---
    if elapsed is not None and elapsed > TIMEOUT_SEC:
        return "timeout"
    if predicted == "" and elapsed is not None and elapsed > TIMEOUT_SEC * 0.9:
        return "timeout"

    # --- no_answer ---
    if is_no_answer(predicted):
        return "no_answer"

    # --- unit_error ---
    if is_unit_error(predicted, gold):
        return "unit_error"

    # Trajectory-based signals (may not always be available)
    traj_info = analyse_trajectory(traj_steps) if traj_steps else {}

    # --- db_error ---
    if traj_info.get("has_db_error"):
        return "db_error"

    # --- search_exhaustion ---
    # Either from the results row tool_calls field, or from counting traj tool calls
    search_count = traj_info.get("search_call_count", 0)
    effective_calls = tool_call_count or len(traj_info.get("tool_calls", []))
    if effective_calls >= 15 and search_count >= effective_calls * 0.4:
        return "search_exhaustion"
    if tool_call_count is not None and tool_call_count >= 15 and not traj_steps:
        # No trajectory available; just use the count
        return "search_exhaustion"

    # --- computation_error ---
    if traj_steps and intermediate_close_to_gold(traj_info.get("intermediate_values", []), gold):
        return "computation_error"

    # --- wrong_evidence (default for numeric mismatches) ---
    return "wrong_evidence"


def classify_all(rows: list[dict]) -> list[dict]:
    """Classify every failed task row. Return list of result dicts."""
    classified = []

    for row in rows:
        if not is_task_row(row):
            continue

        uid = row.get("uid", "?")
        predicted = normalise_predicted(row.get("predicted"))
        gold = str(row.get("gold", "")).strip()

        # Determine if task passed
        passed = predicted == gold
        if passed:
            classified.append(
                {
                    "uid": uid,
                    "status": "pass",
                    "category": None,
                    "predicted": predicted,
                    "gold": gold,
                }
            )
            continue

        # Load trajectory if available
        traj_path = row.get("trajectory_path", "")
        traj_steps = load_trajectory(traj_path)

        category = classify_task(row, traj_steps)

        detail = {
            "uid": uid,
            "status": "fail",
            "category": category,
            "predicted": predicted,
            "gold": gold,
            "elapsed_sec": get_elapsed(row),
            "tool_calls": get_tool_calls(row),
            "question_snippet": str(row.get("question", ""))[:120],
        }
        classified.append(detail)

    return classified


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------


def truncate_uid_list(uids: list[str], max_inline: int = 6) -> str:
    shown = uids[:max_inline]
    rest = len(uids) - max_inline
    s = ", ".join(shown)
    if rest > 0:
        s += f" (+{rest} more)"
    return s


def print_summary(classified: list[dict], source_path: Path) -> None:
    failures = [r for r in classified if r["status"] == "fail"]
    total = len(classified)
    n_fail = len(failures)

    # Bucket by category
    buckets: dict[str, list[str]] = defaultdict(list)
    for r in failures:
        buckets[r["category"]].append(r["uid"])

    # Category display order
    order = [
        "timeout",
        "no_answer",
        "unit_error",
        "db_error",
        "search_exhaustion",
        "computation_error",
        "wrong_evidence",
        "unknown",
    ]
    # Include any unexpected categories at end
    for cat in buckets:
        if cat not in order:
            order.append(cat)

    print()
    print(f"Failure Classification Summary  (N={n_fail} failures out of {total} total)")
    print(f"Source: {source_path}")
    print("\u2500" * 80)

    cat_col = 22
    cnt_col = 7
    pct_col = 7
    print(f"{'Category':<{cat_col}} {'Count':>{cnt_col}} {'%':>{pct_col}}   UIDs")
    print("\u2500" * 80)

    for cat in order:
        uids = buckets.get(cat, [])
        if not uids:
            continue
        count = len(uids)
        pct = count / max(n_fail, 1) * 100
        uid_str = truncate_uid_list(uids)
        print(f"{cat:<{cat_col}} {count:>{cnt_col}} {pct:>{6}.1f}%   {uid_str}")

    print("\u2500" * 80)
    pass_count = total - n_fail
    print(
        f"{'PASS':<{cat_col}} {pass_count:>{cnt_col}} {pass_count / max(total, 1) * 100:>{6}.1f}%"
    )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Classify failures in a JSONL results file.")
    parser.add_argument(
        "results_file",
        nargs="?",
        default=None,
        help="Path to .jsonl results file. Defaults to most recent in results/.",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=str(OUTPUT_PATH),
        help=f"Path for detailed JSON output (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    # Resolve input file
    if args.results_file:
        source_path = Path(args.results_file).resolve()
    else:
        source_path = find_latest_jsonl()
        print(f"No file specified — using latest: {source_path.name}")

    if not source_path.exists():
        print(f"Error: file not found: {source_path}", file=sys.stderr)
        sys.exit(1)

    # Load
    rows = load_jsonl(source_path)
    task_rows = [r for r in rows if is_task_row(r)]
    print(f"Loaded {len(rows)} lines ({len(task_rows)} task rows) from {source_path.name}")

    if not task_rows:
        print("No task rows (uid+gold+predicted) found. Is this the right file?", file=sys.stderr)
        sys.exit(1)

    # Classify
    classified = classify_all(task_rows)

    # Print summary table
    print_summary(classified, source_path)

    # Write detailed JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "source": str(source_path),
                "total": len(classified),
                "failures": len([r for r in classified if r["status"] == "fail"]),
                "tasks": classified,
            },
            f,
            indent=2,
        )
    print(f"Detailed JSON written to: {output_path}")


if __name__ == "__main__":
    main()
