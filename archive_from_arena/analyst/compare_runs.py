#!/usr/bin/env python3
"""Compare two arena test runs and show what changed.

Usage:
    python3 scripts/compare_runs.py <run_a> <run_b>
    python3 scripts/compare_runs.py --list
    python3 scripts/compare_runs.py <run_a> <run_b> --csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POOL_ROOT = ROOT / "results" / "runner_pool"
ARENA_RUNS = ROOT / ".arena" / "runs"

RESULT_RE = re.compile(r"\s+(PASS|FAIL)\s+(officeqa-uid\d+)\s+\(reward=([0-9.]+)\)")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def load_json_safe(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Task record: normalised shape used throughout this script
#   task_id   : str
#   passed    : bool | None  (None = unknown / missing)
#   reward    : float | None
#   cost_usd  : float | None
#   elapsed_s : float | None
# ---------------------------------------------------------------------------


def _make_record(
    task_id: str,
    passed: bool | None,
    reward: float | None,
    cost_usd: float | None,
    elapsed_s: float | None,
) -> dict:
    return dict(
        task_id=task_id,
        passed=passed,
        reward=reward,
        cost_usd=cost_usd,
        elapsed_s=elapsed_s,
    )


# ---------------------------------------------------------------------------
# Pool run loader
# ---------------------------------------------------------------------------


def load_pool_run(label: str) -> tuple[dict, dict[str, dict]]:
    """Return (metadata_dict, task_map) for a pool run.

    Fast path: read summary.json if it already exists (written by
    summarize_pool_run.py).  Fall back to scanning lane*.jsonl files and
    lane*.log files directly so the script works without a pre-built summary.
    """
    run_root = POOL_ROOT / label

    metadata = {
        "label": label,
        "source": "pool",
        "git_sha": None,
        "tag": None,
    }

    # Read metadata.json if present
    meta = load_json_safe(run_root / "metadata.json")
    if meta:
        metadata["git_sha"] = meta.get("git_sha") or meta.get("git")
        metadata["tag"] = meta.get("tag")

    # Fast path via existing summary.json
    summary = load_json_safe(run_root / "summary.json")
    if summary and summary.get("tasks"):
        tasks: dict[str, dict] = {}
        for t in summary["tasks"]:
            tid = t.get("task_id", "")
            if not tid:
                continue
            status = t.get("status")
            passed: bool | None = None
            if status == "passed":
                passed = True
            elif status == "failed":
                passed = False
            tasks[tid] = _make_record(
                task_id=tid,
                passed=passed,
                reward=t.get("reward"),
                cost_usd=t.get("cost_usd"),
                elapsed_s=t.get("elapsed_s"),
            )
        if not metadata["tag"] and summary.get("tasks"):
            first_tag = summary["tasks"][0].get("tag") or ""
            if first_tag:
                metadata["tag"] = first_tag
        return metadata, tasks

    # Slow path: scan lane*.jsonl + lane*.log directly
    tasks = {}

    for lane_jsonl in run_root.glob("*/results/lane*.jsonl"):
        for row in load_jsonl(lane_jsonl):
            tid = row.get("task_id", "")
            if not tid:
                continue
            tasks[tid] = _make_record(
                task_id=tid,
                passed=None,  # exit_code 0 is not necessarily pass
                reward=None,
                cost_usd=row.get("cost_usd"),
                elapsed_s=row.get("elapsed_s"),
            )
            if not metadata["tag"] and row.get("tag"):
                metadata["tag"] = row["tag"]

    for lane_log in run_root.glob("*/logs/lane*.log"):
        text = lane_log.read_text(errors="replace")
        for m in RESULT_RE.finditer(text):
            status_str, tid, reward_str = m.groups()
            if tid not in tasks:
                tasks[tid] = _make_record(
                    task_id=tid,
                    passed=None,
                    reward=None,
                    cost_usd=None,
                    elapsed_s=None,
                )
            tasks[tid]["passed"] = status_str == "PASS"
            tasks[tid]["reward"] = float(reward_str)

    return metadata, tasks


# ---------------------------------------------------------------------------
# Arena run loader
# ---------------------------------------------------------------------------


def load_arena_run(run_id: str) -> tuple[dict, dict[str, dict]]:
    """Return (metadata_dict, task_map) for an arena run."""
    run_root = ARENA_RUNS / run_id

    metadata: dict = {
        "label": run_id,
        "source": "arena",
        "git_sha": None,
        "tag": None,
    }

    meta = load_json_safe(run_root / "meta.json")
    if meta:
        metadata["tag"] = meta.get("tag")
        cfg = meta.get("config_snapshot", {})
        metadata["git_sha"] = cfg.get("git_sha") or cfg.get("git")

    results = load_json_safe(run_root / "results.json")
    tasks: dict[str, dict] = {}

    if results and results.get("tasks"):
        for t in results["tasks"]:
            tid = t.get("task_id", "")
            if not tid:
                continue
            reward = t.get("reward")
            passed: bool | None = None
            if reward is not None:
                passed = float(reward) >= 1.0
            status = t.get("status")
            if status == "passed":
                passed = True
            elif status == "failed":
                passed = False
            tasks[tid] = _make_record(
                task_id=tid,
                passed=passed,
                reward=reward,
                cost_usd=t.get("cost_usd"),
                elapsed_s=t.get("latency_sec"),
            )

    # Also scan per-task result.json files for completeness
    for task_dir in run_root.iterdir():
        if not task_dir.is_dir():
            continue
        result_json = task_dir / "result.json"
        if not result_json.exists():
            continue
        r = load_json_safe(result_json)
        if not r:
            continue
        # Infer task_id from directory name (format: officeqa-uidXXXX__suffix)
        dirname = task_dir.name
        m = re.match(r"(officeqa-uid\d+)", dirname)
        if not m:
            continue
        tid = m.group(1)
        if tid in tasks:
            continue  # results.json already has it
        reward = r.get("reward")
        passed = None
        if reward is not None:
            passed = float(reward) >= 1.0
        tasks[tid] = _make_record(
            task_id=tid,
            passed=passed,
            reward=reward,
            cost_usd=r.get("cost_usd"),
            elapsed_s=r.get("latency_sec"),
        )

    return metadata, tasks


# ---------------------------------------------------------------------------
# Run discovery
# ---------------------------------------------------------------------------


def list_available_runs() -> list[tuple[str, str]]:
    """Return list of (label, source) tuples sorted by label."""
    runs: list[tuple[str, str]] = []
    if POOL_ROOT.exists():
        for d in sorted(POOL_ROOT.iterdir()):
            if d.is_dir():
                runs.append((d.name, "pool"))
    if ARENA_RUNS.exists():
        for d in sorted(ARENA_RUNS.iterdir()):
            if d.is_dir():
                runs.append((d.name, "arena"))
    return runs


def resolve_run(spec: str) -> tuple[str, str]:
    """Resolve a run specifier (label or substring) to (label, source).

    Returns (label, source) or raises SystemExit if ambiguous / not found.
    """
    # Exact match first
    if (POOL_ROOT / spec).is_dir():
        return spec, "pool"
    if (ARENA_RUNS / spec).is_dir():
        return spec, "arena"

    # Substring match
    matches: list[tuple[str, str]] = []
    if POOL_ROOT.exists():
        for d in POOL_ROOT.iterdir():
            if d.is_dir() and spec in d.name:
                matches.append((d.name, "pool"))
    if ARENA_RUNS.exists():
        for d in ARENA_RUNS.iterdir():
            if d.is_dir() and spec in d.name:
                matches.append((d.name, "arena"))

    if not matches:
        sys.exit(f"ERROR: No run found matching '{spec}'")
    if len(matches) > 1:
        labels = [f"  {src}: {lbl}" for lbl, src in sorted(matches)]
        sys.exit(
            f"ERROR: '{spec}' is ambiguous – matches:\n"
            + "\n".join(labels)
            + "\n\nPlease use a more specific label."
        )
    return matches[0]


def load_run(spec: str) -> tuple[dict, dict[str, dict]]:
    label, source = resolve_run(spec)
    if source == "pool":
        return load_pool_run(label)
    return load_arena_run(label)


# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------


def compare_runs(
    meta_a: dict,
    tasks_a: dict[str, dict],
    meta_b: dict,
    tasks_b: dict[str, dict],
) -> dict:
    all_ids = sorted(set(tasks_a) | set(tasks_b))

    gained: list[str] = []  # fail→pass
    lost: list[str] = []  # pass→fail
    cost_changes: list[dict] = []
    only_in_a: list[str] = []
    only_in_b: list[str] = []

    for tid in all_ids:
        in_a = tid in tasks_a
        in_b = tid in tasks_b
        if in_a and not in_b:
            only_in_a.append(tid)
            continue
        if in_b and not in_a:
            only_in_b.append(tid)
            continue
        ta = tasks_a[tid]
        tb = tasks_b[tid]
        pa, pb = ta["passed"], tb["passed"]
        if pa is False and pb is True:
            gained.append(tid)
        elif pa is True and pb is False:
            lost.append(tid)

        ca = ta.get("cost_usd")
        cb = tb.get("cost_usd")
        if ca is not None and cb is not None and ca > 0:
            delta_pct = (cb - ca) / ca * 100
            if abs(delta_pct) >= 50:
                cost_changes.append(dict(task_id=tid, cost_a=ca, cost_b=cb, delta_pct=delta_pct))

    cost_changes.sort(key=lambda x: abs(x["delta_pct"]), reverse=True)

    def _count(tasks: dict, passed_val: bool) -> int:
        return sum(1 for t in tasks.values() if t["passed"] is passed_val)

    def _total_cost(tasks: dict) -> float | None:
        vals = [t["cost_usd"] for t in tasks.values() if t["cost_usd"] is not None]
        return sum(vals) if vals else None

    def _avg_elapsed(tasks: dict) -> float | None:
        vals = [t["elapsed_s"] for t in tasks.values() if t["elapsed_s"] is not None]
        return sum(vals) / len(vals) if vals else None

    passed_a = _count(tasks_a, True)
    passed_b = _count(tasks_b, True)
    failed_a = _count(tasks_a, False)
    failed_b = _count(tasks_b, False)
    total_a = len(tasks_a)
    total_b = len(tasks_b)
    acc_a = passed_a / total_a if total_a else None
    acc_b = passed_b / total_b if total_b else None
    cost_a = _total_cost(tasks_a)
    cost_b = _total_cost(tasks_b)
    elapsed_a = _avg_elapsed(tasks_a)
    elapsed_b = _avg_elapsed(tasks_b)

    return dict(
        meta_a=meta_a,
        meta_b=meta_b,
        passed_a=passed_a,
        passed_b=passed_b,
        failed_a=failed_a,
        failed_b=failed_b,
        total_a=total_a,
        total_b=total_b,
        acc_a=acc_a,
        acc_b=acc_b,
        cost_a=cost_a,
        cost_b=cost_b,
        elapsed_a=elapsed_a,
        elapsed_b=elapsed_b,
        gained=gained,
        lost=lost,
        cost_changes=cost_changes,
        only_in_a=only_in_a,
        only_in_b=only_in_b,
        all_ids=all_ids,
        tasks_a=tasks_a,
        tasks_b=tasks_b,
    )


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _fmt_pct(val: float | None) -> str:
    return f"{val:.1%}" if val is not None else "n/a"


def _fmt_cost(val: float | None) -> str:
    return f"${val:.2f}" if val is not None else "n/a"


def _fmt_elapsed(val: float | None) -> str:
    return f"{val:.0f}s" if val is not None else "n/a"


def _delta_pct(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return ""
    d = b - a
    return f"{d:+.1%}"


def _delta_cost(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return ""
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}${d:.2f}"


def _delta_elapsed(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return ""
    d = b - a
    return f"{d:+.0f}s"


def _delta_int(a: int, b: int) -> str:
    d = b - a
    return f"{d:+d}"


def _wrap_ids(ids: list[str], width: int = 80, indent: int = 2) -> str:
    if not ids:
        return "  (none)"
    lines: list[str] = []
    line = " " * indent
    for uid in ids:
        if len(line) + len(uid) + 2 > width:
            lines.append(line.rstrip())
            line = " " * indent
        line += uid + "  "
    if line.strip():
        lines.append(line.rstrip())
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Output: text
# ---------------------------------------------------------------------------


def print_text_report(cmp: dict) -> None:
    ma, mb = cmp["meta_a"], cmp["meta_b"]

    def _run_label(meta: dict) -> str:
        parts = [meta["label"]]
        if meta.get("git_sha"):
            parts.append(f"git: {meta['git_sha'][:7]}")
        if meta.get("tag"):
            parts.append(f"tag: {meta['tag']}")
        return f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else parts[0]

    print("=== Run Comparison ===")
    print(f"Run A: {_run_label(ma)}")
    print(f"Run B: {_run_label(mb)}")
    print()

    COL_W = 12
    lbl_w = 20

    def row(label: str, va: str, vb: str, delta: str = "") -> None:
        print(f"  {label:<{lbl_w}}{va:<{COL_W}}{vb:<{COL_W}}{delta}")

    header = f"  {'':20}{'Run A':<{COL_W}}{'Run B':<{COL_W}}{'Delta':<{COL_W}}"
    print(header)
    print("  " + "-" * (lbl_w + COL_W * 3))

    row(
        "Passed:",
        str(cmp["passed_a"]),
        str(cmp["passed_b"]),
        _delta_int(cmp["passed_a"], cmp["passed_b"]),
    )
    row(
        "Failed:",
        str(cmp["failed_a"]),
        str(cmp["failed_b"]),
        _delta_int(cmp["failed_a"], cmp["failed_b"]),
    )
    row(
        "Total tasks:",
        str(cmp["total_a"]),
        str(cmp["total_b"]),
        _delta_int(cmp["total_a"], cmp["total_b"]),
    )
    row(
        "Accuracy:",
        _fmt_pct(cmp["acc_a"]),
        _fmt_pct(cmp["acc_b"]),
        _delta_pct(cmp["acc_a"], cmp["acc_b"]),
    )
    row(
        "Total Cost:",
        _fmt_cost(cmp["cost_a"]),
        _fmt_cost(cmp["cost_b"]),
        _delta_cost(cmp["cost_a"], cmp["cost_b"]),
    )
    row(
        "Avg Time/Task:",
        _fmt_elapsed(cmp["elapsed_a"]),
        _fmt_elapsed(cmp["elapsed_b"]),
        _delta_elapsed(cmp["elapsed_a"], cmp["elapsed_b"]),
    )
    print()

    gained = cmp["gained"]
    lost = cmp["lost"]
    cost_changes = cmp["cost_changes"]
    only_a = cmp["only_in_a"]
    only_b = cmp["only_in_b"]

    print(f"=== Tasks Gained (fail→pass): {len(gained)} ===")
    print(_wrap_ids(gained))
    print()

    print(f"=== Tasks Lost (pass→fail): {len(lost)} ===")
    print(_wrap_ids(lost))
    print()

    print(f"=== Tasks Changed Cost (>50% delta): {len(cost_changes)} ===")
    if cost_changes:
        for c in cost_changes[:20]:
            sign = "+" if c["delta_pct"] >= 0 else ""
            print(
                f"  {c['task_id']}: ${c['cost_a']:.2f} → ${c['cost_b']:.2f} ({sign}{c['delta_pct']:.0f}%)"
            )
        if len(cost_changes) > 20:
            print(f"  ... and {len(cost_changes) - 20} more")
    else:
        print("  (none)")
    print()

    if only_a:
        print(f"=== Tasks only in Run A: {len(only_a)} ===")
        print(_wrap_ids(only_a))
        print()

    if only_b:
        print(f"=== Tasks only in Run B: {len(only_b)} ===")
        print(_wrap_ids(only_b))
        print()


# ---------------------------------------------------------------------------
# Output: CSV
# ---------------------------------------------------------------------------


def print_csv_report(cmp: dict) -> None:
    tasks_a = cmp["tasks_a"]
    tasks_b = cmp["tasks_b"]
    all_ids = cmp["all_ids"]

    writer = csv.writer(sys.stdout)
    writer.writerow(
        [
            "task_id",
            "passed_a",
            "passed_b",
            "outcome_change",
            "cost_a",
            "cost_b",
            "cost_delta_pct",
            "elapsed_a",
            "elapsed_b",
            "elapsed_delta_s",
            "reward_a",
            "reward_b",
        ]
    )

    for tid in all_ids:
        ta = tasks_a.get(tid, {})
        tb = tasks_b.get(tid, {})

        pa = ta.get("passed")
        pb = tb.get("passed")

        if pa is False and pb is True:
            change = "gained"
        elif pa is True and pb is False:
            change = "lost"
        elif pa is True and pb is True:
            change = "stable_pass"
        elif pa is False and pb is False:
            change = "stable_fail"
        elif tid not in tasks_a:
            change = "only_in_b"
        elif tid not in tasks_b:
            change = "only_in_a"
        else:
            change = "unknown"

        ca = ta.get("cost_usd")
        cb = tb.get("cost_usd")
        cost_delta = ""
        if ca is not None and cb is not None and ca > 0:
            cost_delta = f"{(cb - ca) / ca * 100:.1f}"

        ea = ta.get("elapsed_s")
        eb = tb.get("elapsed_s")
        elapsed_delta = ""
        if ea is not None and eb is not None:
            elapsed_delta = f"{eb - ea:.1f}"

        writer.writerow(
            [
                tid,
                "" if pa is None else ("pass" if pa else "fail"),
                "" if pb is None else ("pass" if pb else "fail"),
                change,
                "" if ca is None else f"{ca:.4f}",
                "" if cb is None else f"{cb:.4f}",
                cost_delta,
                "" if ea is None else f"{ea:.1f}",
                "" if eb is None else f"{eb:.1f}",
                elapsed_delta,
                "" if ta.get("reward") is None else ta["reward"],
                "" if tb.get("reward") is None else tb["reward"],
            ]
        )


# ---------------------------------------------------------------------------
# --list
# ---------------------------------------------------------------------------


def cmd_list() -> None:
    runs = list_available_runs()
    if not runs:
        print("No runs found.")
        return
    max_len = max(len(lbl) for lbl, _ in runs)
    for label, source in runs:
        print(f"  {label:<{max_len}}  [{source}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two arena test runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/compare_runs.py --list
  python3 scripts/compare_runs.py pool-20260402T094504Z pool-20260402T165000Z-oom-fix
  python3 scripts/compare_runs.py run-20260401-063225 run-20260402-214124
  python3 scripts/compare_runs.py baseline evidence-rerank-v1 --csv
""",
    )
    parser.add_argument("run_a", nargs="?", help="First run label or substring")
    parser.add_argument("run_b", nargs="?", help="Second run label or substring")
    parser.add_argument("--list", action="store_true", help="List available runs and exit")
    parser.add_argument("--csv", action="store_true", help="Output per-task CSV to stdout")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        cmd_list()
        return

    if not args.run_a or not args.run_b:
        sys.exit("ERROR: Provide two run labels (or use --list to see available runs).")

    meta_a, tasks_a = load_run(args.run_a)
    meta_b, tasks_b = load_run(args.run_b)

    if not tasks_a:
        print(f"WARNING: Run A '{meta_a['label']}' has no task data.", file=sys.stderr)
    if not tasks_b:
        print(f"WARNING: Run B '{meta_b['label']}' has no task data.", file=sys.stderr)

    cmp = compare_runs(meta_a, tasks_a, meta_b, tasks_b)

    if args.csv:
        print_csv_report(cmp)
    else:
        print_text_report(cmp)


if __name__ == "__main__":
    main()
