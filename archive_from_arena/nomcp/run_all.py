#!/usr/bin/env python3
"""
Standalone runner for OfficeQA Arena — No Docker, No Arena CLI.

Calls OpenRouter API (MiniMax M2.5) with tool-use to simulate the Goose harness.
The model gets shell command execution as a tool. It runs solve.py, search.py,
python3 computations, and writes answers.

Usage:
    # Run all 246 tasks
    python3 nomcp/run_all.py --corpus /path/to/corpus

    # Run specific UIDs
    python3 nomcp/run_all.py --corpus /path/to/corpus --uids UID0001,UID0002

    # Run with concurrency
    python3 nomcp/run_all.py --corpus /path/to/corpus --concurrency 4

    # Dry run (show questions, don't call API)
    python3 nomcp/run_all.py --corpus /path/to/corpus --dry-run

Environment:
    OPENROUTER_API_KEY  — Required. Your OpenRouter API key.
    CORPUS_DIR          — Path to corpus (or use --corpus flag).
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import jinja2
import requests

# ── Constants ──────────────────────────────────────────────────────────
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "minimax/minimax-m2.5"
MAX_TURNS = 15  # Match arena.yaml config
TIMEOUT_PER_TASK = 280  # seconds (leave margin for the 300s Arena timeout)
NOMCP_DIR = Path(__file__).parent
PROJECT_ROOT = NOMCP_DIR.parent

# ── Tool Definition (OpenAI function-calling format) ──────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Execute a shell command. Use for: grep, python3 /installed-agent/solve.py, python3 /installed-agent/search.py table, python3 -c, printf.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute (e.g., 'python3 /installed-agent/solve.py \"question\"')",
                    }
                },
                "required": ["command"],
            },
        },
    }
]


def load_system_prompt(instruction: str) -> str:
    """Render the Jinja2 system prompt with the given instruction."""
    template_path = NOMCP_DIR / "prompts" / "system.j2"
    template_str = template_path.read_text()
    template = jinja2.Template(template_str)
    return template.render(instruction=instruction)


def load_skills_context() -> str:
    """Load all skill files as additional context."""
    skills_dir = NOMCP_DIR / "skills"
    if not skills_dir.is_dir():
        return ""

    parts = ["\n\n# DOMAIN KNOWLEDGE (from skills/)\n"]
    for md_file in sorted(skills_dir.glob("*.md")):
        content = md_file.read_text().strip()
        parts.append(f"\n## {md_file.stem}\n{content}\n")
    return "\n".join(parts)


def load_questions(csv_path: str, uids: list[str] | None = None) -> list[dict]:
    """Load questions from the full CSV."""
    questions = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if uids and row["uid"].upper() not in [u.upper() for u in uids]:
                continue
            questions.append(
                {
                    "uid": row["uid"],
                    "question": row["question"],
                    "answer": row["answer"],
                    "difficulty": row.get("difficulty", "unknown"),
                    "source_files": row.get("source_files", ""),
                }
            )
    return questions


def _extract_fallback_answer(messages: list[dict], question: str) -> str | None:
    """Last-resort: scan recent tool outputs for a plausible numeric answer."""
    # Extract years from question
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", question)

    # Look at the last 6 tool responses for table data
    candidates = []
    for msg in reversed(messages[-12:]):
        content = msg.get("content", "")
        if not content or msg.get("role") != "tool":
            continue
        # Find rows with pipe-delimited data containing target years
        for year in years:
            for line in content.split("\n"):
                if "|" in line and year in line and "---" not in line:
                    # Extract numeric values from this row
                    cells = [c.strip() for c in line.split("|") if c.strip()]
                    for cell in cells[1:]:  # skip row label
                        cleaned = re.sub(r"[rRpP*/]+/?$", "", cell).strip()
                        cleaned = cleaned.replace(",", "").replace("$", "")
                        # Handle parenthetical negatives
                        m = re.match(r"^\(([0-9,.]+)\)$", cleaned)
                        if m:
                            cleaned = "-" + m.group(1).replace(",", "")
                        try:
                            float(cleaned)
                            # Re-format with commas if original had them
                            if "," in cell:
                                try:
                                    fval = float(cleaned)
                                    if fval == int(fval) and "." not in cleaned:
                                        candidates.append(f"{int(fval):,}")
                                    else:
                                        candidates.append(cleaned)
                                except ValueError:
                                    candidates.append(cleaned)
                            else:
                                candidates.append(cleaned)
                        except ValueError:
                            continue

    # Return first candidate (from most recent table, first data column)
    return candidates[0] if candidates else None


def execute_shell(command: str, corpus_dir: str, work_dir: str, agent_dir: str) -> str:
    """Execute a shell command with path rewriting for local environment."""
    cmd = command
    # Rewrite ALL arena paths to local equivalents
    cmd = cmd.replace("/app/corpus", corpus_dir)
    cmd = cmd.replace("/installed-agent", agent_dir)
    cmd = cmd.replace("/app/answer.txt", os.path.join(work_dir, "answer.txt"))
    # Goose skills path also points to agent dir
    cmd = cmd.replace("~/.config/goose/skills", agent_dir)
    cmd = re.sub(r"\$HOME/\.config/goose/skills", agent_dir, cmd)
    # Rewrite find commands that search the whole filesystem
    cmd = re.sub(r'find\s+/\s+-name\s+"?solve\.py"?', f'echo "{agent_dir}/solve.py"', cmd)
    cmd = re.sub(r'find\s+/\s+-name\s+"?search\.py"?', f'echo "{agent_dir}/search.py"', cmd)
    # Fix macOS echo -n bug: replace with printf '%s' which works everywhere
    cmd = re.sub(r"\becho\s+-n\s+", "printf '%s' ", cmd)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=180,
            cwd=work_dir,
            env={
                **os.environ,
                "CORPUS_DIR": corpus_dir,
                "INDEX_PATH": os.path.join(work_dir, "table_index.jsonl"),
                "KEYWORD_INDEX_PATH": os.path.join(work_dir, "keyword_index.txt"),
                "BUILD_SCRIPT": os.path.join(agent_dir, "build_index.py"),
                "ANSWER_PATH": os.path.join(work_dir, "answer.txt"),
            },
        )
        output = result.stdout + result.stderr
        # Moderate truncation — enough for smart-sliced tables
        if len(output) > 8000:
            output = (
                output[:4000]
                + "\n\n... [TRUNCATED — use more specific grep or read a smaller range] ...\n\n"
                + output[-3000:]
            )
        return output if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 60 seconds."
    except Exception as e:
        return f"ERROR: {e}"


_total_tokens_used = 0
_total_api_calls = 0


def call_openrouter(
    messages: list[dict], api_key: str, include_tools: bool = True, tools_override: list = None
) -> dict:
    """Call OpenRouter chat completions API."""
    global _total_tokens_used, _total_api_calls
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/officeqa-arena",
        "X-Title": "OfficeQA Arena Runner",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0.0,
        "max_tokens": 4096,
    }
    if include_tools:
        payload["tools"] = tools_override if tools_override else TOOLS

    for attempt in range(4):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            result = resp.json()
            # Track token usage
            usage = result.get("usage", {})
            tokens = usage.get("total_tokens", 0)
            _total_tokens_used += tokens
            _total_api_calls += 1
            return result
        except requests.exceptions.RequestException as e:
            if attempt < 3:
                wait = 2 ** (attempt + 1)
                print(f"    API error: {e}, retrying in {wait}s...", file=sys.stderr)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("OpenRouter API failed after 4 retries")


def run_single_task(
    task: dict, api_key: str, corpus_dir: str, agent_dir: str, shared_index_dir: str | None = None
) -> dict:
    """Run a single question through the agentic loop.

    MiniMax gets direct tool access (search_raw_corpus, compute_expression, etc.)
    instead of a shell. No nested LLM calls — one loop, one model.
    """
    # Import tools from solve.py
    import importlib.util

    spec = importlib.util.spec_from_file_location("solve", os.path.join(agent_dir, "solve.py"))
    solve_mod = importlib.util.module_from_spec(spec)
    # Set env before loading
    os.environ["CORPUS_DIR"] = corpus_dir
    os.environ["ANSWER_PATH"] = "/tmp/_answer_placeholder.txt"
    spec.loader.exec_module(solve_mod)

    uid = task["uid"]
    question = task["question"]
    expected = task["answer"]
    start_time = time.time()

    # Create work directory for this task
    work_dir = tempfile.mkdtemp(prefix=f"officeqa_{uid}_")
    answer_path = os.path.join(work_dir, "answer.txt")

    # If shared index exists, symlink it to avoid rebuilding per task
    if shared_index_dir:
        idx_src = os.path.join(shared_index_dir, "table_index.jsonl")
        kw_src = os.path.join(shared_index_dir, "keyword_index.txt")
        if os.path.exists(idx_src) and not os.path.exists(
            os.path.join(work_dir, "table_index.jsonl")
        ):
            os.symlink(idx_src, os.path.join(work_dir, "table_index.jsonl"))
        if os.path.exists(kw_src) and not os.path.exists(
            os.path.join(work_dir, "keyword_index.txt")
        ):
            os.symlink(kw_src, os.path.join(work_dir, "keyword_index.txt"))

    # Direct tool definitions — same tools as solve.py but given directly to MiniMax
    direct_tools = [
        {
            "type": "function",
            "function": {
                "name": "search_raw_corpus",
                "description": "Search Treasury Bulletin TXT files. Returns table data in vertical format (ROW: label, column: value). Use 2-3 keywords + year.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "keywords": {
                            "type": "string",
                            "description": "2-3 keywords e.g. 'national defense expenditures'",
                        },
                        "year": {"type": "integer", "description": "Year to search for"},
                        "period_hint": {
                            "type": "string",
                            "description": "calendar or fiscal — helps find the right table type",
                        },
                    },
                    "required": ["keywords"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "compute_expression",
                "description": "Evaluate math. Functions: sum, mean, median, stdev, cagr, geometric_mean, correlation, percentile, yoy_growth, theil_index, gini, herfindahl, abs, round, sqrt, log, min, max.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Math expression e.g. 'sum([132, 129, 143])'",
                        },
                        "variables": {
                            "type": "object",
                            "description": 'Named variables e.g. {"a": 71}',
                        },
                    },
                    "required": ["expression"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "lookup_cpi",
                "description": "Look up CPI-U index (1982-84=100). For inflation/real dollar questions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "year": {"type": "integer"},
                        "month": {"type": "integer", "description": "1-12, omit for annual avg"},
                        "year_start": {"type": "integer"},
                        "year_end": {"type": "integer"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_answer",
                "description": "Submit final answer. ALWAYS call this. A wrong answer beats no answer.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "answer": {"type": "string", "description": "Final numeric answer"}
                    },
                    "required": ["answer"],
                },
            },
        },
    ]

    # Direct tool dispatch — calls Python functions, no shell, no nested LLM
    def dispatch_tool(name, args):
        if name == "search_raw_corpus":
            return solve_mod.search_raw_corpus(
                keywords=args.get("keywords", ""),
                year=args.get("year"),
                limit=args.get("limit", 5),
                period_hint=args.get("period_hint"),
            )
        elif name == "compute_expression":
            try:
                result = solve_mod.safe_eval_finance(args["expression"], args.get("variables", {}))
                return {"result": result}
            except Exception as e:
                return {"error": str(e)}
        elif name == "lookup_cpi":
            if "year_start" in args and "year_end" in args:
                return solve_mod.lookup_cpi_range(
                    args["year_start"], args["year_end"], args.get("month")
                )
            return solve_mod.lookup_cpi(args.get("year", 2000), args.get("month"))
        elif name == "submit_answer":
            ans = str(args.get("answer", "")).strip()
            try:
                Path(answer_path).write_text(ans)
            except Exception:
                pass
            return {"status": "submitted", "answer": ans}
        else:
            return {"error": f"Unknown tool: {name}"}

    # System prompt — matches solve.py's SYSTEM_PROMPT
    system_prompt = solve_mod.SYSTEM_PROMPT

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]

    tool_calls_count = 0
    answer = None
    MAX_TOOL_CALLS = 10

    for turn in range(MAX_TURNS):
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT_PER_TASK:
            print(f"  [{uid}] Timeout after {elapsed:.0f}s", file=sys.stderr)
            break

        if tool_calls_count >= MAX_TOOL_CALLS + 3 and not answer:
            if os.path.exists(answer_path) and os.path.getsize(answer_path) > 0:
                answer = Path(answer_path).read_text().strip()
                print(f"  [{uid}] Budget exceeded, using written answer: {answer}", file=sys.stderr)
            break

        try:
            response = call_openrouter(
                messages, api_key, include_tools=True, tools_override=direct_tools
            )
        except Exception as e:
            print(f"  [{uid}] API error: {e}", file=sys.stderr)
            break

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})

        tool_calls = message.get("tool_calls")
        if tool_calls:
            messages.append(message)

            for tc in tool_calls:
                func = tc.get("function", {})
                func_name = func.get("name", "")
                try:
                    args = json.loads(func.get("arguments", "{}"))
                except json.JSONDecodeError:
                    args = {}

                tool_calls_count += 1
                remaining = MAX_TOOL_CALLS - tool_calls_count
                print(
                    f"  [{uid}] T{tool_calls_count}: {func_name}({json.dumps(args, default=str)[:80]})",
                    file=sys.stderr,
                )

                result = dispatch_tool(func_name, args)

                # Truncate large results
                result_str = json.dumps(result, default=str)
                if len(result_str) > 6000:
                    result_str = result_str[:6000] + '..."}'

                # Budget nudge
                if remaining <= 0:
                    result_str = f"[BUDGET GONE — call submit_answer NOW]\n{result_str}"
                elif remaining <= 2:
                    result_str = f"[{remaining} calls left — submit soon]\n{result_str}"

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_str,
                    }
                )

                # Check for submit
                if func_name == "submit_answer":
                    answer = args.get("answer", "")
                    break

            if answer:
                break
        else:
            content = message.get("content", "")
            messages.append({"role": "assistant", "content": content})

            if os.path.exists(answer_path) and os.path.getsize(answer_path) > 0:
                answer = Path(answer_path).read_text().strip()

            if choice.get("finish_reason") == "stop":
                # Try to extract answer from text
                if not answer and content:
                    m = re.search(
                        r"(?:answer|result|value)\s*(?:is|=|:)\s*([\-\d,]+\.?\d*%?)",
                        content,
                        re.IGNORECASE,
                    )
                    if m:
                        answer = m.group(1)
                break

    elapsed = time.time() - start_time

    # Fallback
    if not answer and os.path.exists(answer_path) and os.path.getsize(answer_path) > 0:
        answer = Path(answer_path).read_text().strip()

    correct = score_answer(answer, expected) if answer else False

    result = {
        "uid": uid,
        "question": question[:100] + "...",
        "expected": expected,
        "answer": answer,
        "correct": correct,
        "tool_calls": tool_calls_count,
        "elapsed_s": round(elapsed, 1),
        "difficulty": task["difficulty"],
    }

    # Save full conversation trace for debugging
    trace_dir = os.path.join(os.path.dirname(__file__), "traces")
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, f"{uid}.jsonl")
    with open(trace_path, "w") as tf:
        for msg in messages:
            # Serialize: handle non-serializable message dicts from API
            try:
                tf.write(json.dumps(msg, default=str) + "\n")
            except (TypeError, ValueError):
                tf.write(
                    json.dumps(
                        {"role": msg.get("role", "?"), "content": str(msg.get("content", ""))[:500]}
                    )
                    + "\n"
                )

    status = "PASS" if correct else "FAIL"
    print(
        f"  [{uid}] {status} | answer={answer} | expected={expected} | {tool_calls_count} calls | {elapsed:.1f}s",
        file=sys.stderr,
    )

    # Cleanup work dir (but keep answer for debugging)
    return result


def score_answer(predicted: str | None, expected: str) -> bool:
    """Fuzzy numeric matching with 1% tolerance."""
    if predicted is None:
        return False

    # Normalize both
    pred_clean = re.sub(r"[,%$\s]", "", predicted.replace("−", "-").strip().rstrip("%"))
    exp_clean = re.sub(r"[,%$\s]", "", expected.replace("−", "-").strip().rstrip("%"))

    try:
        pred_val = float(pred_clean)
        exp_val = float(exp_clean)
    except (ValueError, TypeError):
        # Fall back to exact string match
        return predicted.strip().lower() == expected.strip().lower()

    if exp_val == 0:
        return abs(pred_val) < 0.01

    return abs(pred_val - exp_val) / abs(exp_val) <= 0.01


def pre_build_index(corpus_dir: str, agent_dir: str, index_dir: str):
    """Pre-build the search index once for all tasks."""
    idx_path = os.path.join(index_dir, "table_index.jsonl")
    kw_path = os.path.join(index_dir, "keyword_index.txt")

    if os.path.exists(idx_path) and os.path.getsize(idx_path) > 1000:
        print(f"Index already exists at {idx_path}", file=sys.stderr)
        return

    print(f"Pre-building index from {corpus_dir}...", file=sys.stderr)
    build_script = os.path.join(agent_dir, "build_index.py")

    result = subprocess.run(
        [sys.executable, build_script, corpus_dir, idx_path],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "CORPUS_DIR": corpus_dir,
            "INDEX_PATH": idx_path,
            "KEYWORD_INDEX_PATH": kw_path,
        },
    )
    print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        print(f"WARNING: Index build failed: {result.stderr}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="OfficeQA Arena standalone runner")
    parser.add_argument(
        "--corpus",
        type=str,
        default=os.environ.get("CORPUS_DIR", "/app/corpus"),
        help="Path to Treasury Bulletin corpus directory",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=str(PROJECT_ROOT / "data" / "officeqa_full.csv"),
        help="Path to questions CSV",
    )
    parser.add_argument(
        "--uids", type=str, default=None, help="Comma-separated UIDs to run (default: all)"
    )
    parser.add_argument("--concurrency", type=int, default=1, help="Number of concurrent tasks")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL file for results")
    parser.add_argument("--dry-run", action="store_true", help="Show questions without calling API")
    parser.add_argument("--skip-index", action="store_true", help="Skip pre-building the index")
    args = parser.parse_args()

    # Validate
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: Set OPENROUTER_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    corpus_dir = os.path.abspath(args.corpus)
    if not os.path.isdir(corpus_dir) and not args.dry_run:
        print(f"ERROR: Corpus directory not found: {corpus_dir}", file=sys.stderr)
        print("  Download the corpus and pass --corpus /path/to/corpus", file=sys.stderr)
        sys.exit(1)

    agent_dir = str(NOMCP_DIR)

    # Load questions
    uid_list = [u.strip() for u in args.uids.split(",")] if args.uids else None
    questions = load_questions(args.cases, uid_list)

    if not questions:
        print("ERROR: No questions loaded. Check --cases and --uids.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(questions)} questions", file=sys.stderr)
    print(f"Corpus: {corpus_dir}", file=sys.stderr)
    print(f"Model: {MODEL}", file=sys.stderr)
    print(f"Concurrency: {args.concurrency}", file=sys.stderr)

    if args.dry_run:
        for q in questions:
            print(f"  {q['uid']} [{q['difficulty']}] {q['question'][:80]}...")
        print(f"\nTotal: {len(questions)} questions")
        return

    # Pre-build shared index
    shared_index_dir = tempfile.mkdtemp(prefix="officeqa_index_")
    if not args.skip_index:
        pre_build_index(corpus_dir, agent_dir, shared_index_dir)

    # Output file
    output_path = args.output or f"results_nomcp_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Output: {output_path}", file=sys.stderr)
    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Starting run: {len(questions)} tasks", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    results = []
    correct = 0
    total = 0

    if args.concurrency <= 1:
        # Sequential
        for task in questions:
            result = run_single_task(task, api_key, corpus_dir, agent_dir, shared_index_dir)
            results.append(result)
            total += 1
            if result["correct"]:
                correct += 1

            # Write incrementally
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")

            print(
                f"  Progress: {correct}/{total} correct ({correct / total * 100:.1f}%)\n",
                file=sys.stderr,
            )
    else:
        # Concurrent
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {
                executor.submit(
                    run_single_task, task, api_key, corpus_dir, agent_dir, shared_index_dir
                ): task
                for task in questions
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                total += 1
                if result["correct"]:
                    correct += 1

                with open(output_path, "a") as f:
                    f.write(json.dumps(result) + "\n")

                print(
                    f"  Progress: {correct}/{total} correct ({correct / total * 100:.1f}%)\n",
                    file=sys.stderr,
                )

    # Final summary
    print(f"\n{'=' * 60}", file=sys.stderr)
    print("FINAL RESULTS", file=sys.stderr)
    print(f"{'=' * 60}", file=sys.stderr)
    print(f"Total:   {total}", file=sys.stderr)
    print(f"Correct: {correct} ({correct / total * 100:.1f}%)", file=sys.stderr)
    print(f"Wrong:   {total - correct}", file=sys.stderr)

    easy = [r for r in results if r["difficulty"] == "easy"]
    hard = [r for r in results if r["difficulty"] == "hard"]
    if easy:
        easy_correct = sum(1 for r in easy if r["correct"])
        print(
            f"Easy:    {easy_correct}/{len(easy)} ({easy_correct / len(easy) * 100:.1f}%)",
            file=sys.stderr,
        )
    if hard:
        hard_correct = sum(1 for r in hard if r["correct"])
        print(
            f"Hard:    {hard_correct}/{len(hard)} ({hard_correct / len(hard) * 100:.1f}%)",
            file=sys.stderr,
        )

    avg_time = sum(r["elapsed_s"] for r in results) / len(results) if results else 0
    avg_calls = sum(r["tool_calls"] for r in results) / len(results) if results else 0
    print(f"Avg time: {avg_time:.1f}s", file=sys.stderr)
    print(f"Avg tool calls: {avg_calls:.1f}", file=sys.stderr)
    print(f"\nResults saved to: {output_path}", file=sys.stderr)

    # Estimate Arena score
    # Score = correct_tasks × (1.0 + cost_adj + time_adj), max 282.9
    # Rough estimate: cost_adj + time_adj ≈ 0.15 (best case)
    estimated_score = correct * 1.15
    print(f"\nEstimated Arena score: ~{estimated_score:.1f} (max 282.9)", file=sys.stderr)

    # Token usage and cost summary
    # MiniMax M2.5 pricing: $0.50/1M input, $2.00/1M output (approx via OpenRouter)
    print("\nAPI Usage:", file=sys.stderr)
    print(f"  Total API calls: {_total_api_calls}", file=sys.stderr)
    print(f"  Total tokens:    {_total_tokens_used:,}", file=sys.stderr)
    est_cost = _total_tokens_used * 1.0 / 1_000_000  # ~$1/1M tokens blended estimate
    print(f"  Est. cost:       ${est_cost:.2f} (blended ~$1/1M tokens)", file=sys.stderr)


if __name__ == "__main__":
    main()
