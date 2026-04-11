#!/usr/bin/env python3
"""Local test runner for the Analyst architecture.

Simulates the Goose harness: MiniMax gets a shell tool, the system.j2 prompt,
and 20 turns with no budget-panic messaging. The 280s timeout is the only limit.

Usage:
    python3 analyst/run_all.py --corpus /path/to/corpus --uids UID0023,UID0041
    python3 analyst/run_all.py --corpus /path/to/corpus --concurrency 2
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
MAX_TURNS = 25  # Generous — let MiniMax think
TIMEOUT_PER_TASK = 280  # seconds (300s arena timeout minus margin)
ANALYST_DIR = Path(__file__).parent
PROJECT_ROOT = ANALYST_DIR.parent

# ── Tool Definition ────────────────────────────────────────────────────
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": (
                "Execute a shell command. Use for: "
                'python3 /installed-agent/solve_briefing.py "question", '
                'python3 /installed-agent/solve.py "question", '
                'grep -i "phrase" /app/corpus/*.txt, '
                'python3 -c "print(1+2)", '
                "printf '%s' \"answer\" > /app/answer.txt"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The shell command to execute"}
                },
                "required": ["command"],
            },
        },
    }
]


def load_system_prompt(instruction: str) -> str:
    """Render the Jinja2 system prompt."""
    template_path = ANALYST_DIR / "prompts" / "system.j2"
    template_str = template_path.read_text()
    template = jinja2.Template(template_str)
    return template.render(instruction=instruction)


def load_skills_context() -> str:
    """Load skill files as additional context."""
    skills_dir = ANALYST_DIR / "skills"
    if not skills_dir.is_dir():
        return ""
    parts = ["\n\n# DOMAIN KNOWLEDGE\n"]
    for md_file in sorted(skills_dir.glob("*.md")):
        content = md_file.read_text().strip()
        parts.append(f"\n## {md_file.stem}\n{content}\n")
    return "\n".join(parts)


def load_questions(csv_path: str, uids: list[str] | None = None) -> list[dict]:
    """Load questions from CSV."""
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
                }
            )
    return questions


def execute_shell(command: str, corpus_dir: str, work_dir: str, agent_dir: str) -> str:
    """Execute a shell command with path rewriting."""
    cmd = command.replace("/app/corpus", corpus_dir)
    cmd = cmd.replace("/installed-agent/", agent_dir + "/")
    cmd = cmd.replace("/app/answer.txt", os.path.join(work_dir, "answer.txt"))
    cmd = re.sub(r"\becho\s+-n\s+", "printf '%s' ", cmd)

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
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
        if len(output) > 10000:
            output = output[:5000] + "\n\n... [TRUNCATED] ...\n\n" + output[-4000:]
        return output if output.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "ERROR: Command timed out after 60 seconds."
    except Exception as e:
        return f"ERROR: {e}"


def call_openrouter(messages: list[dict], api_key: str) -> dict:
    """Call OpenRouter API."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/officeqa-arena",
        "X-Title": "OfficeQA Analyst Runner",
    }
    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": TOOLS,
        "temperature": 0.0,
        "max_tokens": 4096,
    }

    for attempt in range(4):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=120)
            if resp.status_code == 429:
                wait = 2 ** (attempt + 1)
                print(f"    Rate limited, waiting {wait}s...", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException:
            if attempt < 3:
                time.sleep(2 ** (attempt + 1))
            else:
                raise
    raise RuntimeError("OpenRouter API failed after 4 retries")


def score_answer(predicted: str | None, expected: str) -> bool:
    """Fuzzy numeric matching with 1% tolerance."""
    if predicted is None:
        return False
    pred_clean = re.sub(r"[,%$\s]", "", predicted.replace("−", "-").strip().rstrip("%"))
    exp_clean = re.sub(r"[,%$\s]", "", expected.replace("−", "-").strip().rstrip("%"))
    try:
        pred_val = float(pred_clean)
        exp_val = float(exp_clean)
    except (ValueError, TypeError):
        return predicted.strip().lower() == expected.strip().lower()
    if exp_val == 0:
        return abs(pred_val) < 0.01
    return abs(pred_val - exp_val) / abs(exp_val) <= 0.01


def run_single_task(
    task: dict, api_key: str, corpus_dir: str, agent_dir: str, shared_index_dir: str | None = None
) -> dict:
    """Run a single question through the analyst loop."""
    uid = task["uid"]
    question = task["question"]
    expected = task["answer"]
    start_time = time.time()

    work_dir = tempfile.mkdtemp(prefix=f"analyst_{uid}_")

    # Symlink shared index if available
    if shared_index_dir:
        for fname in ["table_index.jsonl", "keyword_index.txt"]:
            src = os.path.join(shared_index_dir, fname)
            if os.path.exists(src):
                os.symlink(src, os.path.join(work_dir, fname))

    # Build prompt — mentor framing, no budget warnings
    full_system = load_system_prompt(question) + load_skills_context()

    messages = [
        {"role": "system", "content": full_system},
        {"role": "user", "content": question},
    ]

    tool_calls_count = 0
    answer = None

    for turn in range(MAX_TURNS):
        elapsed = time.time() - start_time
        if elapsed > TIMEOUT_PER_TASK:
            print(f"  [{uid}] Timeout after {elapsed:.0f}s", file=sys.stderr)
            break

        try:
            response = call_openrouter(messages, api_key)
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

                if func_name == "run_shell":
                    command = args.get("command", "")
                    tool_calls_count += 1
                    print(f"  [{uid}] T{tool_calls_count}: {command[:120]}", file=sys.stderr)

                    result = execute_shell(command, corpus_dir, work_dir, agent_dir)

                    # No budget warnings — just a quiet counter for our logs
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                    )

                    # Check if answer was written
                    answer_path = os.path.join(work_dir, "answer.txt")
                    if os.path.exists(answer_path):
                        answer = Path(answer_path).read_text().strip()
                else:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"Unknown tool: {func_name}",
                        }
                    )
        else:
            # Text response, no tool calls
            content = message.get("content", "")
            messages.append({"role": "assistant", "content": content})

            answer_path = os.path.join(work_dir, "answer.txt")
            if os.path.exists(answer_path):
                answer = Path(answer_path).read_text().strip()

            if choice.get("finish_reason") == "stop":
                # Try to extract answer from text
                if not answer and content:
                    for pat in [
                        r'printf\s+\'%s\'\s+"([^"]+)"\s*>\s*/app/answer\.txt',
                        r'echo\s+-n\s+"([^"]+)"\s*>\s*/app/answer\.txt',
                    ]:
                        m = re.search(pat, content)
                        if m:
                            answer = m.group(1)
                            break
                break

    elapsed = time.time() - start_time

    # Final check for answer file
    if not answer:
        answer_path = os.path.join(work_dir, "answer.txt")
        if os.path.exists(answer_path) and os.path.getsize(answer_path) > 0:
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

    # Save trace
    trace_dir = ANALYST_DIR / "traces"
    trace_dir.mkdir(exist_ok=True)
    trace_path = trace_dir / f"{uid}.jsonl"
    with open(trace_path, "w") as tf:
        for msg in messages:
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
        f"  [{uid}] {status} | answer={answer} | expected={expected} | "
        f"{tool_calls_count} calls | {elapsed:.1f}s",
        file=sys.stderr,
    )

    return result


def pre_build_index(corpus_dir: str, agent_dir: str, index_dir: str):
    """Pre-build search index once."""
    idx_path = os.path.join(index_dir, "table_index.jsonl")
    if os.path.exists(idx_path) and os.path.getsize(idx_path) > 1000:
        print(f"Index exists at {idx_path}", file=sys.stderr)
        return

    print(f"Building index from {corpus_dir}...", file=sys.stderr)
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
            "KEYWORD_INDEX_PATH": os.path.join(index_dir, "keyword_index.txt"),
        },
    )
    print(result.stderr, file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="OfficeQA Analyst local runner")
    parser.add_argument("--corpus", type=str, default=os.environ.get("CORPUS_DIR", "/app/corpus"))
    parser.add_argument(
        "--cases", type=str, default=str(PROJECT_ROOT / "data" / "officeqa_full.csv")
    )
    parser.add_argument("--uids", type=str, default=None, help="Comma-separated UIDs to run")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-index", action="store_true")
    args = parser.parse_args()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and not args.dry_run:
        print("ERROR: Set OPENROUTER_API_KEY", file=sys.stderr)
        sys.exit(1)

    corpus_dir = os.path.abspath(args.corpus)
    agent_dir = str(ANALYST_DIR)

    uid_list = [u.strip() for u in args.uids.split(",")] if args.uids else None
    questions = load_questions(args.cases, uid_list)

    if not questions:
        print("ERROR: No questions loaded.", file=sys.stderr)
        sys.exit(1)

    print(f"Analyst Runner — {len(questions)} questions", file=sys.stderr)
    print(f"Corpus: {corpus_dir}", file=sys.stderr)
    print(f"Model: {MODEL}", file=sys.stderr)
    print(f"Max turns: {MAX_TURNS} (no budget warnings)", file=sys.stderr)

    if args.dry_run:
        for q in questions:
            print(f"  {q['uid']} [{q['difficulty']}] {q['question'][:80]}...")
        return

    # Pre-build index
    shared_index_dir = tempfile.mkdtemp(prefix="analyst_index_")
    if not args.skip_index:
        pre_build_index(corpus_dir, agent_dir, shared_index_dir)

    output_path = args.output or f"results_analyst_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Output: {output_path}\n{'=' * 60}\n", file=sys.stderr)

    results = []
    correct = 0
    total = 0

    if args.concurrency <= 1:
        for task in questions:
            result = run_single_task(task, api_key, corpus_dir, agent_dir, shared_index_dir)
            results.append(result)
            total += 1
            if result["correct"]:
                correct += 1
            with open(output_path, "a") as f:
                f.write(json.dumps(result) + "\n")
            print(
                f"  Progress: {correct}/{total} ({correct / total * 100:.1f}%)\n", file=sys.stderr
            )
    else:
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
                    f"  Progress: {correct}/{total} ({correct / total * 100:.1f}%)\n",
                    file=sys.stderr,
                )

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"RESULTS: {correct}/{total} ({correct / total * 100:.1f}%)", file=sys.stderr)
    avg_calls = sum(r["tool_calls"] for r in results) / len(results) if results else 0
    avg_time = sum(r["elapsed_s"] for r in results) / len(results) if results else 0
    print(f"Avg tool calls: {avg_calls:.1f} | Avg time: {avg_time:.1f}s", file=sys.stderr)
    print(f"Results: {output_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
