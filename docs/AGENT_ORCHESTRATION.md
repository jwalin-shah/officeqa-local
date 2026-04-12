# Agent Orchestration

How to split officeqa-local work across multiple coding agents running in
headless mode inside isolated git worktrees.

## Goals
- Parallelize bounded workstreams without merge chaos.
- Keep `main` stable; each agent works on its own branch + worktree.
- Use each tool for what it's best at.
- One human (you) is the merger/reviewer.

## Shared Traversal Policy

Before doing repo exploration, every agent should read [AGENTS.md](../AGENTS.md)
and follow its traversal rules.

Default policy:
- `llm-tldr` first for structure, code context, call graphs, and impact analysis
- `rtk` first for compact reads, grep, diff, pytest, and command output
- raw shell tools only when exact unfiltered output is necessary

Human rule: when you launch Claude, Gemini, or Cursor, include:

```text
Read AGENTS.md first and follow its Traversal Policy.
```

## Role assignment

| Agent              | Strengths                                   | Use for                                              |
|--------------------|---------------------------------------------|------------------------------------------------------|
| Claude Code (Max)  | Long autonomous loops, strong reasoning     | Decompose/extract strategy, prompt engineering       |
| Codex CLI          | Fast implementation, refactors              | Retrieval, fast-path, test writing, mechanical edits |
| Gemini CLI         | Large context, independent second opinion   | Reviews, synthesis, cross-file critique              |
| Cursor Agent (CLI) | Quick iterative edits with local context    | Small fixes, targeted refactors                      |

Human role: define tasks, merge PRs, resolve conflicts, own `PLAN.md`.

## Workstream → Agent map

| # | Workstream             | Owner        | Reviewer | Files in scope                                      |
|---|------------------------|--------------|----------|------------------------------------------------------|
| 1 | Retrieval reranking    | Codex        | Gemini   | `retrieve_v2.py`, `retrieve.py`, retrieval tests     |
| 2 | Decompose + spec check | Claude       | Gemini   | `solve.py` decompose slice, `validate_decompose.py`  |
| 3 | Fast-path resolver     | Codex        | Claude   | `find.py`, `solve.py` fast-path slice                |
| 4 | Extraction quality     | Claude       | Gemini   | `extract.py`, extraction tests                       |
| 5 | Eval harness           | Codex        | Claude   | `eval_*.py`, `batch_test.py`                         |
| 6 | Ingestion integrity    | Codex        | Gemini   | `build_ledger.py`, `migrate_*.py`                    |

Rule: each workstream touches its files only. If a task needs to cross
boundaries, it becomes an integration PR the human merges by hand.

## Worktree layout

One worktree per workstream. All siblings of the main repo so agents don't
trip over each other.

```
~/projects/officeqa-local/              # main (you)
~/projects/officeqa-wt/retrieval/       # agent 1
~/projects/officeqa-wt/decompose/       # agent 2
~/projects/officeqa-wt/fastpath/        # agent 3
~/projects/officeqa-wt/extract/         # agent 4
~/projects/officeqa-wt/eval/            # agent 5
~/projects/officeqa-wt/ingest/          # agent 6
```

Create them (equivalent to the loop below):

```bash
./scripts/setup_worktrees.sh
```

Or manually:

```bash
mkdir -p ~/projects/officeqa-wt
cd ~/projects/officeqa-local
for stream in retrieval decompose fastpath extract eval ingest; do
  git worktree add -b "agent/$stream" "../officeqa-wt/$stream" main
done
git worktree list
```

Use `./scripts/setup_worktrees.sh --list` to print branch names and paths without creating directories. Override the parent directory with `OFFICEQA_WT_ROOT` and the base ref with `OFFICEQA_WT_BASE` (default `main`).

Default in-repo layout (gitignored): `OFFICEQA_WT_ROOT=$REPO/.agent-worktrees` so worktrees stay next to the clone. After you commit new tooling on `main`, run `./scripts/refresh_worktrees_from_main.sh` so each `agent/*` branch picks it up.

Tear down when a branch is merged:

```bash
git worktree remove ../officeqa-wt/retrieval
git branch -d agent/retrieval
```

### Sharing the ledger

`ledger.sqlite` (~2.7 GB) is gitignored and not copied per worktree. Either:
- symlink it into each worktree: `ln -s ~/projects/officeqa-local/ledger.sqlite .`
- or export `OFFICEQA_LEDGER=/absolute/path/ledger.sqlite` and have code
  respect that env var.

Prefer the symlink until the code path is stable.

## Siloed stages: what you can fake

Upstream and downstream stages do **not** need to be “running” if you preserve **contracts**:

| Stage | Others become… |
|-------|------------------|
| Decompose | Cached `QuestionSpec` / `decompose_eval.full.jsonl`; no live retrieve. |
| Retrieval | Cached specs + `ledger.sqlite`; no LLM decompose or extract. |
| Extraction | Hand-built or snapshot `per_dr` / table entries; mock `extract_structured` LLM. |
| Compute | Values dict + template only (`compute.py` tests). |
| Verify | Answer string + small context dict (`verify.py` tests). |

Fakes must match **real field shapes** (file names, `html` vs prose entries, keys on retrieval dicts). Wrong fixtures give false-green silos.

## Automated check rounds (local machine only)

The repo cannot run Codex/Claude/Cursor on your behalf from GitHub Actions. On your laptop:

```bash
./scripts/setup_worktrees.sh
./scripts/link_worktree_artifacts.sh
DRY_RUN=1 ./scripts/agent_iterate.sh          # checks only, no agent
MAX_ROUNDS=3 ./scripts/agent_iterate.sh       # checks; exits if anything fails

# Optional: after a failure, re-run with a headless CLI (example — use your real flags):
export OFFICEQA_AGENT='codex exec --sandbox workspace-write --full-auto'
MAX_ROUNDS=2 ./scripts/agent_iterate.sh
```

Task text for each workstream lives in `scripts/agent_task_specs/<stream>.txt`. The helper `scripts/_invoke_agent.py` runs `shlex.split($OFFICEQA_AGENT)` and appends the prompt as a **single argv string** (fine for Codex-style CLIs).

### Cursor Agent CLI (one shot per worktree)

`cursor-agent` must be on `PATH` or at `~/.local/bin/cursor-agent` (already logged in via `cursor-agent whoami`).

```bash
# Read-only / planning pass (no `--force`)
OFFICEQA_WT_ROOT="$PWD/.agent-worktrees" CURSOR_AGENT_MODE=plan \
  ./scripts/cursor_agent_once.sh retrieval

# Headless edit pass in that worktree (uses --force + --trust)
OFFICEQA_WT_ROOT="$PWD/.agent-worktrees" \
  ./scripts/cursor_agent_once.sh extract

# Optional: pin model or relax sandbox (see cursor-agent --help)
CURSOR_AGENT_MODEL=sonnet-4 OFFICEQA_CURSOR_SANDBOX=disabled \
  ./scripts/cursor_agent_once.sh eval
```

From `agent_iterate.sh` after failed checks: `OFFICEQA_AGENT_CURSOR=1` (no `OFFICEQA_AGENT` needed).

## Headless invocation cheatsheet

Each agent should receive: goal, in-scope files, out-of-scope files, commands
to run, acceptance criteria. Terse prompts produce shallow work — brief each
agent like a new hire.

### Claude Code (`claude -p`)

```bash
cd ~/projects/officeqa-wt/extract
claude -p "$(cat <<'EOF'
Goal: improve oracle extraction accuracy in extract.py without touching
retrieval or compute.

Read AGENTS.md first and follow its Traversal Policy.

In scope: extract.py, tests/test_extraction.py
Out of scope: everything else — do not modify.

Run: uv run pytest tests/test_extraction.py -x
     uv run python eval_extract_oracle.py --n 30

Acceptance: oracle extraction accuracy ≥ current baseline on the 30-q subset;
no regressions in tests/test_extraction.py; no new LLM calls in inner loops.

Return: short summary of what changed and before/after metrics.
EOF
)" \
  --allowedTools "Read,Edit,Bash,Grep,Glob" \
  --max-turns 25
```

Key flags:
- `-p` / `--print` — non-interactive, exits when done
- `--allowedTools` — pre-approved tools, no prompts mid-run
- `--max-turns N` — hard cap (prevents runaway loops)
- `--output-format json` — parseable for scripting
- `--dangerously-skip-permissions` — only for sandboxes you trust

### Codex CLI (`codex exec`)

```bash
cd ~/projects/officeqa-wt/retrieval
codex exec --sandbox workspace-write --full-auto \
  "Read AGENTS.md first and follow its Traversal Policy. \
   Improve retrieval reranking for summary tables in retrieve_v2.py. \
   Do not touch extract.py or solve.py. \
   Run: uv run pytest tests/test_retrieval.py and uv run python eval_retrieve.py \
   Acceptance: recall@10 improves on retrieve_eval fixtures, no test regressions."
```

Key flags:
- `exec` — non-interactive
- `--sandbox {read-only|workspace-write|danger-full-access}`
- `--full-auto` — no approval prompts
- `-m MODEL` — pin a model (e.g. `gpt-5-codex`)

Codex writes a rollout transcript under `~/.codex/sessions/` — grep that for
audit trails.

### Gemini CLI (`gemini`)

Used as independent reviewer. Read-only workflow by default.

```bash
cd ~/projects/officeqa-wt/retrieval
gemini --yolo -p "$(cat <<'EOF'
Read AGENTS.md first and follow its Traversal Policy.
You are a code reviewer. Read retrieve_v2.py and the diff on this branch vs
main. Focus only on reranking logic. List concrete risks, regressions, and
a single prioritized fix list. Do not modify files.
EOF
)"
```

Key flags:
- `-p` — single-prompt non-interactive mode
- `--yolo` — accept all actions (still read-only if prompt says so)
- `-m gemini-2.5-pro` — pin model
- `-a` / `--all-files` — include full workspace in context (large context
  window; use when cross-file reasoning matters)

### Cursor Agent (`cursor-agent`)

Prefer **`./scripts/cursor_agent_once.sh <stream>`** so `--workspace` and the task file stay consistent (see earlier “Cursor Agent CLI” subsection).

Manual equivalent:

```bash
cd ~/projects/officeqa-wt/fastpath
cursor-agent --workspace "$(pwd)" -p --force --trust -- \
  "Read AGENTS.md first and follow its Traversal Policy. \
   Formalize the deterministic fast-path: extract the fast-path block from \
   solve.py into fast_path.py with a resolve_all(data_requests) entry point. \
   Do not change behavior. Run: uv run pytest tests/ -x"
```

Key flags:
- `-p` / `--print` — non-interactive
- `--force` / `--yolo` — auto-approve tool use (pair with `--trust` for headless)
- `--workspace <path>` — run inside a git worktree checkout
- `--trust` — trust workspace without an interactive prompt
- `--output-format {text|json|stream-json}` — pick based on tooling needs
- `--model` — override model

## Task spec template

Hand every agent the same shape:

```
GOAL:         <one sentence>
IN SCOPE:     <explicit file list>
OUT OF SCOPE: <files or areas the agent must not touch>
COMMANDS:     <tests / evals to run>
ACCEPTANCE:   <objective pass/fail criteria>
RETURN:       <format: summary + metrics + file list>
```

Vague briefs get vague diffs. Concrete briefs get clean PRs.

## Merge protocol

1. Agent finishes, branch is on `agent/<name>` in its worktree.
2. From `main`, create a PR: `gh pr create --base main --head agent/<name>`.
3. Reviewer agent (Gemini or Claude) runs in review-only mode on the diff.
4. Human merges if green; squash commits by default.
5. `git worktree remove` and delete the branch.

Never let two agents edit the same file in parallel. If you need to, serialize
them — finish and merge one before spawning the next.

## Starting order (recommended)

1. **Retrieval first** — highest leverage, easiest to measure cleanly.
2. **Decompose + fast-path in parallel** — independent files once retrieval
   lands.
3. **Extraction** — after upstream signals are clean.
4. **Eval harness hardening** — ongoing background task.
5. **Ingestion integrity audit** — run last, since earlier streams may
   reshape the schema.

## What not to do

- Don't spawn an agent with a repo-wide brief ("fix everything").
- Don't let two agents edit the same file.
- Don't run `--dangerously-skip-permissions` / `--full-auto` outside a worktree.
- Don't commit generated artifacts (`*.sqlite`, `*.jsonl`, `*.log`) — gitignore covers them.
- Don't skip the acceptance-criteria section in a task spec. Without it,
  agents declare victory too early.
