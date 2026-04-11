# Agent Orchestration

How to split officeqa-local work across multiple coding agents running in
headless mode inside isolated git worktrees.

## Goals
- Parallelize bounded workstreams without merge chaos.
- Keep `main` stable; each agent works on its own branch + worktree.
- Use each tool for what it's best at.
- One human (you) is the merger/reviewer.

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

Create them:

```bash
mkdir -p ~/projects/officeqa-wt
cd ~/projects/officeqa-local
for stream in retrieval decompose fastpath extract eval ingest; do
  git worktree add -b "agent/$stream" "../officeqa-wt/$stream" main
done
git worktree list
```

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
  "Improve retrieval reranking for summary tables in retrieve_v2.py. \
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

```bash
cd ~/projects/officeqa-wt/fastpath
cursor-agent -p --force \
  "Formalize the deterministic fast-path: extract the fast-path block from \
   solve.py into fast_path.py with a resolve_all(data_requests) entry point. \
   Do not change behavior. Run: uv run pytest tests/ -x"
```

Key flags:
- `-p` / `--print` — non-interactive
- `--force` — auto-approve edits
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
