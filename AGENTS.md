# Agent Rules

Shared repo-level instructions for headless coding agents working in
`officeqa-local`. This file is meant to be referenced by Claude CLI, Gemini
CLI, Cursor Agent, and any other prompt-driven coding assistant used on this
repo.

## Traversal Policy

Default to token-efficient traversal. Prefer these tools before raw shell
commands:

1. `llm-tldr` for codebase structure and cross-file analysis
2. `rtk` for compact file reads, trees, grep, diffs, and command output
3. raw shell tools (`rg`, `sed`, `git`, `sqlite3`, `python`) only when the
   compact tools are insufficient or exact unfiltered output is required

## Default Tool Choices

Use `llm-tldr` for:
- repo tree and structural overview
- symbol search
- relevant-context extraction
- caller/callee impact checks
- architectural or dependency traversal

Common commands:

```bash
llm-tldr tree .
llm-tldr structure . --lang python
llm-tldr search "resolve_cells|retrieve" .
llm-tldr context solve --project .
llm-tldr calls solve.py solve
llm-tldr impact solve
```

Use `rtk` for:
- compact directory listings
- compact file reads
- compact grep output
- compact git diff/status/log output
- compact pytest / lint / error output

Common commands:

```bash
rtk tree
rtk read README.md
rtk grep "retrieve_v2|resolve_cells" .
rtk diff git diff -- retrieve_v2.py
rtk pytest tests/test_retrieval.py -q --no-cov
rtk err uv run python eval_retrieve.py
```

Use raw tools when needed:
- `rg` for exact/fast search or when `rtk grep` is too lossy
- `sed -n`, `sqlite3`, or direct Python for exact inspection
- plain `git diff` when the full patch matters

## Reading Discipline

- Do not start by dumping large files end-to-end.
- Start with `llm-tldr tree` / `structure` or `rtk tree`.
- Read only the relevant slices of large files.
- Prefer structural queries before content-heavy reads.
- When investigating a function, use `llm-tldr context`, `calls`, or `impact`
  before opening multiple files manually.

## Editing Discipline

- Keep changes scoped to the assigned workstream.
- Do not widen surface area unless the fix truly requires it.
- If a task mentions a bounded file list, treat everything else as read-only.
- Preserve existing evaluation shortcuts: avoid live LLM calls in hot eval loops.

## Validation Discipline

- Run the smallest test or eval command that proves the change.
- Prefer `rtk pytest ...` or `rtk err ...` for compact output first.
- If compact output hides needed detail, rerun the exact raw command once.

## Real-Time Progress Output

Every eval/test loop **must** print one line per question/item as it completes — not at the end.

Rules:
- Use `print(..., flush=True)` on every per-item progress line.
- Call `sys.stdout.reconfigure(line_buffering=True)` at the top of `main()` in all eval scripts.
- Never buffer results and print only at the end — background tasks and monitors depend on live output.
- In `concurrent.futures` loops, print inside the `as_completed` callback, not after the pool closes.

## Repo-Specific Notes

- Primary corpus is `corpus_json/`; `corpus/` is legacy fallback.
- Prefer ledger-backed workflows over old grep-only paths unless explicitly
  working on a legacy baseline.
- For retrieval work, avoid introducing new LLM calls in retrieval hot paths.
- For eval work, prefer cached specs in `decompose_eval.full.jsonl`.

## Prompting Rule For External Agents

When launching an external agent on this repo, tell it explicitly:

```text
Read AGENTS.md first and follow its Traversal Policy.
```
