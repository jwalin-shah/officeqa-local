# One worktree per agent — use with Cursor

Each row is a **self-contained assignment**: open the worktree as its own Cursor window, then attach or paste the linked spec file into an Agent / Composer chat.

## Before you start

From the **main** repo (not inside a worktree):

```bash
./scripts/setup_worktrees.sh
./scripts/link_worktree_artifacts.sh
```

Default worktree parent is **`~/projects/officeqa-wt`**. To keep worktrees **inside** the clone (gitignored):
`OFFICEQA_WT_ROOT="$PWD/.agent-worktrees" ./scripts/setup_worktrees.sh`
then the same for `link_worktree_artifacts.sh`.

Each worktree is on branch **`agent/<stream>`** branched from `main`.

## In the Cursor app

1. **File → Open Folder…** → choose the **worktree directory** in the table below (not the main monorepo, unless you want one agent in main).
2. Start **Agent** (or Composer) in that window.
3. **Attach** the spec file (`@` → pick `retrieval.txt`, etc.) **or** paste the full contents of that file into the first message.
4. Add one line: *“Follow the spec only; do not touch OUT OF SCOPE files.”*

Shared data (symlinks from `link_worktree_artifacts.sh`): `ledger.sqlite`, `corpus_json/` when present on the main clone.

## Assignments

| Workstream | Worktree folder (default) | Spec file (attach or paste) |
|------------|---------------------------|-----------------------------|
| Retrieval | `<WT_ROOT>/retrieval` | [retrieval.txt](retrieval.txt) |
| Decompose | `<WT_ROOT>/decompose` | [decompose.txt](decompose.txt) |
| Fast-path | `<WT_ROOT>/fastpath` | [fastpath.txt](fastpath.txt) |
| Extraction | `<WT_ROOT>/extract` | [extract.txt](extract.txt) |
| Eval harness | `<WT_ROOT>/eval` | [eval.txt](eval.txt) |
| Ingestion | `<WT_ROOT>/ingest` | [ingest.txt](ingest.txt) |

Replace `<WT_ROOT>` with your parent path, e.g. `~/projects/officeqa-wt` or `/…/officeqa-local/.agent-worktrees`.

## Optional: headless instead of the app

- One shot: `./scripts/cursor_agent_once.sh retrieval` (see repo root `README.md` and `docs/AGENT_ORCHESTRATION.md`).
- Check loop + Cursor after failures: `OFFICEQA_AGENT_CURSOR=1 ./scripts/agent_iterate.sh`.

## Merge reminder

When an agent finishes: review diff in that worktree, push branch `agent/<stream>`, open PR to `main`, merge, then remove worktree or run `./scripts/refresh_worktrees_from_main.sh` before the next run.
