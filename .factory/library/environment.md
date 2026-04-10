# Environment

**What belongs here:** Required env vars, external dependencies, setup notes.
**What does NOT belong here:** Service ports/commands (use `.factory/services.yaml`).

---

## Required Environment Variables
- `DEDALUS_API_KEY` — API key for DeepSeek via Dedalus
- `DEDALUS_API_BASE` — `https://api.dedaluslabs.ai/v1`
- `OFFICEQA_MODEL` — Model name, defaults to `deepseek/deepseek-chat` in solve.py, `deepseek-chat` in extract.py (minor inconsistency, both work)

## Python
- Python 3.14.3 via `uv`
- Virtual env: `.venv/` (managed by uv)
- Key deps: openai, instructor, lxml, pandas, rapidfuzz, scipy, pydantic, pint, python-dotenv

## Data Files
- `corpus_json/` — 697 parsed bulletin JSONs (gitignored, primary corpus)
- `corpus/` — 697 `.txt` files (legacy fallback)
- `ledger.sqlite` — SQLite ledger built from corpus_json (gitignored, ~7.8GB pre-optimization)
- `officeqa_full.csv` — 246 benchmark questions with gold answers
- `decompose_eval.full.jsonl` — 246 pre-cached QuestionSpecs for deterministic testing
- `corpus_index.pkl` — legacy BM25 index (535MB, to be retired)

## Machine
- 8GB RAM, 8 cores (Apple Silicon)
- Resource-constrained: avoid loading full DB into memory
