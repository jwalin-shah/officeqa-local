# OfficeQA Local

Run the OfficeQA benchmark locally without Arena infrastructure.

Uses Claude + orchestrator logic to answer U.S. Treasury Bulletin questions.

## Setup

```bash
# Copy .env and add your API key
cp .env.example .env

# Activate venv
source .venv/bin/activate
```

## Usage

### Answer a single question

```bash
python agent.py "What were the total expenditures for U.S national defense in 1940?"
```

### Evaluate on full benchmark

```bash
python evaluate.py           # All 246 questions
python evaluate.py 10        # First 10 questions (for testing)
```

## Files

- `agent.py` — Main orchestrator agent
- `evaluate.py` — Benchmark evaluation script
- `corpus/` — Treasury Bulletin txt files
- `officeqa_full.csv` — 246 test Q&A pairs
- `cpi.py` — CPI calculator utility

## Performance

Target: Beat 75% (previous best with Goose)

Current: Building...
