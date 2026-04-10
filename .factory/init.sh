#!/bin/bash
set -e

cd /Users/jwalinshah/projects/officeqa-local

# Install dependencies (idempotent)
uv sync

# Verify .env exists
if [ ! -f .env ]; then
    echo "ERROR: .env file missing. Copy .env.example and fill in API keys."
    exit 1
fi

# Verify corpus_json exists
if [ ! -d corpus_json ] || [ $(ls corpus_json/*.json 2>/dev/null | wc -l) -lt 600 ]; then
    echo "ERROR: corpus_json/ missing or incomplete. Need 697 JSON files."
    exit 1
fi

# Verify ledger exists
if [ ! -f ledger.sqlite ]; then
    echo "WARNING: ledger.sqlite missing. Run 'uv run python build_ledger.py' to build."
fi

# Verify pre-cached decompose exists
if [ ! -f decompose_eval.full.jsonl ]; then
    echo "WARNING: decompose_eval.full.jsonl missing. Retrieval tests need this."
fi

echo "Environment ready."
