**DEPRECATED:** This component was part of an earlier bottom-up pipeline approach. The current approach uses the lean MCP server (`nomcp/mcp_server.py`).

# Bottom-Up Validation

Layer-by-layer validation of the solve pipeline, starting from the bottom (LLM extraction) and working up.

## Layers (bottom to top)

1. **Layer 1: LLM Extraction** — Given perfect evidence (hand-curated), can the LLM extract the right values and compute the answer?
2. **Layer 2: Evidence Formatting** — Does the serialization format cause misreads? (vertical vs table vs raw)
3. **Layer 3: Row/Evidence Selection** — Does grep return the right rows from the right file?
4. **Layer 4: Table Selection** — Does grep even hit the right file?
5. **Layer 5: Query Formulation** — Are we searching for the right terms?

## Usage

```bash
# Run layer 1 tests (requires OPENROUTER_API_KEY)
python3 layer1_extraction.py
```
