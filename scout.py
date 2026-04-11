#!/usr/bin/env python3
"""Pre-decompose index scout.

Given a raw question, peek at the table index and return a compact summary
of the top-k candidate tables. This grounds the decompose call in labels
that actually exist in the corpus instead of letting the LLM guess row
labels, column headers, and year offsets from scratch.

Deterministic, no LLM call, <100ms once the index is warm.
"""

from retrieve_v2 import retrieve_from_question


def scout(question: str, top_k: int = 5) -> str:
    """Return a short human-readable summary of top-k candidate tables.

    Empty string if the index lookup fails or returns nothing — caller
    treats that as "no hint available" and decomposes unguided.
    """
    try:
        hits = retrieve_from_question(question, top_k=top_k)
    except Exception:
        return ""
    if not hits:
        return ""

    lines = ["CORPUS SCOUT — top candidate tables found for this question:"]
    for i, e in enumerate(hits, 1):
        file = e.get("file") or "?"
        section = (e.get("section") or "").strip()
        caption = (e.get("caption") or "").strip()
        headers = e.get("column_headers") or []
        rows = e.get("row_labels") or []
        years = sorted(y for y in (e.get("years") or []) if isinstance(y, int))
        year_str = f"{min(years)}–{max(years)}" if years else "no years"

        lines.append(f"  [{i}] {file}")
        if section:
            lines.append(f"      section: {section[:140]}")
        if caption:
            lines.append(f"      caption: {caption[:140]}")
        if headers:
            lines.append(f"      cols: {' | '.join(h for h in headers[:6] if h)}")
        if rows:
            lines.append(f"      rows: {' | '.join(r for r in rows[:6] if r)}")
        lines.append(f"      years in table: {year_str}")

    lines.append("")
    lines.append(
        "Use these to ground row_hint/column_hint in labels that actually "
        "appear in the corpus. If none of these look relevant, trust your "
        "own judgment."
    )
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        print(scout(" ".join(sys.argv[1:])))
    else:
        print("Usage: uv run python scout.py <question>")
