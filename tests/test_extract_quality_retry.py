"""Quality-retry path for extract_structured() — second LLM call when first JSON is unusable."""

from __future__ import annotations

import json
from unittest.mock import patch

from extract import extract_structured

_MIN_HTML = "<table><tr><th>Row</th><th>Val</th></tr><tr><td>A</td><td>100</td></tr></table>"


def test_quality_retry_triggers_second_llm_when_all_null_with_labels() -> None:
    """First response: labels filled but all values null → missing DR + retry with feedback."""
    spec: dict = {
        "computation": {},
        "data_requests": [
            {
                "id": "v1",
                "label": "Test value",
                "source": "corpus",
                "years": [1940],
            }
        ],
    }
    per_dr_entries = {
        "v1": [
            {"file": "stub.json", "html": _MIN_HTML},
        ]
    }

    first_json = json.dumps({"extractions": {"v1": {"values": [None], "labels": ["A"]}}})
    second_json = json.dumps({"extractions": {"v1": {"values": [100.0]}}})

    llm_calls: list[tuple[str, str]] = []

    def fake_llm(system: str, user: str, max_tokens: int = 1200, temperature: float = 0.0) -> str:
        llm_calls.append((system, user))
        if len(llm_calls) == 1:
            return first_json
        return second_json

    with patch("extract.llm", side_effect=fake_llm):
        out = extract_structured(spec, per_dr_entries, question="stub question")

    assert len(llm_calls) == 2
    assert "REVIEW FEEDBACK" in llm_calls[1][1]

    assert out is not None
    first_pass = json.loads(first_json)["extractions"]["v1"]
    assert first_pass["values"] == [None]
    final = out["extractions"]["v1"]
    assert final["values"] == [100.0]
    assert any(v is not None for v in final["values"])
