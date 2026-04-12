"""Pytest hooks — keep CI green without a multi-gigabyte ledger.sqlite checkout."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_LEDGER = _REPO_ROOT / "ledger.sqlite"


@pytest.fixture(autouse=True)
def _stub_bottomup_in_extraction_tests(request: pytest.FixtureRequest) -> Generator[None]:
    """Unit tests mock resolve_cells; keep bottom-up search off the real ledger."""
    node_path = getattr(request.node, "path", None) or getattr(request.node, "fspath", None)
    if node_path is None or Path(str(node_path)).name != "test_extraction.py":
        yield
        return
    with patch("solve.search_cells_bottomup", return_value=[]):
        yield


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip integration modules that always touch the ledger when it is absent."""
    if _LEDGER.exists():
        return
    skip = pytest.mark.skip(
        reason="ledger.sqlite not present (run build_ledger.py locally for full suite)"
    )
    for item in items:
        raw = getattr(item, "path", None) or getattr(item, "fspath", None)
        if raw is None:
            continue
        name = Path(str(raw)).name
        if name in ("test_feedback_loop.py", "test_ledger.py"):
            item.add_marker(skip)
