"""Pytest hooks — keep CI green without a multi-gigabyte ledger.sqlite checkout."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import suppress
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
    with patch("find.search_cells_bottomup", return_value=[]):
        yield


@pytest.fixture(autouse=True)
def _reset_find_tls_conn(request: pytest.FixtureRequest) -> Generator[None]:
    """Reset find._TLS.conn after each test that may have swapped LEDGER_PATH.

    Tests in test_extraction.py spin up temp ledgers and monkeypatch LEDGER_PATH.
    After teardown monkeypatch restores the path, but the thread-local connection
    still points at the now-deleted temp file. Subsequent tests then get a stale
    connection and hit 'no such table' errors. Clear it after every test so the
    next caller gets a fresh connection to the real ledger.
    """
    node_path = getattr(request.node, "path", None) or getattr(request.node, "fspath", None)
    if node_path is None or Path(str(node_path)).name != "test_extraction.py":
        yield
        return
    yield
    try:
        import find as find_mod

        conn = getattr(find_mod._TLS, "conn", None)
        if conn is not None:
            with suppress(Exception):
                conn.close()
            with suppress(AttributeError):
                delattr(find_mod._TLS, "conn")
    except ImportError:
        pass


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
