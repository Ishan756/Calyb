"""Pytest bootstrap: start the mock Semantic Scholar API for the session.

tests/test_fetch_papers.py drives the real fetch_papers code against a local
mock of the S2 endpoints, selected through the S2_BASE_URL environment variable.
When that file is run directly its own main() starts the mock; pytest never
calls main(), so without this conftest those tests would silently fall back to
the real api.semanticscholar.org and die on the anonymous rate limit (HTTP 429).

Run:  python -m pytest
"""

from __future__ import annotations

import os
import sys
import threading
from http.server import HTTPServer
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent


@pytest.fixture(scope="session", autouse=True)
def mock_s2_server():
    """One mock server for the whole session, on a random local port."""
    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    from test_fetch_papers import MockS2Handler  # local import: needs sys.path above

    server = HTTPServer(("127.0.0.1", 0), MockS2Handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    previous = os.environ.get("S2_BASE_URL")
    os.environ["S2_BASE_URL"] = f"http://127.0.0.1:{port}"
    try:
        yield
    finally:
        server.shutdown()
        if previous is None:
            os.environ.pop("S2_BASE_URL", None)
        else:
            os.environ["S2_BASE_URL"] = previous
