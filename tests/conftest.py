"""Shared fixtures.

CI never touches the network: every provider adapter test replays a recorded JSON fixture
through ``respx``. The single opt-in live test is marked ``live`` and excluded by default.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> Any:
    """Load a recorded response fixture by filename."""
    return json.loads((FIXTURES / name).read_text())


@pytest.fixture
def fx() -> Any:
    """Fixture loader callable, injected into tests as ``fx``."""
    return load_fixture


@pytest.fixture(autouse=True)
def _no_env_keys(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep developer API keys out of tests unless a test sets them explicitly."""
    for var in (
        "TICKERKIT_COINGECKO_API_KEY",
        "TICKERKIT_COINGECKO_PRO_API_KEY",
        "TICKERKIT_CMC_API_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    yield