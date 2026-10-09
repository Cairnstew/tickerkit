"""The user-facing query store: save/load whole fetches by ticker key.

This is the "connection" users point at a file inside their own repo. It layers
ticker-shaped keys and full model round-tripping (:class:`TickerSnapshot` /
:class:`HistoryResult`) over the low-level :class:`SQLiteCache`, and it also speaks the
:class:`Cache` protocol, so ``TickerKit(cache=QueryCache("..."))`` persists provider
calls automatically on the same file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cache import CacheLimits, SQLiteCache
from .models import HistoryResult, TickerSnapshot


class QueryCache:
    """The user-facing cache for whole ticker queries — save fetches, load them back.

    Same SQLite backing as :class:`SQLiteCache`, but aimed at application code: you open
    one with a path inside your own repo, save complete snapshot/history results under
    their natural ticker keys, and load them back as real model objects
    (:class:`~tickerkit.TickerSnapshot` / :class:`~tickerkit.HistoryResult`) — no
    re-fetching, no rate-limit spend. The file is git-trackable, so a repo can commit it
    and replay queries offline.

    It *also* speaks the low-level :class:`Cache` protocol (``get``/``set``/``clear``),
    so ``TickerKit(cache=QueryCache("my_cache.sqlite3"))`` both persists provider calls
    automatically and gives you ``save_snapshot``/``load_history`` on the same file.

    Default TTL for a saved query is **never expires** (``ttl=-1``) — that is what you
    want for a committed artifact. Pass a positive ``ttl`` to expire it later.

    Usable as an async context manager: ``async with QueryCache("c.sqlite3") as qc:``
    closes the underlying database on exit.
    """

    __slots__ = ("_db",)

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        limits: CacheLimits | None = None,
    ) -> None:
        self._db = SQLiteCache(path=path, limits=limits)

    # -- Cache protocol (so TickerKit(cache=QueryCache(...)) works too) ------

    async def get(self, key: str) -> Any | None:
        return await self._db.get(key)

    async def set(self, key: str, value: Any, ttl: float) -> None:
        await self._db.set(key, value, ttl)

    async def clear(self) -> None:
        await self._db.clear()

    # -- whole-query API -----------------------------------------------------

    @staticmethod
    def _snapshot_key(ticker: str) -> str:
        return f"query:snapshot:{ticker.strip().upper()}"

    @staticmethod
    def _history_key(
        ticker: str, timeframe: str, start: str | None, end: str | None
    ) -> str:
        return (
            f"query:history:{ticker.strip().upper()}:{timeframe.strip().lower()}"
            f":{start or 'any'}:{end or 'any'}"
        )

    async def save_snapshot(
        self,
        ticker: str,
        snapshot: TickerSnapshot | dict[str, Any],
        *,
        ttl: float = -1.0,
    ) -> None:
        """Persist a snapshot (or its ``to_dict()`` form) under ``ticker``."""
        data = snapshot.to_dict() if isinstance(snapshot, TickerSnapshot) else snapshot
        await self.set(self._snapshot_key(ticker), data, ttl)

    async def load_snapshot(self, ticker: str) -> TickerSnapshot | None:
        """Return the saved snapshot as a real :class:`TickerSnapshot`, or ``None``."""
        data = await self.get(self._snapshot_key(ticker))
        return TickerSnapshot.from_dict(data) if data is not None else None

    async def save_history(
        self,
        ticker: str,
        timeframe: str,
        *,
        start: str | None = None,
        end: str | None = None,
        history: HistoryResult | dict[str, Any] | None = None,
        ttl: float = -1.0,
    ) -> None:
        """Persist a history result (or its ``to_dict()`` form) under its key.

        ``start``/``end`` identify the window in the key; pass the same values to
        :meth:`load_history`. ``history`` is the result to store; it is required.
        """
        if history is None:
            raise ValueError("save_history requires history=<result>")
        data = history.to_dict() if isinstance(history, HistoryResult) else history
        await self.set(self._history_key(ticker, timeframe, start, end), data, ttl)

    async def load_history(
        self,
        ticker: str,
        timeframe: str,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> HistoryResult | None:
        """Return the saved history for that window as a :class:`HistoryResult`."""
        data = await self.get(self._history_key(ticker, timeframe, start, end))
        return HistoryResult.from_dict(data) if data is not None else None

    # -- administration -----------------------------------------------------

    def prune(self) -> dict[str, Any]:
        return self._db.prune()

    def stats(self) -> dict[str, Any]:
        return self._db.stats()

    def vacuum(self) -> None:
        self._db.vacuum()

    def close(self) -> None:
        self._db.close()

    def __len__(self) -> int:
        return len(self._db)

    async def __aenter__(self) -> QueryCache:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close()


__all__ = ["QueryCache"]