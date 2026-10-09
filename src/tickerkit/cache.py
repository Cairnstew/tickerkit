"""TTL caches behind a single pluggable interface.

Three implementations of the async :class:`Cache` protocol:

* :class:`MemoryCache` — the default. Fast, per-process, ephemeral.
* :class:`SQLiteCache` — a single-file disk cache that can be committed to git. Values
  are stored as canonical JSON (deterministic bytes for identical data), the database is
  bounded by :class:`CacheLimits`, and eviction is TTL-first then LRU. The file is the
  whole cache — no WAL/shm sidecars — so it stays a single small artifact on disk.
* :class:`NullCache` — stores nothing; used when caching is disabled.

The cache exists purely so repeated calls do not burn free-tier quota.
"""

from __future__ import annotations

import asyncio
import base64
import json
import pickle
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class CacheEntry:
    value: Any
    expires_at: float


@runtime_checkable
class Cache(Protocol):
    """Minimal async cache contract."""

    async def get(self, key: str) -> Any | None:
        """Return the cached value, or ``None`` when absent or expired."""
        ...

    async def set(self, key: str, value: Any, ttl: float) -> None:
        """Store ``value`` for ``ttl`` seconds."""
        ...

    async def clear(self) -> None:
        """Drop everything."""
        ...


class MemoryCache:
    """TTL cache guarded by an ``asyncio.Lock``.

    The lock matters: the orchestrator fetches many capabilities concurrently and two
    tasks must not both miss and both fetch.
    """

    __slots__ = ("_data", "_lock")

    def __init__(self) -> None:
        self._data: dict[str, CacheEntry] = {}
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.monotonic():
                self._data.pop(key, None)
                return None
            return entry.value

    async def set(self, key: str, value: Any, ttl: float) -> None:
        async with self._lock:
            self._data[key] = CacheEntry(value=value, expires_at=time.monotonic() + ttl)

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        return len(self._data)


class NullCache:
    """A cache that stores nothing; used when caching is disabled."""

    __slots__ = ()

    async def get(self, key: str) -> Any | None:
        return None

    async def set(self, key: str, value: Any, ttl: float) -> None:
        return None

    async def clear(self) -> None:
        return None


# --------------------------------------------------------------------------- disk


@dataclass(frozen=True, slots=True)
class CacheLimits:
    """Hard bounds that keep the cache "not too big".

    ``max_bytes`` counts JSON payload bytes only (row overhead is a few hundred bytes per
    entry at most). Eviction runs after every write: expired entries first, then the
    least-recently-used until both bounds hold.
    """

    max_entries: int = 2000
    max_bytes: int = 10 * 1024 * 1024  # 10 MiB


def _now_ms() -> float:
    # Wall clock, not monotonic: the file outlives processes, and monotonic resets.
    return time.time() * 1000.0


def _encode(value: Any) -> tuple[str, str]:
    """Canonical JSON for git-friendly, deterministic bytes.

    ``sort_keys`` + compact separators mean identical data always produces identical
    payload bytes. Anything not JSON-native falls back to pickled base64 (second tuple
    element carries the format tag).
    """
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True),
            "json",
        )
    except (TypeError, ValueError):
        blob = base64.b64encode(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)).decode(
            "ascii"
        )
        return blob, "pickle"


def _decode(payload: str, fmt: str) -> Any:
    if fmt == "json":
        return json.loads(payload)
    # Pickle payloads are only created by this process from trusted in-process values;
    # never point the cache at a file you do not control.
    return pickle.loads(base64.b64decode(payload.encode("ascii")))


class SQLiteCache:
    """Async TTL cache persisted to one SQLite file — git-trackable and bounded.

    Layout (single ``entries`` table, no sidecar files):

    * ``key``            TEXT PRIMARY KEY — provider-assigned, e.g. ``cg:coins/list``.
    * ``payload``        TEXT — canonical JSON (or base64 pickle in the fallback path).
    * ``payload_format`` TEXT — ``json`` or ``pickle``.
    * ``ttl``            REAL — seconds. Negative means **never expire** (immortal,
      useful for long-lived history data you want to keep in git). Zero matches
      :class:`MemoryCache` and expires immediately. Positive expires ``ttl`` seconds
      after write.
    * ``stored_ms`` / ``last_used_ms`` — wall-clock millis; ``last_used_ms`` feeds LRU.

    After every write, :meth:`SQLiteCache.set` evicts beyond :class:`CacheLimits`
    (expired first, then LRU), and ``auto_vacuum`` shrinks the file so the committed
    artifact stays small. Reads bump ``last_used_ms`` so hot entries survive eviction.

    Thread-safety: callers on one event loop (as tickerkit is) share this instance
    freely; a process-level lock serialises accesses and SQLite's own locking handles
    the (rare) multi-writer case.
    """

    __slots__ = ("_conn", "_limits", "_lock", "_path")

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        limits: CacheLimits | None = None,
    ) -> None:
        self._path = Path(path) if path is not None else Path("tickerkit_cache.sqlite3")
        self._limits = limits if limits is not None else CacheLimits()
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self._path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Single-file cache: DELETE journal leaves no -wal/-shm sidecars, and
        # auto_vacuum=FULL keeps the file itself small after eviction (it must be set
        # before the table is created to take effect on a fresh database).
        self._conn.execute("PRAGMA journal_mode=DELETE")
        self._conn.execute("PRAGMA auto_vacuum=FULL")
        self._conn.execute("PRAGMA busy_timeout=3000")
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS entries(
                    key TEXT PRIMARY KEY,
                    payload TEXT NOT NULL,
                    payload_format TEXT NOT NULL,
                    ttl REAL NOT NULL,
                    stored_ms REAL NOT NULL,
                    last_used_ms REAL NOT NULL
                )"""
            )
            self._conn.commit()

    # -- protocol -----------------------------------------------------------

    async def get(self, key: str) -> Any | None:
        now = _now_ms()
        with self._lock:
            row = self._conn.execute(
                "SELECT payload, payload_format, ttl, stored_ms FROM entries WHERE key = ?",
                (key,),
            ).fetchone()
            if row is None:
                return None
            if row["ttl"] >= 0 and row["stored_ms"] + row["ttl"] * 1000.0 <= now:
                self._conn.execute("DELETE FROM entries WHERE key = ?", (key,))
                self._conn.commit()
                return None
            self._conn.execute(
                "UPDATE entries SET last_used_ms = ? WHERE key = ?", (now, key)
            )
            self._conn.commit()
            return _decode(row["payload"], row["payload_format"])

    async def set(self, key: str, value: Any, ttl: float) -> None:
        payload, fmt = _encode(value)
        now = _now_ms()
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO entries(
                    key, payload, payload_format, ttl, stored_ms, last_used_ms
                ) VALUES (?, ?, ?, ?, ?, ?)""",
                (key, payload, fmt, ttl, now, now),
            )
            self._prune_locked(now, protected_key=key)
            self._conn.commit()

    async def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM entries")
            self._conn.commit()

    # -- administration -----------------------------------------------------

    def prune(self) -> dict[str, Any]:
        """Evict expired/beyond-limit entries now. Returns stats after pruning."""
        now = _now_ms()
        with self._lock:
            self._prune_locked(now)
            self._conn.commit()
        return self.stats()

    def stats(self) -> dict[str, Any]:
        """Current state: size, bounds, expiry breakdown. Cheap, no pruning."""
        with self._lock:
            row = self._conn.execute(
                """SELECT COUNT(*) AS n,
                          COALESCE(SUM(LENGTH(payload)), 0) AS bytes,
                          MIN(stored_ms) AS oldest,
                          MAX(stored_ms) AS newest
                   FROM entries"""
            ).fetchone()
            expired = self._conn.execute(
                "SELECT COUNT(*) AS n FROM entries WHERE ttl >= 0"
            ).fetchone()["n"]
            now = _now_ms()
            already_expired = self._conn.execute(
                "SELECT COUNT(*) AS n FROM entries WHERE ttl >= 0"
                " AND stored_ms + ttl * 1000.0 <= ?",
                (now,),
            ).fetchone()["n"]
        return {
            "path": str(self._path),
            "entries": row["n"],
            "bytes": row["bytes"],
            "max_entries": self._limits.max_entries,
            "max_bytes": self._limits.max_bytes,
            "oldest_ms": row["oldest"],
            "newest_ms": row["newest"],
            "expired_now": already_expired,
            "expiring_entries": expired,
        }

    def vacuum(self) -> None:
        """Reclaim file pages. Mostly useful before committing a large cache to git."""
        with self._lock:
            self._conn.execute("VACUUM")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __len__(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS n FROM entries").fetchone()
        return row["n"]

    # -- internals ----------------------------------------------------------

    def _prune_locked(self, now: float, protected_key: str | None = None) -> None:
        # 1) anything past its TTL
        self._conn.execute(
            "DELETE FROM entries WHERE ttl >= 0 AND stored_ms + ttl * 1000.0 <= ?",
            (now,),
        )
        # 2) still over a bound? shed the least-recently-used, deleting only as much as
        #    needed so a write never evicts more than necessary (a fixed batch size is
        #    wrong on a small table — it would empty the cache to satisfy a tight cap).
        #    The key that was just written is never a candidate, so a single oversized
        #    value (e.g. DefiLlama's full protocols list) survives its own refresh and
        #    older entries are shed around it.
        while True:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n, COALESCE(SUM(LENGTH(payload)), 0) AS b FROM entries"
            ).fetchone()
            if row["n"] <= self._limits.max_entries and row["b"] <= self._limits.max_bytes:
                return
            if row["n"] > self._limits.max_entries:
                shed = row["n"] - self._limits.max_entries
            else:
                # over the byte budget with the count already fine: shed an eighth of
                # what is left (at least one) rather than one row per loop.
                shed = max(1, row["n"] // 8)
            query = (
                "DELETE FROM entries WHERE key IN ("
                "SELECT key FROM entries"
                + (" WHERE key != ?" if protected_key is not None else "")
                + " ORDER BY last_used_ms ASC LIMIT ?)"
            )
            params: tuple[Any, ...] = (
                (protected_key, shed) if protected_key is not None else (shed,)
            )
            cur = self._conn.execute(query, params)
            # A single oversized entry (alone over the byte budget) cannot be shed; stop
            # rather than looping forever. It stays stored — bounded by the entry itself.
            if cur.rowcount == 0:
                return


__all__ = [
    "Cache",
    "CacheEntry",
    "CacheLimits",
    "MemoryCache",
    "NullCache",
    "SQLiteCache",
]