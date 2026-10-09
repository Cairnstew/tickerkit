"""Tests for the SQLite disk cache: bounded, TTL + LRU, git-trackable bytes."""

from __future__ import annotations

import asyncio
import json

from tickerkit import Cache, CacheLimits, MemoryCache, NullCache, SQLiteCache


def _cache(tmp_path, **limits) -> SQLiteCache:
    return SQLiteCache(
        tmp_path / "cache.sqlite3",
        limits=CacheLimits(**limits),
    )


async def test_protocol_conformance() -> None:
    assert isinstance(SQLiteCache, Cache)
    assert isinstance(MemoryCache(), Cache)
    assert isinstance(NullCache(), Cache)


async def test_json_roundtrip(tmp_path) -> None:
    cache = _cache(tmp_path)
    try:
        await cache.set("k", {"a": [1, 2, {"b": None}], "c": "x"}, ttl=60)
        assert await cache.get("k") == {"a": [1, 2, {"b": None}], "c": "x"}
    finally:
        cache.close()


async def test_missing_key_returns_none(tmp_path) -> None:
    cache = _cache(tmp_path)
    try:
        assert await cache.get("nope") is None
    finally:
        cache.close()


async def test_ttl_expiry(tmp_path) -> None:
    cache = _cache(tmp_path)
    try:
        await cache.set("k", "v", ttl=0.05)
        assert await cache.get("k") == "v"
        await asyncio.sleep(0.07)
        assert await cache.get("k") is None
        assert len(cache) == 0
    finally:
        cache.close()


async def test_negative_ttl_is_immortal(tmp_path) -> None:
    cache = _cache(tmp_path)
    try:
        await cache.set("k", "history-bytes", ttl=-1)
        await asyncio.sleep(0.05)
        assert await cache.get("k") == "history-bytes"
    finally:
        cache.close()


async def test_zero_ttl_expires_immediately(tmp_path) -> None:
    cache = _cache(tmp_path)
    try:
        await cache.set("k", "v", ttl=0)
        assert await cache.get("k") is None
    finally:
        cache.close()


async def test_none_value_is_stored(tmp_path) -> None:
    # Matches MemoryCache semantics: a stored None is indistinguishable from a miss to
    # callers (ProviderContext.cached() keys off `is not None`), but it is stored.
    cache = _cache(tmp_path)
    try:
        await cache.set("k", None, ttl=60)
        assert len(cache) == 1
        assert await cache.get("k") is None
    finally:
        cache.close()


async def test_max_entries_lru_eviction(tmp_path) -> None:
    cache = _cache(tmp_path, max_entries=3)
    try:
        for i in range(3):
            await cache.set(f"k{i}", f"v{i}", ttl=-1)
        # touch k0 so it is the most recent
        assert await cache.get("k0") == "v0"
        await cache.set("k3", "v3", ttl=-1)
        assert len(cache) == 3
        assert await cache.get("k0") == "v0"  # survived via LRU bump
        assert await cache.get("k1") is None  # evicted first
        assert await cache.get("k2") == "v2"
        assert await cache.get("k3") == "v3"
    finally:
        cache.close()


async def test_max_bytes_eviction(tmp_path) -> None:
    cache = _cache(tmp_path, max_bytes=4096)
    try:
        # ~3 KiB each -> only one or two fit under 4096 bytes
        big = {"blob": "x" * 3000}
        await cache.set("a", big, ttl=-1)
        await cache.set("b", big, ttl=-1)
        s = cache.stats()
        assert s["bytes"] <= s["max_bytes"], s
        # the older entry (a) was evicted in favour of b because set bumps b's recency
        assert await cache.get("a") is None
        assert await cache.get("b") == big
    finally:
        cache.close()


async def test_clear(tmp_path) -> None:
    cache = _cache(tmp_path)
    try:
        await cache.set("a", 1, ttl=60)
        await cache.set("b", 2, ttl=60)
        await cache.clear()
        assert len(cache) == 0
        assert cache.stats()["entries"] == 0
    finally:
        cache.close()


async def test_persistence_across_reopen(tmp_path) -> None:
    path = tmp_path / "cache.sqlite3"
    first = SQLiteCache(path)
    await first.set("k", {"persisted": True}, ttl=3600)
    first.close()

    second = SQLiteCache(path)
    try:
        assert await second.get("k") == {"persisted": True}
    finally:
        second.close()


async def test_pickle_fallback_for_non_json(tmp_path) -> None:
    cache = _cache(tmp_path)
    try:
        value = {"dt": __import__("datetime").datetime(2024, 1, 1)}
        await cache.set("k", value, ttl=60)
        assert await cache.get("k") == value
    finally:
        cache.close()


async def test_canonical_json_bytes_are_deterministic(tmp_path) -> None:
    """The same data must produce identical stored bytes (git-friendly)."""
    cache = _cache(tmp_path)
    try:
        for _ in range(2):
            await cache.set(
                "k",
                {"z": [3, 1, {"a": 1}], "a": "x"},
                ttl=60,
            )
        with cache._lock:
            row = cache._conn.execute(
                "SELECT payload, payload_format FROM entries WHERE key = 'k'"
            ).fetchone()
        assert row["payload_format"] == "json"
        parsed = json.loads(row["payload"])
        assert parsed == {"z": [3, 1, {"a": 1}], "a": "x"}
        assert row["payload"] == json.dumps(
            {"z": [3, 1, {"a": 1}], "a": "x"}, sort_keys=True, separators=(",", ":")
        )
    finally:
        cache.close()


async def test_expired_entries_are_pruned_before_lru(tmp_path) -> None:
    cache = _cache(tmp_path, max_entries=2)
    try:
        await cache.set("old", "v", ttl=0.01)
        await asyncio.sleep(0.02)
        await cache.set("keep1", "v", ttl=-1)
        await cache.set("keep2", "v", ttl=-1)
        # Expired entry removed first; immortal ones fit within the limit.
        assert await cache.get("old") is None
        assert len(cache) == 2
    finally:
        cache.close()


async def test_file_shrinks_after_prune(tmp_path) -> None:
    """auto_vacuum keeps the committed artifact small."""
    cache = _cache(tmp_path, max_entries=500)
    path = tmp_path / "cache.sqlite3"
    try:
        for i in range(200):
            await cache.set(f"k{i}", "x" * 1000, ttl=0.01)
        await asyncio.sleep(0.02)
        cache.prune()
        size = path.stat().st_size
        assert size < 64 * 1024, f"file should shrink after prune, got {size} bytes"
        assert len(cache) == 0
    finally:
        cache.close()


async def test_concurrent_set_smoke(tmp_path) -> None:
    cache = _cache(tmp_path, max_entries=100)
    try:
        await asyncio.gather(
            *(cache.set(f"k{i}", i, ttl=-1) for i in range(50))
        )
        assert len(cache) == 50
        assert await cache.get("k49") == 49
    finally:
        cache.close()


async def test_just_written_entry_survives_its_own_eviction(tmp_path) -> None:
    """A single oversized write (e.g. DefiLlama's protocols list) must not shed
    itself: older entries around it are evicted, it is kept."""
    cache = _cache(tmp_path, max_bytes=4096)
    try:
        big = {"blob": "x" * 3000}
        await cache.set("younger_other", {"blob": "z" * 1500}, ttl=-1)
        await cache.set("big_list", big, ttl=-1)
        # youngest other was written right before big_list, so with plain LRU it would
        # be shed first; the protection rule keeps big_list regardless.
        assert await cache.get("big_list") == big
        s = cache.stats()
        assert s["bytes"] <= s["max_bytes"], s
    finally:
        cache.close()


async def test_oversized_single_entry_is_stored_and_does_not_loop(tmp_path) -> None:
    """A value that alone exceeds max_bytes is still stored (bounded by the entry
    itself); the eviction loop must not spin forever."""
    cache = _cache(tmp_path, max_bytes=1024)
    try:
        big = {"blob": "x" * 5000}
        await cache.set("huge", big, ttl=-1)
        assert await cache.get("huge") == big
    finally:
        cache.close()