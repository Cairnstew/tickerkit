"""Partial-failure tolerance, the cache, and the HTTP retry policy."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from tickerkit.cache import MemoryCache, NullCache
from tickerkit.capabilities import Capability
from tickerkit.config import Config
from tickerkit.core import TickerKit
from tickerkit.http import HttpClient, RateLimiter, RetryPolicy
from tickerkit.models import Identity, Market, VenueTicker, utcnow
from tickerkit.providers.base import Provider
from tickerkit.providers.coingecko import DEMO_ROOT


class ExplodingProvider(Provider):
    """A provider that fails on every capability it claims."""

    name = "exploding"
    supports = frozenset(
        {Capability.META, Capability.VENUE_TICKER, Capability.ORDERBOOK, Capability.TRADES}
    )

    async def fetch_identity(self, asset: Any) -> Identity:
        raise RuntimeError("identity backend is down")

    async def fetch_venue_tickers(self, asset: Any) -> tuple[VenueTicker, ...]:
        raise TimeoutError("venue gateway timeout")

    async def fetch_orderbooks(self, asset: Any) -> tuple[Any, ...]:
        raise RuntimeError("orderbook backend is down")

    async def fetch_trades(self, asset: Any) -> tuple[Any, ...]:
        raise RuntimeError("trades backend is down")


class HealthyProvider(Provider):
    """A provider that returns a little of everything."""

    name = "healthy"
    supports = frozenset({Capability.RESOLVE, Capability.META, Capability.AGG_MARKET,
                          Capability.VENUE_TICKER})

    async def resolve(self, query: str) -> Any:
        from tickerkit.resolve import ResolvedAsset

        return ResolvedAsset(query=query, symbol=query.upper(), coingecko_id="bitcoin",
                             name="Bitcoin", market_cap=1.0)

    async def fetch_identity(self, asset: Any) -> Identity:
        return Identity(symbol="BTC", name="Bitcoin", coingecko_id="bitcoin")

    async def fetch_market(self, asset: Any) -> Market:
        return Market(quote_currency="USD", aggregated_price=81779.0, market_cap=1234.0)

    async def fetch_venue_tickers(self, asset: Any) -> tuple[VenueTicker, ...]:
        # Priced consistently with the fake aggregator above, so the snapshot is coherent.
        return (
            VenueTicker(exchange="healthy", symbol="BTC/USD", quote_currency="USD",
                        bid=81779.0, ask=81779.1, timestamp=utcnow()),
            VenueTicker(exchange="healthy2", symbol="BTC/USD", quote_currency="USD",
                        bid=81778.9, ask=81779.0, timestamp=utcnow()),
        )


@pytest.mark.asyncio
async def test_a_broken_provider_still_yields_a_valid_snapshot() -> None:
    """The headline guarantee: one provider failing must not fail the snapshot."""
    kit = TickerKit(Config(cache_enabled=False), providers=(HealthyProvider, ExplodingProvider))
    snap = await kit.get_ticker("BTC")

    assert snap.identity.coingecko_id == "bitcoin"
    assert snap.market.aggregated_price == 81779.0
    # market.price prefers the venue consensus and falls back to the aggregator.
    assert snap.market.price == snap.market.consensus_price
    assert snap.market.price == pytest.approx(81779.0, rel=1e-4)
    assert len(snap.venues) == 2

    # Every failure is recorded, with enough detail to act on.
    providers_failed = {e.provider for e in snap.data_quality.errors}
    assert providers_failed == {"exploding"}
    assert {e.capability for e in snap.data_quality.errors} == {
        Capability.META,
        Capability.VENUE_TICKER,
        Capability.ORDERBOOK,
        Capability.TRADES,
    }
    types = {e.exception_type for e in snap.data_quality.errors}
    assert types == {"RuntimeError", "TimeoutError"}
    assert all(e.message for e in snap.data_quality.errors)
    assert all(e.occurred_at.tzinfo is not None for e in snap.data_quality.errors)


@pytest.mark.asyncio
async def test_errors_are_not_double_counted() -> None:
    kit = TickerKit(Config(cache_enabled=False), providers=(HealthyProvider, ExplodingProvider))
    snap = await kit.get_ticker("BTC")
    pairs = [(e.provider, e.capability) for e in snap.data_quality.errors]
    assert len(pairs) == len(set(pairs)), "each failing capability must appear once"


@pytest.mark.asyncio
async def test_all_providers_failing_still_returns_a_snapshot() -> None:
    kit = TickerKit(Config(cache_enabled=False), providers=(ExplodingProvider,))
    snap = await kit.get_ticker("NOTHING")

    assert snap.data_quality.errors
    assert snap.market.price is None
    # A missing price is explained, not silently zero.
    assert any(m.field == "market.price" for m in snap.data_quality.missing)
    # And the snapshot is still fully serialisable.
    assert snap.to_json()


@pytest.mark.asyncio
@respx.mock
async def test_explicit_id_survives_a_throttled_resolver() -> None:
    """A caller-supplied id must hold even when every resolver is rate limited."""
    respx.get(f"{DEMO_ROOT}/coins/markets").mock(return_value=httpx.Response(429))
    kit = TickerKit(Config(cache_enabled=False, max_retries=0))
    snap = await kit.get_ticker("BTC", coingecko_id="bitcoin")

    assert snap.identity.coingecko_id == "bitcoin"


@pytest.mark.asyncio
async def test_unresolvable_ticker_is_reported_not_raised() -> None:
    class NoResolver(Provider):
        name = "none"

        supports = frozenset({Capability.RESOLVE})

        async def resolve(self, query: str) -> None:
            return None

    kit = TickerKit(Config(cache_enabled=False), providers=(NoResolver,))
    snap = await kit.get_ticker("MADEUPCOIN")

    assert snap.identity.coingecko_id is None
    assert snap.data_quality.providers_used == ("none",)


class GatedProvider(Provider):
    """A provider that reports itself unavailable, as CoinMarketCap does without a key."""

    name = "gated"
    supports = frozenset({Capability.META, Capability.AGG_MARKET})

    @classmethod
    def available(cls, config: Config) -> tuple[bool, str | None]:
        return False, "no credentials configured"


@pytest.mark.asyncio
async def test_unavailable_provider_is_skipped_with_a_reason() -> None:
    kit = TickerKit(Config(cache_enabled=False), providers=(GatedProvider, HealthyProvider))
    snap = await kit.get_ticker("BTC")

    assert snap.market.price == pytest.approx(81779.0, rel=1e-4)
    assert [(s.provider, s.reason) for s in snap.data_quality.providers_skipped] == [
        ("gated", "no credentials configured")
    ]
    assert "gated" not in snap.data_quality.providers_used


class TestCache:
    @pytest.mark.asyncio
    async def test_round_trip_and_expiry(self) -> None:
        cache = MemoryCache()
        assert await cache.get("k") is None
        await cache.set("k", {"v": 1}, ttl=60)
        assert await cache.get("k") == {"v": 1}
        await cache.set("expired", 1, ttl=-1)
        assert await cache.get("expired") is None
        await cache.clear()
        assert await cache.get("k") is None

    @pytest.mark.asyncio
    async def test_null_cache_never_stores(self) -> None:
        cache = NullCache()
        await cache.set("k", 1, 60)
        assert await cache.get("k") is None


class TestRetryPolicy:
    @pytest.mark.asyncio
    @respx.mock
    async def test_retries_429_then_succeeds(self) -> None:
        route = respx.get("https://example.test/x")
        route.side_effect = [
            httpx.Response(429),
            httpx.Response(503),
            httpx.Response(200, json={"ok": True}),
        ]
        config = Config(max_retries=3, retry_base_delay=0.001, cache_enabled=False)
        async with HttpClient(config=config) as http:
            data = await http.get_json("https://example.test/x")
        assert data == {"ok": True}
        assert route.call_count == 3

    @pytest.mark.asyncio
    @respx.mock
    async def test_does_not_retry_404(self) -> None:
        """A 404 is a caller/credential problem; retrying it only burns quota."""
        route = respx.get("https://example.test/x").mock(return_value=httpx.Response(404))
        config = Config(max_retries=3, retry_base_delay=0.001, cache_enabled=False)
        async with HttpClient(config=config) as http:
            with pytest.raises(httpx.HTTPStatusError):
                await http.get_json("https://example.test/x")
        assert route.call_count == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_does_not_retry_401(self) -> None:
        route = respx.get("https://example.test/x").mock(return_value=httpx.Response(401))
        config = Config(max_retries=3, retry_base_delay=0.001, cache_enabled=False)
        async with HttpClient(config=config) as http:
            with pytest.raises(httpx.HTTPStatusError):
                await http.get_json("https://example.test/x")
        assert route.call_count == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_gives_up_after_max_retries(self) -> None:
        route = respx.get("https://example.test/x").mock(return_value=httpx.Response(503))
        config = Config(max_retries=2, retry_base_delay=0.001, cache_enabled=False)
        async with HttpClient(config=config) as http:
            with pytest.raises(httpx.HTTPStatusError):
                await http.get_json("https://example.test/x")
        assert route.call_count == 3  # initial + 2 retries

    @pytest.mark.asyncio
    @respx.mock
    async def test_retries_timeouts(self) -> None:
        route = respx.get("https://example.test/x")
        route.side_effect = [httpx.ReadTimeout("slow"), httpx.Response(200, json=1)]
        config = Config(max_retries=2, retry_base_delay=0.001, cache_enabled=False)
        async with HttpClient(config=config) as http:
            assert await http.get_json("https://example.test/x") == 1


class TestRateLimiter:
    def test_rejects_non_positive_rpm(self) -> None:
        with pytest.raises(ValueError):
            RateLimiter(0)

    @pytest.mark.asyncio
    async def test_serialises_slots(self) -> None:
        import time

        limiter = RateLimiter(600, name="t")  # 10 slots/second -> 100ms apart
        start = time.monotonic()
        for _ in range(3):
            await limiter.acquire()
        # Three acquisitions must take at least two intervals.
        assert time.monotonic() - start >= 0.15


def test_retry_delay_is_jittered_and_capped() -> None:
    import random

    policy = RetryPolicy(max_retries=5, base_delay=0.5, max_delay=2.0)
    rng = random.Random(0)
    for attempt in range(5):
        ceiling = min(2.0, 0.5 * (2**attempt))
        for _ in range(20):
            delay = policy.delay_for(attempt, rng=rng)
            assert 0.0 <= delay <= ceiling


@pytest.mark.asyncio
async def test_unidentified_asset_is_explicitly_flagged() -> None:
    """A price with no resolved identity must be labelled, not passed off as resolved.

    Guards the exchange-only path: if the aggregator is unavailable the snapshot still
    carries venue prices, but the identity gap has to be stated explicitly.
    """

    class NoResolverNoId(Provider):
        name = "venue-only"
        supports = frozenset({Capability.VENUE_TICKER})

        async def fetch_venue_tickers(self, asset: Any) -> tuple[VenueTicker, ...]:
            return (
                VenueTicker(exchange="stub", symbol="BTC/USD", quote_currency="USD",
                            bid=100.0, ask=100.1, timestamp=utcnow()),
            )

    kit = TickerKit(Config(cache_enabled=False), providers=(NoResolverNoId,))
    snap = await kit.get_ticker("SOMETHING")

    assert snap.market.price is not None          # a price is still available
    assert snap.identity.coingecko_id is None     # but the asset was never identified
    flagged = [m for m in snap.data_quality.missing if m.field == "identity.coingecko_id"]
    assert len(flagged) == 1
    assert "different asset" in flagged[0].reason


@pytest.mark.asyncio
async def test_flag_is_not_duplicated_when_the_provider_already_explained_it() -> None:
    class Explaining(Provider):
        name = "explaining"
        supports = frozenset({Capability.RESOLVE})

        async def resolve(self, query: str) -> None:
            return None

    kit = TickerKit(Config(cache_enabled=False), providers=(Explaining,))
    snap = await kit.get_ticker("MYSTERY")

    flagged = [m for m in snap.data_quality.missing if m.field == "identity.coingecko_id"]
    assert len(flagged) == 1


@pytest.mark.asyncio
async def test_resolution_failure_is_recorded() -> None:
    """Regression: a resolver that raises must surface in data_quality.errors.

    Resolution failures were written into each provider's context but read back from a
    local list that nothing appended to, so a rate-limited aggregator produced a
    snapshot with an unidentified asset, a real venue price and zero recorded errors.
    """

    class FailingResolver(Provider):
        name = "failing-resolver"
        supports = frozenset({Capability.RESOLVE, Capability.VENUE_TICKER})

        async def resolve(self, query: str) -> None:
            raise RuntimeError("aggregator rate limited")

        async def fetch_venue_tickers(self, asset: Any) -> tuple[VenueTicker, ...]:
            return (
                VenueTicker(exchange="stub", symbol="BTC/USD", quote_currency="USD",
                            bid=100.0, ask=100.1, timestamp=utcnow()),
            )

    kit = TickerKit(Config(cache_enabled=False), providers=(FailingResolver,))
    snap = await kit.get_ticker("BTC")

    assert snap.identity.coingecko_id is None
    # Exactly one resolve error: the provider's own, not a redundant synthetic one.
    resolve_errors = [e for e in snap.data_quality.errors if e.capability.value == "resolve"]
    assert len(resolve_errors) == 1
    assert resolve_errors[0].provider == "failing-resolver"
    assert resolve_errors[0].exception_type == "RuntimeError"
    assert "rate limited" in resolve_errors[0].message
    # The venue price still came through, and the identity gap is explained.
    assert snap.market.price is not None
    assert any(m.field == "identity.coingecko_id" for m in snap.data_quality.missing)


@pytest.mark.asyncio
async def test_resolution_errors_are_not_double_counted() -> None:
    class FailingResolver(Provider):
        name = "failing-resolver"
        supports = frozenset({Capability.RESOLVE})

        async def resolve(self, query: str) -> None:
            raise RuntimeError("boom")

    kit = TickerKit(Config(cache_enabled=False), providers=(FailingResolver,))
    snap = await kit.get_ticker("BTC")

    pairs = [(e.provider, e.capability, e.message) for e in snap.data_quality.errors]
    assert len(pairs) == len(set(pairs))
