"""ccxt adapter, exercised against a stub exchange object (no network).

ccxt itself is a third-party library with its own live tests; what matters here is that
tickerkit reads ``exchange.has`` correctly, prefers perpetuals over dated futures, and
never fabricates a value for an unsupported or emulated sub-capability.
"""

from __future__ import annotations

from typing import Any

import pytest

from tickerkit.cache import MemoryCache
from tickerkit.config import Config
from tickerkit.http import HttpClient
from tickerkit.providers.base import ProviderContext
from tickerkit.providers.ccxt_provider import CcxtProvider, _prefer_quote, _rank_markets
from tickerkit.resolve import MarketRef, ResolvedAsset

NOW_MS = 1_760_000_000_000


class StubExchange:
    """A minimal stand-in for a ccxt async exchange."""

    def __init__(self, has: dict[str, Any], markets: dict[str, dict[str, Any]]) -> None:
        self.has = has
        self.markets = markets
        self.urls = {"api": {"public": "https://api.example.test", "private": ""}}
        self.calls: list[tuple[str, str]] = []

    async def load_markets(self) -> dict[str, dict[str, Any]]:
        return self.markets

    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("fetchTicker", symbol))
        return {
            "symbol": symbol,
            "bid": 99.9,
            "ask": 100.1,
            "last": 100.0,
            "baseVolume": 1234.5,
            "quoteVolume": 123450.0,
            "timestamp": NOW_MS,
        }

    async def fetch_order_book(self, symbol: str, limit: int = 20) -> dict[str, Any]:
        self.calls.append(("fetchOrderBook", symbol))
        return {
            "bids": [[99.9, 1.0], [99.8, 2.0], [99.0, 4.0]],
            "asks": [[100.1, 1.5], [100.2, 2.5], [101.0, 4.0]],
            "timestamp": NOW_MS,
        }

    async def fetch_trades(self, symbol: str, limit: int = 100) -> list[dict[str, Any]]:
        self.calls.append(("fetchTrades", symbol))
        return [
            {"timestamp": NOW_MS - 1000, "price": 100.0, "amount": 0.5, "side": "buy", "id": "1"},
            {"timestamp": NOW_MS - 2000, "price": 100.1, "amount": 0.25, "side": "sell", "id": "2"},
            # Malformed rows must be skipped, not crash the adapter.
            {"timestamp": None, "price": 1.0, "amount": 1.0, "id": "3"},
        ]

    async def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 200) -> list[list[float]]:
        self.calls.append(("fetchOHLCV", f"{symbol}|{timeframe}"))
        return [
            [NOW_MS - 600_000, 99.0, 101.0, 98.5, 100.0, 12.0],
            [NOW_MS - 300_000, 100.0, 101.5, 99.5, 101.0, 13.0],
        ]

    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("fetchFundingRate", symbol))
        return {"symbol": symbol, "fundingRate": 0.0001, "fundingInterval": 8 * 3600_000,
                "timestamp": NOW_MS}

    async def fetch_open_interest(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("fetchOpenInterest", symbol))
        return {"symbol": symbol, "openInterestAmount": 5000.0, "timestamp": NOW_MS}

    async def fetch_mark_price(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("fetchMarkPrice", symbol))
        return {"symbol": symbol, "markPrice": 100.0, "timestamp": NOW_MS}

    async def fetch_index_price(self, symbol: str) -> dict[str, Any]:
        self.calls.append(("fetchIndexPrice", symbol))
        return {"symbol": symbol, "indexPrice": 99.95, "timestamp": NOW_MS}


SPOT_MARKETS: dict[str, dict[str, Any]] = {
    "BTC/USD": {"base": "BTC", "quote": "USD", "type": "spot", "active": True},
    "BTC/USDT": {"base": "BTC", "quote": "USDT", "type": "spot", "active": True},
    "BTC/EUR": {"base": "BTC", "quote": "EUR", "type": "spot", "active": True},
    "BTC/GBP": {"base": "BTC", "quote": "GBP", "type": "spot", "active": False},
    "BTC/USDT:USDT": {"base": "BTC", "quote": "USDT", "type": "swap", "swap": True,
                      "active": True, "contractSize": 1.0},
    "BTC/USDT:USDT-261225": {"base": "BTC", "quote": "USDT", "type": "future",
                             "active": True, "contractSize": 1.0},
    "BTC/USD:BTC": {"base": "BTC", "quote": "USD", "type": "swap", "swap": True,
                    "active": True, "contractSize": 100.0},
    "ETH/USD": {"base": "ETH", "quote": "USD", "type": "spot", "active": True},
}

FULL_HAS: dict[str, Any] = {
    "fetchTicker": True,
    "fetchOrderBook": True,
    "fetchTrades": True,
    "fetchOHLCV": True,
    "fetchFundingRate": True,
    "fetchOpenInterest": True,
    "fetchMarkPrice": True,
    "fetchIndexPrice": True,
    "swap": True,
    "future": True,
}


def make_provider(
    stub: StubExchange, **config_kwargs: Any
) -> tuple[CcxtProvider, ProviderContext, HttpClient]:
    config = Config(exchanges=("stub",), perp_exchanges=("stub",),
                    ohlcv_timeframes=("5m",), cache_enabled=False, **config_kwargs)
    http = HttpClient(config=config)
    provider = CcxtProvider(ProviderContext(config=config, http=http, cache=MemoryCache()))
    provider._exchanges["stub"] = stub  # bypass real ccxt instantiation
    return provider, provider.ctx, http


def asset() -> ResolvedAsset:
    return ResolvedAsset(
        query="BTC",
        symbol="BTC",
        spot_markets=(MarketRef("stub", "BTC/USD", "USD", "spot"),),
        derivative_markets=(
            MarketRef("stub", "BTC/USDT:USDT", "USDT", "swap"),
            MarketRef("stub", "BTC/USDT:USDT-261225", "USDT", "future"),
            MarketRef("stub", "BTC/USD:BTC", "USD", "swap"),
        ),
    )


class TestCapabilityGating:
    @pytest.mark.asyncio
    async def test_never_calls_an_unsupported_method(self) -> None:
        stub = StubExchange(
            {"fetchTicker": True, "fetchOrderBook": False, "fetchTrades": False,
             "fetchOHLCV": False, "swap": False, "future": False},
            SPOT_MARKETS,
        )
        provider, ctx, http = make_provider(stub)

        assert await provider.fetch_venue_tickers(asset())
        # The unsupported methods are never invoked...
        for method in ("fetchOrderBook", "fetchTrades", "fetchOHLCV"):
            assert not any(call[0] == method for call in stub.calls)

        # ...and the resulting capability failures are recorded rather than raised,
        # which is what keeps one venue from sinking the snapshot.
        assert await provider.fetch_orderbooks(asset()) == ()
        assert await provider.fetch_trades(asset()) == ()
        assert await provider.fetch_ohlcv(asset()) == ()
        errors = [e for e in ctx.errors if e.capability.value in
                  {"orderbook", "trades", "ohlcv"}]
        assert len(errors) == 3
        assert all("does not support" in e.message for e in errors)
        await http.aclose()

    @pytest.mark.asyncio
    async def test_emulated_methods_count_as_unsupported(self) -> None:
        """ccxt marks synthesised responses as 'emulated'; reporting them would fabricate."""
        provider, _ctx, http = make_provider(StubExchange(FULL_HAS, SPOT_MARKETS))
        stub = provider._exchanges["stub"]
        assert provider._supports(stub, "fetchFundingRate") is True
        emulated = StubExchange({**FULL_HAS, "fetchFundingRate": "emulated"}, SPOT_MARKETS)
        assert provider._supports(emulated, "fetchFundingRate") is False
        assert provider._supports(StubExchange({"fetchTicker": False}, {}), "fetchTicker") is False
        await http.aclose()


class TestVenueTicker:
    @pytest.mark.asyncio
    async def test_maps_documented_ticker_fields(self) -> None:
        provider, _ctx, http = make_provider(StubExchange(FULL_HAS, SPOT_MARKETS))
        venues = await provider.fetch_venue_tickers(asset())

        assert len(venues) == 1
        v = venues[0]
        assert v.bid == 99.9
        assert v.ask == 100.1
        assert v.last == 100.0
        assert v.spread == pytest.approx(0.2)
        assert v.spread_bps == pytest.approx(20.0, rel=1e-3)
        assert v.volume_24h_quote == 123450.0
        assert v.timestamp is not None and v.timestamp.tzinfo is not None
        assert v.provenance is not None
        assert v.provenance.source == "ccxt:stub:fetch_ticker"
        assert v.provenance.url == "https://api.example.test"
        await http.aclose()


class TestOrderBook:
    @pytest.mark.asyncio
    async def test_derives_spread_mid_depth_and_imbalance(self) -> None:
        provider, _ctx, http = make_provider(StubExchange(FULL_HAS, SPOT_MARKETS))
        books = await provider.fetch_orderbooks(asset())

        assert len(books) == 1
        book = books[0]
        assert book.best_bid == 99.9
        assert book.best_ask == 100.1
        assert book.mid == pytest.approx(100.0)
        assert book.bid_depth_base == 7.0
        assert book.ask_depth_base == 8.0
        assert book.imbalance == pytest.approx((7 - 8) / 15)
        bands = {b.pct: b for b in book.depth_quote_within}
        # mid = 100.0. Within 0.5% => [99.5, 100.5]: bids 99.9 + 99.8, asks 100.1 + 100.2.
        assert bands[0.005].bid_base == 3.0
        assert bands[0.005].ask_base == 4.0
        # Within 2% => [98.0, 102.0]: every level, boundary levels included.
        assert bands[0.02].bid_base == 7.0
        assert bands[0.02].ask_base == 8.0
        await http.aclose()

    @pytest.mark.asyncio
    async def test_empty_book_is_reported_not_zeroed(self) -> None:
        stub = StubExchange(FULL_HAS, SPOT_MARKETS)

        async def empty(symbol: str, limit: int = 20) -> dict[str, Any]:
            return {"bids": [], "asks": [], "timestamp": NOW_MS}

        stub.fetch_order_book = empty  # type: ignore[method-assign]
        provider, ctx, http = make_provider(stub)

        assert await provider.fetch_orderbooks(asset()) == ()
        assert any("empty book" in m.reason for m in ctx.missing)
        await http.aclose()


class TestTradesAndOhlcv:
    @pytest.mark.asyncio
    async def test_skips_malformed_trade_rows(self) -> None:
        provider, _ctx, http = make_provider(StubExchange(FULL_HAS, SPOT_MARKETS))
        trades = await provider.fetch_trades(asset())
        assert len(trades) == 2
        assert all(t.timestamp.tzinfo is not None for t in trades)
        await http.aclose()

    @pytest.mark.asyncio
    async def test_ohlcv_candles_are_utc(self) -> None:
        provider, _ctx, http = make_provider(StubExchange(FULL_HAS, SPOT_MARKETS))
        series = await provider.fetch_ohlcv(asset())
        assert len(series) == 1
        s = series[0]
        assert s.timeframe == "5m"
        assert len(s.candles) == 2
        assert all(c.timestamp.tzinfo is not None for c in s.candles)
        assert s.provenance is not None and "timeframe=5m" in s.provenance.notes
        await http.aclose()


class TestDerivatives:
    @pytest.mark.asyncio
    async def test_full_support_yields_every_field(self) -> None:
        provider, _ctx, http = make_provider(StubExchange(FULL_HAS, SPOT_MARKETS))
        derivs = await provider.fetch_derivatives(asset())

        assert len(derivs) == 1
        d = derivs[0]
        # The perpetual must be chosen, not the dated future.
        assert d.symbol == "BTC/USDT:USDT"
        assert d.contract_type == "swap"
        assert d.funding_rate == 0.0001
        assert d.funding_rate_pct == pytest.approx(0.01)
        assert d.funding_interval_hours == 8.0
        assert d.open_interest_contracts == 5000.0
        assert d.open_interest_base == 5000.0
        assert d.open_interest_quote == pytest.approx(500_000.0)
        assert d.mark_price == 100.0
        assert d.index_price == 99.95
        await http.aclose()

    @pytest.mark.asyncio
    async def test_emulated_funding_is_not_reported(self) -> None:
        """bybit returns a constant 0.0001 from an emulated method; that is not real data."""
        stub = StubExchange({**FULL_HAS, "fetchFundingRate": "emulated",
                             "fetchIndexPrice": "emulated"}, SPOT_MARKETS)
        provider, ctx, http = make_provider(stub)
        derivs = await provider.fetch_derivatives(asset())

        assert derivs[0].funding_rate is None
        assert derivs[0].index_price is None
        reasons = " ".join(m.reason for m in ctx.missing)
        assert "emulated" in reasons
        await http.aclose()

    @pytest.mark.asyncio
    async def test_all_sub_capabilities_unsupported_is_recorded_not_raised(self) -> None:
        """With every sub-capability emulated there is nothing real to report."""
        stub = StubExchange({**FULL_HAS, "fetchFundingRate": "emulated",
                             "fetchOpenInterest": "emulated",
                             "fetchMarkPrice": "emulated",
                             "fetchIndexPrice": "emulated"}, SPOT_MARKETS)
        provider, ctx, http = make_provider(stub)

        assert await provider.fetch_derivatives(asset()) == ()
        assert any(
            e.capability.value == "derivatives" and "fabricated" in e.message
            for e in ctx.errors
        )
        await http.aclose()

    @pytest.mark.asyncio
    async def test_inverse_contract_size_is_not_converted(self) -> None:
        """contractSize=100 on an inverse market is a fiat amount, not 100 BTC."""
        stub = StubExchange(FULL_HAS, SPOT_MARKETS)
        provider, ctx, http = make_provider(stub)
        inverse = ResolvedAsset(
            query="BTC", symbol="BTC",
            derivative_markets=(MarketRef("stub", "BTC/USD:BTC", "USD", "swap"),),
        )
        stub.markets["BTC/USD:BTC"] = {**stub.markets["BTC/USD:BTC"], "contractSize": 1000.0}
        derivs = await provider.fetch_derivatives(inverse)

        assert derivs[0].open_interest_base is None
        assert any("wrong-unit" in m.reason or "inverse" in m.reason for m in ctx.missing)
        await http.aclose()


class TestMarketRanking:
    def test_perpetual_beats_dated_future(self) -> None:
        refs = [
            MarketRef("x", "BTC/USDT:USDT-261225", "USDT", "future"),
            MarketRef("x", "BTC/USDT:USDT", "USDT", "swap"),
        ]
        assert _rank_markets(refs, True, "USD")[0].symbol == "BTC/USDT:USDT"

    def test_linear_perp_beats_inverse_even_for_usd_quote(self) -> None:
        """A linear contract's contractSize is base units; an inverse one is not."""
        refs = [
            MarketRef("x", "BTC/USD:BTC", "USD", "swap"),
            MarketRef("x", "BTC/USDT:USDT", "USDT", "swap"),
        ]
        assert _rank_markets(refs, True, "USD")[0].symbol == "BTC/USDT:USDT"

    def test_spot_prefers_the_exact_quote_currency(self) -> None:
        refs = [
            MarketRef("x", "BTC/USDT", "USDT", "spot"),
            MarketRef("x", "BTC/USD", "USD", "spot"),
            MarketRef("x", "BTC/EUR", "EUR", "spot"),
        ]
        assert _rank_markets(refs, False, "USD")[0].symbol == "BTC/USD"

    def test_prefer_quote_falls_back_sensibly(self) -> None:
        assert _prefer_quote((), "USD") is None
        refs = (MarketRef("x", "BTC/EUR", "EUR", "spot"),)
        assert _prefer_quote(refs, "USD").symbol == "BTC/EUR"


@pytest.mark.asyncio
async def test_scan_prefers_spot_matching_the_quote_currency() -> None:
    provider, _ctx, http = make_provider(StubExchange(FULL_HAS, SPOT_MARKETS))
    resolved = await provider.resolve("BTC")
    assert resolved is not None
    assert resolved.symbol == "BTC"
    assert "BTC/USD" in {m.symbol for m in resolved.spot_markets}
    # Inactive markets are never offered.
    assert "BTC/GBP" not in {m.symbol for m in resolved.spot_markets}
    await http.aclose()