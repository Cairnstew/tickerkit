"""CoinPaprika adapter, replayed from recorded fixtures.

CoinPaprika is the keyless second opinion on aggregated market data. The tests that
matter most here are the ones about *degradation*: when CoinGecko throttles, this
provider must keep price and market cap flowing, and must never invent the fields the
endpoint does not publish.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from tickerkit.cache import MemoryCache
from tickerkit.config import Config
from tickerkit.http import HttpClient
from tickerkit.providers.base import ProviderContext
from tickerkit.providers.coinpaprika import BASE_URL as PAPRIKA_URL
from tickerkit.providers.coinpaprika import CoinPaprikaProvider, parse_epoch
from tickerkit.resolve import ResolvedAsset

from ..conftest import load_fixture

SEARCH = f"{PAPRIKA_URL}/search/"


def _ctx(**config_kwargs: Any) -> tuple[Config, HttpClient, ProviderContext]:
    config = Config(cache_enabled=False, max_retries=0, **config_kwargs)
    http = HttpClient(config=config, source="coinpaprika")
    return config, http, ProviderContext(config=config, http=http, cache=MemoryCache())


def _mock_resolve(symbol: str = "BTC", fixture: str = "coinpaprika_search_btc.json") -> None:
    respx.get(SEARCH).mock(
        return_value=httpx.Response(200, json=load_fixture(fixture))
    )


class TestResolve:
    @pytest.mark.asyncio
    @respx.mock
    async def test_prefers_the_highest_ranked_coin(self) -> None:
        """BTC is shared by 16 coins here; the rank-1 one must win, not the first row."""
        _mock_resolve()
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        asset = await provider.resolve("BTC")

        assert asset is not None
        assert asset.symbol == "BTC"
        assert asset.name == "Bitcoin"
        assert not asset.unresolved
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_uses_the_trailing_slash_endpoint(self) -> None:
        """/search without the trailing slash 301s; the request URL must be correct."""
        _mock_resolve()
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        await provider.resolve("BTC")

        route = respx.routes[0]
        assert route.calls.last.request.url.path.endswith("/search/")
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_does_not_invent_a_coingecko_id(self) -> None:
        """CoinPaprika ids are `btc-bitcoin`; claiming a CoinGecko id here would be a lie.

        DefiLlama joins on gecko_id, so a fabricated value would silently attach the
        wrong protocol's TVL to the asset.
        """
        _mock_resolve()
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        asset = await provider.resolve("BTC")

        assert asset is not None
        assert asset.coingecko_id is None
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_rivals_are_recorded_without_faking_their_namespace(self) -> None:
        _mock_resolve()
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        await provider.resolve("BTC")

        # Ambiguity.coingecko_id is CoinGecko-namespaced; a paprika id must not land there.
        assert ctx.ambiguities == []
        note = next(m for m in ctx.missing if m.field == "identity.coinpaprika_id")
        assert "btc-buy-the-cat" in note.reason
        assert "btc-bitcoin" in note.reason
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_unknown_symbol_is_a_clean_miss(self) -> None:
        respx.get(SEARCH).mock(return_value=httpx.Response(200, json={"currencies": []}))
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        assert await provider.resolve("NOTAREALCOIN") is None
        assert any(m.field == "identity.coinpaprika_id" for m in ctx.missing)
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_symbol_matching_is_exact_not_substring(self) -> None:
        """The fixture contains WBTC; a BTC query must not return it."""
        _mock_resolve()
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        await provider.resolve("BTC")

        assert ctx.ambiguities == []
        await http.aclose()


class TestMarket:
    @pytest.mark.asyncio
    @respx.mock
    async def test_parses_price_cap_volume_and_ath(self) -> None:
        _mock_resolve()
        respx.get(f"{PAPRIKA_URL}/tickers/btc-bitcoin").mock(
            return_value=httpx.Response(200, json=load_fixture("coinpaprika_ticker_btc.json"))
        )
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        market = await provider.fetch_market(ResolvedAsset(query="BTC", symbol="BTC"))

        assert market is not None
        assert market.aggregated_price == pytest.approx(83043.05)
        assert market.market_cap == pytest.approx(1668809670234.0)
        assert market.volume_24h == pytest.approx(29387330244.06)
        assert market.change_24h_pct == pytest.approx(0.83)
        assert market.ath == pytest.approx(126173.18)
        assert market.ath_date is not None and market.ath_date.year == 2025
        assert "coinpaprika" in market.provenance
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_unpublished_fields_are_explained_not_zeroed(self) -> None:
        """No circulating supply or FDV on this endpoint: say so rather than report 0."""
        _mock_resolve()
        respx.get(f"{PAPRIKA_URL}/tickers/btc-bitcoin").mock(
            return_value=httpx.Response(200, json=load_fixture("coinpaprika_ticker_btc.json"))
        )
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        market = await provider.fetch_market(ResolvedAsset(query="BTC", symbol="BTC"))

        assert market is not None
        assert market.circulating_supply is None  # not 0
        assert market.fdv is None
        explained = {m.field for m in ctx.missing}
        assert "market.coinpaprika.fdv" in explained
        assert "market.coinpaprika.circulating_supply" in explained
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_unknown_id_404_is_surfaced_as_an_error(self) -> None:
        _mock_resolve()
        respx.get(f"{PAPRIKA_URL}/tickers/btc-bitcoin").mock(
            return_value=httpx.Response(404, json={"error": "id not found"})
        )
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        with pytest.raises(httpx.HTTPStatusError):
            await provider.fetch_market(ResolvedAsset(query="BTC", symbol="BTC"))
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_missing_quote_block_is_explained(self) -> None:
        _mock_resolve()
        respx.get(f"{PAPRIKA_URL}/tickers/btc-bitcoin").mock(
            return_value=httpx.Response(200, json={"id": "btc-bitcoin", "quotes": {}})
        )
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        assert await provider.fetch_market(ResolvedAsset(query="BTC", symbol="BTC")) is None
        assert any("no USD quote block" in m.reason for m in ctx.missing)
        await http.aclose()


class TestIdentity:
    @pytest.mark.asyncio
    @respx.mock
    async def test_maps_the_real_link_keys(self) -> None:
        """The API uses `website` and `source_code`, not `homepage` and `github`."""
        _mock_resolve()
        respx.get(f"{PAPRIKA_URL}/coins/btc-bitcoin").mock(
            return_value=httpx.Response(200, json=load_fixture("coinpaprika_coin_btc.json"))
        )
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        identity = await provider.fetch_identity(ResolvedAsset(query="BTC", symbol="BTC"))

        assert identity is not None
        assert identity.name == "Bitcoin"
        assert identity.links.homepage == ("https://bitcoin.org/",)
        assert identity.links.github == ("https://github.com/bitcoin/bitcoin",)
        assert identity.links.reddit == "https://www.reddit.com/r/bitcoin"
        assert identity.links.blockchain == ("https://blockstream.info/",)
        assert "Proof Of Work" in identity.categories
        assert identity.description is not None
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_blog_link_comes_from_the_links_extended_list(self) -> None:
        """links_extended is a list of {url,type}, and the blog is only in there."""
        _mock_resolve()
        respx.get(f"{PAPRIKA_URL}/coins/btc-bitcoin").mock(
            return_value=httpx.Response(200, json=load_fixture("coinpaprika_coin_btc.json"))
        )
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        identity = await provider.fetch_identity(ResolvedAsset(query="BTC", symbol="BTC"))

        assert identity is not None
        assert "https://bitcoin.org/en/blog" in identity.links.other
        await http.aclose()


class TestEpochTimestamps:
    def test_parses_unix_seconds(self) -> None:
        parsed = parse_epoch(1791549889)
        assert parsed is not None and parsed.year == 2026

    @pytest.mark.parametrize("value", [None, 0, -1, "", "nonsense", True])
    def test_non_timestamps_are_ignored(self, value: Any) -> None:
        assert parse_epoch(value) is None


class TestDegradation:
    """The whole reason this provider exists."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_market_survives_a_throttled_coingecko(self) -> None:
        """With CoinGecko refusing everything, CoinPaprika still yields price and cap."""
        respx.get(url__startswith="https://api.coingecko.com").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "60"})
        )
        _mock_resolve()
        respx.get(f"{PAPRIKA_URL}/tickers/btc-bitcoin").mock(
            return_value=httpx.Response(200, json=load_fixture("coinpaprika_ticker_btc.json"))
        )
        _config, http, ctx = _ctx()
        provider = CoinPaprikaProvider(ctx)

        market = await provider.fetch_market(ResolvedAsset(query="BTC", symbol="BTC"))

        assert market is not None
        assert market.price is not None
        assert market.market_cap is not None
        await http.aclose()