"""CoinGecko adapter, replayed from recorded fixtures."""

from __future__ import annotations

import httpx
import pytest
import respx

from tickerkit.cache import MemoryCache
from tickerkit.config import Config
from tickerkit.http import HttpClient
from tickerkit.providers.base import ProviderContext
from tickerkit.providers.coingecko import DEMO_ROOT, PRO_ROOT, CoinGeckoProvider

from ..conftest import load_fixture


def _context() -> tuple[Config, HttpClient, ProviderContext]:
    config = Config(cache_enabled=False, max_retries=0)
    http = HttpClient(config=config)
    return config, http, ProviderContext(config=config, http=http, cache=MemoryCache())


@pytest.mark.asyncio
@respx.mock
async def test_resolve_picks_highest_market_cap_and_records_alternatives() -> None:
    """Ambiguous symbols resolve to the biggest asset; the rest become ambiguities."""
    respx.get(f"{DEMO_ROOT}/coins/markets").mock(
        return_value=httpx.Response(200, json=load_fixture("coingecko_markets_ambiguous.json"))
    )
    _config, http, ctx = _context()
    provider = CoinGeckoProvider(ctx)

    asset = await provider.resolve("BTC")

    assert asset is not None
    assert asset.coingecko_id == "bitcoin"
    assert asset.symbol == "BTC"
    # The other two tokens sharing the BTC symbol must be surfaced, not silently dropped.
    assert {a.coingecko_id for a in ctx.ambiguities} == {"bitcoin-cash-sv", "bitcoin-2"}
    # A token with no market cap must sort below one that has one.
    caps = {a.coingecko_id: a.market_cap for a in ctx.ambiguities}
    assert caps["bitcoin-cash-sv"] == 940000000.0
    assert caps["bitcoin-2"] is None
    await http.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_resolve_records_provenance() -> None:
    respx.get(f"{DEMO_ROOT}/coins/markets").mock(
        return_value=httpx.Response(200, json=load_fixture("coingecko_markets_btc.json"))
    )
    _config, http, ctx = _context()
    provider = CoinGeckoProvider(ctx)

    await provider.resolve("BTC")

    prov = ctx.provenance["resolve"]
    assert prov.source == "coingecko:/coins/markets"
    assert prov.capability.value == "resolve"
    assert prov.fetched_at.tzinfo is not None
    assert prov.url and prov.url.startswith(DEMO_ROOT)
    await http.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_resolve_returns_none_for_unknown_symbol_and_explains_why() -> None:
    respx.get(f"{DEMO_ROOT}/coins/markets").mock(return_value=httpx.Response(200, json=[]))
    _config, http, ctx = _context()
    provider = CoinGeckoProvider(ctx)

    assert await provider.resolve("NOTACOIN") is None
    assert any("NOTACOIN" in m.reason for m in ctx.missing)
    await http.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_market_maps_every_documented_field() -> None:
    respx.get(f"{DEMO_ROOT}/coins/markets").mock(
        return_value=httpx.Response(200, json=load_fixture("coingecko_markets_btc.json"))
    )
    _config, http, ctx = _context()
    provider = CoinGeckoProvider(ctx)

    from tickerkit.resolve import ResolvedAsset

    market = await provider.fetch_market(
        ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")
    )

    assert market is not None
    assert market.aggregated_price == 81779.0
    assert market.market_cap == 1643365584205.0
    assert market.fdv == 1643365584205.0
    assert market.circulating_supply == 20030493.0
    assert market.max_supply == 21000000.0
    assert market.ath == 126080.0
    assert market.atl == 67.81
    assert market.ath_date is not None and market.ath_date.tzinfo is not None
    assert market.provenance["aggregated"].exchange_timestamp is not None
    await http.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_fetch_identity_parses_html_description_and_links() -> None:
    respx.get(f"{DEMO_ROOT}/coins/bitcoin").mock(
        return_value=httpx.Response(
            200, json=load_fixture("coingecko_coin_bitcoin.json")
        )
    )
    _config, http, ctx = _context()
    provider = CoinGeckoProvider(ctx)

    from tickerkit.resolve import ResolvedAsset

    identity = await provider.fetch_identity(
        ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")
    )

    assert identity is not None
    assert identity.coingecko_id == "bitcoin"
    assert identity.categories == ("Layer 1 (L1)", "Proof of Work (PoW)")
    # HTML tags stripped, blank link entries dropped.
    assert "<" not in (identity.description or "")
    assert identity.links.homepage == ("http://www.bitcoin.org",)
    assert identity.links.whitepaper == ("https://bitcoin.org/bitcoin.pdf",)
    assert identity.links.github == ("https://github.com/bitcoin/bitcoin",)
    assert identity.links.twitter == "bitcoin"
    assert identity.provenance is not None
    await http.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_pro_api_key_switches_base_url_and_header() -> None:
    """A Pro key must use pro-api + x-cg-pro-api-key; a Demo key api + x-cg-demo-api-key."""
    from tickerkit.providers.base import ProviderContext as PC

    for key_attr, root, header in (
        ("coingecko_api_key", DEMO_ROOT, "x-cg-demo-api-key"),
        ("coingecko_pro_api_key", PRO_ROOT, "x-cg-pro-api-key"),
    ):
        config = Config(cache_enabled=False, **{key_attr: "test-key"})
        http = HttpClient(config=config)
        provider = CoinGeckoProvider(
            PC(config=config, http=http, cache=MemoryCache())
        )
        assert provider.root == root
        assert provider._headers == {header: "test-key"}
        await http.aclose()


    def test_rate_limit_reflects_keyless_vs_demo() -> None:
        """Keyless access is IP-shared and far tighter than the documented 100/min demo.

        12 rpm, not the previous 25: the bucket is shared with any other consumer on the
        same IP, and a measured 15-ticker sweep at 25 rpm tripped a genuine 429 mid-run.
        """
        assert CoinGeckoProvider.rate_limit_rpm(Config()) == 12
    assert CoinGeckoProvider.rate_limit_rpm(Config(coingecko_api_key="k")) == 100
    assert CoinGeckoProvider.rate_limit_rpm(Config(coingecko_pro_api_key="k")) == 100


@pytest.mark.asyncio
@respx.mock
async def test_cache_prevents_a_second_market_request() -> None:
    """Repeated calls must not burn free-tier quota."""
    route = respx.get(f"{DEMO_ROOT}/coins/markets").mock(
        return_value=httpx.Response(200, json=load_fixture("coingecko_markets_btc.json"))
    )
    config = Config(cache_enabled=True, cache_ttl=60.0, max_retries=0)
    http = HttpClient(config=config)
    ctx = ProviderContext(config=config, http=http, cache=MemoryCache())
    provider = CoinGeckoProvider(ctx)

    from tickerkit.resolve import ResolvedAsset

    asset = ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")
    await provider.fetch_market(asset)
    first = route.call_count
    await provider.fetch_market(asset)

    assert route.call_count == first, "second fetch must be served from the cache"
    await http.aclose()


