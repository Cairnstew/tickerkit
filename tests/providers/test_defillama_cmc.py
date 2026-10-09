"""DefiLlama and CoinMarketCap adapters, replayed from recorded fixtures."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
import respx

from tickerkit.cache import MemoryCache
from tickerkit.config import Config
from tickerkit.http import HttpClient
from tickerkit.providers.base import ProviderContext
from tickerkit.providers.coinmarketcap import BASE_URL as CMC_URL
from tickerkit.providers.coinmarketcap import CoinMarketCapProvider
from tickerkit.providers.defillama import BASE_URL as LLAMA_URL
from tickerkit.providers.defillama import (
    DefiLlamaProvider,
    is_chain_key,
    parent_slug,
    parse_tvl,
)
from tickerkit.resolve import ResolvedAsset

from ..conftest import load_fixture


def _ctx(**config_kwargs: Any) -> tuple[Config, HttpClient, ProviderContext]:
    config = Config(cache_enabled=False, max_retries=0, **config_kwargs)
    http = HttpClient(config=config)
    return config, http, ProviderContext(config=config, http=http, cache=MemoryCache())


class TestDefiLlama:
    @pytest.mark.asyncio
    @respx.mock
    async def test_falls_back_to_chain_tvl_and_explains_the_gap(self) -> None:
        """Bitcoin is a chain, not a DeFi protocol: report chain TVL and say so."""
        respx.get(f"{LLAMA_URL}/v2/chains").mock(
            return_value=httpx.Response(200, json=load_fixture("defillama_chains_bitcoin.json"))
        )
        respx.get(f"{LLAMA_URL}/protocol/bitcoin").mock(
            return_value=httpx.Response(
                200, json={"id": "bitcoin", "name": "Bitcoin", "tvl": None, "chainTvls": {}}
            )
        )
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        data = await provider.fetch_onchain(
            ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin",
                          defillama_slug="bitcoin")
        )

        assert data is not None
        assert data.tvl == pytest.approx(4369531827.234072)
        assert data.chains[0].chain == "Bitcoin"
        assert data.provenance is not None
        # The protocol itself has no TVL; that must be stated, not silently replaced.
        assert any(m.field == "onchain.protocol_tvl" for m in ctx.missing)
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_protocol_tvl_excludes_non_chain_buckets(self) -> None:
        respx.get(f"{LLAMA_URL}/v2/chains").mock(return_value=httpx.Response(200, json=[]))
        respx.get(f"{LLAMA_URL}/protocol/lido").mock(
            return_value=httpx.Response(200, json=load_fixture("defillama_protocol_lido.json"))
        )
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        data = await provider.fetch_onchain(
            ResolvedAsset(query="LDO", symbol="LDO", coingecko_id="lido-dao",
                          defillama_slug="lido")
        )

        assert data is not None
        # tvl is a history array on /protocol/{slug}; the last point is current.
        assert data.tvl == 24347452807.0
        assert data.category == "Liquid Staking"
        # currentChainTvls is the flat per-chain map; zero-TVL chains are still chains.
        assert {c.chain for c in data.chains} == {"Ethereum", "Terra", "Moonbeam"}
        assert data.chains[0].tvl == 24341758740.8232
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_tvl_is_read_from_the_history_array(self) -> None:
        """Regression: /protocol/{slug} returns tvl as a list, not a scalar.

        Parsing only a scalar silently produced tvl=None for every real protocol, which
        was masked by the chain-level fallback in the other tests.
        """
        respx.get(f"{LLAMA_URL}/v2/chains").mock(return_value=httpx.Response(200, json=[]))
        respx.get(f"{LLAMA_URL}/protocol/lido").mock(
            return_value=httpx.Response(200, json=load_fixture("defillama_protocol_lido.json"))
        )
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        data = await provider.fetch_onchain(
            ResolvedAsset(query="LDO", symbol="LDO", coingecko_id="lido-dao",
                          defillama_slug="lido")
        )

        assert data is not None
        assert data.tvl == 24347452807.0  # last point, not the first
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_versioned_gecko_id_resolves_to_the_parent_aggregate(self) -> None:
        """Regression: gecko_id 'aave' sits on aave-v2 ($109M), not the real $18.4B.

        Reporting the version row would understate the protocol's TVL by two orders of
        magnitude, so the parent slug is preferred.
        """
        respx.get(f"{LLAMA_URL}/protocols").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocols_versions.json")
            )
        )
        respx.get(f"{LLAMA_URL}/protocol/aave").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocol_aave.json")
            )
        )
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        data = await provider.fetch_onchain(
            ResolvedAsset(query="AAVE", symbol="AAVE", coingecko_id="aave")
        )

        assert data is not None
        assert data.defillama_slug == "aave"          # parent, not aave-v2
        assert data.tvl == 18366126691.0              # aggregate, not 109M
        # Sub-bucket keys in currentChainTvls are excluded.
        assert {c.chain for c in data.chains} == {"Ethereum", "Avalanche"}
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_symbol_only_protocol_uses_the_parent_and_says_so(self) -> None:
        """Uniswap has no gecko_id row and four versioned siblings."""
        respx.get(f"{LLAMA_URL}/protocols").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocols_versions.json")
            )
        )
        respx.get(f"{LLAMA_URL}/protocol/uniswap").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocol_uniswap.json")
            )
        )
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        data = await provider.fetch_onchain(
            ResolvedAsset(query="UNI", symbol="UNI", coingecko_id="uniswap")
        )

        assert data is not None
        assert data.defillama_slug == "uniswap"
        assert data.tvl == 3948674870.0  # v1+v2+v3+v4, not v3 alone
        assert any(m.field == "onchain.protocol_scope" for m in ctx.missing)
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_parent_missing_falls_back_to_the_version_row(self) -> None:
        """A protocol with versions but no resolvable parent still reports something."""
        respx.get(f"{LLAMA_URL}/protocols").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocols_versions.json")
            )
        )
        respx.get(f"{LLAMA_URL}/protocol/uniswap").mock(return_value=httpx.Response(404))
        respx.get(f"{LLAMA_URL}/protocol/uniswap-v3").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocol_uniswap_v3.json")
            )
        )
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        data = await provider.fetch_onchain(
            ResolvedAsset(query="UNI", symbol="UNI", coingecko_id="uniswap")
        )

        assert data is not None
        assert data.defillama_slug == "uniswap-v3"
        assert data.tvl == 1614399124.0
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_mismatched_protocol_symbol_is_flagged_not_trusted(self) -> None:
        respx.get(f"{LLAMA_URL}/v2/chains").mock(return_value=httpx.Response(200, json=[]))
        respx.get(f"{LLAMA_URL}/protocol/lido").mock(
            return_value=httpx.Response(
                200,
                json={"name": "Something Else", "symbol": "XXX",
                      "tvl": [{"date": 1, "totalLiquidityUSD": 5.0}]},
            )
        )
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        await provider.fetch_onchain(
            ResolvedAsset(query="LDO", symbol="LDO", coingecko_id="lido-dao",
                          defillama_slug="lido")
        )

        assert any(
            m.field == "onchain.protocol_scope" and "not this asset" in m.reason
            for m in ctx.missing
        )
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_slug_is_derived_from_gecko_id(self) -> None:
        respx.get(f"{LLAMA_URL}/protocols").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocols_bitcoin.json")
            )
        )
        respx.get(f"{LLAMA_URL}/protocol/bitcoin").mock(
            return_value=httpx.Response(200, json={"name": "Bitcoin", "tvl": None})
        )
        respx.get(f"{LLAMA_URL}/v2/chains").mock(return_value=httpx.Response(200, json=[]))
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)
        data = await provider.fetch_onchain(
            ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")
        )
        # gecko_id join finds the Bitcoin row regardless of how many share the symbol.
        assert data is not None
        assert data.defillama_slug == "bitcoin"
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_ambiguous_symbol_does_not_resolve(self) -> None:
        """Three protocols share symbol BTC; that is not a resolution."""
        respx.get(f"{LLAMA_URL}/protocols").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocols_bitcoin.json")
            )
        )
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        resolved = await provider.resolve("BTC")

        assert resolved is None
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_unique_symbol_does_resolve(self) -> None:
        respx.get(f"{LLAMA_URL}/protocols").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocols_bitcoin.json")
            )
        )
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        resolved = await provider.resolve("LDO")

        assert resolved is not None
        assert resolved.defillama_slug == "lido"
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_no_match_is_explained(self) -> None:
        respx.get(f"{LLAMA_URL}/v2/chains").mock(return_value=httpx.Response(200, json=[]))
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        assert await provider.fetch_onchain(
            ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")
        ) is None
        assert any("neither a DefiLlama chain nor a protocol" in m.reason for m in ctx.missing)
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_http_failure_is_tolerated(self) -> None:
        respx.get(f"{LLAMA_URL}/v2/chains").mock(return_value=httpx.Response(500))
        respx.get(f"{LLAMA_URL}/protocol/lido").mock(return_value=httpx.Response(500))
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        # A 500 must not propagate out of fetch_onchain.
        assert await provider.fetch_onchain(
            ResolvedAsset(query="LDO", symbol="LDO", coingecko_id="lido-dao",
                          defillama_slug="lido")
        ) is None
        assert ctx.missing
        await http.aclose()


class TestCoinMarketCap:
    def test_is_skipped_without_a_key(self) -> None:
        ok, reason = CoinMarketCapProvider.available(Config())
        assert ok is False
        assert reason is not None
        assert "TICKERKIT_CMC_API_KEY" in reason

    def test_is_available_with_a_key(self) -> None:
        ok, _reason = CoinMarketCapProvider.available(Config(cmc_api_key="k"))
        assert ok is True

    @pytest.mark.asyncio
    @respx.mock
    async def test_quotes_are_mapped_with_provenance(self) -> None:
        quote_route = respx.get(
            f"{CMC_URL}/v3/cryptocurrency/quotes/latest",
            params={"id": "1", "convert": "usd"},
        ).mock(return_value=httpx.Response(200, json=load_fixture("cmc_quotes_btc.json")))
        map_route = respx.get(f"{CMC_URL}/v1/cryptocurrency/map").mock(
            return_value=httpx.Response(200, json=load_fixture("cmc_map_btc.json"))
        )
        _config, http, ctx = _ctx(cmc_api_key="test-key")
        provider = CoinMarketCapProvider(ctx)

        market = await provider.fetch_market(
            ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")
        )

        assert market is not None
        assert market.aggregated_price == 81765.5
        assert market.market_cap == 1643000000000.0
        assert market.max_supply == 21000000.0
        assert market.provenance["cmc"].source.endswith("quotes/latest")
        # Both requests carried the credential header.
        for route in (quote_route, map_route):
            assert route.calls[0].request.headers["X-CMC_PRO_API_KEY"] == "test-key"
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_unknown_symbol_is_explained(self) -> None:
        respx.get(f"{CMC_URL}/v1/cryptocurrency/map").mock(
            return_value=httpx.Response(200, json={"data": []})
        )
        _config, http, ctx = _ctx(cmc_api_key="test-key")
        provider = CoinMarketCapProvider(ctx)

        assert await provider.fetch_market(
            ResolvedAsset(query="ZZZ", symbol="ZZZ")
        ) is None
        assert any("no CoinMarketCap id" in m.reason for m in ctx.missing)
        await http.aclose()

class TestLlamaParsers:
    """Unit coverage for the two shapes DefiLlama actually returns."""

    def test_parse_tvl_accepts_a_scalar(self) -> None:
        assert parse_tvl(9500000000.0) == 9500000000.0

    def test_parse_tvl_takes_the_last_history_point(self) -> None:
        history = [
            {"date": 1, "totalLiquidityUSD": 1.0},
            {"date": 2, "totalLiquidityUSD": 2.5},
            {"date": 3, "totalLiquidityUSD": 9.0},
        ]
        assert parse_tvl(history) == 9.0

    def test_parse_tvl_of_unknown_shape_is_none_not_zero(self) -> None:
        assert parse_tvl(None) is None
        assert parse_tvl("nope") is None
        assert parse_tvl([]) is None
        assert parse_tvl([{"date": 1}]) is None
        assert parse_tvl([1, 2, 3]) is None

    def test_parse_tvl_skips_trailing_unusable_points(self) -> None:
        assert parse_tvl([{"totalLiquidityUSD": 4.0}, {"date": 2}, None]) == 4.0

    @pytest.mark.parametrize(
        ("slug", "expected"),
        [
            ("aave-v2", "aave"),
            ("uniswap-v3", "uniswap"),
            ("aave-v3.1", "aave"),
            ("lido", None),
            ("rocket-pool", None),
        ],
    )
    def test_parent_slug(self, slug: str, expected: str | None) -> None:
        assert parent_slug(slug) == expected

    @pytest.mark.parametrize(
        ("key", "expected"),
        [
            ("Ethereum", True),
            ("Arbitrum", True),
            ("Ethereum-staking", False),
            ("Ethereum-pool2", False),
            ("Polygon-borrowed", False),
        ],
    )
    def test_is_chain_key(self, key: str, expected: bool) -> None:
        assert is_chain_key(key) is expected


class TestAmbiguousProtocolRefusal:
    """A ticker shared by several protocols must not yield an arbitrary protocol's TVL."""

    @pytest.mark.asyncio
    @respx.mock
    async def test_unidentified_asset_refuses_to_guess(self) -> None:
        """SOL also matches 'solana-farm' ($238); without an identity that is a wrong number."""
        respx.get(f"{LLAMA_URL}/protocols").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocols_solana.json")
            )
        )
        respx.get(f"{LLAMA_URL}/v2/chains").mock(return_value=httpx.Response(200, json=[]))
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        data = await provider.fetch_onchain(
            ResolvedAsset(query="SOL", symbol="SOL")   # unresolved
        )

        assert data is None
        assert any(
            m.field == "onchain.protocol_scope" and "refusing to guess" in m.reason
            for m in ctx.missing
        )
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_resolved_name_selects_the_right_protocol(self) -> None:
        respx.get(f"{LLAMA_URL}/protocols").mock(
            return_value=httpx.Response(
                200, json=load_fixture("defillama_protocols_solana.json")
            )
        )
        respx.get(f"{LLAMA_URL}/protocol/solana").mock(
            return_value=httpx.Response(
                200,
                json={"name": "Solana", "symbol": "SOL", "tvl": None,
                      "currentChainTvls": {"Solana": 6210773694.47}},
            )
        )
        # A resolved id also unlocks the chain-level TVL from /v2/chains, which is the
        # honest source for a base-layer asset like SOL.
        respx.get(f"{LLAMA_URL}/v2/chains").mock(
            return_value=httpx.Response(
                200,
                json=[{"gecko_id": "solana", "name": "Solana", "tvl": 6210773694.47,
                       "tokenSymbol": "SOL", "cmcId": "5426", "chainId": None}],
            )
        )
        _config, http, ctx = _ctx()
        provider = DefiLlamaProvider(ctx)

        data = await provider.fetch_onchain(
            ResolvedAsset(query="SOL", symbol="SOL", name="Solana", coingecko_id="solana")
        )

        assert data is not None
        assert data.defillama_slug == "solana"   # not solana-farm
        # Chain TVL, not the $238 of some unrelated protocol.
        assert data.tvl == 6210773694.47
        await http.aclose()
