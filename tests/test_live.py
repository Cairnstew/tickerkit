"""Opt-in live smoke test against the real free APIs.

Deselected by default (``addopts = "-m 'not live'"``). Run it with:

    pytest -m live

It hits CoinGecko (keyless), ccxt and DefiLlama. The assertions are deliberately about
*shape and traceability*, not about price levels, which change every second.
"""

from __future__ import annotations

import pytest

import tickerkit

pytestmark = pytest.mark.live


def _cg_throttled(snap: object) -> bool:
    """Whether CoinGecko refused us (keyless access is IP-rate-limited)."""
    return any(
        e.provider == "coingecko" for e in snap.data_quality.errors  # type: ignore[attr-defined]
    )


@pytest.mark.asyncio
async def test_live_btc_snapshot() -> None:
    snap = await tickerkit.aget_ticker("BTC", cache_enabled=False)

    if _cg_throttled(snap):
        # Keyless CoinGecko is IP-rate-limited and shared, so a refusal during a
        # test run is expected behaviour, not a failure — but it must be recorded.
        pytest.skip("CoinGecko rate-limited this run; failures were correctly recorded")

    assert snap.identity.coingecko_id == "bitcoin"
    assert snap.identity.name == "Bitcoin"
    assert snap.market.price is not None and snap.market.price > 0
    assert snap.market.consensus_price is not None
    assert len(snap.venues) >= 1
    assert snap.orderbooks, "at least one venue should publish a book"
    assert snap.trades

    # Every field group must be attributable.
    assert snap.data_quality.provenance, "provenance map must not be empty"
    for key, prov in snap.data_quality.provenance.items():
        assert prov.source, key
        assert prov.fetched_at.tzinfo is not None, key

    for venue in snap.venues:
        if venue.bid is not None and venue.ask is not None:
            assert venue.ask >= venue.bid, f"{venue.exchange} book is crossed"
        assert venue.provenance is not None

    assert snap.onchain is not None
    assert snap.onchain.tvl is not None and snap.onchain.tvl > 0

    # Serialisable and DataFrame-able.
    assert snap.to_json()
    assert snap.to_dataframes()


@pytest.mark.asyncio
async def test_live_consensus_agrees_with_the_aggregator() -> None:
    """Venue consensus and CoinGecko's aggregate should be within a few percent."""
    snap = await tickerkit.aget_ticker("ETH", cache_enabled=False)
    if _cg_throttled(snap):
        pytest.skip("CoinGecko rate-limited this run")

    assert snap.market.price is not None
    if snap.market.consensus_price and snap.market.aggregated_price:
        deviation = abs(
            snap.market.consensus_price - snap.market.aggregated_price
        ) / snap.market.consensus_price
        assert deviation < 0.05, f"consensus/aggregator divergence {deviation:.2%}"


@pytest.mark.asyncio
async def test_live_cli_json_path() -> None:
    import json

    snap = await tickerkit.aget_ticker("SOL", cache_enabled=False)
    payload = json.loads(snap.to_json())
    assert payload["identity"]["symbol"] == "SOL"
    assert payload["data_quality"]["providers_used"]
    if _cg_throttled(snap):
        pytest.skip("CoinGecko rate-limited this run")
    assert payload["identity"]["coingecko_id"] == "solana"


@pytest.mark.asyncio
async def test_live_unknown_ticker_still_returns_a_snapshot() -> None:
    """A ticker nobody has heard of must not raise."""
    snap = await tickerkit.aget_ticker("ZZZZNOTACOIN", cache_enabled=False)
    assert snap.market.price is None
    assert snap.data_quality.errors or snap.data_quality.missing
    assert snap.to_json()