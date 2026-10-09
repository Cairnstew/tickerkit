"""QueryCache: save/load whole snapshot & history results, and model round-trips.

The round-trip asserts are the real safety net for ``from_jsonified``: every field
(including datetimes, StrEnums, nested series, coverage objects, provenance dicts) must
survive ``to_dict -> from_dict -> to_dict`` unchanged.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tickerkit import (
    Cache,
    Candle,
    Capability,
    ConsensusCandle,
    DataQuality,
    Discrepancy,
    HistoryCandle,
    HistoryConsensus,
    HistoryCoverage,
    HistoryRequest,
    HistoryResult,
    HistorySeries,
    Identity,
    IdentityLinks,
    Market,
    MissingField,
    OHLCVSeries,
    Provenance,
    ProviderErrorRecord,
    QueryCache,
    RateLimitInfo,
    SkippedProvider,
    TickerSnapshot,
    VenueTicker,
)

DT = datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC)


def _provenance(source: str = "coingecko") -> Provenance:
    return Provenance(
        source=source,
        capability=Capability.META,
        fetched_at=DT,
        cached=False,
        notes=("x",),
    )


def _identity() -> Identity:
    return Identity(
        symbol="BTC",
        name="Bitcoin",
        coingecko_id="bitcoin",
        categories=("currency", "mineable"),
        links=IdentityLinks(homepage=("https://bitcoin.org",), github=("bitcoin/bitcoin",)),
        description="A cryptocurrency.",
        contract_address=None,
        asset_platform_id=None,
        provenance=_provenance(),
    )


def _history_result() -> HistoryResult:
    series = HistorySeries(
        source="ccxt:binance",
        symbol="BTC",
        timeframe="1d",
        quote_currency="USD",
        kind="ohlcv",
        candles=(
            HistoryCandle(timestamp=DT, close=40_000.0, open=39_000.0, high=41_000.0, low=38_000.0),
            HistoryCandle(timestamp=DT + timedelta(days=1), close=41_000.0),
        ),
        coverage=HistoryCoverage(
            requested_start=DT,
            requested_end=DT + timedelta(days=1),
            first=DT,
            last=DT + timedelta(days=1),
            candles=2,
            expected=2,
            gaps=(),
            truncated=False,
            calls=1,
            complete=True,
        ),
    )
    consensus = HistoryConsensus(
        timeframe="1d",
        quote_currency="USD",
        method="median",
        inputs=("ccxt:binance", "ccxt:coinbase"),
        candles=tuple(
            ConsensusCandle(
                timestamp=c.timestamp, close=c.close, open=c.open,
                high=c.high, low=c.low, volume=None, sources=("ccxt:binance",), n=1,
            )
            for c in series.candles
        ),
        deviations_pct={"ccxt:binance": 0.01, "ccxt:coinbase": 0.02},
    )
    return HistoryResult(
        identity=_identity(),
        request=HistoryRequest(timeframe="1d", start=DT, end=DT + timedelta(days=1)),
        data_quality=DataQuality(
            errors=(
                ProviderErrorRecord(
                    provider="ccxt:kraken",
                    capability=Capability.HISTORY,
                    exception_type="RateLimitExceeded",
                    message="kraken ignored since",
                    occurred_at=DT,
                ),
            ),
            rate_limited=(
                RateLimitInfo(
                    source="coingecko",
                    capability=Capability.AGG_MARKET,
                    message="429",
                    retry_after=30.0,
                    remedy="wait",
                    occurred_at=DT,
                ),
            ),
            providers_used=("ccxt:binance",),
            providers_skipped=(SkippedProvider(provider="coinmarketcap", reason="no key"),),
        ),
        series=(series,),
        consensus=consensus,
    )


def _snapshot() -> TickerSnapshot:
    venue = VenueTicker(
        exchange="binance",
        symbol="BTC",
        quote_currency="USD",
        bid=40_000.0,
        ask=40_001.0,
        last=40_000.5,
        timestamp=DT,
        is_stale=False,
        is_outlier=False,
        deviation_pct=0.01,
        provenance=_provenance("ccxt"),
    )
    return TickerSnapshot(
        identity=_identity(),
        market=Market(
            quote_currency="USD",
            consensus_price=40_000.5,
            consensus_method="median",
            consensus_inputs=3,
            aggregated_price=40_000.5,
            change_24h=100.0,
            change_24h_pct=0.25,
            high_24h=41_000.0,
            low_24h=39_000.0,
            volume_24h=1e9,
            market_cap=8e11,
            ath=69_000.0,
            ath_date=DT,
            provenance={"agg": _provenance("coingecko")},
        ),
        data_quality=DataQuality(
            missing=(MissingField(field="fdv", reason="not supplied by source"),),
            discrepancies=(
                Discrepancy(
                    kind="history_reference_vs_consensus",
                    detail="divergence",
                    left_source="a",
                    left_value=1.0,
                    right_source="b",
                    right_value=2.0,
                    deviation_pct=1.0,
                ),
            ),
        ),
        venues=(venue,),
        ohlcv=(
            OHLCVSeries(
                exchange="binance",
                symbol="BTC",
                timeframe="1h",
                quote_currency="USD",
                candles=(
                    Candle(timestamp=DT, open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0),
                ),
                provenance=_provenance("ccxt"),
            ),
        ),
    )


async def test_history_roundtrip() -> None:
    original = _history_result()
    rebuilt = HistoryResult.from_dict(original.to_dict())
    assert rebuilt == original
    assert rebuilt.to_dict() == original.to_dict()
    assert isinstance(rebuilt.consensus, HistoryConsensus)
    cov = rebuilt.series[0].coverage
    assert cov is not None and cov.complete is True


async def test_history_json_roundtrip() -> None:
    original = _history_result()
    rebuilt = HistoryResult.from_json(original.to_json())
    assert rebuilt == original
    # enum + datetime survived the string path
    assert rebuilt.data_quality.errors[0].capability is Capability.HISTORY
    assert rebuilt.data_quality.errors[0].occurred_at == DT


async def test_snapshot_roundtrip_ignores_derived_mid() -> None:
    original = _snapshot()
    data = original.to_dict()
    # VenueTicker._json_hook adds a derived mid; from_dict must ignore it
    assert "mid" in data["venues"][0]
    rebuilt = TickerSnapshot.from_dict(data)
    assert rebuilt == original
    assert rebuilt.to_dict() == original.to_dict()
    assert rebuilt.venues[0].mid == original.venues[0].mid


async def test_snapshot_json_roundtrip() -> None:
    original = _snapshot()
    rebuilt = TickerSnapshot.from_json(original.to_json())
    assert rebuilt == original
    assert rebuilt.market.provenance["agg"].source == "coingecko"


async def test_query_cache_save_load_snapshot(tmp_path) -> None:
    original = _snapshot()
    qc = QueryCache(tmp_path / "queries.sqlite3")
    try:
        await qc.save_snapshot("BTC", original)
        loaded = await qc.load_snapshot("BTC")
        assert loaded is not None
        assert loaded == original
        assert loaded.to_dict() == original.to_dict()
        # case-insensitive ticker, immortal default TTL
        assert await qc.load_snapshot("btc") == loaded
        assert await qc.load_snapshot("ETH") is None
    finally:
        qc.close()


async def test_query_cache_save_load_history(tmp_path) -> None:
    qc = QueryCache(tmp_path / "queries.sqlite3")
    try:
        hist = _history_result()
        await qc.save_history("BTC", "1d", start="2024-01-15", history=hist)
        loaded = await qc.load_history("BTC", "1d", start="2024-01-15")
        assert loaded is not None and loaded == hist
        # window is part of the key
        assert await qc.load_history("BTC", "1d", start="2023-01-15") is None
        # dict form is accepted
        await qc.save_history("ETH", "1d", end="2024-02-01", history=hist.to_dict())
        assert await qc.load_history("ETH", "1d", end="2024-02-01") == hist
    finally:
        qc.close()


async def test_save_history_requires_result(tmp_path) -> None:
    qc = QueryCache(tmp_path / "q.sqlite3")
    try:
        with pytest.raises(ValueError):
            await qc.save_history("BTC", "1d")
    finally:
        qc.close()


async def test_query_cache_is_a_cache_protocol(tmp_path) -> None:
    qc = QueryCache(tmp_path / "q.sqlite3")
    try:
        assert isinstance(qc, Cache)
        await qc.set("k", {"a": 1}, 60)
        assert await qc.get("k") == {"a": 1}
        assert len(qc) == 1
        await qc.clear()
        assert len(qc) == 0
    finally:
        qc.close()


async def test_async_context_manager(tmp_path) -> None:
    async with QueryCache(tmp_path / "q.sqlite3") as qc:
        await qc.save_snapshot("BTC", _snapshot())
        assert await qc.load_snapshot("BTC") is not None


async def test_query_cache_persists_across_reopen(tmp_path) -> None:
    path = tmp_path / "queries.sqlite3"
    qc = QueryCache(path)
    await qc.save_snapshot("BTC", _snapshot())
    qc.close()
    qc2 = QueryCache(path)
    try:
        assert await qc2.load_snapshot("BTC") is not None
        assert isinstance(qc2.stats()["entries"], int)
    finally:
        qc2.close()