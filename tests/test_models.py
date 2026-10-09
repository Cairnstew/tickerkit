"""Schema invariants: serialisation, provenance, DataFrame views, config."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from tickerkit.capabilities import Capability
from tickerkit.config import Config
from tickerkit.models import (
    Candle,
    DataQuality,
    Identity,
    Market,
    OHLCVSeries,
    OrderBook,
    Provenance,
    TickerSnapshot,
    VenueTicker,
    finite,
    jsonify,
    to_iso,
    utcnow,
)


def rich_snapshot() -> TickerSnapshot:
    now = utcnow()
    prov = Provenance(
        source="coingecko:/coins/markets",
        capability=Capability.AGG_MARKET,
        fetched_at=now,
        exchange_timestamp=now,
        url="https://api.coingecko.com/api/v3/coins/markets",
    )
    return TickerSnapshot(
        identity=Identity(symbol="BTC", name="Bitcoin", coingecko_id="bitcoin"),
        market=Market(
            quote_currency="USD",
            consensus_price=100.0,
            aggregated_price=100.5,
            market_cap=1234.0,
            provenance={"aggregated": prov},
        ),
        venues=(
            VenueTicker(
                exchange="kraken", symbol="BTC/USD", quote_currency="USD",
                bid=99.9, ask=100.1, last=100.0, timestamp=now, provenance=prov,
            ),
        ),
        orderbooks=(
            OrderBook(
                exchange="kraken", symbol="BTC/USD", quote_currency="USD",
                best_bid=99.9, best_ask=100.1, mid=100.0, spread=0.2,
                bid_depth_base=3.0, ask_depth_base=1.0, imbalance=0.5, timestamp=now,
            ),
        ),
        ohlcv=(
            OHLCVSeries(
                exchange="kraken", symbol="BTC/USD", timeframe="1h", quote_currency="USD",
                candles=(
                    Candle(timestamp=now, open=1.0, high=2.0, low=0.5, close=1.5, volume=10.0),
                ),
                provenance=prov,
            ),
        ),
        data_quality=DataQuality(provenance={"market:aggregated": prov}),
        query="BTC",
    )


class TestSerialisation:
    def test_to_dict_is_json_serialisable(self) -> None:
        payload = rich_snapshot().to_dict()
        assert json.loads(json.dumps(payload)) == payload

    def test_to_json_round_trips(self) -> None:
        snap = rich_snapshot()
        assert json.loads(snap.to_json()) == snap.to_dict()

    def test_to_json_compact_form(self) -> None:
        out = rich_snapshot().to_json(indent=None)
        assert "\n" not in out

    def test_timestamps_are_iso_and_utc(self) -> None:
        payload = rich_snapshot().to_dict()
        fetched = payload["data_quality"]["provenance"]["market:aggregated"]["fetched_at"]
        assert fetched.endswith("+00:00")
        assert datetime.fromisoformat(fetched).tzinfo == UTC

    def test_nested_dataclasses_are_expanded(self) -> None:
        payload = rich_snapshot().to_dict()
        assert payload["identity"]["coingecko_id"] == "bitcoin"
        assert payload["venues"][0]["mid"] == pytest.approx(100.0)


class TestJsonify:
    def test_non_finite_floats_become_null_not_invalid_json(self) -> None:
        assert json.dumps({"x": float("nan")}) != '{"x": null}'
        assert jsonify(float("nan")) is None
        assert jsonify(float("inf")) is None

    def test_handles_nested_structures(self) -> None:
        assert jsonify({"a": (1, 2), "b": [3.0]}) == {"a": [1, 2], "b": [3.0]}


class TestFinite:
    @pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True, "abc", object()])
    def test_rejects_unusable(self, value: object) -> None:
        assert finite(value) is None

    def test_accepts_numbers_and_numeric_strings(self) -> None:
        assert finite(1.5) == 1.5
        assert finite("1.5") == 1.5
        assert finite(0) == 0.0


def test_to_iso_handles_none() -> None:
    assert to_iso(None) is None
    assert to_iso(datetime(2024, 1, 1, tzinfo=UTC)).startswith("2024-01-01")


class TestDataFrames:
    def test_frames_have_expected_names_and_columns(self) -> None:
        frames = rich_snapshot().to_dataframes()
        assert {"venues", "orderbook", "ohlcv"} <= set(frames)
        assert "source" in frames["venues"].columns
        assert "fetched_at" in frames["venues"].columns
        assert frames["ohlcv"].index.name == "timestamp"

    def test_ohlcv_index_is_utc(self) -> None:
        frame = rich_snapshot().to_dataframes()["ohlcv"]
        assert str(frame.index.tz) == "UTC"

    def test_empty_snapshot_yields_no_frames_rather_than_empty_ones(self) -> None:
        snap = TickerSnapshot(
            identity=Identity(symbol="X"),
            market=Market(quote_currency="USD"),
            data_quality=DataQuality(),
        )
        assert snap.to_dataframes() == {}

    def test_every_venue_row_is_traceable(self) -> None:
        """The stated invariant: every number in the output is traceable to a source."""
        frame = rich_snapshot().to_dataframes()["venues"]
        assert frame["source"].notna().all()
        assert frame["fetched_at"].notna().all()


class TestConfig:
    def test_defaults(self) -> None:
        config = Config()
        assert config.quote_currency == "USD"
        assert "binance" in config.exchanges
        assert config.is_fiat_equivalent("USDT")
        assert not config.is_fiat_equivalent("EUR")

    def test_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TICKERKIT_QUOTE_CURRENCY", "eur")
        monkeypatch.setenv("TICKERKIT_EXCHANGES", "kraken, bybit")
        monkeypatch.setenv("TICKERKIT_CACHE_TTL", "90")
        monkeypatch.setenv("TICKERKIT_MAX_RETRIES", "not-a-number")

        config = Config.from_env()
        assert config.quote_currency == "EUR"
        assert config.exchanges == ("kraken", "bybit")
        assert config.cache_ttl == 90.0
        assert config.max_retries == 3  # unparseable falls back to the default

    def test_with_overrides_ignores_none(self) -> None:
        base = Config()
        assert base.with_overrides(quote_currency=None) is base
        assert base.with_overrides(quote_currency="EUR").quote_currency == "EUR"

    def test_outlier_threshold(self) -> None:
        assert float(Config(outlier_pct=5.0).outlier_threshold()) == 0.05

    def test_never_hardcodes_a_secret(self) -> None:
        """Config must not carry credentials unless they came from the environment."""
        assert Config().coingecko_api_key is None
        assert Config().coingecko_pro_api_key is None
        assert Config().cmc_api_key is None


class TestSchemaIsFrozen:
    def test_snapshots_are_immutable(self) -> None:
        snap = rich_snapshot()
        with pytest.raises(AttributeError):
            snap.query = "ETH"  # type: ignore[misc]

    def test_slots_reject_unknown_fields(self) -> None:
        with pytest.raises(TypeError):
            Identity(symbol="BTC", nonsense=True)  # type: ignore[call-arg]