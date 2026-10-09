"""Deep-history assembly: coverage accounting, consensus and cross-source checks.

History's hard problem is not fetching numbers, it is knowing whether you got all of
them. These tests pin the honesty guarantees: a short series is never reported as
complete, a rogue venue is excluded rather than averaged in, and a close-only reference
never contributes a bar it cannot support.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tickerkit.config import Config
from tickerkit.history import (
    build_coverage,
    build_history_consensus,
    cross_check_references,
    expected_intervals,
    find_gaps,
    floor_to_interval,
    interval_seconds,
    normalise_candles,
)
from tickerkit.models import (
    HistoryCandle,
    HistoryRequest,
    HistorySeries,
    utcnow,
)

DAY = 86_400


def _days(*days: int, month: int = 1) -> list[datetime]:
    return [datetime(2024, month, d, tzinfo=UTC) for d in days]


def _candle(moment: datetime, price: float, *, ohlcv: bool = True) -> HistoryCandle:
    if ohlcv:
        return HistoryCandle(
            timestamp=moment, close=price, open=price, high=price, low=price, volume=1.0
        )
    return HistoryCandle(timestamp=moment, close=price)


def _request(start: datetime | None = None, end: datetime | None = None) -> HistoryRequest:
    return HistoryRequest(timeframe="1d", start=start, end=end)


def _series(
    source: str, candles: list[HistoryCandle], *, kind: str = "ohlcv", request: HistoryRequest | None = None
) -> HistorySeries:
    request = request or _request()
    return HistorySeries(
        source=source,
        symbol="BTC",
        timeframe="1d",
        quote_currency="USD",
        kind=kind,
        candles=tuple(candles),
        coverage=build_coverage([c.timestamp for c in candles], request),
    )


class TestIntervalMath:
    def test_known_timeframes(self) -> None:
        assert interval_seconds("1d") == DAY
        assert interval_seconds("4h") == 14_400
        assert interval_seconds("1M") is not None

    def test_unknown_timeframe_is_none_not_a_guess(self) -> None:
        assert interval_seconds("7s") is None

    def test_floor_to_interval_snaps_onto_the_boundary(self) -> None:
        moment = datetime(2024, 1, 1, 13, 45, tzinfo=UTC)
        assert floor_to_interval(moment, DAY) == datetime(2024, 1, 1, tzinfo=UTC)

    def test_expected_intervals_counts_inclusive_boundaries(self) -> None:
        request = _request(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC))
        # 1st..5th inclusive is five daily intervals, not four.
        assert expected_intervals(request, DAY) == 5

    def test_expected_intervals_needs_both_bounds(self) -> None:
        assert expected_intervals(_request(datetime(2024, 1, 1, tzinfo=UTC), None), DAY) is None


class TestGapDetection:
    def test_contiguous_series_has_no_gaps(self) -> None:
        assert find_gaps(_days(1, 2, 3, 4), DAY) == ()

    def test_single_missing_interval_is_a_gap(self) -> None:
        gaps = find_gaps(_days(1, 2, 4, 5), DAY)
        assert len(gaps) == 1
        assert gaps[0].missing == 1
        assert gaps[0].start == datetime(2024, 1, 2, tzinfo=UTC)
        assert gaps[0].end == datetime(2024, 1, 4, tzinfo=UTC)

    def test_multi_interval_gap_counts_each_missing_candle(self) -> None:
        gaps = find_gaps(_days(1, 5), DAY)
        assert gaps[0].missing == 3


class TestCoverage:
    def test_full_window_is_complete(self) -> None:
        request = _request(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC))
        coverage = build_coverage(_days(1, 2, 3, 4, 5), request)
        assert coverage.complete is True
        assert coverage.expected == 5 and coverage.missing == 0
        assert coverage.gaps == ()

    def test_missing_interior_candle_is_incomplete(self) -> None:
        request = _request(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC))
        coverage = build_coverage(_days(1, 2, 4, 5), request)
        assert coverage.complete is False
        assert coverage.missing == 1
        assert len(coverage.gaps) == 1

    def test_truncated_is_never_complete(self) -> None:
        """Even a full-looking window is not complete when a cap cut the walk short."""
        request = _request(datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 1, 5, tzinfo=UTC))
        coverage = build_coverage(_days(1, 2, 3, 4, 5), request, truncated=True)
        assert coverage.complete is False
        assert coverage.truncated is True

    def test_unbounded_request_is_complete_without_gaps(self) -> None:
        coverage = build_coverage(_days(1, 2, 3), _request())
        assert coverage.complete is True
        assert coverage.expected is None
        assert coverage.missing is None

    def test_empty_series_is_never_complete(self) -> None:
        coverage = build_coverage([], _request(datetime(2024, 1, 1, tzinfo=UTC)))
        assert coverage.complete is False
        assert coverage.candles == 0


class TestNormaliseCandles:
    def test_trims_to_requested_window(self) -> None:
        request = _request(datetime(2024, 1, 2, tzinfo=UTC), datetime(2024, 1, 4, tzinfo=UTC))
        candles = [_candle(m, 100.0) for m in _days(1, 2, 3, 4, 5)]
        trimmed = normalise_candles(candles, request)
        assert [c.timestamp.day for c in trimmed] == [2, 3, 4]

    def test_duplicate_timestamps_last_write_wins(self) -> None:
        moment = datetime(2024, 1, 1, tzinfo=UTC)
        candles = [_candle(moment, 100.0), _candle(moment, 111.0)]
        out = normalise_candles(candles, _request())
        assert len(out) == 1 and out[0].close == 111.0

    def test_align_snaps_price_points_onto_the_boundary(self) -> None:
        off = datetime(2024, 1, 1, 9, 30, tzinfo=UTC)
        out = normalise_candles([_candle(off, 100.0, ohlcv=False)], _request(), align=True)
        assert out[0].timestamp == datetime(2024, 1, 1, tzinfo=UTC)

    def test_limit_keeps_the_most_recent(self) -> None:
        request = HistoryRequest(timeframe="1d", limit=2)
        candles = [_candle(m, float(d)) for m, d in zip(_days(1, 2, 3, 4), (1, 2, 3, 4), strict=True)]
        out = normalise_candles(candles, request)
        assert [c.timestamp.day for c in out] == [3, 4]


class TestHistoryConsensus:
    def test_no_ohlcv_source_yields_no_consensus(self) -> None:
        price_only = _series("coingecko:bitcoin", [_candle(m, 1.0, ohlcv=False) for m in _days(1, 2)], kind="price")
        consensus, discrepancies = build_history_consensus([price_only], Config(), _request())
        assert consensus is None and discrepancies == ()

    def test_single_source_still_produces_a_consensus(self) -> None:
        source = _series("ccxt:binance", [_candle(m, 100.0) for m in _days(1, 2, 3)])
        consensus, discrepancies = build_history_consensus([source], Config(), _request())
        assert consensus is not None
        assert consensus.method == "single_source"
        assert consensus.inputs == ("ccxt:binance",)
        assert len(consensus.candles) == 3
        assert discrepancies == ()

    def test_agreeing_sources_median_cleanly(self) -> None:
        a = _series("ccxt:binance", [_candle(m, p) for m, p in zip(_days(1, 2), (100.0, 101.0), strict=True)])
        b = _series("ccxt:okx", [_candle(m, p) for m, p in zip(_days(1, 2), (100.2, 100.8), strict=True)])
        consensus, discrepancies = build_history_consensus([a, b], Config(), _request())
        assert consensus is not None
        assert consensus.method == "median"
        assert consensus.candles[0].n == 2
        assert consensus.candles[0].sources == ("ccxt:binance", "ccxt:okx")
        assert discrepancies == ()

    def test_rogue_venue_is_excluded_not_averaged(self) -> None:
        days = _days(1, 2, 3)
        a = _series("ccxt:binance", [_candle(m, 100.0) for m in days])
        b = _series("ccxt:okx", [_candle(m, 100.0) for m in days])
        # 50% above the others on every interval: a wrong feed, not a market move.
        c = _series("ccxt:bybit", [_candle(m, 150.0) for m in days])
        consensus, discrepancies = build_history_consensus([a, b, c], Config(), _request())
        assert consensus is not None
        assert consensus.method == "median_outlier_trimmed"
        assert "ccxt:bybit" not in consensus.inputs
        assert consensus.candles[0].n == 2
        assert consensus.candles[0].close == 100.0
        assert any(d.kind == "history_source_outlier" for d in discrepancies)

    def test_two_sources_are_never_trimmed(self) -> None:
        """With two venues a disagreement is ambiguous, so nothing is dropped."""
        days = _days(1, 2)
        a = _series("ccxt:binance", [_candle(m, 100.0) for m in days])
        b = _series("ccxt:okx", [_candle(m, 150.0) for m in days])
        consensus, _ = build_history_consensus([a, b], Config(), _request())
        assert consensus is not None
        assert consensus.method == "median"
        assert len(consensus.inputs) == 2

    def test_spread_is_reported_per_interval(self) -> None:
        day = _days(1)[0]
        a = _series("ccxt:binance", [_candle(day, 100.0)])
        b = _series("ccxt:okx", [_candle(day, 101.0)])
        consensus, _ = build_history_consensus([a, b], Config(), _request())
        assert consensus is not None
        # (101 - 100) / median(100.5) * 100, not a plain 1% of either leg.
        assert consensus.candles[0].spread_pct == pytest.approx(0.995, abs=0.01)

    def test_price_kind_is_never_a_consensus_input(self) -> None:
        ohlcv = _series("ccxt:binance", [_candle(m, 100.0) for m in _days(1, 2)])
        price = _series("coingecko:bitcoin", [_candle(m, 100.0, ohlcv=False) for m in _days(1, 2)], kind="price")
        consensus, _ = build_history_consensus([ohlcv, price], Config(), _request())
        assert consensus is not None
        assert consensus.inputs == ("ccxt:binance",)


class TestCrossCheckReferences:
    def test_agreeing_reference_is_not_reported(self) -> None:
        days = _days(1, 2)
        ohlcv = _series("ccxt:binance", [_candle(m, 100.0) for m in days])
        consensus, _ = build_history_consensus([ohlcv], Config(), _request())
        reference = _series("coingecko:bitcoin", [_candle(m, 100.05, ohlcv=False) for m in days], kind="price")
        assert cross_check_references(consensus, [reference], 1.0) == ()

    def test_diverging_reference_is_recorded(self) -> None:
        days = _days(1, 2)
        ohlcv = _series("ccxt:binance", [_candle(m, 100.0) for m in days])
        consensus, _ = build_history_consensus([ohlcv], Config(), _request())
        reference = _series("coingecko:bitcoin", [_candle(m, 120.0, ohlcv=False) for m in days], kind="price")
        discrepancies = cross_check_references(consensus, [reference], 1.0)
        assert len(discrepancies) == 1
        assert discrepancies[0].kind == "history_reference_vs_consensus"
        assert discrepancies[0].left_source == "coingecko:bitcoin"

    def test_no_consensus_no_report(self) -> None:
        reference = _series("coingecko:bitcoin", [_candle(m, 100.0, ohlcv=False) for m in _days(1)], kind="price")
        assert cross_check_references(None, [reference], 1.0) == ()

    def test_reference_without_shared_intervals_is_silent(self) -> None:
        ohlcv = _series("ccxt:binance", [_candle(m, 100.0) for m in _days(1, 2)])
        consensus, _ = build_history_consensus([ohlcv], Config(), _request())
        elsewhere = _series(
            "coingecko:bitcoin", [_candle(datetime(2020, 6, 1, tzinfo=UTC), 1.0, ohlcv=False)], kind="price"
        )
        assert cross_check_references(consensus, [elsewhere], 1.0) == ()


class TestTimestampHandling:
    def test_utc_floor_does_not_drift_with_offset_input(self) -> None:
        """A naive-looking near-midnight timestamp must not shift a day on flooring."""
        moment = datetime(2024, 1, 2, 0, 0, 1, tzinfo=UTC)
        assert floor_to_interval(moment, DAY) == datetime(2024, 1, 2, tzinfo=UTC)

    def test_recent_window_covers_now(self) -> None:
        now = utcnow()
        coverage = build_coverage([now - timedelta(days=1), now], _request())
        assert coverage.complete is True
        assert coverage.first is not None
