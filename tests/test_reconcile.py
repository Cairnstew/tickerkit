"""Reconciliation: consensus, outlier exclusion, staleness, aggregator cross-check."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from tickerkit.config import Config
from tickerkit.models import (
    DataQuality,
    Identity,
    Market,
    TickerSnapshot,
    VenueTicker,
    utcnow,
)
from tickerkit.reconcile import (
    apply_consensus,
    build_consensus,
    cross_check,
    robust_z_scores,
    summarise,
)


def venue(
    exchange: str,
    bid: float | None,
    ask: float | None,
    *,
    last: float | None = None,
    age_seconds: float | None = 0.0,
) -> VenueTicker:
    now = utcnow()
    timestamp = None if age_seconds is None else now - timedelta(seconds=age_seconds)
    return VenueTicker(
        exchange=exchange,
        symbol="BTC/USD",
        quote_currency="USD",
        bid=bid,
        ask=ask,
        last=last,
        timestamp=timestamp,
    )


def snapshot_with(venues: tuple[VenueTicker, ...], **market_kwargs: object) -> TickerSnapshot:
    market = Market(quote_currency="USD", **market_kwargs)  # type: ignore[arg-type]
    return TickerSnapshot(
        identity=Identity(symbol="BTC", coingecko_id="bitcoin"),
        market=market,
        data_quality=DataQuality(),
        venues=venues,
    )


class TestRobustZScores:
    def test_zero_mad_yields_no_outliers(self) -> None:
        """Identical venues have MAD 0; nothing should be flagged as an outlier."""
        assert robust_z_scores([100.0, 100.0, 100.0], 100.0) == [0.0, 0.0, 0.0]

    def test_identifies_the_single_extreme_value(self) -> None:
        values = [100.0, 100.1, 99.9, 100.0, 500.0]
        centre = 100.0
        scores = robust_z_scores(values, centre)
        assert scores[-1] > 3.5
        assert all(abs(s) < 3.5 for s in scores[:-1])


class TestConsensus:
    def test_median_of_agreeing_venues(self) -> None:
        venues = (
            venue("a", 100.0, 100.1),
            venue("b", 100.2, 100.3),
            venue("c", 99.9, 100.0),
        )
        # mids are 100.05, 100.25, 99.95 -> median 100.05
        result = build_consensus(venues, Config())
        assert result.price == pytest.approx(100.05, rel=1e-6)
        assert result.inputs == 3
        assert result.excluded == ()

    def test_injected_outlier_is_excluded_and_recorded(self) -> None:
        """One venue priced 20x too high must not drag the consensus."""
        venues = (
            venue("kraken", 100.0, 100.1),
            venue("binance", 100.1, 100.2),
            venue("coinbase", 99.9, 100.0),
            venue("rogue", 2000.0, 2001.0),  # <- injected outlier
        )
        result = build_consensus(venues, Config(outlier_pct=5.0, outlier_mad_z=3.5))

        assert result.excluded == ("rogue",)
        assert result.inputs == 3
        # Median of the three honest venues, not something dragged upward.
        assert result.price == pytest.approx(100.05, rel=1e-3)
        assert result.method == "median_outlier_trimmed"

    def test_outlier_on_both_sides_is_excluded(self) -> None:
        venues = (
            venue("a", 100.0, 100.0),
            venue("b", 100.1, 100.1),
            venue("c", 100.1, 100.1),
            venue("high", 300.0, 300.0),
            venue("low", 3.0, 3.0),
        )
        result = build_consensus(venues, Config(outlier_pct=5.0))
        assert set(result.excluded) == {"high", "low"}

    def test_small_deviation_within_threshold_is_kept_but_flagged(self) -> None:
        """A 2% gap is not an outlier, but must still be recorded as a deviation."""
        venues = (
            venue("a", 100.0, 100.0),
            venue("b", 100.0, 100.0),
            venue("c", 102.0, 102.0),
        )
        result = build_consensus(venues, Config(outlier_pct=5.0))
        assert result.excluded == ()
        assert result.deviations_pct["c"] == pytest.approx(2.0, rel=1e-2)

    def test_fewer_than_three_venues_is_not_trimmed(self) -> None:
        """With two venues nothing is an outlier by definition; median stands."""
        venues = (venue("a", 100.0, 100.0), venue("b", 200.0, 200.0))
        result = build_consensus(venues, Config(outlier_pct=1.0))
        assert result.excluded == ()
        assert result.price == pytest.approx(150.0)

    def test_falls_back_to_last_then_bid_then_ask(self) -> None:
        venues = (
            venue("a", None, None, last=100.0),
            venue("b", None, 101.0),
            venue("c", 99.0, None),
        )
        result = build_consensus(venues, Config())
        assert result.price == pytest.approx(100.0)

    def test_no_usable_prices_yields_none_not_zero(self) -> None:
        result = build_consensus((venue("a", None, None),), Config())
        assert result.price is None
        assert result.inputs == 0

    def test_zero_and_negative_prices_are_ignored(self) -> None:
        result = build_consensus((venue("a", 0.0, 0.0), venue("b", -1.0, -1.0)), Config())
        assert result.price is None

    def test_tight_cluster_is_not_mangled_by_the_mad_criterion(self) -> None:
        """Regression: three venues agreeing to 5 significant figures.

        Their MAD is ~1e-5 of the price, so a naive robust z-score scores an ordinary
        venue > 3.5 and excludes it. Only the gross percentage test may exclude a venue
        below min_venues_for_mad.
        """
        venues = (
            venue("binance", 81709.76, 81709.79),   # mid 81709.775
            venue("kraken", 81707.24, 81707.26),     # mid 81707.250
            venue("coinbase", 81706.94, 81706.95),   # mid 81706.945
        )
        result = build_consensus(venues, Config(outlier_pct=5.0))
        assert result.excluded == ()
        assert result.inputs == 3
        assert result.price == pytest.approx(81707.25, rel=1e-9)

    def test_mad_criterion_does_apply_once_there_are_enough_venues(self) -> None:
        """With 5+ venues the MAD is estimable, so a subtle rogue is still caught."""
        venues = (
            venue("a", 100.00, 100.00),
            venue("b", 100.01, 100.01),
            venue("c", 99.99, 99.99),
            venue("d", 100.00, 100.00),
            venue("e", 140.00, 140.00),
        )
        result = build_consensus(venues, Config(outlier_pct=5.0, min_venues_for_mad=5))
        assert result.excluded == ("e",)

    def test_mad_floor_is_configurable(self) -> None:
        """Lowering the floor re-enables the statistical test at three venues."""
        venues = (
            venue("a", 100.00, 100.00),
            venue("b", 100.01, 100.01),
            venue("c", 140.00, 140.00),
        )
        strict = Config(outlier_pct=50.0, min_venues_for_mad=5)
        assert build_consensus(venues, strict).excluded == ()

        sensitive = Config(outlier_pct=50.0, outlier_mad_z=1.0, min_venues_for_mad=3)
        assert build_consensus(venues, sensitive).excluded == ("c",)


class TestStaleness:
    def test_old_data_is_flagged(self) -> None:
        venues = (
            venue("fresh", 100.0, 100.0, age_seconds=1),
            venue("old", 100.0, 100.0, age_seconds=3600),
        )
        out = apply_consensus(snapshot_with(venues), Config(stale_seconds=120.0), utcnow())
        by_exchange = {v.exchange: v for v in out.venues}
        assert by_exchange["fresh"].is_stale is False
        assert by_exchange["old"].is_stale is True
        assert any("old" in flag for flag in out.data_quality.stale_flags)

    def test_venue_without_a_timestamp_is_not_claimed_stale(self) -> None:
        venues = (venue("nots", 100.0, 100.0, age_seconds=None),)
        out = apply_consensus(snapshot_with(venues), Config(), utcnow())
        assert out.venues[0].is_stale is False
        assert out.data_quality.stale_flags == ()


class TestAggregatorCrossCheck:
    def test_within_threshold_records_nothing(self) -> None:
        venues = (venue("a", 100.0, 100.0), venue("b", 100.0, 100.0), venue("c", 100.0, 100.0))
        consensus = build_consensus(venues, Config())
        assert cross_check(consensus, 100.5, "coingecko", 5.0) == ()

    def test_divergence_is_recorded(self) -> None:
        venues = (venue("a", 100.0, 100.0), venue("b", 100.0, 100.0), venue("c", 100.0, 100.0))
        consensus = build_consensus(venues, Config())
        found = cross_check(consensus, 130.0, "coingecko", 5.0)
        assert len(found) == 1
        assert found[0].kind == "aggregator_vs_consensus"
        assert found[0].deviation_pct == pytest.approx(30.0, rel=1e-6)
        assert found[0].left_source == "coingecko"

    def test_missing_aggregator_price_records_nothing(self) -> None:
        venues = (venue("a", 100.0, 100.0),)
        consensus = build_consensus(venues, Config())
        assert cross_check(consensus, None, "coingecko", 5.0) == ()


class TestApplyConsensus:
    def test_end_to_end_with_injected_outlier(self) -> None:
        venues = (
            venue("kraken", 100.0, 100.0),
            venue("binance", 100.0, 100.0),
            venue("rogue", 5000.0, 5000.0),
        )
        snap = snapshot_with(venues, aggregated_price=100.0)
        out = apply_consensus(snap, Config(outlier_pct=5.0), utcnow())

        assert out.market.consensus_price == pytest.approx(100.0)
        assert out.market.consensus_inputs == 2
        assert out.data_quality.excluded_outliers == ("rogue",)
        assert any(d.kind == "outlier_excluded" for d in out.data_quality.discrepancies)
        rogue = next(v for v in out.venues if v.exchange == "rogue")
        assert rogue.is_outlier is True

    def test_market_price_falls_back_to_aggregator(self) -> None:
        snap = snapshot_with((), aggregated_price=42.0)
        out = apply_consensus(snap, Config(), utcnow())
        assert out.market.price == 42.0
        assert out.market.consensus_price is None

    def test_summarise_reports_the_outcome(self) -> None:
        venues = (
            venue("a", 100.0, 100.0),
            venue("b", 100.0, 100.0),
            venue("rogue", 9000.0, 9000.0),
        )
        out = apply_consensus(snapshot_with(venues), Config(outlier_pct=5.0), utcnow())
        summary = summarise(out)
        assert summary["price"] == pytest.approx(100.0)
        assert summary["excluded_outliers"] == ["rogue"]
        assert str(summary["method"]).startswith("median_outlier_trimmed")

    def test_input_snapshot_is_not_mutated(self) -> None:
        venues = (venue("a", 100.0, 100.0), venue("b", 100.1, 100.1))
        snap = snapshot_with(venues)
        apply_consensus(snap, Config(), utcnow())
        assert snap.market.consensus_price is None


def test_utcnow_is_timezone_aware() -> None:
    now = utcnow()
    assert now.tzinfo is not None
    assert now.utcoffset() == timedelta(0)
    assert now <= datetime.now(UTC) + timedelta(seconds=1)