"""The output layer: formatters, sections and pipeline shapes."""

from __future__ import annotations

import json
from datetime import timedelta

import pytest

from tickerkit.capabilities import Capability
from tickerkit.models import (
    Ambiguity,
    Candle,
    DataQuality,
    DerivativesData,
    Discrepancy,
    Identity,
    Market,
    MissingField,
    OHLCVSeries,
    OnchainData,
    OrderBook,
    Provenance,
    ProviderError,
    SkippedProvider,
    TickerSnapshot,
    Trade,
    VenueTicker,
    utcnow,
)
from tickerkit.render import (
    NA,
    derivatives_table,
    flat_metrics,
    fmt_age,
    fmt_bps,
    fmt_compact,
    fmt_number,
    fmt_pct,
    fmt_price,
    fmt_time,
    onchain_block,
    orderbook_table,
    plural,
    provenance_table,
    quality_block,
    render_markdown,
    render_report,
    summary_block,
    table,
    to_records,
    venues_table,
)


class TestFormatters:
    def test_missing_values_never_render_as_zero(self) -> None:
        for fn in (fmt_price, fmt_number, fmt_compact, fmt_pct, fmt_bps, fmt_age, fmt_time):
            assert fn(None) == NA

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (1_643_365_584_205.0, "1.64T"),
            (42_805_527_096.0, "42.81B"),
            (1_500_000.0, "1.50M"),
            (12_345.0, "12.35K"),
            (999.0, "999.00"),
            (0.0, "0.00"),
            (-2_500_000_000.0, "-2.50B"),
        ],
    )
    def test_fmt_compact(self, value: float, expected: str) -> None:
        assert fmt_compact(value) == expected

    def test_fmt_compact_prefix(self) -> None:
        assert fmt_compact(1_500_000.0, prefix="$") == "$1.50M"

    def test_fmt_price_keeps_small_magnitudes_useful(self) -> None:
        # A naive fixed-point format would render this as 0.00000.
        assert fmt_price(0.000003925) == "3.925e-06"
        assert fmt_price(0.000003925, "USD") == "3.925e-06 USD"

    def test_fmt_pct_sign(self) -> None:
        assert fmt_pct(-1.85) == "-1.85%"
        assert fmt_pct(1.85) == "+1.85%"
        assert fmt_pct(1.85, signed=False) == "1.85%"

    def test_fmt_age_buckets(self) -> None:
        now = utcnow()
        assert fmt_age(now - timedelta(seconds=5), now=now) == "5s"
        assert fmt_age(now - timedelta(minutes=4), now=now) == "4m"
        assert fmt_age(now - timedelta(hours=2), now=now) == "2.0h"
        assert fmt_age(now - timedelta(days=3), now=now) == "3.0d"

    def test_fmt_age_of_none_is_not_stale(self) -> None:
        assert fmt_age(None) == NA

    def test_fmt_time_is_utc_iso(self) -> None:
        stamp = utcnow()
        text = fmt_time(stamp)
        assert text.endswith("+00:00")
        assert len(fmt_time(stamp, short=True)) == 8

    def test_plural(self) -> None:
        assert plural(1, "venue") == "1 venue"
        assert plural(3, "venue") == "3 venues"
        assert plural(0, "trade") == "0 trades"


class TestTable:
    def test_aligns_and_rules(self) -> None:
        out = table(["a", "bbb"], [["1", "2"], ["333", "4"]])
        lines = out.splitlines()
        assert lines[0] == "a    bbb"
        assert lines[1] == "---  ---"
        assert lines[2] == "1    2"
        assert lines[3] == "333  4"

    def test_empty_rows(self) -> None:
        assert table(["a"], []) == "(none)"

    def test_indent_is_applied_to_every_line(self) -> None:
        out = table(["a"], [["1"]], indent="  ")
        assert all(line.startswith("  ") for line in out.splitlines())

    def test_none_values_do_not_break_alignment(self) -> None:
        """A rendered "-" is a real cell, so it participates in column width."""
        out = table(["x", "y"], [["-", "1.0"], ["longvalue", "2.0"]])
        lines = out.splitlines()
        assert lines[0] == "x          y"
        assert lines[2] == "-          1.0"


def sample_snapshot() -> TickerSnapshot:
    now = utcnow()
    prov = Provenance(
        source="coingecko:/coins/markets",
        capability=Capability.AGG_MARKET,
        fetched_at=now,
        exchange_timestamp=now,
        url="https://api.coingecko.com/api/v3/coins/markets",
    )
    venue_prov = Provenance(
        source="ccxt:kraken:fetch_ticker",
        capability=Capability.VENUE_TICKER,
        fetched_at=now,
        exchange_timestamp=now - timedelta(seconds=5),
    )
    return TickerSnapshot(
        identity=Identity(symbol="BTC", name="Bitcoin", coingecko_id="bitcoin"),
        market=Market(
            quote_currency="USD",
            consensus_price=81795.5,
            consensus_method="median(n=2, excluded=1)",
            consensus_inputs=2,
            aggregated_price=81790.0,
            change_24h_pct=-1.85,
            volume_24h=42_805_527_096.0,
            market_cap=1_643_365_584_205.0,
            fdv=1_643_365_584_205.0,
            circulating_supply=20_030_493.0,
            max_supply=21_000_000.0,
            ath=126_080.0,
            ath_date=now - timedelta(days=300),
            atl=67.81,
            atl_date=now - timedelta(days=4000),
            provenance={"aggregated": prov},
        ),
        venues=(
            VenueTicker(exchange="kraken", symbol="BTC/USD", quote_currency="USD",
                        bid=81795.0, ask=81796.0, timestamp=now - timedelta(seconds=3),
                        provenance=venue_prov, deviation_pct=0.001),
            VenueTicker(exchange="rogue", symbol="BTC/USD", quote_currency="USD",
                        bid=9.0, ask=9.1, timestamp=now - timedelta(hours=2),
                        is_outlier=True, is_stale=True, deviation_pct=-99.9),
        ),
        orderbooks=(
            OrderBook(exchange="kraken", symbol="BTC/USD", quote_currency="USD",
                      best_bid=81795.0, best_ask=81796.0, mid=81795.5, spread=1.0,
                      spread_bps=0.12, bid_depth_base=3.0, ask_depth_base=1.0,
                      imbalance=0.5, timestamp=now),
        ),
        trades=(
            Trade(exchange="kraken", symbol="BTC/USD", timestamp=now, price=1.0,
                  amount=0.5, side="buy", trade_id="1"),
        ),
        ohlcv=(
            OHLCVSeries(exchange="kraken", symbol="BTC/USD", timeframe="1h",
                        quote_currency="USD",
                        candles=(Candle(timestamp=now, open=1.0, high=2.0, low=0.5,
                                        close=1.5, volume=10.0),),
                        provenance=prov),
        ),
        derivatives=(
            DerivativesData(exchange="binance", symbol="BTC/USDT:USDT",
                            quote_currency="USD", funding_rate=0.0001,
                            funding_rate_pct=0.01, open_interest_quote=7.6e9,
                            mark_price=81800.0, index_price=None, timestamp=now,
                            provenance=prov),
        ),
        onchain=OnchainData(defillama_slug="bitcoin", name="Bitcoin", tvl=4.3e9,
                            category="Canonical Bridge",
                            chains=(), url=None, provenance=prov),
        data_quality=DataQuality(
            provenance={"market:aggregated": prov, "venue:kraken": venue_prov},
            errors=(ProviderError(provider="rogue", capability=Capability.TRADES,
                                  exception_type="TimeoutError", message="timed out"),),
            ambiguities=(Ambiguity(symbol="BTC", coingecko_id="bitcoin-cash-sv",
                                   name="Bitcoin Cash SV", market_cap=9.4e8),),
            discrepancies=(Discrepancy(kind="outlier_excluded",
                                        detail="rogue deviated", left_source="rogue"),),
            stale_flags=("rogue (age=2.0h)",),
            excluded_outliers=("rogue",),
            missing=(MissingField(field="onchain.tvl", reason="no TVL reported"),),
            providers_used=("coingecko", "ccxt"),
            providers_skipped=(SkippedProvider(provider="coinmarketcap",
                                                reason="no API key"),),
        ),
        query="BTC",
    )


class TestSections:
    def test_summary_contains_the_headline_numbers(self) -> None:
        out = summary_block(sample_snapshot())
        assert "BTC" in out and "Bitcoin" in out
        assert "81,795.5" in out
        assert "1.64T" in out          # market cap compacted
        assert "median(n=2, excluded=1)" in out
        assert "-1.85%" in out
        assert "3 venues" in out or "2 venues" in out

    def test_summary_says_when_there_is_no_price(self) -> None:
        snap = TickerSnapshot(identity=Identity(symbol="ZZZ"),
                              market=Market(quote_currency="USD"),
                              data_quality=DataQuality())
        out = summary_block(snap)
        assert NA in out
        assert "unresolved" in out

    def test_venues_table_flags_outliers_and_staleness(self) -> None:
        out = venues_table(sample_snapshot())
        assert "kraken" in out and "rogue" in out
        assert "OUTLIER,STALE" in out

    def test_derivatives_table_never_invents_index_price(self) -> None:
        out = derivatives_table(sample_snapshot())
        assert "binance" in out
        # index price is genuinely unsupported here and must print as "-"
        assert NA in out

    def test_orderbook_table(self) -> None:
        out = orderbook_table(sample_snapshot())
        assert "kraken" in out
        assert "0.12bps" in out

    def test_onchain_block_handles_absence(self) -> None:
        snap = TickerSnapshot(identity=Identity(symbol="X"),
                              market=Market(quote_currency="USD"),
                              data_quality=DataQuality())
        assert "not applicable" in onchain_block(snap)

    def test_quality_block_reports_everything(self) -> None:
        out = quality_block(sample_snapshot())
        assert "TimeoutError" in out
        assert "coinmarketcap" in out
        assert "rogue" in out
        assert "onchain.tvl" in out
        assert "bitcoin-cash-sv" in out

    def test_quality_block_says_clean_when_clean(self) -> None:
        snap = TickerSnapshot(identity=Identity(symbol="X"),
                              market=Market(quote_currency="USD", consensus_price=1.0),
                              data_quality=DataQuality())
        assert "clean" in quality_block(snap)

    def test_provenance_table_lists_every_group(self) -> None:
        out = provenance_table(sample_snapshot())
        assert "coingecko:/coins/markets" in out
        assert "ccxt:kraken:fetch_ticker" in out
        assert "old" in out  # age summary


class TestComposites:
    def test_render_report_default_sections(self) -> None:
        out = render_report(sample_snapshot())
        for header in ("SUMMARY", "VENUES", "ORDER BOOKS", "DERIVATIVES",
                       "ONCHAIN", "DATA QUALITY", "PROVENANCE"):
            assert header in out

    def test_render_report_section_subset_and_order(self) -> None:
        out = render_report(sample_snapshot(), sections=("quality", "summary"))
        assert out.index("DATA QUALITY") < out.index("SUMMARY")
        assert "VENUES" not in out

    def test_render_report_rejects_unknown_section(self) -> None:
        with pytest.raises(ValueError, match="unknown section"):
            render_report(sample_snapshot(), sections=("nope",))

    def test_markdown_is_fenced(self) -> None:
        out = render_markdown(sample_snapshot())
        assert out.startswith("# BTC snapshot")
        assert "```" in out and out.rstrip().endswith("```")


class TestPipelineShapes:
    def test_to_records_keys_and_row_counts(self) -> None:
        records = to_records(sample_snapshot())
        assert set(records) == {"venues", "orderbooks", "trades", "ohlcv", "derivatives"}
        assert len(records["venues"]) == 2
        assert len(records["trades"]) == 1
        assert len(records["ohlcv"]) == 1

    def test_to_records_rows_are_json_serialisable(self) -> None:
        records = to_records(sample_snapshot())
        json.dumps(records, default=str)  # must not raise

    def test_to_records_carries_provenance(self) -> None:
        records = to_records(sample_snapshot())
        assert records["venues"][0]["source"] == "ccxt:kraken:fetch_ticker"

    def test_flat_metrics_is_one_flat_row(self) -> None:
        row = flat_metrics(sample_snapshot())
        assert row["symbol"] == "BTC"
        assert row["price"] == 81795.5
        assert row["venue_count"] == 2
        assert row["outlier_count"] == 1
        assert row["error_count"] == 1
        assert row["price_source"] == "coingecko:/coins/markets"
        # Every value is a scalar, so it can be a DataFrame cell.
        for key, value in row.items():
            assert not isinstance(value, (list, dict, tuple, set)), key

    def test_flat_metrics_json_serialisable(self) -> None:
        json.dumps(flat_metrics(sample_snapshot()), default=str)

    def test_flat_metrics_uses_none_for_missing(self) -> None:
        snap = TickerSnapshot(identity=Identity(symbol="X"),
                              market=Market(quote_currency="USD"),
                              data_quality=DataQuality())
        row = flat_metrics(snap)
        assert row["price"] is None
        assert row["tvl"] is None
        assert row["mean_funding_rate"] is None
        assert row["error_count"] == 0

    def test_flat_metrics_derived_aggregates(self) -> None:
        row = flat_metrics(sample_snapshot())
        # Venue prices are 81795.5 (mid of kraken) and 9.05 (mid of rogue).
        assert row["min_venue_price"] == pytest.approx(9.05)
        assert row["max_venue_price"] == pytest.approx(81795.5)
        assert row["mean_funding_rate"] == pytest.approx(0.0001)
        assert row["total_open_interest_quote"] == pytest.approx(7.6e9)


class TestEmptySnapshot:
    """Every renderer must survive a completely empty snapshot."""

    def empty(self) -> TickerSnapshot:
        return TickerSnapshot(identity=Identity(symbol="X"),
                              market=Market(quote_currency="USD"),
                              data_quality=DataQuality())

    def test_all_sections_render(self) -> None:
        snap = self.empty()
        assert summary_block(snap)
        assert venues_table(snap) == "(none)"
        assert orderbook_table(snap) == "(none)"
        assert derivatives_table(snap) == "(none)"
        assert onchain_block(snap)
        assert quality_block(snap)
        assert provenance_table(snap) == "(none)"
        assert render_report(snap)
        assert render_markdown(snap)
        assert to_records(snap)["venues"] == []
        assert flat_metrics(snap)["price"] is None