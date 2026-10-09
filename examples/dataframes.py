#!/usr/bin/env python3
"""Notebook-style walkthrough of the pandas views.

Run as a script, or paste the cells into a notebook:

    uv run python examples/dataframes.py ETH

Every DataFrame carries `source` and `fetched_at`, so any number you take from these
frames can be traced back to the provider and moment it came from.
"""

from __future__ import annotations

import asyncio

import pandas as pd

import tickerkit


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


async def main() -> None:
    # -- cell: fetch ------------------------------------------------------------------
    section("1. Fetch a snapshot")
    snap = await tickerkit.aget_ticker("ETH")
    print(snap.summary())
    print(f"\nproviders used   : {snap.data_quality.providers_used}")
    skipped = [(s.provider, s.reason) for s in snap.data_quality.providers_skipped]
    print(f"providers skipped: {skipped}")

    frames = snap.to_dataframes()
    print(f"\navailable frames : {sorted(frames)}")

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    # -- cell: venues -----------------------------------------------------------------
    section("2. venues — cross-venue tickers with provenance")
    venues = frames["venues"]
    print(venues[["exchange", "symbol", "bid", "ask", "mid", "spread_bps",
                  "deviation_pct", "is_outlier", "is_stale", "source"]].to_string(index=False))
    print(f"\nconsensus: {snap.market.consensus_price} ({snap.market.consensus_method})")

    # -- cell: ohlcv ------------------------------------------------------------------
    section("3. ohlcv — UTC-indexed candles, multiple timeframes")
    ohlcv = frames["ohlcv"]
    print("timeframes:", sorted(ohlcv["timeframe"].unique()))
    print("\ndaily tail:")
    print(ohlcv[ohlcv["timeframe"] == "1d"].tail(5).to_string())
    print(f"\nindex tz: {ohlcv.index.tz}  (all timestamps are timezone-aware UTC)")

    # A quick derived view: realised spread and returns from the same frame.
    hourly = ohlcv[ohlcv["timeframe"] == "1h"].copy()
    if not hourly.empty:
        hourly["range_pct"] = (hourly["high"] - hourly["low"]) / hourly["low"] * 100
        hourly["ret_pct"] = hourly["close"].pct_change() * 100
        print("\nderived columns:")
        print(hourly[["exchange", "close", "range_pct", "ret_pct"]].tail(5).to_string())

    # -- cell: order book -------------------------------------------------------------
    section("4. orderbook — spread and depth bands")
    if "orderbook" in frames:
        print(frames["orderbook"][["exchange", "best_bid", "best_ask", "mid",
                                   "spread_bps", "bid_depth_base", "ask_depth_base",
                                   "imbalance"]].to_string(index=False))
    if "orderbook_depth" in frames:
        print("\ndepth bands:")
        print(
            "(bands can look identical on liquid pairs: with 20 levels requested, "
            "every level already sits within 0.5% of the mid)"
        )
        print(frames["orderbook_depth"][["exchange", "within_pct", "bid_base",
                                         "ask_base", "bid_quote", "ask_quote"]]
              .to_string(index=False))

    # -- cell: trades -----------------------------------------------------------------
    section("5. trades — most recent first")
    if "trades" in frames:
        trades = frames["trades"]
        print(trades.head(8).to_string(index=False))
        print("\nper-venue notional:")
        notional = trades.assign(quote=trades["price"] * trades["amount"])
        print(notional.groupby("exchange")["quote"]
              .agg(["count", "sum"]).rename(columns={"sum": "quote_volume"}).to_string())

    # -- cell: derivatives ------------------------------------------------------------
    section("6. derivatives — funding, open interest, mark/index")
    if "derivatives" in frames:
        print(frames["derivatives"].to_string(index=False))

    # -- cell: traceability -----------------------------------------------------------
    section("7. traceability — where did a number come from?")
    prov = snap.market.provenance["aggregated"]
    print(f"market.aggregated_price = {snap.market.aggregated_price}")
    print(f"  source              : {prov.source}")
    print(f"  fetched_at (UTC)    : {prov.fetched_at.isoformat()}")
    print(f"  exchange_timestamp  : {prov.exchange_timestamp}")
    print(f"  url                 : {prov.url}")

    section("8. reconciliation")
    from tickerkit.reconcile import summarise

    print(summarise(snap))

    section("9. missing data is explained, never zeroed")
    if snap.data_quality.missing:
        for item in snap.data_quality.missing[:10]:
            print(f"  {item.field}: {item.reason}")
    else:
        print("  (nothing missing)")


if __name__ == "__main__":
    asyncio.run(main())