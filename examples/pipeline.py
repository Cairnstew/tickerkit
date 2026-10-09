#!/usr/bin/env python3
"""Pipeline output: one flat row per asset, ready for a DataFrame or a CSV.

    uv run python examples/pipeline.py BTC ETH SOL
    uv run python examples/pipeline.py BTC ETH --csv > prices.csv

Shows the two shapes built for machines rather than humans:

* :func:`tickerkit.flat_metrics` — one flat ``dict`` of scalars per asset. Append a
  snapshot per ticker and you have a tidy DataFrame; missing measurements are ``None``
  (``null`` in JSON), never ``0``.
* :func:`tickerkit.to_records` — normalised row lists for the per-venue / per-trade
  tables, with provenance columns attached, so any number stays attributable.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
from typing import Any

import tickerkit
from tickerkit.models import TickerSnapshot
from tickerkit.render import flat_metrics, render_report, to_records


async def collect(
    tickers: list[str], delay: float, **kwargs: Any
) -> list[TickerSnapshot]:
    """One snapshot per ticker, spaced out to respect free-tier rate limits."""
    snaps: list[TickerSnapshot] = []
    for i, ticker in enumerate(tickers):
        if i:
            # Keyless CoinGecko is IP-rate-limited; spacing keeps us inside it.
            await asyncio.sleep(delay)
        snaps.append(await tickerkit.aget_ticker(ticker, **kwargs))
    return snaps


def to_csv(rows: list[dict[str, object]]) -> str:
    if not rows:
        return ""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: "" if row.get(k) is None else row.get(k) for k in columns})
    return buffer.getvalue()


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tickers", nargs="*", default=["BTC", "ETH", "SOL"])
    parser.add_argument("--csv", action="store_true", help="emit CSV instead of JSON")
    parser.add_argument("--delay", type=float, default=6.0, help="seconds between tickers")
    parser.add_argument("--report", metavar="TICKER", help="also print a full report")
    parser.add_argument("--venues", action="store_true", help="print per-venue records")
    args = parser.parse_args()

    snaps = await collect(args.tickers, args.delay)
    rows = [flat_metrics(s) for s in snaps]

    if args.csv:
        print(to_csv(rows), end="")
    else:
        print(json.dumps(rows, indent=2, default=str))

    if args.venues and snaps:
        records = to_records(snaps[0])
        print("\nper-venue records for the first ticker:", file=sys.stderr)
        print(json.dumps(records["venues"], indent=2, default=str), file=sys.stderr)

    if args.report:
        by_symbol = {s.identity.symbol: s for s in snaps}
        snap = by_symbol.get(args.report.upper())
        if snap is None:
            print(f"no snapshot for {args.report}", file=sys.stderr)
            return 1
        print(render_report(snap))

    # Degraded snapshots are surfaced rather than hidden: a row with errors still returns.
    degraded = [r for r in rows if r["error_count"]]
    if degraded:
        print(
            f"\n{len(degraded)} of {len(rows)} snapshots had tolerated provider errors; "
            "see error_count and data_quality.errors",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))