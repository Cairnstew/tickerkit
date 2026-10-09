#!/usr/bin/env python3
"""Deep history with cross-referencing, as a pipeline.

    uv run python examples/history.py BTC --timeframe 1d --start 2026-08-01 --end 2026-10-01
    uv run python examples/history.py BTC --timeframe 1d \
      --start 2017-01-01 --end 2024-01-01 --csv > btc.csv

Walks the same resolution-first flow as the snapshot pipeline, then cross-references the
exchanges against the aggregators. What makes this pipeline different is accountability:
every source says how much of the window it actually covered (``coverage``), sources that
agree merge into a per-interval median (``consensus``), and close-only aggregators audit
that consensus with divergence reported, never averaged.

Missing measurements stay ``None`` in JSON — never ``0``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import pandas as pd

import tickerkit
from tickerkit.render import history_flat_metrics, render_history_report


async def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="history.py", description="tickerkit deep-history pipeline."
    )
    parser.add_argument("ticker", help="e.g. BTC, ETH")
    parser.add_argument("--timeframe", default="1d", help="1h, 4h, 1d, 1w (default 1d)")
    parser.add_argument("--start", default=None, help="window start (ISO date or epoch)")
    parser.add_argument("--end", default=None, help="window end (ISO date or epoch)")
    parser.add_argument("--limit", type=int, default=None, help="max candles per source")
    parser.add_argument("--json", action="store_true", help="print the full JSON result")
    parser.add_argument("--csv", action="store_true", help="emit the consensus as CSV")
    parser.add_argument("--report", action="store_true", help="print the human report")
    args = parser.parse_args(argv)

    result = await tickerkit.aget_history(
        args.ticker,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        limit=args.limit,
    )

    if args.csv:
        consensus = result.consensus
        assert consensus is not None, "no consensus series to emit"
        pd.DataFrame(
            [
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                    "volume": c.volume,
                    "sources": ",".join(c.sources),
                }
                for c in consensus.candles
            ]
        ).to_csv(sys.stdout, index=False)
        return 0

    if args.json:
        print(result.to_json())
    elif args.report:
        print(render_history_report(result))
    else:
        print(tickerkit.render_history_report(result, sections=("summary", "sources")))

    print(json.dumps(json.loads(json.dumps(history_flat_metrics(result), default=str)), indent=2))
    return 0 if result.series else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))