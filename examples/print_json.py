#!/usr/bin/env python3
"""Print a full tickerkit snapshot as JSON.

    uv run python examples/print_json.py BTC
    uv run python examples/print_json.py ETH --exchanges kraken --indent 0
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

import tickerkit


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ticker", nargs="?", default="BTC")
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--exchanges", default=None, help="comma-separated ccxt ids")
    parser.add_argument("--quote", default=None)
    args = parser.parse_args()

    snap = await tickerkit.aget_ticker(
        args.ticker,
        exchanges=args.exchanges,
        quote_currency=args.quote,
    )

    print(snap.to_json(indent=args.indent if args.indent >= 0 else None))

    # The snapshot is a plain dict, so anything downstream can consume it directly.
    payload = json.loads(snap.to_json())
    assert payload["identity"]["symbol"] == snap.identity.symbol

    if snap.data_quality.errors:
        print(
            f"\n{len(snap.data_quality.errors)} provider error(s) were tolerated; "
            "see data_quality.errors",
            file=sys.stderr,
        )
        return 0 if snap.market.price is not None else 1
    return 0 if snap.market.price is not None else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))