"""Command-line interface.

    tickerkit BTC --json
    tickerkit ETH --summary
    tickerkit BTC --frames ohlcv venues
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from . import __version__, aget_history, aget_ticker
from .cache import SQLiteCache
from .config import Config
from .models import HistoryResult, TickerSnapshot, jsonify
from .render import (
    flat_metrics,
    history_flat_metrics,
    history_summary_block,
    render_history_markdown,
    render_history_report,
    render_markdown,
    render_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tickerkit",
        description="Fetch a complete, provenance-tracked snapshot for a crypto ticker.",
    )
    parser.add_argument("ticker", help="e.g. BTC, ETH, SOL, BTC/USDT")
    parser.add_argument(
        "--history",
        action="store_true",
        help="fetch a deep, cross-referenced history series instead of a snapshot",
    )
    parser.add_argument(
        "--timeframe", default="1d", help="history interval, e.g. 1h, 4h, 1d, 1w (default 1d)"
    )
    parser.add_argument(
        "--start", default=None, help="history window start (ISO-8601 date or epoch seconds)"
    )
    parser.add_argument(
        "--end", default=None, help="history window end (ISO-8601 date or epoch seconds)"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="maximum candles in the history series"
    )
    parser.add_argument("--json", action="store_true", help="print the full JSON snapshot")
    parser.add_argument("--summary", action="store_true", help="print a short digest (default)")
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the full human-readable report (summary, venues, quality, ...)",
    )
    parser.add_argument(
        "--markdown", action="store_true", help="print the report as a fenced markdown block"
    )
    parser.add_argument(
        "--metrics",
        action="store_true",
        help="print one flat JSON row of scalar metrics (for pipelines)",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        metavar="NAME",
        help="print the named pandas DataFrames (venues, trades, ohlcv, orderbook, derivatives)",
    )
    parser.add_argument("--quote", default=None, help="quote currency (default USD)")
    parser.add_argument(
        "--exchanges", default=None, help="comma-separated ccxt ids to query"
    )
    parser.add_argument(
        "--perp-exchanges", default=None, help="comma-separated ccxt ids for derivatives"
    )
    parser.add_argument("--coingecko-id", default=None, help="force a CoinGecko id")
    parser.add_argument("--defillama-slug", default=None, help="force a DefiLlama slug")
    parser.add_argument(
        "--outlier-pct", type=float, default=None, help="outlier threshold, percent"
    )
    parser.add_argument(
        "--stale-seconds", type=float, default=None, help="staleness threshold, seconds"
    )
    parser.add_argument("--no-cache", action="store_true", help="bypass the TTL cache")
    parser.add_argument(
        "--show-errors",
        action="store_true",
        help="print tolerated provider errors and missing-field explanations",
    )
    parser.add_argument("--version", action="version", version=f"tickerkit {__version__}")
    return parser


async def _fetch(args: argparse.Namespace) -> TickerSnapshot:
    config = Config.from_env()
    if args.no_cache:
        config = config.with_overrides(cache_enabled=False)
    return await aget_ticker(args.ticker, config=config, **_overrides(args))


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "quote_currency": args.quote,
        "exchanges": args.exchanges,
        "perp_exchanges": args.perp_exchanges,
        "coingecko_id": args.coingecko_id,
        "defillama_slug": args.defillama_slug,
        "outlier_pct": args.outlier_pct,
        "stale_seconds": args.stale_seconds,
    }


def _print_quality(snapshot: Any) -> None:
    dq = snapshot.data_quality
    if dq.providers_skipped:
        print("\nproviders skipped:", file=sys.stderr)
        for skipped in dq.providers_skipped:
            print(f"  - {skipped.provider}: {skipped.reason}", file=sys.stderr)
    if dq.errors:
        print("\ntolerated errors:", file=sys.stderr)
        for err in dq.errors:
            print(
                f"  - [{err.provider}/{err.capability}] {err.exception_type}: {err.message}",
                file=sys.stderr,
            )
    if dq.missing:
        print("\nmissing data:", file=sys.stderr)
        for item in dq.missing:
            print(f"  - {item.field}: {item.reason}", file=sys.stderr)
    if dq.ambiguities:
        print("\nambiguous symbol — alternatives:", file=sys.stderr)
        for amb in dq.ambiguities[:10]:
            print(
                f"  - {amb.symbol} -> {amb.coingecko_id} ({amb.name})"
                f" mcap={amb.market_cap}",
                file=sys.stderr,
            )
    if dq.discrepancies:
        print("\ndiscrepancies:", file=sys.stderr)
        for disc in dq.discrepancies:
            print(f"  - [{disc.kind}] {disc.detail}", file=sys.stderr)


def _render(frame: Any) -> str:
    """Render a DataFrame for the terminal."""
    return frame.to_string()


def _print_frames(frames: dict[str, Any], requested: list[str] | None) -> None:
    """Print the named DataFrames, or all of them when none were named."""
    if not frames:
        print("no tabular data available (see --show-errors)", file=sys.stderr)
    for name in requested or sorted(frames):
        frame = frames.get(name)
        if frame is None:
            print(f"no frame named {name!r}; available: {sorted(frames)}", file=sys.stderr)
            continue
        print(f"\n== {name} ==")
        print(_render(frame))


async def _run_history(args: argparse.Namespace) -> int:
    config = Config.from_env()
    if args.no_cache:
        config = config.with_overrides(cache_enabled=False)
    result: HistoryResult = await aget_history(
        args.ticker,
        config=config,
        timeframe=args.timeframe,
        start=args.start,
        end=args.end,
        limit=args.limit,
        **_overrides(args),
    )

    if args.frames is not None:
        _print_frames(result.to_dataframes(), args.frames)
    elif args.metrics:
        print(json.dumps(jsonify(history_flat_metrics(result)), indent=2, default=str))
    elif args.report:
        print(render_history_report(result))
    elif args.markdown:
        print(render_history_markdown(result))
    elif args.json:
        print(result.to_json())
    else:
        print(history_summary_block(result))
        consensus = result.consensus
        if consensus is not None and consensus.deviations_pct:
            worst = max(consensus.deviations_pct.items(), key=lambda kv: kv[1])
            print(f"  spread    widest source deviation: {worst[0]} {worst[1]:.3f}%")

    if args.show_errors:
        _print_quality(result)

    # A history request that returned nothing at all is a failure worth a non-zero exit.
    return 0 if result.series else 1


async def _run(args: argparse.Namespace) -> int:
    if args.history:
        return await _run_history(args)

    snapshot = await _fetch(args)

    if args.frames is not None:
        _print_frames(snapshot.to_dataframes(), args.frames)
    elif args.metrics:
        # One flat row per asset: the pipeline-friendly shape.
        print(json.dumps(jsonify(flat_metrics(snapshot)), indent=2, default=str))
    elif args.report:
        print(render_report(snapshot))
    elif args.markdown:
        print(render_markdown(snapshot))
    elif args.json:
        print(snapshot.to_json())
    else:
        print(snapshot.summary())
        price_prov = snapshot.market.provenance.get("aggregated")
        if price_prov is not None:
            print(f"  source    {price_prov.source} @ {price_prov.fetched_at.isoformat()}")

    if args.show_errors:
        _print_quality(snapshot)

    # A snapshot with no price at all is a failure worth a non-zero exit.
    return 0 if snapshot.market.price is not None else 1


def _build_cache_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tickerkit cache",
        description="Inspect and maintain the disk (SQLite) cache.",
    )
    parser.add_argument(
        "action",
        nargs="?",
        default="stats",
        choices=["stats", "prune", "clear", "vacuum"],
        help="what to do (default: stats)",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="cache file (default: TICKERKIT_CACHE_PATH env or tickerkit_cache.sqlite3)",
    )
    return parser


def _run_cache(argv: list[str]) -> int:
    args = _build_cache_parser().parse_args(argv)
    config = Config.from_env()
    path = args.path or config.cache_path or "tickerkit_cache.sqlite3"
    cache = SQLiteCache(path=path)
    try:
        if args.action == "clear":
            asyncio.run(cache.clear())
            print(f"cleared {path}")
        elif args.action == "vacuum":
            cache.vacuum()
            print(f"vacuumed {path}")
        else:
            if args.action == "prune":
                cache.prune()
            s = cache.stats()
            print(
                f"{s['entries']} entries, {s['bytes'] / 1024:.1f} KiB "
                f"(max {s['max_entries']} / {s['max_bytes'] / (1024 * 1024):.1f} MiB) "
                f"at {path}"
            )
    finally:
        cache.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["cache"]:
        return _run_cache(argv[1:])
    args = build_parser().parse_args(argv)
    try:
        return asyncio.run(_run(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
