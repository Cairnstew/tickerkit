"""tickerkit — one call, one complete, provenance-tracked crypto snapshot.

    import asyncio, tickerkit
    snap = asyncio.run(tickerkit.aget_ticker("BTC"))
    print(snap.market.price, snap.data_quality.provenance["market:aggregated"].fetched_at)

Only free tiers and open-source libraries are used on the core path. No paid key is
required; optional providers light up when their env var is present.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any, cast

from .cache import Cache, CacheLimits, MemoryCache, NullCache, SQLiteCache
from .capabilities import Capability
from .config import Config
from .core import TickerKit
from .errors import (
    ConfigurationError,
    ProviderError,
    RateLimitExceeded,
    ResolutionError,
    TickerKitError,
    UnsupportedCapability,
)
from .models import (
    Ambiguity,
    BookLevel,
    Candle,
    CandleGap,
    ChainTVL,
    ConsensusCandle,
    DataQuality,
    DepthBand,
    DerivativesData,
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
    OnchainData,
    OrderBook,
    Provenance,
    RateLimitInfo,
    SkippedProvider,
    TickerSnapshot,
    Trade,
    VenueTicker,
)
from .models import (
    ProviderError as ProviderErrorRecord,
)
from .providers import PROVIDERS, register, registry
from .providers.base import Provider, ProviderContext
from .render import (
    flat_metrics,
    history_flat_metrics,
    history_to_records,
    render_history_markdown,
    render_history_report,
    render_markdown,
    render_report,
    to_records,
)
from .resolve import MarketRef, ResolvedAsset
from .store import QueryCache

__version__ = "0.1.0"

__all__ = [
    "PROVIDERS",
    "Ambiguity",
    "BookLevel",
    "Cache",
    "CacheLimits",
    "Candle",
    "CandleGap",
    "Capability",
    "ChainTVL",
    "Config",
    "ConfigurationError",
    "ConsensusCandle",
    "DataQuality",
    "DepthBand",
    "DerivativesData",
    "Discrepancy",
    "HistoryCandle",
    "HistoryConsensus",
    "HistoryCoverage",
    "HistoryRequest",
    "HistoryResult",
    "HistorySeries",
    "Identity",
    "IdentityLinks",
    "Market",
    "MarketRef",
    "MemoryCache",
    "MissingField",
    "NullCache",
    "OHLCVSeries",
    "OnchainData",
    "OrderBook",
    "Provenance",
    "Provider",
    "ProviderContext",
    "ProviderError",
    "ProviderErrorRecord",
    "QueryCache",
    "RateLimitExceeded",
    "RateLimitInfo",
    "ResolutionError",
    "ResolvedAsset",
    "SQLiteCache",
    "SkippedProvider",
    "TickerKit",
    "TickerKitError",
    "TickerSnapshot",
    "Trade",
    "UnsupportedCapability",
    "VenueTicker",
    "__version__",
    "aget_history",
    "aget_ticker",
    "aget_tickers",
    "flat_metrics",
    "get_history",
    "get_ticker",
    "get_tickers",
    "history_flat_metrics",
    "history_to_records",
    "register",
    "registry",
    "render_history_markdown",
    "render_history_report",
    "render_markdown",
    "render_report",
    "to_records",
]


async def aget_ticker(ticker: str, **kwargs: Any) -> TickerSnapshot:
    """Fetch a snapshot for ``ticker`` asynchronously.

    Accepts the same overrides as :meth:`TickerKit.get_ticker`, e.g.
    ``aget_ticker("BTC", coingecko_id="bitcoin", exchanges=["kraken"])``.
    """
    config = kwargs.pop("config", None)
    cache = kwargs.pop("cache", None)
    kit = TickerKit(config=config, cache=cache)
    return await kit.get_ticker(ticker, **kwargs)


def get_ticker(ticker: str, **kwargs: Any) -> TickerSnapshot:
    """Fetch a snapshot for ``ticker``.

    A synchronous convenience wrapper; inside an event loop use :func:`aget_ticker`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aget_ticker(ticker, **kwargs))
    raise ConfigurationError(
        "get_ticker() cannot run inside an event loop; await aget_ticker() instead"
    )


async def aget_tickers(tickers: Sequence[str], **kwargs: Any) -> list[TickerSnapshot]:
    """Fetch snapshots for many tickers concurrently.

    Adding tickers adds concurrency, not request rate: the whole batch shares one
    connection pool, one rate limiter per provider and one circuit breaker per source.
    Results are returned in the order given, and a ticker repeated in the input is
    fetched only once.

        snaps = await aget_tickers(["BTC", "ETH", "SOL", "BTC"], concurrency=4)

    Accepts the same overrides as :func:`aget_ticker`, plus ``concurrency`` and
    ``return_exceptions``.
    """
    config = kwargs.pop("config", None)
    cache = kwargs.pop("cache", None)
    kit = TickerKit(config=config, cache=cache)
    # The module-level helpers promise a plain list of snapshots, so a caller-supplied
    # ``return_exceptions`` would contradict the signature; use ``TickerKit.get_tickers``
    # directly if you want the exceptions handed back instead of raised.
    return cast("list[TickerSnapshot]", await kit.get_tickers(tickers, **kwargs))


def get_tickers(tickers: Sequence[str], **kwargs: Any) -> list[TickerSnapshot]:
    """Fetch snapshots for many tickers.

    A synchronous convenience wrapper; inside an event loop use :func:`aget_tickers`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aget_tickers(tickers, **kwargs))
    raise ConfigurationError(
        "get_tickers() cannot run inside an event loop; await aget_tickers() instead"
    )


async def aget_history(ticker: str, **kwargs: Any) -> HistoryResult:
    """Fetch a deep, cross-referenced history series for ``ticker`` asynchronously.

    Accepts the same overrides as :meth:`TickerKit.get_history`, e.g.
    ``aget_history("BTC", timeframe="1d", start="2017-01-01")``.
    """
    config = kwargs.pop("config", None)
    cache = kwargs.pop("cache", None)
    kit = TickerKit(config=config, cache=cache)
    return await kit.get_history(ticker, **kwargs)


def get_history(ticker: str, **kwargs: Any) -> HistoryResult:
    """Fetch a deep history series for ``ticker``.

    A synchronous convenience wrapper; inside an event loop use :func:`aget_history`.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(aget_history(ticker, **kwargs))
    raise ConfigurationError(
        "get_history() cannot run inside an event loop; await aget_history() instead"
    )