"""Provider capabilities.

Each provider declares which capabilities it implements. The orchestrator runs one
concurrent task per (provider, capability) pair that is actually available, so adding a
provider never requires touching the orchestrator.
"""

from __future__ import annotations

from enum import StrEnum


class Capability(StrEnum):
    """A unit of data a provider may be able to supply."""

    RESOLVE = "resolve"
    """Map a user ticker / coin name onto this provider's own id space."""

    META = "meta"
    """Static-ish identity: name, categories, description, links."""

    AGG_MARKET = "agg_market"
    """Aggregated cross-venue market data: price, cap, supply, ATH/ATL."""

    VENUE_TICKER = "venue_ticker"
    """Per-exchange best bid/ask/last for a market."""

    ORDERBOOK = "orderbook"
    """Per-exchange top-of-book / L2 depth."""

    TRADES = "trades"
    """Recent public trades."""

    OHLCV = "ohlcv"
    """The most recent candles, a shallow tail used to decorate a snapshot."""

    HISTORY = "history"
    """A deep, windowed candlestick/price series over a caller-supplied date range.

    Distinct from :attr:`OHLCV`: ``OHLCV`` is a fixed-size recent tail fetched for every
    snapshot, whereas ``HISTORY`` is a paginated walk over an arbitrary window and is
    fetched only by :meth:`tickerkit.TickerKit.get_history`.
    """

    DERIVATIVES = "derivatives"
    """Perpetual/future funding rate, open interest, mark & index price."""

    ONCHAIN = "onchain"
    """TVL / protocol / chain metrics."""


#: Capabilities whose absence never invalidates a snapshot, only annotates it.
OPTIONAL_CAPABILITIES: frozenset[Capability] = frozenset(
    {
        Capability.ORDERBOOK,
        Capability.TRADES,
        Capability.OHLCV,
        Capability.HISTORY,
        Capability.DERIVATIVES,
        Capability.ONCHAIN,
    }
)