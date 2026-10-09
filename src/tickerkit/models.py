"""The canonical, frozen snapshot schema.

Design rules enforced here:

* Every field group carries a :class:`Provenance` (source + fetched_at + the provider's own
  timestamp when it exposes one), so any number in the output is traceable.
* Missing data is ``None`` and is explained in ``DataQuality`` — never a silent ``0`` or
  ``0.0``.
* Timestamps are timezone-aware UTC :class:`datetime` throughout.
* Prices/vols are IEEE-754 doubles (see README "Precision"). Every numeric group also
  names its ``quote_currency`` explicitly.
"""

from __future__ import annotations

import json
import math
import types
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Union, get_args, get_origin, get_type_hints

from .capabilities import Capability

if TYPE_CHECKING:
    import pandas as pd


# --------------------------------------------------------------------------- helpers


def utcnow() -> datetime:
    """Timezone-aware current UTC time."""
    return datetime.now(UTC)


def as_utc(value: datetime) -> datetime:
    """Coerce a timestamp to timezone-aware UTC.

    Naive datetimes are *assumed* UTC, which is what every provider in this project
    documents; aware datetimes are converted.
    """
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def to_iso(value: datetime | None) -> str | None:
    """ISO-8601 rendering of an optional timestamp."""
    return as_utc(value).isoformat() if value is not None else None


def finite(value: object) -> float | None:
    """Coerce to ``float`` or ``None``, rejecting NaN/inf so it never reaches JSON."""
    if value is None or isinstance(value, bool):
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def jsonify(value: Any) -> Any:
    """Recursively convert a schema object into JSON-serialisable primitives."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, datetime):
        return to_iso(value)
    if is_dataclass(value) and not isinstance(value, type):
        # A class may define _json_hook to shape its own output (e.g. adding a derived
        # property). Classes without one are expanded field-by-field. TickerSnapshot
        # deliberately has no hook, so jsonify(self) cannot recurse through it.
        hook = getattr(value, "_json_hook", None)
        if callable(hook):
            return hook()
        return {f.name: jsonify(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): jsonify(v) for k, v in value.items()}  # type: ignore[var-annotated]
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonify(v) for v in value]  # type: ignore[var-annotated]
    return str(value)


def from_jsonified(value: Any, hint: Any) -> Any:
    """Rebuild a schema object from :func:`jsonify` output — its exact inverse.

    Generic over the frozen dataclasses in this module: datetimes parse back from ISO,
    tuples/sets from lists, ``StrEnum`` members by value, ``Optional[X]`` unwraps to
    ``X``, unknown dict keys are ignored (e.g. the derived ``mid`` a ``_json_hook``
    adds), and missing keys simply fall back to the field's dataclass default.

    ``hint`` must be a *resolved* type (``get_type_hints`` output), not a string.
    """
    if value is None:
        return None
    if hint is Any or hint is object:
        return value
    hint = _unwrap_optional(hint)
    if hint is Any or hint is object:
        return value
    if hint in (str, bool, int, float, bytes):
        return value
    if hint is datetime:
        return value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    if isinstance(hint, type) and issubclass(hint, Enum):
        # StrEnum/IntEnum: serialize as the value, revive by value.
        try:
            return hint(value)
        except ValueError:
            return value
    origin = get_origin(hint)
    args = get_args(hint)
    if origin in (list, tuple, set, frozenset):
        element = args[0] if args and args[0] is not ... else Any
        items = (from_jsonified(v, element) for v in value)
        if origin is list:
            return list(items)
        if origin is tuple:
            return tuple(items)
        if origin is set:
            return set(items)
        return frozenset(items)
    if origin is dict:
        value_hint = args[1] if len(args) > 1 else Any
        return {str(k): from_jsonified(v, value_hint) for k, v in value.items()}
    if isinstance(hint, type) and is_dataclass(hint):
        hints = get_type_hints(hint)
        kwargs = {
            f.name: from_jsonified(value[f.name], hints[f.name])
            for f in fields(hint)
            if f.name in value
        }
        return hint(**kwargs)
    return value


def _unwrap_optional(hint: Any) -> Any:
    """Pull the non-``None`` branch out of ``X | None`` (or ``Optional[X]``)."""
    origin = get_origin(hint)
    if origin is not types.UnionType and origin is not Union:
        return hint
    args = [a for a in get_args(hint) if a is not type(None)]
    return args[0] if len(args) == 1 else hint


# --------------------------------------------------------------------------- provenance


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a field group came from and how old it is."""

    source: str
    """Human-readable origin, e.g. ``ccxt:binance``, ``coingecko:/coins/markets``."""

    capability: Capability
    fetched_at: datetime
    """When tickerkit retrieved it (UTC, aware)."""

    exchange_timestamp: datetime | None = None
    """The provider's own timestamp, when it reports one."""

    url: str | None = None
    cached: bool = False
    notes: tuple[str, ...] = ()



# --------------------------------------------------------------------------- identity


@dataclass(frozen=True, slots=True)
class IdentityLinks:
    homepage: tuple[str, ...] = ()
    blockchain: tuple[str, ...] = ()
    whitepaper: tuple[str, ...] = ()
    twitter: str | None = None
    reddit: str | None = None
    github: tuple[str, ...] = ()
    telegram: str | None = None
    other: tuple[str, ...] = ()



@dataclass(frozen=True, slots=True)
class Identity:
    symbol: str
    name: str | None = None
    coingecko_id: str | None = None
    categories: tuple[str, ...] = ()
    links: IdentityLinks = field(default_factory=IdentityLinks)
    description: str | None = None
    contract_address: str | None = None
    asset_platform_id: str | None = None
    provenance: Provenance | None = None



# --------------------------------------------------------------------------- market


@dataclass(frozen=True, slots=True)
class Market:
    """Aggregated market view. Consensus price is venue-derived; see :mod:`tickerkit.reconcile`."""

    quote_currency: str
    consensus_price: float | None = None
    consensus_method: str | None = None
    consensus_inputs: int = 0

    aggregated_price: float | None = None
    """Price from an aggregator (CoinGecko/CMC), kept separate for reconciliation."""

    change_24h: float | None = None
    change_24h_pct: float | None = None
    high_24h: float | None = None
    low_24h: float | None = None
    volume_24h: float | None = None

    market_cap: float | None = None
    fdv: float | None = None
    circulating_supply: float | None = None
    total_supply: float | None = None
    max_supply: float | None = None

    ath: float | None = None
    ath_date: datetime | None = None
    ath_change_pct: float | None = None
    atl: float | None = None
    atl_date: datetime | None = None
    atl_change_pct: float | None = None

    provenance: dict[str, Provenance] = field(default_factory=dict)

    @property
    def price(self) -> float | None:
        """Best available price: venue consensus, else aggregator. Never fabricated."""
        return self.consensus_price if self.consensus_price is not None else self.aggregated_price



# --------------------------------------------------------------------------- venues


@dataclass(frozen=True, slots=True)
class VenueTicker:
    exchange: str
    symbol: str
    quote_currency: str
    bid: float | None = None
    ask: float | None = None
    last: float | None = None
    spread: float | None = None
    spread_bps: float | None = None
    volume_24h_base: float | None = None
    volume_24h_quote: float | None = None
    timestamp: datetime | None = None
    is_stale: bool = False
    is_outlier: bool = False
    deviation_pct: float | None = None
    deviation_bps: float | None = None
    provenance: Provenance | None = None

    @property
    def mid(self) -> float | None:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return None

    def _json_hook(self) -> dict[str, Any]:
        """Include the derived ``mid``, which is not a dataclass field."""
        out: dict[str, Any] = {
            f.name: jsonify(getattr(self, f.name)) for f in fields(self)
        }
        out["mid"] = self.mid
        return out


# --------------------------------------------------------------------------- order book


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: float
    amount: float


@dataclass(frozen=True, slots=True)
class OrderBook:
    exchange: str
    symbol: str
    quote_currency: str
    bids: tuple[BookLevel, ...] = ()
    asks: tuple[BookLevel, ...] = ()

    best_bid: float | None = None
    best_ask: float | None = None
    mid: float | None = None
    spread: float | None = None
    spread_bps: float | None = None

    bid_depth_base: float | None = None
    ask_depth_base: float | None = None
    depth_quote_within: tuple[DepthBand, ...] = ()
    imbalance: float | None = None
    """(bid - ask) / (bid + ask) over the requested levels; +1 = all bid."""

    timestamp: datetime | None = None
    provenance: Provenance | None = None



@dataclass(frozen=True, slots=True)
class DepthBand:
    """Cumulative two-sided depth within a percentage band of the mid."""

    pct: float
    bid_base: float
    ask_base: float
    bid_quote: float
    ask_quote: float



# --------------------------------------------------------------------------- trades / ohlcv


@dataclass(frozen=True, slots=True)
class Trade:
    exchange: str
    symbol: str
    timestamp: datetime
    price: float
    amount: float
    side: str | None = None
    trade_id: str | None = None



@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True, slots=True)
class OHLCVSeries:
    exchange: str
    symbol: str
    timeframe: str
    quote_currency: str
    candles: tuple[Candle, ...] = ()
    provenance: Provenance | None = None



# --------------------------------------------------------------------------- history


@dataclass(frozen=True, slots=True)
class HistoryRequest:
    """What the caller asked for. Echoed on the result so coverage can be judged.

    ``start``/``end`` are inclusive bounds on candle timestamps; ``limit`` caps the number
    of candles. Any of the three may be ``None``, meaning "as far back / as recent / as
    many as the source will give".
    """

    timeframe: str = "1d"
    start: datetime | None = None
    end: datetime | None = None
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class HistoryCandle:
    """One point of a history series.

    ``close`` is always present. ``open``/``high``/``low``/``volume`` are ``None`` when the
    source only publishes a single price per interval (CoinGecko's chart, DefiLlama). They
    are never back-filled from ``close``: collapsing all four onto one number would make a
    sampled price indistinguishable from a real OHLC bar, which is exactly the kind of
    fabrication this project refuses.
    """

    timestamp: datetime
    close: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None

    @property
    def is_ohlcv(self) -> bool:
        """True when the source supplied a genuine OHLC bar, not just a price sample."""
        return self.open is not None and self.high is not None and self.low is not None


@dataclass(frozen=True, slots=True)
class CandleGap:
    """A missing stretch strictly inside a series' covered range.

    ``missing`` is the number of expected intervals absent between ``start`` and ``end``
    (exclusive of both), so a gap of one missing candle reads ``start`` .. ``end`` two
    intervals apart with ``missing=1``.
    """

    start: datetime
    end: datetime
    missing: int


@dataclass(frozen=True, slots=True)
class HistoryCoverage:
    """How complete a series is, stated rather than left for the caller to guess.

    This is the answer to "did I actually get everything I asked for?": the requested
    window, the window actually covered, the interval count, any interior gaps, and
    whether a provider cap cut the walk short.
    """

    requested_start: datetime | None = None
    requested_end: datetime | None = None
    first: datetime | None = None
    last: datetime | None = None
    candles: int = 0
    #: Intervals expected across the requested window, when it is bounded.
    expected: int | None = None
    #: ``expected - candles`` when ``expected`` is known, else ``None``.
    missing: int | None = None
    gaps: tuple[CandleGap, ...] = ()
    #: The walk stopped at a provider/pagination cap before reaching the requested start.
    truncated: bool = False
    #: Upstream requests the series cost.
    calls: int = 0
    #: True only when the requested window is fully covered with no gaps or truncation.
    complete: bool = False
    note: str | None = None


@dataclass(frozen=True, slots=True)
class HistorySeries:
    """One source's answer for one ``(asset, timeframe)`` window."""

    source: str
    """Origin, e.g. ``ccxt:binance`` or ``coingecko:market_chart/range``."""

    symbol: str
    timeframe: str
    quote_currency: str
    kind: str = "ohlcv"
    """``"ohlcv"`` when bars carry open/high/low, ``"price"`` when only a close exists."""

    candles: tuple[HistoryCandle, ...] = ()
    coverage: HistoryCoverage | None = None
    provenance: Provenance | None = None


@dataclass(frozen=True, slots=True)
class ConsensusCandle:
    """A single interval of the merged series, with the sources that agreed on it."""

    timestamp: datetime
    close: float
    open: float | None = None
    high: float | None = None
    low: float | None = None
    volume: float | None = None
    sources: tuple[str, ...] = ()
    n: int = 0
    #: (max close - min close) / median close across sources, in percent; ``None`` for n=1.
    spread_pct: float | None = None


@dataclass(frozen=True, slots=True)
class HistoryConsensus:
    """The cross-referenced series: a per-interval median over agreeing venues."""

    timeframe: str
    quote_currency: str
    method: str
    inputs: tuple[str, ...] = ()
    candles: tuple[ConsensusCandle, ...] = ()
    #: Per-source mean absolute percent deviation from the consensus close.
    deviations_pct: dict[str, float] = field(default_factory=dict)
    coverage: HistoryCoverage | None = None


@dataclass(frozen=True, slots=True)
class HistoryResult:
    """The history analogue of :class:`TickerSnapshot`."""

    identity: Identity
    request: HistoryRequest
    data_quality: DataQuality
    series: tuple[HistorySeries, ...] = ()
    consensus: HistoryConsensus | None = None
    generated_at: datetime = field(default_factory=utcnow)
    query: str = ""
    schema_version: str = "1.0"

    @property
    def ohlcv_series(self) -> tuple[HistorySeries, ...]:
        """Series carrying genuine OHLC bars (the consensus inputs)."""
        return tuple(s for s in self.series if s.kind == "ohlcv")

    @property
    def reference_series(self) -> tuple[HistorySeries, ...]:
        """Close-only series, used to cross-check rather than to build the consensus."""
        return tuple(s for s in self.series if s.kind != "ohlcv")

    def available_range(self) -> tuple[datetime | None, datetime | None]:
        """Earliest and latest candle across every series, for a coverage headline."""
        firsts = [s.coverage.first for s in self.series if s.coverage and s.coverage.first]
        lasts = [s.coverage.last for s in self.series if s.coverage and s.coverage.last]
        return (min(firsts) if firsts else None, max(lasts) if lasts else None)

    def to_dict(self) -> dict[str, Any]:
        """Plain dict, JSON-serialisable."""
        return jsonify(self)

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = False) -> str:
        """JSON-serialise the whole result."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HistoryResult:
        """Rebuild a result from :meth:`to_dict` output (the inverse of it).

        Round-trips every field, including datetimes, coverage objects, enums and
        nested series. Unknown keys (e.g. derived ``mid``) are ignored. Raises
        ``TypeError``/``ValueError`` if the dict is not a result-shaped dict.
        """
        return from_jsonified(data, cls)

    @classmethod
    def from_json(cls, text: str) -> HistoryResult:
        """Parse JSON produced by :meth:`to_json` back into a result object."""
        return cls.from_dict(json.loads(text))

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Pandas views: ``history`` (long form, every source) and ``consensus``."""
        import pandas as pd

        frames: dict[str, pd.DataFrame] = {}
        rows: list[dict[str, Any]] = []
        for series in self.series:
            for candle in series.candles:
                rows.append(
                    {
                        "source": series.source,
                        "symbol": series.symbol,
                        "timeframe": series.timeframe,
                        "kind": series.kind,
                        "timestamp": candle.timestamp,
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                    }
                )
        if rows:
            frames["history"] = pd.DataFrame(rows).set_index("timestamp")  # type: ignore[reportUnknownMemberType]

        if self.consensus is not None and self.consensus.candles:
            frames["consensus"] = pd.DataFrame(
                [
                    {
                        "timestamp": c.timestamp,
                        "open": c.open,
                        "high": c.high,
                        "low": c.low,
                        "close": c.close,
                        "volume": c.volume,
                        "sources": ",".join(c.sources),
                        "n": c.n,
                        "spread_pct": c.spread_pct,
                    }
                    for c in self.consensus.candles
                ]
            ).set_index("timestamp")  # type: ignore[reportUnknownMemberType]
        return frames


# --------------------------------------------------------------------------- derivatives


@dataclass(frozen=True, slots=True)
class DerivativesData:
    exchange: str
    symbol: str
    quote_currency: str
    contract_type: str = "perpetual"
    funding_rate: float | None = None
    """Fractional rate (0.0001 == 0.01%), as reported. Documented in the README."""

    funding_rate_pct: float | None = None
    funding_interval_hours: float | None = None
    open_interest_contracts: float | None = None
    open_interest_base: float | None = None
    open_interest_quote: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    basis: float | None = None
    """mark/consensus - 1, annualisation not attempted."""

    timestamp: datetime | None = None
    provenance: Provenance | None = None



# --------------------------------------------------------------------------- onchain


@dataclass(frozen=True, slots=True)
class ChainTVL:
    chain: str
    tvl: float | None = None


@dataclass(frozen=True, slots=True)
class OnchainData:
    defillama_slug: str
    name: str | None = None
    category: str | None = None
    tvl: float | None = None
    chains: tuple[ChainTVL, ...] = ()
    change_1d: float | None = None
    change_7d: float | None = None
    url: str | None = None
    provenance: Provenance | None = None



# --------------------------------------------------------------------------- data quality


@dataclass(frozen=True, slots=True)
class ProviderError:
    """A single tolerated failure. One of these must never sink the snapshot."""

    provider: str
    capability: Capability
    exception_type: str
    message: str
    detail: str | None = None
    occurred_at: datetime = field(default_factory=utcnow)



@dataclass(frozen=True, slots=True)
class Ambiguity:
    """An alternative asset that also matched the requested ticker."""

    symbol: str
    coingecko_id: str
    name: str | None = None
    market_cap: float | None = None



@dataclass(frozen=True, slots=True)
class Discrepancy:
    """A recorded disagreement between two sources or a consensus member."""

    kind: str
    detail: str
    left_source: str
    left_value: float | None = None
    right_source: str | None = None
    right_value: float | None = None
    deviation_pct: float | None = None



@dataclass(frozen=True, slots=True)
class SkippedProvider:
    provider: str
    reason: str



@dataclass(frozen=True, slots=True)
class MissingField:
    """Explains a ``None`` that would otherwise look like a real zero."""

    field: str
    reason: str



@dataclass(frozen=True, slots=True)
class RateLimitInfo:
    """A source that throttled us, described well enough to act on.

    ``remedy`` is deliberately human-readable: the snapshot should tell the caller what
    to change (add a key, slow down, use another source), not merely that something
    failed. ``suppressed`` marks a call that the circuit breaker refused to send, so a
    reader can tell "we were throttled" apart from "we spent retries being told no".
    """

    source: str
    capability: Capability
    message: str
    retry_after: float | None = None
    limit: int | None = None
    remaining: int | None = None
    remedy: str | None = None
    suppressed: bool = False
    occurred_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True, slots=True)
class DataQuality:
    provenance: dict[str, Provenance] = field(default_factory=dict)
    errors: tuple[ProviderError, ...] = ()
    #: Throttling specifically, split out from `errors` so a pipeline can react to a
    #: quota problem without string-matching exception names.
    rate_limited: tuple[RateLimitInfo, ...] = ()
    ambiguities: tuple[Ambiguity, ...] = ()
    discrepancies: tuple[Discrepancy, ...] = ()
    stale_flags: tuple[str, ...] = ()
    excluded_outliers: tuple[str, ...] = ()
    missing: tuple[MissingField, ...] = ()
    providers_used: tuple[str, ...] = ()
    providers_skipped: tuple[SkippedProvider, ...] = ()

    @property
    def ok(self) -> bool:
        """True when no provider failed for this snapshot."""
        return not self.errors

    @property
    def throttled(self) -> bool:
        """True when any source hit a rate limit."""
        return bool(self.rate_limited)

    def _json_hook(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            f.name: jsonify(getattr(self, f.name)) for f in fields(self)
        }
        out["ok"] = self.ok
        return out



# --------------------------------------------------------------------------- snapshot


@dataclass(frozen=True, slots=True)
class TickerSnapshot:
    """Everything tickerkit could obtain, plus a full account of how it got it."""

    identity: Identity
    market: Market
    data_quality: DataQuality
    venues: tuple[VenueTicker, ...] = ()
    orderbooks: tuple[OrderBook, ...] = ()
    trades: tuple[Trade, ...] = ()
    ohlcv: tuple[OHLCVSeries, ...] = ()
    derivatives: tuple[DerivativesData, ...] = ()
    onchain: OnchainData | None = None
    generated_at: datetime = field(default_factory=utcnow)
    query: str = ""
    schema_version: str = "1.0"


    def to_dict(self) -> dict[str, Any]:
        """Plain dict, JSON-serialisable. Deliberately not a ``_json_hook`` so that
        ``jsonify(self)`` below cannot recurse back into this method."""
        return jsonify(self)

    def to_json(self, *, indent: int | None = 2, sort_keys: bool = False) -> str:
        """JSON-serialise the whole snapshot."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=sort_keys)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TickerSnapshot:
        """Rebuild a snapshot from :meth:`to_dict` output (the inverse of it).

        Round-trips every field, including the derived ``mid``/``is_stale`` flags on
        venues (ignored — recomputed properties) and datetimes. Raises
        ``TypeError``/``ValueError`` if the dict is not a snapshot-shaped dict.
        """
        return from_jsonified(data, cls)

    @classmethod
    def from_json(cls, text: str) -> TickerSnapshot:
        """Parse JSON produced by :meth:`to_json` back into a snapshot object."""
        return cls.from_dict(json.loads(text))

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        """Pandas views. Pandas is imported lazily so the core path stays light."""
        import pandas as pd

        frames: dict[str, pd.DataFrame] = {}

        if self.venues:
            frames["venues"] = pd.DataFrame(
                [
                    {
                        "exchange": v.exchange,
                        "symbol": v.symbol,
                        "quote_currency": v.quote_currency,
                        "bid": v.bid,
                        "ask": v.ask,
                        "last": v.last,
                        "mid": v.mid,
                        "spread": v.spread,
                        "spread_bps": v.spread_bps,
                        "volume_24h_quote": v.volume_24h_quote,
                        "timestamp": v.timestamp,
                        "is_stale": v.is_stale,
                        "is_outlier": v.is_outlier,
                        "deviation_pct": v.deviation_pct,
                        "source": v.provenance.source if v.provenance else None,
                        "fetched_at": v.provenance.fetched_at if v.provenance else None,
                    }
                    for v in self.venues
                ]
            )

        if self.trades:
            frames["trades"] = pd.DataFrame(
                [
                    {
                        "exchange": t.exchange,
                        "symbol": t.symbol,
                        "timestamp": t.timestamp,
                        "price": t.price,
                        "amount": t.amount,
                        "side": t.side,
                        "trade_id": t.trade_id,
                    }
                    for t in self.trades
                ]
            ).sort_values("timestamp", ascending=False)

        ohlcv_rows: list[dict[str, Any]] = []
        for series in self.ohlcv:
            for candle in series.candles:
                ohlcv_rows.append(
                    {
                        "exchange": series.exchange,
                        "symbol": series.symbol,
                        "timeframe": series.timeframe,
                        "timestamp": candle.timestamp,
                        "open": candle.open,
                        "high": candle.high,
                        "low": candle.low,
                        "close": candle.close,
                        "volume": candle.volume,
                    }
                )
        if ohlcv_rows:
            ohlcv_frame = pd.DataFrame(ohlcv_rows)
            # pandas' own stubs declare set_index(keys: Unknown), which trips
            # reportUnknownMemberType regardless of the argument; the result is a
            # DataFrame either way.
            indexed = ohlcv_frame.set_index("timestamp")  # type: ignore[reportUnknownMemberType]
            frames["ohlcv"] = indexed

        if self.orderbooks:
            depth_rows: list[dict[str, Any]] = []
            for book in self.orderbooks:
                for band in book.depth_quote_within:
                    depth_rows.append(
                        {
                            "exchange": book.exchange,
                            "symbol": book.symbol,
                            "within_pct": band.pct,
                            "bid_base": band.bid_base,
                            "ask_base": band.ask_base,
                            "bid_quote": band.bid_quote,
                            "ask_quote": band.ask_quote,
                            "mid": book.mid,
                            "spread_bps": book.spread_bps,
                        }
                    )
            if depth_rows:
                frames["orderbook_depth"] = pd.DataFrame(depth_rows)

            frames["orderbook"] = pd.DataFrame(
                [
                    {
                        "exchange": b.exchange,
                        "symbol": b.symbol,
                        "best_bid": b.best_bid,
                        "best_ask": b.best_ask,
                        "mid": b.mid,
                        "spread": b.spread,
                        "spread_bps": b.spread_bps,
                        "bid_depth_base": b.bid_depth_base,
                        "ask_depth_base": b.ask_depth_base,
                        "imbalance": b.imbalance,
                        "levels_bid": len(b.bids),
                        "levels_ask": len(b.asks),
                        "timestamp": b.timestamp,
                    }
                    for b in self.orderbooks
                ]
            )

        if self.derivatives:
            frames["derivatives"] = pd.DataFrame(
                [
                    {
                        "exchange": d.exchange,
                        "symbol": d.symbol,
                        "contract_type": d.contract_type,
                        "funding_rate": d.funding_rate,
                        "funding_rate_pct": d.funding_rate_pct,
                        "open_interest_quote": d.open_interest_quote,
                        "mark_price": d.mark_price,
                        "index_price": d.index_price,
                        "basis": d.basis,
                        "timestamp": d.timestamp,
                    }
                    for d in self.derivatives
                ]
            )

        return frames

    def summary(self) -> str:
        """A short human-readable digest for the CLI."""
        price = self.market.price
        lines = [
            f"{self.identity.symbol.upper()}  {self.identity.name or '?'}"
            f"  ({self.identity.coingecko_id or 'unresolved'})",
        ]
        if price is not None:
            lines.append(f"  price     {price:,.6g} {self.market.quote_currency}")
            if self.market.change_24h_pct is not None:
                lines.append(f"  24h       {self.market.change_24h_pct:+.2f}%")
        else:
            lines.append("  price     unavailable (see data_quality.errors)")
        lines.append(
            f"  venues    {len(self.venues)}  books {len(self.orderbooks)}"
            f"  trades {len(self.trades)}  ohlcv {len(self.ohlcv)}"
            f"  derivs {len(self.derivatives)}"
        )
        if self.onchain is not None:
            lines.append(f"  tvl       {self.onchain.tvl:,.0f} ({self.onchain.defillama_slug})")
        if self.data_quality.errors:
            lines.append(f"  errors    {len(self.data_quality.errors)} (tolerated)")
        return "\n".join(lines)


__all__ = [
    "Ambiguity",
    "BookLevel",
    "Candle",
    "CandleGap",
    "ChainTVL",
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
    "MissingField",
    "OHLCVSeries",
    "OnchainData",
    "OrderBook",
    "Provenance",
    "ProviderError",
    "SkippedProvider",
    "TickerSnapshot",
    "Trade",
    "VenueTicker",
    "as_utc",
    "finite",
    "to_iso",
    "utcnow",
]