"""The provider contract.

A provider declares which :class:`~tickerkit.capabilities.Capability` values it implements
and implements only those methods. The orchestrator introspects ``supports`` and never
calls a method a provider did not claim, so adding a provider requires no orchestrator
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, ClassVar

from ..cache import Cache
from ..capabilities import Capability
from ..config import Config
from ..errors import RateLimitExceeded
from ..models import (
    Ambiguity,
    DerivativesData,
    HistoryRequest,
    HistorySeries,
    Identity,
    Market,
    MissingField,
    OHLCVSeries,
    OnchainData,
    OrderBook,
    Provenance,
    ProviderError,
    RateLimitInfo,
    Trade,
    VenueTicker,
    utcnow,
)
from ..resolve import ResolvedAsset

if True:  # pragma: no cover - import alias keeps the cycle explicit and typed
    from ..http import HttpClient


@dataclass(slots=True)
class ProviderContext:
    """Everything a provider needs, injected by the orchestrator."""

    config: Config
    http: HttpClient
    cache: Cache
    #: Per-provider rate limiter, applied before each HTTP call. Optional because ccxt
    #: manages its own limiting internally.
    limiter: Any | None = None

    provenance: dict[str, Provenance] = field(default_factory=dict)
    errors: list[ProviderError] = field(default_factory=list)
    #: Throttled calls, recorded separately so `DataQuality.rate_limited` can be
    #: consulted without inspecting exception types.
    rate_limited: list[RateLimitInfo] = field(default_factory=list)
    missing: list[MissingField] = field(default_factory=list)
    ambiguities: list[Ambiguity] = field(default_factory=list)

    def record(self, key: str, provenance: Provenance) -> None:
        """Record provenance for a field group."""
        self.provenance[key] = provenance

    def note_missing(self, field_name: str, reason: str) -> None:
        """Explain a ``None`` so it is never mistaken for a real zero."""
        self.missing.append(MissingField(field=field_name, reason=reason))

    def note_error(
        self, provider: str, capability: Capability, exc: BaseException
    ) -> ProviderError:
        """Record a tolerated failure and return it."""
        message = str(exc) or repr(exc)
        err = ProviderError(
            provider=provider,
            capability=capability,
            exception_type=type(exc).__name__,
            message=message,
        )
        self.errors.append(err)
        if isinstance(exc, RateLimitExceeded):
            self.rate_limited.append(
                RateLimitInfo(
                    source=exc.source or provider,
                    capability=capability,
                    message=message,
                    retry_after=exc.retry_after,
                    limit=exc.limit,
                    remaining=exc.remaining,
                    remedy=exc.hint,
                    suppressed=exc.suppressed,
                )
            )
        return err

    async def throttle(self) -> None:
        """Wait for this provider's rate-limit slot, if one is configured."""
        if self.limiter is not None:
            await self.limiter.acquire()

    async def cached(self, key: str, ttl: float) -> tuple[Any | None, bool]:
        """Return ``(value, was_cached)`` from the TTL cache."""
        value = await self.cache.get(key)
        return value, value is not None

    async def remember(self, key: str, value: Any, ttl: float | None = None) -> None:
        """Store a value in the TTL cache."""
        await self.cache.set(key, value, self.config.cache_ttl if ttl is None else ttl)


class Provider:
    """Base class for all data providers.

    Deliberately *not* an ``abc.ABC``: every capability hook has a working no-op
    default, so a provider only overrides what it actually supports. Capability
    enforcement lives in ``supports`` and in the orchestrator, which never calls a
    method outside that set.
    """

    #: Stable identifier used in provenance and error records.
    name: ClassVar[str] = "provider"

    #: Capabilities this provider implements. Subclasses must narrow this.
    supports: ClassVar[frozenset[Capability]] = frozenset()

    def __init__(self, context: ProviderContext) -> None:
        self.ctx = context

    @classmethod
    def available(cls, config: Config) -> tuple[bool, str | None]:
        """Whether this provider can run, and the reason if it cannot.

        Default: always available. Override to gate on credentials (e.g. CMC).
        """
        return True, None

    # -- capability hooks; default no-ops so partial implementations stay short ----------

    async def resolve(self, query: str) -> ResolvedAsset | None:
        return None

    async def fetch_identity(self, asset: ResolvedAsset) -> Identity | None:
        return None

    async def fetch_market(self, asset: ResolvedAsset) -> Market | None:
        return None

    async def fetch_venue_tickers(self, asset: ResolvedAsset) -> tuple[VenueTicker, ...]:
        return ()

    async def fetch_orderbooks(self, asset: ResolvedAsset) -> tuple[OrderBook, ...]:
        return ()

    async def fetch_trades(self, asset: ResolvedAsset) -> tuple[Trade, ...]:
        return ()

    async def fetch_ohlcv(self, asset: ResolvedAsset) -> tuple[OHLCVSeries, ...]:
        return ()

    async def fetch_history(
        self, asset: ResolvedAsset, request: HistoryRequest
    ) -> tuple[HistorySeries, ...]:
        """Deep, windowed series for :meth:`tickerkit.TickerKit.get_history`.

        Unlike :meth:`fetch_ohlcv`, this is only called for an explicit history request,
        so a provider may paginate and spend many upstream calls. Each returned series
        must carry a :class:`~tickerkit.models.HistoryCoverage` built with
        ``tickerkit.history.build_coverage`` so truncation is never silent.
        """
        return ()

    async def fetch_derivatives(self, asset: ResolvedAsset) -> tuple[DerivativesData, ...]:
        return ()

    async def fetch_onchain(self, asset: ResolvedAsset) -> OnchainData | None:
        return None


def provenance_for(
    *,
    source: str,
    capability: Capability,
    fetched_at: datetime | None = None,
    exchange_timestamp: datetime | None = None,
    url: str | None = None,
    cached: bool = False,
    notes: tuple[str, ...] = (),
) -> Provenance:
    """Build a :class:`Provenance` with sane defaults."""
    return Provenance(
        source=source,
        capability=capability,
        fetched_at=fetched_at or utcnow(),
        exchange_timestamp=exchange_timestamp,
        url=url,
        cached=cached,
        notes=notes,
    )


__all__ = ["Provider", "ProviderContext", "provenance_for"]