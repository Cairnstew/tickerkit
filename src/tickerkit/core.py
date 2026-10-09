"""The orchestrator.

Resolution runs first (providers contribute, highest-priority first), then every
capability fetch runs concurrently. A provider that raises is recorded in
``DataQuality.errors`` and never sinks the snapshot.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Iterable, Sequence
from dataclasses import replace
from datetime import datetime
from typing import Any, Protocol, cast, runtime_checkable

from .cache import Cache, CacheLimits, MemoryCache, NullCache, SQLiteCache
from .capabilities import Capability
from .config import Config
from .history import build_history_consensus, cross_check_references
from .http import HttpClient, RateLimiter
from .models import (
    DataQuality,
    DerivativesData,
    HistoryCoverage,
    HistoryRequest,
    HistoryResult,
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
    SkippedProvider,
    TickerSnapshot,
    Trade,
    VenueTicker,
    as_utc,
    utcnow,
)
from .normalise import epoch_s_to_utc, parse_iso
from .providers import registry
from .providers.base import Provider, ProviderContext
from .reconcile import apply_consensus
from .resolve import ResolvedAsset, initial_asset


class TickerKit:
    """Fetches and assembles :class:`TickerSnapshot` objects."""

    def __init__(
        self,
        config: Config | None = None,
        *,
        cache: Cache | None = None,
        providers: tuple[type[Provider], ...] | None = None,
    ) -> None:
        self.config = config or Config.from_env()
        if cache is not None:
            self.cache = cache
        elif self.config.cache_path is not None:
            self.cache = SQLiteCache(
                self.config.cache_path,
                limits=CacheLimits(
                    max_entries=self.config.cache_max_entries,
                    max_bytes=self.config.cache_max_bytes,
                ),
            )
        elif self.config.cache_enabled:
            self.cache = MemoryCache()
        else:
            self.cache = NullCache()
        self._providers = providers

    def _provider_classes(self) -> tuple[type[Provider], ...]:
        """The provider set for this instance.

        An explicit set is honoured even when empty — ``providers=()`` means "none", not
        "fall back to the registry". Only ``None`` (the default) selects the registry.
        """
        return self._providers if self._providers is not None else registry()

    # -- public API -------------------------------------------------------------------

    async def get_ticker(self, ticker: str, **overrides: Any) -> TickerSnapshot:
        """Build a complete snapshot for ``ticker``. This is the async core."""
        config = self.config.with_overrides(
            quote_currency=_upper(overrides.get("quote_currency")),
            exchanges=_as_tuple(overrides.get("exchanges")),
            perp_exchanges=_as_tuple(overrides.get("perp_exchanges")),
            outlier_pct=_as_float(overrides.get("outlier_pct")),
            stale_seconds=_as_float(overrides.get("stale_seconds")),
        )
        async with HttpClient(config=config) as http:
            return await self._assemble(ticker, config, http, overrides)

    async def get_history(
        self,
        ticker: str,
        *,
        timeframe: str = "1d",
        start: datetime | str | int | float | None = None,
        end: datetime | str | int | float | None = None,
        limit: int | None = None,
        **overrides: Any,
    ) -> HistoryResult:
        """Build a deep, cross-referenced history series for ``ticker``.

        The shape mirrors :meth:`get_ticker`: the ticker is resolved first (identically,
        reusing every resolver), then each history-capable provider is asked for the same
        window concurrently. Dissimilarities are surfaced, not hidden — a series says how
        much of the window it covers, and the consensus says which venues it trusted.

        ``start``/``end`` accept a ``datetime``, an ISO-8601 string, or an epoch (seconds
        or milliseconds). ``start`` is the one that matters for depth: without it a
        provider returns its most recent page(s), capped by ``limit``.
        """
        config = self.config.with_overrides(
            quote_currency=_upper(overrides.get("quote_currency")),
            exchanges=_as_tuple(overrides.get("exchanges")),
            perp_exchanges=_as_tuple(overrides.get("perp_exchanges")),
        )
        request = HistoryRequest(
            timeframe=timeframe,
            start=_as_datetime(start),
            end=_as_datetime(end),
            limit=_as_int(limit),
        )
        async with HttpClient(config=config) as http:
            asset, resolution_quality = await self._resolve(ticker, config, http, overrides)
            return await self._assemble_history(
                ticker, config, http, asset, resolution_quality, request
            )

    async def _assemble(
        self,
        ticker: str,
        config: Config,
        http: HttpClient,
        overrides: dict[str, Any],
    ) -> TickerSnapshot:
        started = utcnow()
        asset, quality = await self._resolve(ticker, config, http, overrides)

        contexts: dict[str, ProviderContext] = {}
        providers: dict[str, Provider] = {}
        skipped: list[SkippedProvider] = []
        for cls in self._provider_classes():
            ok, reason = cls.available(config)
            if not ok:
                skipped.append(SkippedProvider(provider=cls.name, reason=reason or "unavailable"))
                continue
            context = ProviderContext(
                config=config,
                # A per-provider view over the shared transport: the throttle circuit
                # breaker is per-source, so CoinGecko being exhausted must not stop
                # DefiLlama or the exchanges from answering.
                http=http.view(cls.name),
                cache=self.cache,
                limiter=_limiter_for(cls, config),
            )
            contexts[cls.name] = context
            providers[cls.name] = cls(context)

        # DefiLlama joins on CoinGecko's gecko_id, so hand the resolved id to it.
        if asset.coingecko_id:
            config = replace(config, extra={**config.extra, "coingecko_id": asset.coingecko_id})
            for context in contexts.values():
                context.config = config

        # Every declared capability runs concurrently; failures are collected per call.
        plan: list[tuple[str, Capability]] = [
            (name, capability)
            for name, provider in providers.items()
            for capability in _fetch_plan(provider.supports)
        ]

        results = await asyncio.gather(
            *(_run_capture(providers[name], cap, asset) for name, cap in plan),
            return_exceptions=True,
        )

        venues: list[VenueTicker] = []
        orderbooks: list[OrderBook] = []
        trades: list[Trade] = []
        ohlcv: list[OHLCVSeries] = []
        derivatives: list[DerivativesData] = []
        onchain: OnchainData | None = None
        identity: Identity | None = None
        markets: list[Market] = []

        for (name, capability), outcome in zip(plan, results, strict=True):
            context = contexts[name]
            if isinstance(outcome, BaseException):
                context.note_error(name, capability, outcome)
                continue
            value = outcome
            if value is None:
                continue
            if capability is Capability.META:
                if isinstance(value, Identity):
                    identity = identity or value
            elif capability is Capability.AGG_MARKET:
                if isinstance(value, Market):
                    markets.append(value)
            elif capability is Capability.VENUE_TICKER:
                venues.extend(cast_tuple(value))
            elif capability is Capability.ORDERBOOK:
                orderbooks.extend(cast_tuple(value))
            elif capability is Capability.TRADES:
                trades.extend(cast_tuple(value))
            elif capability is Capability.OHLCV:
                ohlcv.extend(cast_tuple(value))
            elif capability is Capability.DERIVATIVES:
                derivatives.extend(cast_tuple(value))
            elif capability is Capability.ONCHAIN and isinstance(value, OnchainData):
                onchain = onchain or value

        try:
            snapshot = self._build_snapshot(
                config=config,
                query=ticker,
                asset=asset,
                identity=identity,
                markets=markets,
                venues=tuple(venues),
                orderbooks=tuple(orderbooks),
                trades=tuple(trades),
                ohlcv=tuple(ohlcv),
                derivatives=tuple(derivatives),
                onchain=onchain,
                contexts=contexts,
                resolution_quality=quality,
                skipped=skipped,
                generated_at=started,
            )
            return apply_consensus(snapshot, config, utcnow())
        finally:
            await _aclose_providers(providers.values())

    # -- history ----------------------------------------------------------------------

    async def _assemble_history(
        self,
        ticker: str,
        config: Config,
        http: HttpClient,
        asset: ResolvedAsset,
        resolution_quality: DataQuality,
        request: HistoryRequest,
    ) -> HistoryResult:
        """Fan the request out to every history-capable provider, then cross-reference."""
        started = utcnow()
        contexts: dict[str, ProviderContext] = {}
        providers: dict[str, Provider] = {}
        skipped: list[SkippedProvider] = []
        for cls in self._provider_classes():
            if Capability.HISTORY not in cls.supports:
                continue
            ok, reason = cls.available(config)
            if not ok:
                skipped.append(
                    SkippedProvider(provider=cls.name, reason=reason or "unavailable")
                )
                continue
            context = ProviderContext(
                config=config,
                http=http.view(cls.name),
                cache=self.cache,
                limiter=_limiter_for(cls, config),
            )
            contexts[cls.name] = context
            providers[cls.name] = cls(context)

        # DefiLlama joins history on the CoinGecko id, exactly as it does for TVL.
        if asset.coingecko_id:
            config = replace(config, extra={**config.extra, "coingecko_id": asset.coingecko_id})
            for context in contexts.values():
                context.config = config

        results = await asyncio.gather(
            *(provider.fetch_history(asset, request) for provider in providers.values()),
            return_exceptions=True,
        )
        series: list[HistorySeries] = []
        for name, outcome in zip(providers, results, strict=True):
            context = contexts[name]
            if isinstance(outcome, BaseException):
                context.note_error(name, Capability.HISTORY, outcome)
                continue
            series.extend(_as_series(outcome))

        try:
            return self._build_history_result(
                config=config,
                query=ticker,
                asset=asset,
                request=request,
                series=tuple(series),
                contexts=contexts,
                resolution_quality=resolution_quality,
                skipped=skipped,
                generated_at=started,
            )
        finally:
            await _aclose_providers(providers.values())

    def _build_history_result(
        self,
        *,
        config: Config,
        query: str,
        asset: ResolvedAsset,
        request: HistoryRequest,
        series: tuple[HistorySeries, ...],
        contexts: dict[str, ProviderContext],
        resolution_quality: DataQuality,
        skipped: list[SkippedProvider],
        generated_at: datetime,
    ) -> HistoryResult:
        all_provenance = dict(resolution_quality.provenance)
        all_errors: list[ProviderError] = list(resolution_quality.errors)
        all_missing: list[MissingField] = list(resolution_quality.missing)
        all_ambiguities = list(resolution_quality.ambiguities)
        all_throttled: list[RateLimitInfo] = list(resolution_quality.rate_limited)
        used: list[str] = []
        for name, context in contexts.items():
            all_provenance.update(context.provenance)
            all_errors.extend(context.errors)
            all_missing.extend(context.missing)
            all_ambiguities.extend(context.ambiguities)
            all_throttled.extend(context.rate_limited)
            used.append(name)

        # Safety net: a returned series is evidence of a source even if its provider
        # forgot to record provenance, so every series stays traceable.
        recorded = {prov.source for prov in all_provenance.values()}
        for item in series:
            prov = item.provenance
            if prov is not None and prov.source not in recorded:
                all_provenance[f"history:{item.source}"] = prov
                recorded.add(prov.source)

        consensus, discrepancies = build_history_consensus(series, config, request)
        references = tuple(s for s in series if s.kind != "ohlcv")
        discrepancies = discrepancies + cross_check_references(
            consensus, references, config.history_discrepancy_pct
        )

        if not series:
            all_missing.append(
                _missing(
                    "history",
                    f"no provider returned a {request.timeframe} series for this asset",
                )
            )
        elif consensus is not None and consensus.coverage is not None:
            # A bounded request that was not fully covered must say so: the whole point
            # of the coverage object is that a short series is never mistaken for a full
            # one.
            bounded = request.start is not None or request.end is not None
            if bounded and not consensus.coverage.complete:
                all_missing.append(
                    _missing("history.coverage", _coverage_reason(consensus.coverage))
                )

        identity = Identity(
            symbol=asset.symbol,
            name=asset.name,
            coingecko_id=asset.coingecko_id,
            contract_address=asset.contract_address,
            asset_platform_id=asset.asset_platform_id,
        )
        quality = DataQuality(
            provenance=all_provenance,
            errors=tuple(all_errors),
            rate_limited=_dedupe_throttled(all_throttled),
            ambiguities=tuple(all_ambiguities),
            discrepancies=tuple(discrepancies),
            missing=tuple(dict.fromkeys(all_missing)),
            providers_used=tuple(dict.fromkeys(used)),
            providers_skipped=_dedupe_skipped(
                [*resolution_quality.providers_skipped, *skipped]
            ),
        )
        return HistoryResult(
            identity=identity,
            request=request,
            data_quality=quality,
            series=series,
            consensus=consensus,
            generated_at=generated_at,
            query=query,
        )

    # -- resolution -------------------------------------------------------------------

    async def _resolve(
        self,
        ticker: str,
        config: Config,
        http: HttpClient,
        overrides: dict[str, Any],
    ) -> tuple[ResolvedAsset, DataQuality]:
        """Run every resolver, then merge. First non-empty CoinGecko id wins."""
        asset = initial_asset(ticker, config)
        explicit_id = overrides.get("coingecko_id")
        errors: list[ProviderError] = []
        rate_limited: list[RateLimitInfo] = []
        missing: list[MissingField] = []
        ambiguities: list[Any] = []
        provenance: dict[str, Provenance] = {}
        resolved_any = False

        contexts: dict[str, ProviderContext] = {}
        resolvers: dict[str, Provider] = {}
        skipped: list[SkippedProvider] = []
        for cls in self._provider_classes():
            if Capability.RESOLVE not in cls.supports:
                continue
            ok, reason = cls.available(config)
            if not ok:
                skipped.append(SkippedProvider(provider=cls.name, reason=reason or "unavailable"))
                continue
            context = ProviderContext(
                config=config,
                http=http.view(cls.name),
                cache=self.cache,
                limiter=_limiter_for(cls, config),
            )
            contexts[cls.name] = context
            resolvers[cls.name] = cls(context)

        results = await asyncio.gather(
            *(_run_capture(provider, Capability.RESOLVE, None, query=ticker)
              for provider in resolvers.values()),
            return_exceptions=True,
        )

        await _aclose_providers(resolvers.values())

        contributions: list[tuple[str, ResolvedAsset]] = []
        for name, outcome in zip(resolvers, results, strict=True):
            context = contexts[name]
            if isinstance(outcome, BaseException):
                context.note_error(name, Capability.RESOLVE, outcome)
                continue
            if isinstance(outcome, ResolvedAsset):
                contributions.append((name, outcome))
                provenance.update(context.provenance)
                ambiguities.extend(context.ambiguities)
                missing.extend(context.missing)
                resolved_any = True

        merged = asset
        notes: list[str] = list(asset.notes)

        if explicit_id:
            merged = merged.with_coingecko(str(explicit_id))
            notes.append(f"coingecko id overridden to {explicit_id}")
            resolved_any = True

        for name, contribution in contributions:
            if merged.coingecko_id is None and contribution.coingecko_id:
                merged = merged.with_coingecko(
                    str(contribution.coingecko_id), name=contribution.name
                )
                notes.append(f"coingecko id from {name}")
            if merged.market_cap is None and contribution.market_cap is not None:
                merged = replace(merged, market_cap=contribution.market_cap)
            if contribution.spot_markets and not merged.spot_markets:
                merged = replace(merged, spot_markets=contribution.spot_markets)
            if contribution.derivative_markets and not merged.derivative_markets:
                merged = replace(merged, derivative_markets=contribution.derivative_markets)
            if contribution.defillama_slug and not merged.defillama_slug:
                merged = merged.with_defillama(contribution.defillama_slug)
            notes.extend(contribution.notes)
            if contribution.ambiguities and not merged.ambiguities:
                merged = replace(merged, ambiguities=contribution.ambiguities)

        override_slug = overrides.get("defillama_slug")
        if override_slug:
            merged = merged.with_defillama(str(override_slug))
            notes.append(f"defillama slug overridden to {override_slug}")

        override_base = overrides.get("base_symbol")
        if override_base:
            merged = replace(merged, symbol=str(override_base).upper())
            notes.append(f"base symbol overridden to {override_base}")

        # Resolution-phase failures land in each provider's context, so they have to be
        # collected from there. Reading a local list that nothing ever appends to silently
        # dropped every resolution error, which surfaced as an unidentified asset with a
        # real venue price and zero recorded errors.
        for context in contexts.values():
            errors.extend(context.errors)
            missing.extend(context.missing)
            # A throttle during resolution is the single most common reason an asset
            # comes back unidentified, so it must reach the snapshot's quality block
            # just as the matching error does.
            rate_limited.extend(context.rate_limited)

        # Only add the overall "nothing identified this ticker" note when no provider
        # already reported why; otherwise the same fact is recorded twice.
        has_resolve_error = any(e.capability is Capability.RESOLVE for e in errors)
        if not resolved_any and not explicit_id and not has_resolve_error:
            errors.append(
                _resolution_error(ticker, "no provider could identify this ticker")
            )

        merged = replace(merged, notes=tuple(dict.fromkeys(notes)), unresolved=not resolved_any)
        quality = DataQuality(
            provenance=provenance,
            errors=tuple(errors),
            rate_limited=_dedupe_throttled(rate_limited),
            ambiguities=tuple(ambiguities),
            missing=tuple(dict.fromkeys(missing)),
            providers_used=tuple(resolvers),
            providers_skipped=tuple(skipped),
        )
        return merged, quality

    # -- assembly ---------------------------------------------------------------------

    def _build_snapshot(
        self,
        *,
        config: Config,
        query: str,
        asset: ResolvedAsset,
        identity: Identity | None,
        markets: list[Market],
        venues: tuple[VenueTicker, ...],
        orderbooks: tuple[OrderBook, ...],
        trades: tuple[Trade, ...],
        ohlcv: tuple[OHLCVSeries, ...],
        derivatives: tuple[DerivativesData, ...],
        onchain: OnchainData | None,
        contexts: dict[str, ProviderContext],
        resolution_quality: DataQuality,
        skipped: list[SkippedProvider],
        generated_at: Any,
    ) -> TickerSnapshot:
        # The effective config for this call, not the instance default: per-call
        # overrides such as quote_currency="EUR" must be reflected in the snapshot,
        # otherwise the market claims a currency the venues were not fetched in.
        quote = config.quote_currency

        identity = identity or Identity(
            symbol=asset.symbol,
            name=asset.name,
            coingecko_id=asset.coingecko_id,
            contract_address=asset.contract_address,
            asset_platform_id=asset.asset_platform_id,
        )
        # Merge aggregator markets: CoinGecko is primary, CMC fills gaps.
        market_provenance: dict[str, Provenance] = dict(resolution_quality.provenance)
        for m in markets:
            market_provenance.update(m.provenance)

        base = markets[0] if markets else Market(quote_currency=quote)

        market = Market(
            quote_currency=quote,
            consensus_price=base.consensus_price,
            consensus_method=base.consensus_method,
            consensus_inputs=base.consensus_inputs,
            aggregated_price=base.aggregated_price,
            change_24h=base.change_24h,
            change_24h_pct=base.change_24h_pct,
            high_24h=base.high_24h,
            low_24h=base.low_24h,
            volume_24h=base.volume_24h,
            market_cap=base.market_cap,
            fdv=base.fdv,
            circulating_supply=base.circulating_supply,
            total_supply=base.total_supply,
            max_supply=base.max_supply,
            ath=base.ath,
            ath_date=base.ath_date,
            ath_change_pct=base.ath_change_pct,
            atl=base.atl,
            atl_date=base.atl_date,
            atl_change_pct=base.atl_change_pct,
            provenance=market_provenance,
        )

        all_provenance = dict(market_provenance)
        all_errors: list[ProviderError] = list(resolution_quality.errors)
        all_missing: list[MissingField] = list(resolution_quality.missing)
        all_ambiguities = list(resolution_quality.ambiguities)
        all_throttled: list[RateLimitInfo] = list(resolution_quality.rate_limited)
        used: list[str] = []
        for name, context in contexts.items():
            all_provenance.update(context.provenance)
            all_errors.extend(context.errors)
            all_missing.extend(context.missing)
            all_ambiguities.extend(context.ambiguities)
            all_throttled.extend(context.rate_limited)
            used.append(name)

        quality = DataQuality(
            provenance=all_provenance,
            errors=tuple(all_errors),
            rate_limited=_dedupe_throttled(all_throttled),
            ambiguities=tuple(all_ambiguities),
            missing=tuple(all_missing),
            providers_used=tuple(dict.fromkeys(used)),
            providers_skipped=_dedupe_skipped(
                [*resolution_quality.providers_skipped, *skipped]
            ),
        )

        if market.price is None:
            quality = replace(
                quality,
                missing=(*quality.missing,
                         _missing("market.price", "no provider returned a usable price")),
            )

        # Whatever route the snapshot took, an unidentified asset must say so. Without
        # this a transient provider hiccup could yield a snapshot with a price (from the
        # exchange-only path) and no indication that the asset itself was never resolved.
        if identity.coingecko_id is None and not any(
            m.field == "identity.coingecko_id" for m in quality.missing
        ):
            quality = replace(
                quality,
                missing=(
                    *quality.missing,
                    _missing(
                        "identity.coingecko_id",
                        f"no aggregator identified {query!r}; the price below comes from "
                        "exchange data only and may belong to a different asset sharing "
                        "this ticker",
                    ),
                ),
            )

        return TickerSnapshot(
            identity=identity,
            market=market,
            venues=venues,
            orderbooks=orderbooks,
            trades=trades,
            ohlcv=ohlcv,
            derivatives=derivatives,
            onchain=onchain,
            data_quality=quality,
            generated_at=generated_at,
            query=query,
        )


# --------------------------------------------------------------------------- helpers

#: Capabilities actually fetched by the orchestrator, in a stable order.
capability_set: frozenset[Capability] = frozenset(
    {
        Capability.META,
        Capability.AGG_MARKET,
        Capability.VENUE_TICKER,
        Capability.ORDERBOOK,
        Capability.TRADES,
        Capability.OHLCV,
        Capability.DERIVATIVES,
        Capability.ONCHAIN,
    }
)

#: Maps a capability onto the provider method implementing it.
_METHODS: dict[Capability, str] = {
    Capability.META: "fetch_identity",
    Capability.AGG_MARKET: "fetch_market",
    Capability.VENUE_TICKER: "fetch_venue_tickers",
    Capability.ORDERBOOK: "fetch_orderbooks",
    Capability.TRADES: "fetch_trades",
    Capability.OHLCV: "fetch_ohlcv",
    Capability.DERIVATIVES: "fetch_derivatives",
    Capability.ONCHAIN: "fetch_onchain",
}


def _fetch_plan(supported: frozenset[Capability]) -> list[Capability]:
    return [c for c in _ORDER if c in supported]


_ORDER: tuple[Capability, ...] = (
    Capability.META,
    Capability.AGG_MARKET,
    Capability.VENUE_TICKER,
    Capability.ORDERBOOK,
    Capability.TRADES,
    Capability.OHLCV,
    Capability.DERIVATIVES,
    Capability.ONCHAIN,
)


async def _run_capture(
    provider: Provider, capability: Capability, asset: Any, *, query: str | None = None
) -> Any:
    """Invoke one capability, letting exceptions propagate to the caller to record."""
    if capability is Capability.RESOLVE:
        return await provider.resolve(query or "")
    method = getattr(provider, _METHODS[capability])
    return await method(asset)


def cast_tuple(value: object) -> tuple[Any, ...]:
    """Defensive: a provider returning a bare object should not sink the snapshot."""
    out: tuple[object, ...]
    if isinstance(value, tuple):
        out = cast("tuple[object, ...]", value)
    elif isinstance(value, list):
        out = tuple(cast("list[object]", value))
    else:
        out = ()
    return out


def _resolution_error(ticker: str, message: str) -> ProviderError:
    return ProviderError(
        provider="resolve",
        capability=Capability.RESOLVE,
        exception_type="ResolutionError",
        message=f"{ticker!r}: {message}",
    )


def _missing(field_name: str, reason: str) -> MissingField:
    return MissingField(field=field_name, reason=reason)


def _upper(value: object) -> str | None:
    return str(value).upper() if isinstance(value, str) and value else None


def _limiter_for(cls: type[Provider], config: Config) -> RateLimiter | None:
    """Build the provider's own rate limiter, if it documents one.

    ccxt is excluded: it manages request weight internally via ``enableRateLimit``.
    """
    if cls.name == "ccxt":
        return None
    rpm_fn: object = getattr(cls, "rate_limit_rpm", None)
    rpm: object = rpm_fn(config) if callable(rpm_fn) else config.rate_limit_rpm
    return RateLimiter(rpm if isinstance(rpm, int) and rpm > 0 else config.rate_limit_rpm,
                       name=cls.name)


@runtime_checkable
class _Closable(Protocol):
    """A provider that owns resources needing explicit teardown."""

    async def aclose(self) -> None: ...


async def _aclose_providers(providers: Iterable[Provider]) -> None:
    """Close provider-held resources (ccxt keeps aiohttp sessions open)."""
    for provider in providers:
        if not isinstance(provider, _Closable):
            continue
        # Teardown failures must never mask a successfully assembled snapshot.
        with contextlib.suppress(Exception):
            await provider.aclose()


def _dedupe_throttled(items: list[RateLimitInfo]) -> tuple[RateLimitInfo, ...]:
    """Collapse duplicate throttle records for the same (source, capability) pair.

    A provider that fans out over several capabilities records the same exhausted
    source once per call. Listing "coingecko throttled, retry in 30s" eight times tells
    the reader nothing extra, so keep the first (the real 429, not a suppression).
    """
    out: list[RateLimitInfo] = []
    seen: set[tuple[str, Capability]] = set()
    for item in items:
        key = (item.source, item.capability)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return tuple(out)


def _dedupe_skipped(items: list[SkippedProvider]) -> tuple[SkippedProvider, ...]:
    """Drop duplicate (provider, reason) skip records, preserving order."""
    seen: set[tuple[str, str]] = set()
    out: list[SkippedProvider] = []
    for item in items:
        key = (item.provider, item.reason)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return tuple(out)


def _as_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        parts = tuple(p.strip() for p in value.split(",") if p.strip())
        return parts or None
    if isinstance(value, (list, tuple)):
        entries = cast("Sequence[object]", value)
        return tuple(str(v) for v in entries)
    return None


def _as_float(value: object) -> float | None:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _as_int(value: object) -> int | None:
    """A positive integer, or ``None`` (bad/absent limits are ignored, not guessed)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value > 0 else None
    if isinstance(value, float):
        return int(value) if value > 0 else None
    return None


def _as_datetime(value: object) -> datetime | None:
    """Coerce a datetime, ISO-8601 string, or epoch (seconds or milliseconds) to UTC."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return as_utc(value)
    if isinstance(value, (int, float)):
        epoch = float(value)
        if epoch > 100_000_000_000:  # a millisecond epoch, not seconds
            epoch /= 1000.0
        return epoch_s_to_utc(epoch)
    if isinstance(value, str):
        return parse_iso(value)
    return None


def _as_series(value: object) -> tuple[HistorySeries, ...]:
    """Defensive: a provider returning a bare object must not sink the result."""
    if not isinstance(value, (tuple, list)):
        return ()
    entries = cast("Sequence[object]", value)
    return tuple(item for item in entries if isinstance(item, HistorySeries))


def _coverage_reason(coverage: HistoryCoverage) -> str:
    """A one-line explanation of why a history window is not fully covered."""
    parts: list[str] = []
    if coverage.truncated:
        parts.append("a provider cap stopped the walk early")
    if coverage.gaps:
        parts.append(f"{len(coverage.gaps)} interior gap(s)")
    if coverage.missing:
        parts.append(f"{coverage.missing} interval(s) missing")
    if coverage.note:
        parts.append(coverage.note)
    detail = "; ".join(parts) or "the requested window is not fully covered"
    first = coverage.first.isoformat() if coverage.first else "?"
    last = coverage.last.isoformat() if coverage.last else "?"
    return f"requested window not fully covered: {detail} (covered {first} .. {last})"


__all__ = ["TickerKit", "capability_set"]