"""DefiLlama provider: TVL for the protocol and/or chain behind an asset.

Docs (read before writing this adapter, per project rule):
  * Free API reference — https://api-docs.defillama.com/llms-free.txt
  * API docs home     — https://api-docs.defillama.com

Facts this adapter depends on, all from the Free API reference above:
  * Base URL for every endpoint is ``https://api.llama.fi``; no authentication required.
  * ``GET /protocols`` -> list of rows carrying ``id``, ``slug``, ``name``, ``symbol``,
    ``category``, ``chains``, ``tvl``, ``chainTvls``, ``change_1d``, ``change_7d``,
    ``gecko_id``, ``cmcId``, ``url``. The path parameter for the protocol endpoints is
    the **slug**, not the numeric ``id`` — ``/protocol/bitcoin`` resolves, ``/protocol/2786``
    does not.
  * ``GET /protocol/{slug}`` -> ``{id, name, symbol, gecko_id, category, chains,
    currentChainTvls, chainTvls, tvl, ...}``. Verified against the live API: ``tvl`` is a
    **history array** of ``{"date": ..., "totalLiquidityUSD": ...}`` points, not a scalar,
    so the current value is the last element. ``currentChainTvls`` is the flat per-chain
    map of numbers (``{"Ethereum": 2573231226}``); ``chainTvls[chain].tvl`` is that
    chain's history array. Many keys carry ``-staking``/``-pool2``/``-borrowed`` suffixes
    that are sub-buckets, not chains.
  * ``/protocol/{slug}`` resolves a **parent** slug even when no such row appears in
    ``/protocols``: ``/protocol/uniswap`` returns the aggregate across v1-v4, and
    ``/protocol/aave`` the aggregate across its versions. Without this, a ``gecko_id``
    match can land on one version (``gecko_id=aave`` sits on ``aave-v2``) and understate
    the protocol's real TVL by two orders of magnitude.
  * ``GET /v2/chains`` -> ``[{gecko_id, gasTokenGeckoId, tvl, tokenSymbol, cmcId, name,
    chainId}]``.

Resolution strategy: ``gecko_id`` is the join key on both endpoints, so a resolved
CoinGecko id is matched directly rather than guessing from a ticker symbol. An asset can
match both a chain (its own L1 TVL) and a protocol (a DeFi TVL); both are reported when
present, which is why ``fetch_onchain`` consults the chain list first.

Note: DefiLlama is a TVL analytics source, never a price feed. Nothing here influences
the consensus price.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any, cast
from urllib.parse import quote as urlquote
from urllib.parse import urlencode

from ..capabilities import Capability
from ..config import Config
from ..errors import ProviderError
from ..history import build_coverage, interval_seconds, normalise_candles
from ..models import (
    ChainTVL,
    HistoryCandle,
    HistoryRequest,
    HistorySeries,
    OnchainData,
    Provenance,
    as_utc,
    finite,
)
from ..normalise import epoch_s_to_utc
from ..resolve import ResolvedAsset, normalise_query
from .base import Provider, provenance_for

BASE_URL = "https://api.llama.fi"

#: The coins service is a *separate* host from ``api.llama.fi``; ``/chart`` does not exist
#: on the TVL host. Verified live: ``coins.llama.fi/chart/...`` answers, the TVL host 404s.
COINS_BASE_URL = "https://coins.llama.fi"

#: ccxt timeframe -> DefiLlama ``period``. DefiLlama's finest chart period is hourly, so
#: sub-hourly timeframes are declined rather than fetched at the wrong resolution.
_LLAMA_PERIODS: dict[str, str] = {"1h": "1h", "1d": "1d", "1w": "1w"}

#: Maximum points DefiLlama serves in one chart call. Verified live: ``span=500`` is
#: accepted, ``span=1000`` returns HTTP 400. Longer windows are walked in pages.
_MAX_SPAN = 500

#: ``chainTvls`` sub-keys that are not chains and would corrupt a chain breakdown.
_NON_CHAIN_KEYS: frozenset[str] = frozenset(
    {"staking", "pool2", "borrowed", "vesting", "offers", "pool2Details"}
)

_CHAINS_TTL = 3600.0
_PROTOCOLS_TTL = 3600.0


@dataclass(frozen=True, slots=True)
class ChainRow:
    name: str
    tvl: float | None
    gecko_id: str | None


class DefiLlamaProvider(Provider):
    name = "defillama"
    supports = frozenset({Capability.RESOLVE, Capability.ONCHAIN, Capability.HISTORY})

    @classmethod
    def available(cls, config: Config) -> tuple[bool, str | None]:
        return True, None

    async def _protocols(self) -> list[dict[str, Any]]:
        cached, hit = await self.ctx.cached("defillama:protocols", _PROTOCOLS_TTL)
        if hit:
            return cast(list[dict[str, Any]], cached)
        await self.ctx.throttle()
        rows = _as_rows(await self.ctx.http.get_json(f"{BASE_URL}/protocols"))
        await self.ctx.remember("defillama:protocols", rows, _PROTOCOLS_TTL)
        return rows

    async def _chains(self) -> list[dict[str, Any]]:
        cached, hit = await self.ctx.cached("defillama:chains", _CHAINS_TTL)
        if hit:
            return cast(list[dict[str, Any]], cached)
        await self.ctx.throttle()
        rows = _as_rows(await self.ctx.http.get_json(f"{BASE_URL}/v2/chains"))
        await self.ctx.remember("defillama:chains", rows, _CHAINS_TTL)
        return rows

    # -- resolve ----------------------------------------------------------------------

    async def resolve(self, query: str) -> ResolvedAsset | None:
        """Map the asset onto a DefiLlama protocol slug.

        Prefers an exact ``gecko_id`` match (unambiguous); falls back to a symbol match
        only when no CoinGecko id is available, and in that case requires a symbol match
        to be unique.
        """
        symbol = normalise_query(query)
        asset = ResolvedAsset(query=query, symbol=symbol)

        protocols = await self._protocols()
        rows: list[dict[str, Any]] = []
        if True:
            exact = [
                r for r in protocols if str(r.get("symbol") or "").upper() == symbol
            ]
            named = [r for r in exact if str(r.get("name") or "").upper() == symbol]
            # A symbol shared by many protocols is not a resolution; only accept a
            # unique match, otherwise TVL would be arbitrary.
            if len(named) == 1:
                rows = named
            elif len(exact) == 1:
                rows = exact
            else:
                rows = []

        if not rows:
            return None

        rows.sort(key=lambda r: finite(r.get("tvl")) or 0.0, reverse=True)
        slug = _slug_of(rows[0])
        if not slug:
            return None
        self.ctx.record(
            "defillama:resolve",
            provenance_for(
                source="defillama:/protocols",
                capability=Capability.RESOLVE,
                url=f"{BASE_URL}/protocols",
            ),
        )
        return asset.with_defillama(slug)

    # -- onchain ----------------------------------------------------------------------

    async def fetch_onchain(self, asset: ResolvedAsset) -> OnchainData | None:
        chain_tvl = await self._chain_tvl_for(asset)
        slugs = await self._candidates_for_asset(asset)
        protocol = await self._protocol_tvl_for(slugs, asset)
        slug = protocol.slug if protocol is not None else (slugs[0] if slugs else None)
        if chain_tvl is None and protocol is None:
            self.ctx.note_missing(
                "onchain",
                "asset matches neither a DefiLlama chain nor a protocol "
                "(TVL is only defined for chains and DeFi protocols)",
            )
            return None

        tvl = protocol.tvl if protocol and protocol.tvl is not None else None
        if tvl is None and chain_tvl is not None:
            tvl = chain_tvl.tvl
        prov = _pick_provenance(protocol, chain_tvl)
        assert prov is not None  # one of the two always matched here

        # Merge the protocol's per-chain breakdown with the asset's own chain row; the
        # breakdown is only available from /protocol/{slug}, the chain row from /v2/chains.
        by_chain: dict[str, float] = {
            c.chain: c.tvl for c in (protocol.chains if protocol else ()) if c.tvl is not None
        }
        if chain_tvl is not None and chain_tvl.tvl is not None:
            by_chain.setdefault(chain_tvl.name, chain_tvl.tvl)
        chains = tuple(
            ChainTVL(chain=name, tvl=value)
            for name, value in sorted(by_chain.items(), key=lambda kv: kv[1], reverse=True)
        )

        if tvl is None:
            self.ctx.note_missing(
                "onchain.tvl",
                f"DefiLlama reports no TVL for protocol {slug!r}"
                + (f" or chain {chain_tvl.name!r}" if chain_tvl else " and no chain matched"),
            )
        elif protocol is not None and protocol.tvl is None:
            self.ctx.note_missing(
                "onchain.protocol_tvl",
                f"protocol {slug!r} has no TVL of its own; reporting chain-level TVL",
            )
        return OnchainData(
            defillama_slug=slug or "",
            name=(protocol.name if protocol else (chain_tvl.name if chain_tvl else None)),
            category=protocol.category if protocol else "chain",
            tvl=tvl,
            chains=chains,
            url=protocol.url if protocol else None,
            provenance=prov,
        )

    async def _chain_tvl_for(self, asset: ResolvedAsset) -> _ChainResult | None:
        gecko_id = asset.coingecko_id
        if not gecko_id:
            return None
        try:
            rows = await self._chains()
        except Exception as exc:
            self.ctx.note_missing(
                "onchain.chains", f"DefiLlama /v2/chains failed: {type(exc).__name__}"
            )
            return None
        match = next((r for r in rows if r.get("gecko_id") == gecko_id), None)
        if match is None:
            return None
        return _ChainResult(
            name=str(match.get("name") or gecko_id),
            tvl=finite(match.get("tvl")),
            provenance=provenance_for(
                source="defillama:/v2/chains",
                capability=Capability.ONCHAIN,
                url=f"{BASE_URL}/v2/chains",
            ),
        )

    async def _candidate_slugs(self, asset: ResolvedAsset) -> tuple[str, ...]:
        """Slugs derived from ``/protocols``, best first.

        Selection is tiered by how trustworthy each key is:

        1. ``gecko_id`` — unambiguous.
        2. the asset's resolved *name* matching a slug (Solana -> ``solana``).
        3. the slug equal to the ticker (``eth`` -> ``eth``).
        4. otherwise the largest protocol sharing the ticker, with the ambiguity recorded.

        If the asset was never identified (no ``gecko_id``, no name) and several
        protocols share the ticker, nothing is returned. Guessing there is how SOL ends
        up reported as "solana-farm" ($238) instead of Solana, so a missing TVL is
        strongly preferable to a wrong one.
        """
        try:
            rows = await self._protocols()
        except Exception as exc:
            self.ctx.note_missing(
                "onchain", f"DefiLlama /protocols failed: {type(exc).__name__}"
            )
            return ()

        matches: list[dict[str, Any]] = []
        gecko_id = asset.coingecko_id
        if gecko_id:
            matches = [r for r in rows if r.get("gecko_id") == gecko_id]

        if not matches:
            symbol = (asset.symbol or "").upper()
            name = (asset.name or "").lower()
            symbol_rows = [r for r in rows if str(r.get("symbol") or "").upper() == symbol]
            by_name = [r for r in symbol_rows if name and str(r.get("slug") or "").lower() == name]
            by_slug = [
                r for r in symbol_rows
                if symbol and str(r.get("slug") or "").lower() == symbol.lower()
            ]
            if by_name:
                matches = by_name
            elif by_slug:
                matches = by_slug
            elif len(symbol_rows) == 1:
                matches = symbol_rows
            elif not gecko_id and not name:
                names = ", ".join(
                    str(r.get("slug")) for r in symbol_rows[:4]
                )
                self.ctx.note_missing(
                    "onchain.protocol_scope",
                    f"asset is unidentified and {len(symbol_rows)} protocols share "
                    f"ticker {symbol} ({names}); refusing to guess a protocol TVL",
                )
                return ()
            else:
                ordered = sorted(
                    symbol_rows, key=lambda r: finite(r.get("tvl")) or 0.0, reverse=True
                )
                names = ", ".join(str(r.get("slug")) for r in ordered[:4])
                self.ctx.note_missing(
                    "onchain.protocol_scope",
                    f"{len(ordered)} protocols share ticker {symbol} ({names}); "
                    "using the largest, or the parent aggregate where one exists",
                )
                matches = ordered[:1]

        if not matches:
            return ()

        matches.sort(key=lambda r: finite(r.get("tvl")) or 0.0, reverse=True)
        out: list[str] = []
        for row in matches:
            slug = _slug_of(row)
            if not slug:
                continue
            for candidate in (parent_slug(slug), slug):
                if candidate and candidate not in out:
                    out.append(candidate)
        # A slug built from the resolved name is worth trying too (Solana -> solana).
        name_slug = _slug_from_name(asset.name)
        if name_slug and name_slug not in out:
            out.append(name_slug)
        return tuple(out)

    async def _candidates_for_asset(self, asset: ResolvedAsset) -> tuple[str, ...]:
        """Slugs to try: an explicit override first, then everything derivable.

        Within each source the parent of a versioned slug is tried before the version
        itself, because the parent aggregates every version.
        """
        combined: list[str] = []
        if asset.defillama_slug:
            for candidate in (parent_slug(asset.defillama_slug), asset.defillama_slug):
                if candidate and candidate not in combined:
                    combined.append(candidate)
        for slug in await self._candidate_slugs(asset):
            if slug not in combined:
                combined.append(slug)
        return tuple(combined)

    async def _protocol_tvl_for(
        self, slugs: Sequence[str], asset: ResolvedAsset
    ) -> _ProtocolResult | None:
        """Try each candidate slug in order and return the first that resolves."""
        hint = await self._category_hint(asset)
        for slug in slugs:
            result = await self._fetch_protocol(slug, asset, hint)
            if result is not None:
                return result
        return None

    async def _category_hint(self, asset: ResolvedAsset) -> str | None:
        """Category from ``/protocols``; the parent endpoint often omits it."""
        try:
            rows = await self._protocols()
        except Exception:
            return None
        gecko_id = asset.coingecko_id
        symbol = (asset.symbol or "").upper()
        for row in rows:
            if gecko_id and row.get("gecko_id") == gecko_id:
                category = row.get("category")
                if isinstance(category, str) and category:
                    return category
        for row in rows:
            if str(row.get("symbol") or "").upper() == symbol:
                category = row.get("category")
                if isinstance(category, str) and category:
                    return category
        return None

    async def _fetch_protocol(
        self, slug: str, asset: ResolvedAsset, category_hint: str | None = None
    ) -> _ProtocolResult | None:
        url = f"{BASE_URL}/protocol/{slug}"
        try:
            await self.ctx.throttle()
            data = cast("dict[str, Any]", await self.ctx.http.get_json(url))
        except Exception as exc:
            self.ctx.note_missing(
                "onchain", f"DefiLlama /protocol/{slug} failed: {type(exc).__name__}"
            )
            return None

        prov = provenance_for(
            source=f"defillama:/protocol/{slug}",
            capability=Capability.ONCHAIN,
            url=url,
        )
        self.ctx.record("onchain", prov)

        chains = _chain_breakdown(data)

        # Guard against answering with a protocol that is not this asset: reporting
        # another protocol's TVL under this symbol would be a wrong number, not a
        # missing one, so it is surfaced rather than trusted.
        remote_symbol = str(data.get("symbol") or "").upper()
        want_symbol = (asset.symbol or "").upper()
        if remote_symbol and want_symbol and remote_symbol != want_symbol:
            self.ctx.note_missing(
                "onchain.protocol_scope",
                f"DefiLlama slug {slug!r} reports symbol {remote_symbol!r}, "
                f"not {want_symbol!r}; its TVL is not this asset's",
            )

        category = data.get("category")
        return _ProtocolResult(
            name=data.get("name"),
            category=category if isinstance(category, str) and category else category_hint,
            tvl=parse_tvl(data.get("tvl")),
            url=data.get("url"),
            provenance=prov,
            chains=chains,
            slug=slug,
        )

    # -- deep history -----------------------------------------------------------------

    async def fetch_history(
        self, asset: ResolvedAsset, request: HistoryRequest
    ) -> tuple[HistorySeries, ...]:
        """Close-only price history from ``coins.llama.fi/chart``.

        DefiLlama's join key is the CoinGecko id, so this is only attempted for a resolved
        asset; a coin key is never guessed from a bare ticker. It publishes one price per
        period (not OHLC bars), and reaches further back than any free exchange feed
        (verified to 2015 for BTC), which makes it the pre-listing cross-check. It never
        contributes to the consensus, because a close-only series cannot be compared
        bar-for-bar with an OHLC one.
        """
        period = _LLAMA_PERIODS.get(request.timeframe)
        if period is None:
            self.ctx.note_missing(
                f"history.{request.timeframe}",
                f"DefiLlama charts support only {sorted(_LLAMA_PERIODS)}; "
                f"requested {request.timeframe!r}",
            )
            return ()
        coin_id = asset.coingecko_id
        if not coin_id:
            return ()
        key = f"coingecko:{coin_id}"
        interval = interval_seconds(request.timeframe)

        start = as_utc(request.start) if request.start is not None else None
        end = as_utc(request.end) if request.end is not None else None
        points, calls, truncated = await self._chart_walk(
            key, period, interval, start, end, request.limit
        )
        if not points:
            self.ctx.note_missing(
                f"history.{request.timeframe}", f"DefiLlama has no chart for {key}"
            )
            return ()

        candles: list[HistoryCandle] = []
        confidence: float | None = None
        for point in points:
            if not isinstance(point, dict):
                continue
            record = cast("dict[str, Any]", point)
            if confidence is None:
                confidence = finite(record.get("confidence"))
            timestamp = epoch_s_to_utc(record.get("timestamp"))
            price = finite(record.get("price"))
            if timestamp is None or price is None:
                continue
            candles.append(HistoryCandle(timestamp=timestamp, close=price))
        ordered = normalise_candles(candles, request, align=True)

        notes = [f"timeframe={request.timeframe}", "close-only, no OHLC", f"calls={calls}"]
        if confidence is not None:
            notes.append(f"confidence={confidence}")
        prov = provenance_for(
            source=f"defillama:/chart/{key}",
            capability=Capability.HISTORY,
            url=COINS_BASE_URL,
            notes=tuple(notes),
        )
        self.ctx.record(f"history:defillama:{request.timeframe}", prov)
        return (
            HistorySeries(
                source=f"defillama:{coin_id}",
                symbol=asset.symbol,
                timeframe=request.timeframe,
                quote_currency=self.ctx.config.quote_currency,
                kind="price",
                candles=ordered,
                coverage=build_coverage(
                    [c.timestamp for c in ordered],
                    request,
                    calls=calls,
                    truncated=truncated,
                    note=(f"walk stopped after {calls} page(s)" if truncated else None),
                ),
                provenance=prov,
            ),
        )

    async def _chart_walk(
        self,
        key: str,
        period: str,
        interval: int | None,
        start: datetime | None,
        end: datetime | None,
        limit: int | None,
    ) -> tuple[list[Any], int, bool]:
        """Page ``/chart`` forward in ``_MAX_SPAN``-point steps.

        Returns ``(points, calls, truncated)``. When neither bound is given the most
        recent window is returned; a start is walked forward until the end is reached,
        the limit is satisfied, or ``history_max_calls`` is exhausted — in which case the
        caller is told the walk was truncated.
        """
        max_calls = max(1, self.ctx.config.history_max_calls)
        points: list[Any] = []
        seen: set[int] = set()
        calls = 0
        truncated = False
        cursor = int(start.timestamp()) if start is not None else None
        remaining = limit

        while calls < max_calls:
            span = _MAX_SPAN if remaining is None else min(_MAX_SPAN, remaining)
            if span <= 0:
                break
            params: dict[str, Any] = {"period": period, "span": span}
            if cursor is not None:
                params["start"] = cursor
            url = f"{COINS_BASE_URL}/chart/{urlquote(key, safe=':')}?{urlencode(params)}"

            await self.ctx.throttle()
            payload = cast("dict[str, Any]", await self.ctx.http.get_json(url))
            calls += 1
            entry = _llama_coin_entry(payload, key)
            if entry is None:
                break
            page = cast("list[Any]", entry.get("prices") or [])
            newest: int | None = None
            added = 0
            for point in page:
                if not isinstance(point, dict):
                    continue
                raw_ts = cast("dict[str, Any]", point).get("timestamp")
                if not isinstance(raw_ts, (int, float)) or isinstance(raw_ts, bool):
                    continue
                timestamp = int(raw_ts)
                newest = timestamp if newest is None else max(newest, timestamp)
                if timestamp in seen:
                    continue
                seen.add(timestamp)
                points.append(point)
                added += 1

            if cursor is None or newest is None or added == 0:
                # No start: the recent window is all we can serve in one direction.
                break
            # Follow the data stream, not a fixed grid: DefiLlama samples at loose
            # ~daily timestamps, so a fixed anchor skips days the feed actually has, while
            # advancing past the newest returned point stays locked onto what exists.
            cursor = newest + (interval or 86_400)
            if end is not None and cursor > int(end.timestamp()):
                break
            if remaining is not None:
                remaining -= added
        else:
            truncated = True

        points.sort(key=_point_ts)
        return points, calls, truncated


def _as_pairs(payload: dict[Any, Any]) -> list[tuple[str, Any]]:
    """String-keyed view of a JSON object, so callers never see Unknown."""
    entries: list[tuple[Any, Any]] = list(payload.items())
    return [(k, v) for k, v in entries if isinstance(k, str)]


def _chain_breakdown(data: dict[str, Any]) -> tuple[ChainTVL, ...]:
    """Per-chain TVL, preferring the flat ``currentChainTvls`` map.

    ``chainTvls[chain].tvl`` is a history array, so it is only used when the flat map is
    absent. Sub-buckets (``Ethereum-staking``, ``Polygon-borrowed``, …) are excluded.
    """
    flat: dict[str, float] = {}
    current: object = data.get("currentChainTvls")
    if isinstance(current, dict):
        for key, value in _as_pairs(cast("dict[Any, Any]", current)):
            if not is_chain_key(key):
                continue
            parsed = finite(value)
            if parsed is not None:
                flat[key] = parsed
    if not flat:
        chain_tvls: object = data.get("chainTvls")
        if isinstance(chain_tvls, dict):
            for key, payload in _as_pairs(cast("dict[Any, Any]", chain_tvls)):
                if not is_chain_key(key):
                    continue
                record = cast("dict[Any, Any]", payload) if isinstance(payload, dict) else {}
                parsed = parse_tvl(record.get("tvl"))
                if parsed is not None:
                    flat[key] = parsed
    return tuple(
        ChainTVL(chain=key, tvl=value)
        for key, value in sorted(flat.items(), key=lambda kv: kv[1], reverse=True)
    )


@dataclass(frozen=True, slots=True)
class _ChainResult:
    name: str
    tvl: float | None
    provenance: Provenance


@dataclass(frozen=True, slots=True)
class _ProtocolResult:
    name: str | None
    category: str | None
    tvl: float | None
    url: str | None
    provenance: Provenance
    #: Per-chain breakdown, excluding non-chain sub-buckets.
    chains: tuple[ChainTVL, ...]
    #: The slug that actually resolved (may differ from the requested candidate).
    slug: str


def _as_rows(payload: object) -> list[dict[str, Any]]:
    """Coerce a DefiLlama list response, rejecting an unexpected shape loudly."""
    if not isinstance(payload, list):
        raise ProviderError(
            "defillama",
            f"expected a JSON array, got {type(payload).__name__}",
        )
    items = cast("list[object]", payload)
    return [cast("dict[str, Any]", item) for item in items if isinstance(item, dict)]


def _llama_coin_entry(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    """The per-coin object from a ``/chart`` response, or ``None`` when absent."""
    coins = payload.get("coins")
    if not isinstance(coins, dict):
        return None
    entry = cast("dict[str, Any]", coins).get(key)
    return cast("dict[str, Any]", entry) if isinstance(entry, dict) else None


def _point_ts(point: Any) -> int:
    """A ``/chart`` price point's epoch seconds, or ``0`` when unusable."""
    if isinstance(point, dict):
        value = cast("dict[str, Any]", point).get("timestamp")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return int(value)
    return 0


def _pick_provenance(
    protocol: _ProtocolResult | None, chain: _ChainResult | None
) -> Provenance | None:
    """Protocol provenance wins; fall back to the chain row's."""
    if protocol is not None:
        return protocol.provenance
    return chain.provenance if chain is not None else None


#: ``aave-v2`` -> ``aave``; the parent slug aggregates every version.
_VERSION_SUFFIX_RE = re.compile(r"-(?:v\d+(?:\.\d+)?)$")

#: Suffixes in ``currentChainTvls`` that are sub-buckets, not chains.
_CHAIN_SUFFIX_RE = re.compile(r"-(?:staking|pool2|borrowed|vesting|offers|chainlink)$")


def parse_tvl(value: object) -> float | None:
    """Read a TVL value in either of the shapes DefiLlama actually returns.

    ``tvl`` is a scalar on some endpoints and a history array of
    ``{"date": ..., "totalLiquidityUSD": ...}`` points on ``/protocol/{slug}``. The
    current value is the final point. Returning ``None`` rather than 0 when the shape is
    unrecognised keeps "unknown" distinct from "empty".
    """
    direct = finite(value)
    if direct is not None:
        return direct
    if isinstance(value, list):
        points = cast("list[Any]", value)
        for point in reversed(points):
            if not isinstance(point, dict):
                continue
            record = cast("dict[Any, Any]", point)
            for key in ("totalLiquidityUSD", "totalLiquidity", "tvl"):
                parsed = finite(record.get(key))
                if parsed is not None:
                    return parsed
    return None


def parent_slug(slug: str) -> str | None:
    """The parent of a versioned slug (``aave-v2`` -> ``aave``), if any."""
    stripped = _VERSION_SUFFIX_RE.sub("", slug)
    return stripped if stripped != slug else None


def is_chain_key(key: str) -> bool:
    """Whether a ``currentChainTvls`` key names a chain rather than a sub-bucket."""
    return _CHAIN_SUFFIX_RE.search(key) is None


def _slug_from_name(name: str | None) -> str | None:
    """A DefiLlama slug guessed from a resolved asset name, if it looks like one."""
    if not name:
        return None
    slug = name.strip().lower().replace(" ", "-")
    return urlquote(slug) if slug and slug.isascii() else None


def _slug_of(row: dict[str, Any]) -> str | None:
    """The path slug, with a defensive fallback to a name-derived slug."""
    slug = row.get("slug")
    if isinstance(slug, str) and slug.strip():
        return urlquote(slug.strip().lower())
    name = row.get("name")
    if isinstance(name, str) and name.strip():
        return urlquote(name.strip().lower().replace(" ", "-"))
    return None


__all__ = ["BASE_URL", "COINS_BASE_URL", "DefiLlamaProvider"]