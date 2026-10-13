"""CoinGecko provider: asset metadata + aggregated cross-venue market data.

Docs (read before writing this adapter, per project rule):
  * Coins List with Market Data — https://docs.coingecko.com/reference/coins-markets
  * Coin Data by ID             — https://docs.coingecko.com/demo/reference/coins-id
  * Authentication (Demo API)   — https://docs.coingecko.com/demo/reference/authentication
  * Errors & Rate Limits        — https://docs.coingecko.com/docs/errors-and-rate-limits

Facts this adapter depends on, all taken from the pages above:
  * Demo / keyless root URL is ``https://api.coingecko.com/api/v3``; a Pro key requires
    ``https://pro-api.coingecko.com/api/v3`` (error 10010/10011 otherwise).
  * Auth headers: ``x-cg-demo-api-key`` (Demo) and ``x-cg-pro-api-key`` (Pro).
  * Demo plan allows **100 calls/min**; keyless is IP-based. All requests, including
    4xx/5xx, count against the limit — hence the limiter in ``http.py``.
  * ``/coins/markets`` accepts ``vs_currency``, ``ids``, ``names``, ``symbols``,
    ``include_tokens`` (``top`` default, ``all`` for every token sharing a symbol),
    ``per_page`` (max 250), ``price_change_percentage``. Priority when several filters
    are combined is category > ids > names > symbols.
  * ``include_tokens`` only works with ``symbols`` lookups, max 50 symbols per request.
  * ``/coins/markets`` is cached server-side every 60s on Demo/keyless (30s paid).
  * ``/coins/{id}`` returns ``description`` as a language-keyed object, ``links`` with
    ``homepage[]``/``whitepaper``(str)/``blockchain_site[]``/``twitter_screen_name``/
    ``subreddit_url``/``telegram_channel_identifier``/``repos_url.github[]``, ``platforms``
    mapping platform id -> contract address, and ``market_data`` with all the
    per-currency objects plus scalar supply/ATH/ATL fields.
  * ``community_data``/``developer_data`` are no longer returned (28 Aug 2026) and are
    deliberately not requested.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any, cast
from urllib.parse import urlencode

from ..capabilities import Capability
from ..config import Config
from ..errors import ProviderError
from ..history import build_coverage, interval_seconds, normalise_candles
from ..models import (
    HistoryCandle,
    HistoryRequest,
    HistorySeries,
    Identity,
    IdentityLinks,
    Market,
    as_utc,
    finite,
    utcnow,
)
from ..normalise import epoch_ms_to_utc, first_link, link_tuple, parse_iso, strip_html
from ..resolve import ResolvedAsset, normalise_query, pick_best_candidate
from .base import Provider, provenance_for

DEMO_ROOT = "https://api.coingecko.com/api/v3"
PRO_ROOT = "https://pro-api.coingecko.com/api/v3"

#: Demo plan limit from the rate-limit doc ("Demo plan: 100 calls/min").
DEMO_RATE_LIMIT_RPM = 100

#: Conservative default for keyless access. The docs state keyless is "IP-based rate
#: limiting - shared across all users on the same IP", which is far tighter than the Demo
#: plan's 100/min in practice, so tickerkit throttles harder until a key is supplied.
#: 12 rpm rather than the previous 25: the bucket is *shared*, so another consumer on
#: the same IP (a second process, a neighbouring NAT user) draws from it too, and a
#: measured 15-ticker sweep at 25 rpm tripped a genuine 429 mid-run. Halving leaves
#: headroom for a co-existing consumer instead of spending the whole IP budget.
KEYLESS_RATE_LIMIT_RPM = 12


class CoinGeckoProvider(Provider):
    name = "coingecko"
    supports = frozenset(
        {Capability.RESOLVE, Capability.META, Capability.AGG_MARKET, Capability.HISTORY}
    )

    def __init__(self, context: Any) -> None:
        super().__init__(context)
        self._root = PRO_ROOT if self.ctx.config.coingecko_pro_api_key else DEMO_ROOT
        self._headers = self._auth_headers(self.ctx.config)

    @staticmethod
    def _auth_headers(config: Config) -> dict[str, str]:
        if config.coingecko_pro_api_key:
            return {"x-cg-pro-api-key": config.coingecko_pro_api_key}
        if config.coingecko_api_key:
            return {"x-cg-demo-api-key": config.coingecko_api_key}
        # Keyless is documented as working, but IP-rate-limited and shared.
        return {}

    @classmethod
    def rate_limit_rpm(cls, config: Config) -> int:
        """Requests/minute this provider should be throttled to."""
        if config.coingecko_pro_api_key or config.coingecko_api_key:
            return DEMO_RATE_LIMIT_RPM
        return KEYLESS_RATE_LIMIT_RPM

    @property
    def root(self) -> str:
        return self._root

    def _url(self, path: str, params: dict[str, Any] | None = None) -> str:
        url = f"{self._root}{path}"
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url = f"{url}?{urlencode(clean)}"
        return url

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> tuple[Any, str]:
        url = self._url(path, params)
        # CoinGecko counts *every* request, including 4xx/5xx, so throttle before each.
        await self.ctx.throttle()
        payload: Any = await self.ctx.http.get_json(url, headers=self._headers or None)
        self._reject_error_envelope(path, payload)
        return payload, url

    def _reject_error_envelope(self, path: str, payload: Any) -> None:
        """Turn a 200-with-error-body response into a clear, recorded failure.

        CoinGecko spells the same failure two ways: ``/coins/markets`` answers
        ``{"status": {"error_code": ...}}`` while the chart endpoints answer
        ``{"error": {"status": {...}}}``. Both are 200 OK, so a plain
        ``raise_for_status`` never sees them; handing either to the field readers produces
        garbage or an obscure ``AttributeError`` deep in an adapter. Both shapes are
        rejected here and surface as a normal ``ProviderError``.
        """
        if not isinstance(payload, dict):
            return
        envelope = cast("dict[str, Any]", payload)
        status = envelope.get("status")
        if status is None:
            wrapped = envelope.get("error")
            if isinstance(wrapped, dict):
                status = cast("dict[str, Any]", wrapped).get("status")
        if isinstance(status, dict):
            detail = cast("dict[str, Any]", status)
            raise ProviderError(
                self.name,
                f"CoinGecko returned an error envelope for {path} "
                f"(error_code={detail.get('error_code')}): {detail.get('error_message')}",
            )

    # -- resolve ----------------------------------------------------------------------

    async def resolve(self, query: str) -> ResolvedAsset | None:
        """Resolve a symbol (or a ``SYMBOL/QUOTE`` pair) to a CoinGecko id.

        Highest market cap wins; every other match is returned as an alternative.
        """
        symbol = normalise_query(query)
        path = "/coins/markets"
        params: dict[str, Any] = {
            "vs_currency": self.ctx.config.quote_currency.lower(),
            "symbols": symbol.lower(),
            "include_tokens": "all",
            "per_page": 100,
            "order": "market_cap_desc",
            "price_change_percentage": "24h",
        }
        cache_key = f"coingecko:resolve:{symbol}:{self.ctx.config.quote_currency}"
        cached, hit = await self.ctx.cached(cache_key, self.ctx.config.cache_ttl)
        url = self._url(path, params)
        if hit:
            rows = cast(list[dict[str, Any]], cached)
        else:
            data, url = await self._get(path, params)
            rows = cast(list[dict[str, Any]], data or [])
            await self.ctx.remember(cache_key, rows)

        best, alternatives = pick_best_candidate(rows)
        if best is None:
            self.ctx.note_missing(
                "identity.coingecko_id",
                f"CoinGecko has no coin matching symbol {symbol!r}",
            )
            return None

        prov = provenance_for(
            source="coingecko:/coins/markets",
            capability=Capability.RESOLVE,
            url=url,
            cached=hit,
            notes=(f"symbols={symbol.lower()} include_tokens=all",),
        )
        self.ctx.record("resolve", prov)
        self.ctx.ambiguities.extend(alternatives)

        # These rows already carry the market fields; reuse them below instead of
        # issuing a second /coins/markets request and burning free-tier quota.
        resolved = [row for row in rows if str(row.get("id")) == str(best["id"])]
        if resolved:
            await self.ctx.remember(
                f"coingecko:rows:{self.ctx.config.quote_currency}:{best['id']}", resolved
            )

        return ResolvedAsset(
            query=query,
            symbol=symbol,
            coingecko_id=str(best["id"]),
            name=best.get("name"),
            market_cap=finite(best.get("market_cap")),
            ambiguities=tuple(alternatives),
            unresolved=False,
            notes=(f"resolved via coingecko {url}",),
        )

    # -- meta -------------------------------------------------------------------------

    async def fetch_identity(self, asset: ResolvedAsset) -> Identity | None:
        coin_id = asset.coingecko_id
        if not coin_id:
            return None

        path = f"/coins/{coin_id}"
        params = {
            "localization": "false",
            "tickers": "false",
            "market_data": "false",
            "sparkline": "false",
        }
        cache_key = f"coingecko:meta:{coin_id}"
        cached, hit = await self.ctx.cached(cache_key, self.ctx.config.cache_ttl * 5)
        if hit:
            data, url = cast(dict[str, Any], cached), None
        else:
            data, url = await self._get(path, params)
            await self.ctx.remember(cache_key, data, self.ctx.config.cache_ttl * 5)

        links_raw = cast(dict[str, Any], data.get("links") or {})
        repos = cast(dict[str, Any], links_raw.get("repos_url") or {})
        description = cast(dict[str, Any], data.get("description") or {}).get("en")

        prov = provenance_for(
            source="coingecko:/coins/{id}",
            capability=Capability.META,
            exchange_timestamp=parse_iso(data.get("last_updated")),
            url=url,
            cached=hit,
        )
        self.ctx.record("identity", prov)

        platforms = cast(dict[str, Any], data.get("platforms") or {})
        contract = asset.contract_address
        platform_id = asset.asset_platform_id
        if contract is None and platform_id and platform_id in platforms:
            contract = platforms.get(platform_id) or None
        if contract is None:
            # Fall back to the first non-empty platform entry.
            for key, value in platforms.items():
                if key and isinstance(value, str) and value:
                    contract, platform_id = value, key
                    break

        return Identity(
            symbol=str(data.get("symbol") or asset.symbol).upper(),
            name=data.get("name"),
            coingecko_id=str(data.get("id") or coin_id),
            categories=tuple(c for c in cast(list[str], data.get("categories") or []) if c),
            links=IdentityLinks(
                homepage=link_tuple(links_raw.get("homepage")),
                blockchain=link_tuple(links_raw.get("blockchain_site")),
                whitepaper=link_tuple(links_raw.get("whitepaper")),
                twitter=first_link(links_raw.get("twitter_screen_name")),
                reddit=first_link(links_raw.get("subreddit_url")),
                github=link_tuple(repos.get("github")),
                telegram=first_link(links_raw.get("telegram_channel_identifier")),
                other=link_tuple(links_raw.get("official_forum_url"))
                + link_tuple(links_raw.get("announcement_url")),
            ),
            description=strip_html(description),
            contract_address=contract,
            asset_platform_id=platform_id,
            provenance=prov,
        )

    # -- aggregated market ------------------------------------------------------------

    async def fetch_market(self, asset: ResolvedAsset) -> Market | None:
        coin_id = asset.coingecko_id
        if not coin_id:
            return None

        quote = self.ctx.config.quote_currency.lower()
        path = "/coins/markets"
        params = {
            "vs_currency": quote,
            "ids": coin_id,
            "price_change_percentage": "24h",
            "precision": "full",
        }
        # Prefer rows already fetched during resolution.
        rows_key = f"coingecko:rows:{self.ctx.config.quote_currency}:{coin_id}"
        preloaded, preloaded_hit = await self.ctx.cached(rows_key, self.ctx.config.cache_ttl)
        cache_key = f"coingecko:market:{coin_id}:{quote}"
        cached, hit = await self.ctx.cached(cache_key, self.ctx.config.cache_ttl)
        url: str | None = None
        if preloaded_hit:
            rows = cast(list[dict[str, Any]], preloaded)
            hit = False
        elif hit:
            rows = cast(list[dict[str, Any]], cached)
        else:
            data, url = await self._get(path, params)
            rows = cast(list[dict[str, Any]], data or [])
            await self.ctx.remember(cache_key, rows)
            if not rows:
                self.ctx.note_missing(
                    "market",
                    f"CoinGecko /coins/markets returned no row for id={coin_id}",
                )
                return None

        row = rows[0]
        prov = provenance_for(
            source="coingecko:/coins/markets",
            capability=Capability.AGG_MARKET,
            exchange_timestamp=parse_iso(row.get("last_updated")),
            url=url,
            cached=hit,
            notes=(f"vs_currency={quote}",),
        )
        self.ctx.record("market:aggregated", prov)

        market = Market(
            quote_currency=self.ctx.config.quote_currency,
            aggregated_price=finite(row.get("current_price")),
            change_24h=finite(row.get("price_change_24h")),
            change_24h_pct=finite(row.get("price_change_percentage_24h")),
            high_24h=finite(row.get("high_24h")),
            low_24h=finite(row.get("low_24h")),
            volume_24h=finite(row.get("total_volume")),
            market_cap=finite(row.get("market_cap")),
            fdv=finite(row.get("fully_diluted_valuation")),
            circulating_supply=finite(row.get("circulating_supply")),
            total_supply=finite(row.get("total_supply")),
            max_supply=finite(row.get("max_supply")),
            ath=finite(row.get("ath")),
            ath_date=parse_iso(row.get("ath_date")),
            ath_change_pct=finite(row.get("ath_change_percentage")),
            atl=finite(row.get("atl")),
            atl_date=parse_iso(row.get("atl_date")),
            atl_change_pct=finite(row.get("atl_change_percentage")),
            provenance={"aggregated": prov},
        )
        for name in ("aggregated_price", "market_cap", "volume_24h", "ath", "atl"):
            if getattr(market, name) is None:
                self.ctx.note_missing(
                    f"market.{name}",
                    f"CoinGecko returned null for {name} (asset may lack this metric)",
                )
        return market

    # -- deep history -----------------------------------------------------------------

    async def fetch_history(
        self, asset: ResolvedAsset, request: HistoryRequest
    ) -> tuple[HistorySeries, ...]:
        """Close-only history from ``/coins/{id}/market_chart/range``.

        CoinGecko publishes a single price per interval, not OHLC bars, so this adapter
        returns ``kind="price"`` and never invents the open/high/low. It is a cross-check
        source, not a consensus input.

        Keyless and Demo plans are hard-limited to the **past 365 days** (error 10012);
        a request reaching further back is clamped and reported as truncated rather than
        failing, so the exchange series still reconcile against what is available.
        """
        coin_id = asset.coingecko_id
        if not coin_id:
            return ()
        interval = interval_seconds(request.timeframe)
        now = utcnow()
        end = as_utc(request.end) if request.end is not None else now
        if request.start is not None:
            start = as_utc(request.start)
        elif request.limit is not None and interval is not None:
            start = end - timedelta(seconds=interval * request.limit)
        else:
            start = end - timedelta(days=365)

        truncated = False
        note: str | None = None
        if not self.ctx.config.coingecko_pro_api_key:
            floor = now - timedelta(days=365)
            if start < floor:
                start = floor
                truncated = True
                note = (
                    "CoinGecko public API serves only the past 365 days; the requested "
                    "start was clamped (a Pro key lifts this)"
                )
        if start > end:
            # The whole requested window predates the free window, so there is nothing to
            # serve. Returning an empty series with a reason beats issuing a request whose
            # `from` is later than its `to`, which the API rejects as a range error.
            self.ctx.note_missing(
                f"history.{request.timeframe}",
                f"CoinGecko public API cannot serve {start.date()} .. {end.date()}: "
                "older than the past 365 days (a Pro key lifts this)",
            )
            return ()

        path = f"/coins/{coin_id}/market_chart/range"
        params = {
            "vs_currency": self.ctx.config.quote_currency.lower(),
            "from": int(start.timestamp()),
            "to": int(end.timestamp()),
        }
        payload, url = await self._get(path, params)
        data = cast("dict[str, Any]", payload)
        prices = cast("list[Any]", data.get("prices") or [])

        candles: list[HistoryCandle] = []
        for entry in prices:
            if not isinstance(entry, (list, tuple)):
                continue
            pair = cast("tuple[Any, ...]", entry)
            if len(pair) < 2:
                continue
            timestamp = epoch_ms_to_utc(pair[0])
            price = finite(pair[1])
            if timestamp is None or price is None:
                continue
            # total_volumes is a rolling 24h figure, not a per-interval volume, so it is
            # deliberately not attached here: it would be a number of the wrong kind.
            candles.append(HistoryCandle(timestamp=timestamp, close=price))

        ordered = normalise_candles(candles, request, align=True)
        prov = provenance_for(
            source=f"coingecko:{path}",
            capability=Capability.HISTORY,
            url=url,
            notes=(f"timeframe={request.timeframe}", "close-only, no per-interval volume"),
        )
        self.ctx.record(f"history:coingecko:{request.timeframe}", prov)
        return (
            HistorySeries(
                source=f"coingecko:{coin_id}",
                symbol=asset.symbol,
                timeframe=request.timeframe,
                quote_currency=self.ctx.config.quote_currency,
                kind="price",
                candles=ordered,
                coverage=build_coverage(
                    [c.timestamp for c in ordered],
                    request,
                    calls=1,
                    truncated=truncated,
                    note=note,
                ),
                provenance=prov,
            ),
        )


def provenance_note() -> str:
    """Exposed so the README/tests can reference the citation without duplication."""
    return "https://docs.coingecko.com/reference/coins-markets"


__all__ = ["DEMO_RATE_LIMIT_RPM", "DEMO_ROOT", "PRO_ROOT", "CoinGeckoProvider"]