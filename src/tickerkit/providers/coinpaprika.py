"""CoinPaprika provider: a keyless second source for aggregated market data.

CoinPaprika exists here for one reason: CoinGecko's keyless tier is IP-rate-limited and
this one is not, so a throttled CoinGecko degrades the snapshot instead of emptying it.
It is a genuine second opinion on price and market cap, not a fallback bolted on: both
sources contribute a :class:`~tickerkit.models.Market` and the reconcile layer compares
them.

Docs (read before writing this adapter, per project rule):
  * API overview & tiers — https://coinpaprika.com/api/
  * Ticker endpoint      — https://api.coinpaprika.com/v1/tickers/{coin_id}
  * Coin detail          — https://api.coinpaprika.com/v1/coins/{coin_id}
  * Global metrics       — https://api.coinpaprika.com/v1/global
  * Search               — https://api.coinpaprika.com/v1/search/

Facts this adapter depends on, all observed against the live keyless API:

  * Base URL is ``https://api.coinpaprika.com/v1`` and requires **no API key**. Paid
    plans raise the quota; the public tier is advertised at 20,000 calls/month.
  * Every response carries ``ratelimit-limit`` / ``ratelimit-remaining`` /
    ``ratelimit-reset`` headers. ``ratelimit-reset`` is a **count of seconds remaining
    in the monthly window** (observed ~1.9M seconds), *not* a short cooldown and *not* an
    epoch — a client that treats it as an epoch-computed wait will be wrong by orders of
    magnitude, so it is deliberately not read as a per-request delay.
  * Coin ids are ``{symbol}-{slug}`` (``btc-bitcoin``), which is *not* CoinGecko's id.
    Resolution therefore needs its own lookup; it is cached for a long TTL because the
    mapping barely moves and the endpoint that yields it is expensive.
  * ``/coins`` is a ~7.5 MB payload of 62k coins. Fetching it to resolve one symbol would
    burn a large slice of a monthly budget, so ``/search/?q=&c=currencies`` is used
    instead — note the **trailing slash**, without which the API answers 301.
  * ``/tickers/{id}`` returns ``{"id","name","symbol","rank","total_supply",
    "max_supply","quotes":{"USD":{...}}}``. The quote object carries price, volume_24h,
    market_cap, percent_change_* and ath_price/ath_date/ath_date_timestamp/ath_change_percentage,
    but **no** ath_change_percentage under that exact key and **no** circulating supply.
  * An unknown id answers ``404 {"error": "id not found"}`` — a clean miss, not a throttle.
  * ``/coins/{id}`` returns ``links`` as a dict of ``{"key": [url, ...]}`` whose keys are
    observed to be ``website`` (not "homepage"), ``source_code`` (not "github"),
    ``explorer``, ``reddit``, ``facebook``, ``youtube`` and occasionally ``twitter``/
    ``whitepaper``/``telegram``. ``links_extended`` is a **list** of
    ``{"url", "type"}`` objects, not a dict — notably the blog is only in there.
    ``description`` is plain text and ``tags`` is a list of ``{"id","name"}``.
  * ``/global`` is a single small object; it is the cheapest possible keyless sanity check
    that the provider is reachable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from ..capabilities import Capability
from ..config import Config
from ..models import (
    Identity,
    IdentityLinks,
    Market,
    finite,
)
from ..normalise import first_link, link_tuple, parse_iso
from ..resolve import ResolvedAsset, normalise_query
from .base import Provider, provenance_for

BASE_URL = "https://api.coinpaprika.com/v1"

#: The public tier's advertised allowance (https://coinpaprika.com/api/).
MONTHLY_CALL_BUDGET = 20_000

#: Requests/minute tickerkit self-throttles to. The real ceiling is a *monthly* budget,
#: not a per-minute one, so this is deliberately gentle rather than an attempt to spend
#: the quota quickly: it keeps a batch sweep from exhausting the month in a minute.
SELF_LIMIT_RPM = 30

#: Symbol -> id resolution changes rarely, so it is cached far longer than market data.
RESOLVE_TTL_SECONDS = 24 * 3600.0


def _blogs(value: Any) -> tuple[str, ...]:
    """Pull ``{"type": "blog"}`` URLs out of the ``links_extended`` list."""
    if not isinstance(value, list):
        return ()
    out: list[str] = []
    for item in cast("list[Any]", value):
        if not isinstance(item, dict):
            continue
        entry = cast("dict[str, Any]", item)
        url = entry.get("url")
        if entry.get("type") == "blog" and isinstance(url, str) and url:
            out.append(url)
    return tuple(out)


def parse_epoch(value: Any) -> datetime | None:
    """CoinPaprika returns some timestamps as integer unix seconds, not ISO strings."""
    if isinstance(value, int | float) and not isinstance(value, bool) and value > 0:
        return datetime.fromtimestamp(value, tz=UTC)
    return None


class CoinPaprikaProvider(Provider):
    name = "coinpaprika"
    supports = frozenset({Capability.RESOLVE, Capability.META, Capability.AGG_MARKET})

    @classmethod
    def rate_limit_rpm(cls, config: Config) -> int:
        """Requests/minute this provider should be throttled to."""
        return SELF_LIMIT_RPM

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> tuple[Any, str]:
        from urllib.parse import urlencode

        url = f"{BASE_URL}{path}"
        if params:
            clean = {k: v for k, v in params.items() if v is not None}
            if clean:
                url = f"{url}?{urlencode(clean)}"
        await self.ctx.throttle()
        payload: Any = await self.ctx.http.get_json(url)
        return payload, url

    # -- resolve ----------------------------------------------------------------------

    async def _search_ids(self, symbol: str) -> list[dict[str, Any]]:
        """All coins whose symbol matches, best-first by rank.

        Uses ``/search/`` rather than ``/coins``: the full coin list is ~7.5 MB and the
        quota is monthly. The trailing slash is required — without it the API 301s.
        """
        cache_key = f"coinpaprika:search:{symbol}"
        cached, hit = await self.ctx.cached(cache_key, RESOLVE_TTL_SECONDS)
        if hit:
            return cast("list[dict[str, Any]]", cached)

        data, _url = await self._get(
            "/search/", {"q": symbol, "c": "currencies", "limit": 100, "modifier": "symbol_search"}
        )
        envelope = cast("dict[str, Any]", data) if isinstance(data, dict) else {}
        rows = cast("list[dict[str, Any]]", envelope.get("currencies") or [])
        matches = [
            r for r in rows if str(r.get("symbol") or "").upper() == symbol and r.get("id")
        ]
        # Rank 1 is the highest market cap; unranked rows sort last rather than first.
        matches.sort(key=lambda r: (0, int(r["rank"])) if r.get("rank") else (1, 0))
        await self.ctx.remember(cache_key, matches, RESOLVE_TTL_SECONDS)
        return matches

    async def resolve(self, query: str) -> ResolvedAsset | None:
        """Resolve a symbol to a CoinPaprika id (``btc-bitcoin``).

        This does not and cannot produce a CoinGecko id — the two id schemes are
        unrelated — so the contribution is recorded as a ``coingecko_id``-free asset.
        CoinGecko remains the primary resolver; CoinPaprika fills in when it throttles.
        """
        symbol = normalise_query(query)
        try:
            matches = await self._search_ids(symbol)
        except Exception as exc:
            self.ctx.note_error(self.name, Capability.RESOLVE, exc)
            return None

        if not matches:
            self.ctx.note_missing(
                "identity.coinpaprika_id",
                f"CoinPaprika has no coin matching symbol {symbol!r}",
            )
            return None

        best = matches[0]
        # Deliberately NOT pushed onto ctx.ambiguities: Ambiguity.coingecko_id is a
        # CoinGecko-namespace field, and stuffing "btc-buy-the-cat" into it would assert
        # a CoinGecko id that does not exist. The alternatives are recorded as notes,
        # in their own namespace, where they cannot mislead.
        rivals = tuple(
            f"{row['id']} ({row.get('name')})" for row in matches[1:6]
        )
        if rivals:
            self.ctx.note_missing(
                "identity.coinpaprika_id",
                f"{len(matches)} coins share ticker {symbol}; picked rank "
                f"{best.get('rank')} {best['id']}. Also matched: " + ", ".join(rivals),
            )

        url = f"{BASE_URL}/search/?q={symbol}&c=currencies"
        self.ctx.record(
            "resolve:coinpaprika",
            provenance_for(
                source="coinpaprika:/search/",
                capability=Capability.RESOLVE,
                url=url,
                notes=(f"q={symbol} modifier=symbol_search",),
            ),
        )
        return ResolvedAsset(
            query=query,
            symbol=symbol,
            name=best.get("name"),
            unresolved=False,
            notes=(f"resolved via coinpaprika {best['id']}",),
        )

    # -- identity lookup ---------------------------------------------------------------

    async def _paprika_id(self, asset: ResolvedAsset) -> str | None:
        """The CoinPaprika id for this asset, resolved from its symbol."""
        symbol = (asset.symbol or normalise_query(asset.query)).upper()
        if not symbol:
            return None
        matches = await self._search_ids(symbol)
        return str(matches[0]["id"]) if matches else None

    async def fetch_identity(self, asset: ResolvedAsset) -> Identity | None:
        coin_id = await self._paprika_id(asset)
        if not coin_id:
            return None

        cache_key = f"coinpaprika:meta:{coin_id}"
        cached, hit = await self.ctx.cached(cache_key, self.ctx.config.cache_ttl * 10)
        if hit:
            data, url = cast("dict[str, Any]", cached), None
        else:
            data, url = await self._get(f"/coins/{coin_id}")
            await self.ctx.remember(cache_key, data, self.ctx.config.cache_ttl * 10)

        links = cast("dict[str, Any]", data.get("links") or {})
        blog = _blogs(data.get("links_extended"))
        prov = provenance_for(
            source="coinpaprika:/coins/{id}",
            capability=Capability.META,
            exchange_timestamp=parse_iso(data.get("last_data_at")),
            url=url,
            cached=hit,
        )
        self.ctx.record("identity:coinpaprika", prov)

        return Identity(
            symbol=str(data.get("symbol") or asset.symbol).upper(),
            name=data.get("name"),
            coingecko_id=asset.coingecko_id,
            categories=tuple(
                str(t.get("name"))
                for t in cast("list[dict[str, Any]]", data.get("tags") or [])
                if t.get("name")
            )[:12],
            links=IdentityLinks(
                homepage=link_tuple(links.get("website")),
                blockchain=link_tuple(links.get("explorer")),
                whitepaper=link_tuple(links.get("whitepaper")),
                twitter=first_link(links.get("twitter")),
                reddit=first_link(links.get("reddit")),
                github=link_tuple(links.get("source_code")),
                telegram=first_link(links.get("telegram")),
                other=blog + link_tuple(links.get("youtube")) + link_tuple(links.get("facebook")),
            ),
            description=data.get("description") or None,
            contract_address=asset.contract_address,
            asset_platform_id=asset.asset_platform_id,
            provenance=prov,
        )

    # -- aggregated market --------------------------------------------------------------

    async def fetch_market(self, asset: ResolvedAsset) -> Market | None:
        coin_id = await self._paprika_id(asset)
        if not coin_id:
            self.ctx.note_missing(
                "market.coinpaprika",
                "no CoinPaprika id could be resolved for this symbol",
            )
            return None

        quote = self.ctx.config.quote_currency.lower()
        cache_key = f"coinpaprika:market:{coin_id}:{quote}"
        cached, hit = await self.ctx.cached(cache_key, self.ctx.config.cache_ttl)
        url: str | None = None
        if hit:
            data = cast("dict[str, Any]", cached)
        else:
            data, url = await self._get(f"/tickers/{coin_id}", {"quotes": quote})
            await self.ctx.remember(cache_key, data)

        quotes = cast("dict[str, Any]", data.get("quotes") or {})
        # CoinPaprika keys quotes by *uppercase* currency; a non-USD quote request can
        # still come back under "USD", so fall back rather than reporting nothing.
        q = cast("dict[str, Any]", quotes.get(quote.upper()) or quotes.get("USD") or {})
        if not q:
            self.ctx.note_missing(
                "market.coinpaprika",
                f"CoinPaprika returned no {quote.upper()} quote block for {coin_id}",
            )
            return None

        prov = provenance_for(
            source="coinpaprika:/tickers/{id}",
            capability=Capability.AGG_MARKET,
            exchange_timestamp=parse_epoch(data.get("last_updated")),
            url=url,
            cached=hit,
            notes=(f"quotes={quote.upper()}", f"paprika_id={coin_id}"),
        )
        self.ctx.record("market:coinpaprika", prov)

        market = Market(
            quote_currency=self.ctx.config.quote_currency,
            aggregated_price=finite(q.get("price")),
            change_24h_pct=finite(q.get("percent_change_24h")),
            # No circulating supply on this endpoint, so the ratio cannot be derived.
            total_supply=finite(data.get("total_supply")),
            max_supply=finite(data.get("max_supply")),
            volume_24h=finite(q.get("volume_24h")),
            market_cap=finite(q.get("market_cap")),
            ath=finite(q.get("ath_price")),
            ath_date=parse_iso(q.get("ath_date")) or parse_epoch(q.get("ath_date_timestamp")),
            ath_change_pct=finite(q.get("ath_change_percentage")),
            provenance={"coinpaprika": prov},
        )

        # Anything this endpoint genuinely lacks is stated rather than left to look
        # like a zero: there is no FDV, no circulating supply, and no 24h high/low.
        for name in ("circulating_supply", "fdv", "high_24h", "low_24h"):
            if getattr(market, name) is None:
                self.ctx.note_missing(
                    f"market.coinpaprika.{name}",
                    "CoinPaprika /tickers does not publish this field; "
                    "the value comes from another source",
                )
        return market


__all__ = ["BASE_URL", "MONTHLY_CALL_BUDGET", "SELF_LIMIT_RPM", "CoinPaprikaProvider"]