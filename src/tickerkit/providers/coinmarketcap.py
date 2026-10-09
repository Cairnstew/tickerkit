"""CoinMarketCap provider: optional second aggregated source for cross-checking.

Docs (read before writing this adapter, per project rule):
  * API reference — https://coinmarketcap.com/api/documentation/pro-api-reference
  * Quick start   — https://coinmarketcap.com/api/documentation/guides/quick-start

Facts this adapter depends on, all from the pages above:
  * Base URL is ``https://pro-api.coinmarketcap.com``.
  * Auth is the custom header ``X-CMC_PRO_API_KEY`` (preferred over a query param).
  * ``GET /v3/cryptocurrency/quotes/latest?id=1,1027&convert=USD`` returns
    ``{"status": {"timestamp", "error_code", "error_message", ...},
    "data": {"id", "name", "symbol", "quote": {"price": {"usd": ...}, ...}}}``.
  * ``GET /v1/cryptocurrency/map`` returns ``{"data": [{"id", "name", "symbol", ...}]}``
    and CMC's own guidance is to key off the numeric CMC id, never the symbol.
  * Every successful request deducts credits; there is no free tier that needs no key,
    so this provider is gated on ``TICKERKIT_CMC_API_KEY`` and is *skipped with a reason*
    when the key is absent.
"""

from __future__ import annotations

from typing import Any, cast

from ..capabilities import Capability
from ..config import Config
from ..models import Market, finite
from ..normalise import parse_iso
from ..resolve import ResolvedAsset, normalise_query
from .base import Provider, provenance_for

BASE_URL = "https://pro-api.coinmarketcap.com"


class CoinMarketCapProvider(Provider):
    name = "coinmarketcap"
    supports = frozenset({Capability.RESOLVE, Capability.META, Capability.AGG_MARKET})

    @classmethod
    def available(cls, config: Config) -> tuple[bool, str | None]:
        if not config.cmc_api_key:
            return False, (
                "no API key: set TICKERKIT_CMC_API_KEY (CoinMarketCap has no "
                "keyless tier, so this optional cross-check source is disabled)"
            )
        return True, None

    @property
    def _headers(self) -> dict[str, str]:
        key = self.ctx.config.cmc_api_key or ""
        return {"X-CMC_PRO_API_KEY": key, "Accept": "application/json"}

    async def _get(self, path: str, params: dict[str, Any]) -> tuple[dict[str, Any], str]:
        from urllib.parse import urlencode

        url = f"{BASE_URL}{path}?{urlencode(params)}"
        await self.ctx.throttle()
        data = cast(dict[str, Any], await self.ctx.http.get_json(url, headers=self._headers))
        return data, url

    async def _cmc_id_for(self, asset: ResolvedAsset) -> int | None:
        """Resolve the CMC numeric id, preferring an explicit symbol match."""
        symbol = (asset.symbol or normalise_query(asset.query)).upper()
        cache_key = f"cmc:symbol:{symbol}"
        cached, hit = await self.ctx.cached(cache_key, 3600.0)
        if hit:
            return cast(int | None, cached)

        data, _url = await self._get(
            "/v1/cryptocurrency/map", {"symbol": symbol, "limit": 5000, "start": 1}
        )
        rows = cast(list[dict[str, Any]], data.get("data") or [])
        matches = [r for r in rows if str(r.get("symbol") or "").upper() == symbol]
        if not matches:
            await self.ctx.remember(cache_key, None, 600.0)
            return None
        # Prefer the lowest numeric id: CMC ids roughly follow listing seniority.
        matches.sort(key=lambda r: int(r.get("id") or 0))
        cmc_id = int(matches[0]["id"])
        await self.ctx.remember(cache_key, cmc_id, 3600.0)
        return cmc_id

    async def fetch_market(self, asset: ResolvedAsset) -> Market | None:
        cmc_id = await self._cmc_id_for(asset)
        if cmc_id is None:
            self.ctx.note_missing("market.cmc", f"no CoinMarketCap id for symbol {asset.symbol}")
            return None

        convert = self.ctx.config.quote_currency.lower()
        data, url = await self._get(
            "/v3/cryptocurrency/quotes/latest", {"id": str(cmc_id), "convert": convert}
        )
        entries = cast(list[dict[str, Any]], data.get("data") or [])
        if not entries:
            self.ctx.note_missing("market.cmc", f"CoinMarketCap returned no quote for id={cmc_id}")
            return None

        row = entries[0]
        quote = cast(dict[str, Any], row.get("quote") or {})
        price_block = _quote_block(quote, convert)
        status = cast(dict[str, Any], data.get("status") or {})

        prov = provenance_for(
            source="coinmarketcap:/v3/cryptocurrency/quotes/latest",
            capability=Capability.AGG_MARKET,
            exchange_timestamp=parse_iso(status.get("timestamp")),
            url=url,
        )
        self.ctx.record("market:cmc", prov)

        return Market(
            quote_currency=self.ctx.config.quote_currency,
            aggregated_price=finite(price_block.get("price")),
            change_24h=finite(price_block.get("price_change_24h")),
            change_24h_pct=finite(price_block.get("percent_change_24h")),
            high_24h=finite(price_block.get("high_24h")),
            low_24h=finite(price_block.get("low_24h")),
            volume_24h=finite(price_block.get("volume_24h")),
            market_cap=finite(price_block.get("market_cap")),
            fdv=finite(price_block.get("fully_diluted_valuation")),
            circulating_supply=finite(price_block.get("circulating_supply")),
            total_supply=finite(price_block.get("total_supply")),
            max_supply=finite(price_block.get("max_supply")),
            ath=finite(price_block.get("ath")),
            ath_date=parse_iso(price_block.get("ath_date")),
            ath_change_pct=finite(price_block.get("ath_change_percentage")),
            atl=finite(price_block.get("atl")),
            atl_date=parse_iso(price_block.get("atl_date")),
            atl_change_pct=finite(price_block.get("atl_change_percentage")),
            provenance={"cmc": prov},
        )


def _quote_block(quote: dict[str, Any], convert: str) -> dict[str, Any]:
    """Return the per-currency quote block.

    CMC keys its quote blocks by currency code, e.g. ``quote.USD.price``, and the code is
    case-insensitive on the way in. Some responses instead inline the fields and nest the
    currency under each numeric field (``quote.price.usd``); both shapes are accepted so
    a formatting change upstream degrades to ``None`` rather than a wrong number.
    """
    lowered = convert.lower()
    for key, value in quote.items():
        if key.lower() == lowered and isinstance(value, dict):
            return cast(dict[str, Any], value)
    if "price" in quote or "volume_24h" in quote:
        return quote
    return {}


__all__ = ["BASE_URL", "CoinMarketCapProvider"]