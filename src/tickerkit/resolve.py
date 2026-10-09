"""Symbol resolution.

Turns a user-supplied ticker into a :class:`ResolvedAsset`: a CoinGecko id, per-exchange
market symbols, and (when applicable) a DefiLlama slug.

Ambiguity is the normal case in crypto — dozens of tokens share a symbol. The rule is:
highest market cap wins, and every other plausible match is recorded as an
:class:`~tickerkit.models.Ambiguity` so the caller can see what else it might have meant.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from .config import Config
from .models import Ambiguity, finite

#: Query forms we understand, all of which end up as a bare base symbol.
_SEPARATORS = ("/", "-", "_", ":")


def normalise_query(query: str) -> str:
    """Reduce ``BTC/USDT``, ``btc-usd``, ``BTC`` alike to the base symbol ``BTC``."""
    text = query.strip()
    for sep in _SEPARATORS:
        if sep in text:
            text = text.split(sep, 1)[0]
    return text.strip().upper()


@dataclass(frozen=True, slots=True)
class MarketRef:
    """A concrete tradable market on one venue."""

    exchange: str
    symbol: str
    quote_currency: str
    kind: str = "spot"  # "spot" | "swap" | "future"

    @property
    def key(self) -> str:
        return f"{self.exchange}:{self.symbol}"


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    """The outcome of resolution: everything the providers need to know."""

    query: str
    symbol: str
    coingecko_id: str | None = None
    name: str | None = None
    market_cap: float | None = None

    #: ccxt spot markets keyed by exchange id.
    spot_markets: tuple[MarketRef, ...] = ()
    #: ccxt derivative markets keyed by exchange id.
    derivative_markets: tuple[MarketRef, ...] = ()

    defillama_slug: str | None = None
    contract_address: str | None = None
    asset_platform_id: str | None = None

    ambiguities: tuple[Ambiguity, ...] = ()
    #: Human-readable notes about how resolution went.
    notes: tuple[str, ...] = ()

    #: True when no provider could identify the asset at all.
    unresolved: bool = False

    def markets_for(self, exchange: str) -> tuple[MarketRef, ...]:
        return tuple(m for m in self.spot_markets if m.exchange == exchange)

    def derivatives_for(self, exchange: str) -> tuple[MarketRef, ...]:
        return tuple(m for m in self.derivative_markets if m.exchange == exchange)

    def with_coingecko(self, coingecko_id: str, *, name: str | None = None) -> ResolvedAsset:
        return replace(self, coingecko_id=coingecko_id, name=name or self.name)

    def with_defillama(self, slug: str) -> ResolvedAsset:
        return replace(self, defillama_slug=slug)


def pick_best_candidate(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, list[Ambiguity]]:
    """Choose the highest-market-cap candidate; return the rest as ambiguities.

    CoinGecko already orders ``/coins/markets`` by market cap descending by default, but
    we sort explicitly so behaviour does not depend on server-side defaults.
    """
    usable = [c for c in candidates if c.get("id")]
    if not usable:
        return None, []

    def sort_key(row: dict[str, Any]) -> tuple[int, float]:
        cap = finite(row.get("market_cap"))
        # Missing market cap sorts last rather than being read as 0.
        return (1, cap) if cap is not None else (0, 0.0)

    ordered = sorted(usable, key=sort_key, reverse=True)
    winner = ordered[0]
    others = [
        Ambiguity(
            symbol=str(row.get("symbol", "")).upper(),
            coingecko_id=str(row["id"]),
            name=row.get("name"),
            market_cap=finite(row.get("market_cap")),
        )
        for row in ordered[1:]
    ]
    return winner, others


@dataclass(slots=True)
class ResolutionOverrides:
    """Explicit user overrides; every field optional."""

    coingecko_id: str | None = None
    exchanges: tuple[str, ...] | None = None
    perp_exchanges: tuple[str, ...] | None = None
    defillama_slug: str | None = None
    quote_currency: str | None = None
    base_symbol: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> ResolutionOverrides:
        known = {
            "coingecko_id",
            "exchanges",
            "perp_exchanges",
            "defillama_slug",
            "quote_currency",
            "base_symbol",
        }
        extra = {k: v for k, v in kwargs.items() if k not in known and v is not None}
        return cls(
            coingecko_id=kwargs.get("coingecko_id"),
            exchanges=tuple(kwargs["exchanges"]) if kwargs.get("exchanges") else None,
            perp_exchanges=tuple(kwargs["perp_exchanges"])
            if kwargs.get("perp_exchanges")
            else None,
            defillama_slug=kwargs.get("defillama_slug"),
            quote_currency=kwargs.get("quote_currency"),
            base_symbol=kwargs.get("base_symbol"),
            extra=extra,
        )


def initial_asset(query: str, config: Config) -> ResolvedAsset:
    """Seed a :class:`ResolvedAsset` before any provider has run."""
    symbol = normalise_query(query)
    return ResolvedAsset(
        query=query,
        symbol=symbol,
        unresolved=not overrides_unresolved(symbol),
    )


def overrides_unresolved(symbol: str) -> bool:
    """An obviously blank/garbage symbol starts life unresolved."""
    return not symbol or not symbol.isalnum()


__all__ = [
    "MarketRef",
    "ResolutionOverrides",
    "ResolvedAsset",
    "initial_asset",
    "normalise_query",
    "pick_best_candidate",
]