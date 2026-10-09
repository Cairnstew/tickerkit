"""Provider registry.

Adding a provider is a one-line change here plus the adapter module. The orchestrator
discovers capabilities from the registry, so nothing else needs editing.
"""

from __future__ import annotations

from .base import Provider
from .ccxt_provider import CcxtProvider
from .coingecko import CoinGeckoProvider
from .coinmarketcap import CoinMarketCapProvider
from .coinpaprika import CoinPaprikaProvider
from .defillama import DefiLlamaProvider

#: Default provider set, in resolution-priority order.
_providers: list[type[Provider]] = [
    # Aggregated/identity sources first, in resolution-priority order. CoinPaprika sits
    # right after CoinGecko so it can carry price and market cap when CoinGecko's
    # keyless IP limit throttles us — it is the source that makes a throttled snapshot
    # degrade instead of empty.
    CoinGeckoProvider,
    CoinPaprikaProvider,
    CcxtProvider,
    DefiLlamaProvider,
    CoinMarketCapProvider,
]

#: The live registry, in resolution-priority order. Mutated in place by :func:`register`.
PROVIDERS: list[type[Provider]] = list(_providers)

_BY_NAME: dict[str, type[Provider]] = {cls.name: cls for cls in _providers}


def registry() -> tuple[type[Provider], ...]:
    """All registered providers."""
    return tuple(PROVIDERS)


def get(name: str) -> type[Provider]:
    """Look up a provider class by name."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown provider {name!r}; known: {sorted(_BY_NAME)}") from None


def register(cls: type[Provider]) -> None:
    """Register (or replace) a provider class at runtime."""
    PROVIDERS[:] = [p for p in PROVIDERS if p.name != cls.name] + [cls]
    _providers[:] = PROVIDERS
    _BY_NAME[cls.name] = cls


__all__ = [
    "PROVIDERS",
    "CcxtProvider",
    "CoinGeckoProvider",
    "CoinMarketCapProvider",
    "CoinPaprikaProvider",
    "DefiLlamaProvider",
    "Provider",
    "get",
    "register",
    "registry",
]