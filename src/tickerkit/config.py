"""Configuration for tickerkit.

Everything is env-driven with sane defaults. No secrets are ever hardcoded; optional
provider keys are simply absent unless the user exports them.

Environment variables
---------------------
TICKERKIT_QUOTE_CURRENCY
    Target quote currency for every normalised number. Default ``USD``.
TICKERKIT_EXCHANGES
    Comma-separated ccxt exchange ids. Default ``binance,kraken,coinbase``.
TICKERKIT_PERP_EXCHANGES
    Comma-separated ccxt ids queried for derivatives. Default ``binance,bybit,okx``.
TICKERKIT_HTTP_TIMEOUT
    Per-request timeout in seconds. Default ``15``.
TICKERKIT_RATE_LIMIT
    Safety-net requests-per-minute cap. Each provider additionally applies its own
    documented limit, which takes precedence when stricter. Default ``60``.
TICKERKIT_CACHE_TTL
    Default snapshot-cache TTL in seconds. Default ``30``.
TICKERKIT_CACHE_ENABLED
    ``0``/``false`` disables the in-memory cache. Default on.
TICKERKIT_CACHE_PATH
    Path to a SQLite cache file. When set, TickerKit persists the cache to this file
    (git-trackable, bounded by the two limits below) instead of using the in-memory
    cache. Default unset (in-memory).
TICKERKIT_CACHE_MAX_ENTRIES
    Upper bound on cached entries before LRU eviction. Default ``2000``.
TICKERKIT_CACHE_MAX_BYTES
    Upper bound on total JSON payload bytes before LRU eviction. Default ``10485760``
    (10 MiB). Keeps a committed cache file small.
TICKERKIT_MAX_RETRIES
    Retry attempts for 429/5xx/timeouts. Default ``3``.
TICKERKIT_MAX_RETRY_AFTER
    Longest wait honoured from a server's Retry-After, in seconds. Default ``60``.
TICKERKIT_HISTORY_MAX_CALLS
    Ceiling on upstream requests spent paginating one history series. Default ``25``.
    A backfill that would need more is returned **truncated** and says so, rather than
    walking a decade of 1-minute candles in one call.
TICKERKIT_HISTORY_DISCREPANCY_PCT
    Percent deviation at which two history sources are recorded as disagreeing and at
    which a venue is dropped from the history consensus. Default ``1.0``.
TICKERKIT_COINGECKO_API_KEY
    Optional CoinGecko Demo key -> ``x-cg-demo-api-key``.
TICKERKIT_COINGECKO_PRO_API_KEY
    Optional CoinGecko Pro key -> ``x-cg-pro-api-key`` (switches base URL to pro-api).
TICKERKIT_CMC_API_KEY
    Optional CoinMarketCap key. When unset the CMC provider is skipped with a reason.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from decimal import Decimal
from pathlib import Path
from types import MappingProxyType

#: Liquid, widely-covered spot venues used by default.
DEFAULT_EXCHANGES: tuple[str, ...] = ("binance", "kraken", "coinbase")
#: Venues with meaningful perpetual funding data used by default.
DEFAULT_PERP_EXCHANGES: tuple[str, ...] = ("binance", "bybit", "okx")

#: Stablecoins we are willing to read as fiat-equivalent to the quote currency. The
#: assumption is recorded in provenance so it is never silent.
DEFAULT_FIAT_EQUIVALENT_STABLECOINS: frozenset[str] = frozenset({"USDT", "USDC", "DAI", "FDUSD"})

_TRUE = frozenset({"1", "true", "yes", "on"})


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value.strip() if value and value.strip() else default


def _env_opt(name: str) -> str | None:
    value = os.environ.get(name)
    return value.strip() or None if value and value.strip() else None


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in _TRUE


def _env_int(name: str, default: int) -> int:
    raw = _env_opt(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = _env_opt(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_tuple(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = _env_opt(name)
    if raw is None:
        return default
    parts = tuple(p.strip() for p in raw.split(",") if p.strip())
    return parts or default


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value.strip()) if value and value.strip() else None


@dataclass(frozen=True, slots=True)
class Config:
    """Immutable run configuration."""

    quote_currency: str = "USD"
    exchanges: tuple[str, ...] = DEFAULT_EXCHANGES
    perp_exchanges: tuple[str, ...] = DEFAULT_PERP_EXCHANGES
    fiat_equivalent_stablecoins: frozenset[str] = DEFAULT_FIAT_EQUIVALENT_STABLECOINS

    http_timeout: float = 15.0
    rate_limit_rpm: int = 60
    max_retries: int = 3
    retry_base_delay: float = 0.5
    retry_max_delay: float = 8.0
    #: Ceiling on how long a server may park us via Retry-After. Also trips the circuit
    #: breaker for that long, so it doubles as the "stop calling this source" window.
    retry_max_after: float = 60.0

    cache_enabled: bool = True
    cache_ttl: float = 30.0

    #: When set, TickerKit uses a git-trackable SQLite cache at this path instead of the
    #: in-memory cache. The file is bounded by ``cache_max_entries``/``cache_max_bytes``.
    cache_path: Path | None = None
    cache_max_entries: int = 2000
    cache_max_bytes: int = 10 * 1024 * 1024  # 10 MiB

    # Reconciliation thresholds.
    outlier_pct: float = 5.0
    """Absolute percent deviation from the median at which a venue is an outlier."""

    outlier_mad_z: float = 3.5
    """Robust z-score (median absolute deviation) cut-off for outlier rejection."""

    min_venues_for_mad: int = 5
    """Minimum venues before the MAD criterion is trusted.

    With three or four venues the median absolute deviation is not a usable estimate of
    dispersion, and in a tight cluster it is small enough that an ordinary venue a hair
    above the median scores as a statistical outlier. Below this floor only the gross
    percentage test (``outlier_pct``) can exclude a venue.
    """

    stale_seconds: float = 120.0
    """Age beyond which venue data is flagged stale."""

    orderbook_depth: int = 20
    """Levels per side requested from each venue."""

    trades_limit: int = 100
    ohlcv_limit: int = 200
    ohlcv_timeframes: tuple[str, ...] = ("5m", "1h", "1d")

    #: Upper bound on upstream calls spent paginating a single history series.
    history_max_calls: int = 25
    #: Percent deviation at which history sources are flagged as disagreeing.
    history_discrepancy_pct: float = 1.0

    # Provider credentials (never hardcoded; absent => provider skipped or keyless).
    coingecko_api_key: str | None = None
    coingecko_pro_api_key: str | None = None
    cmc_api_key: str | None = None

    extra: Mapping[str, str] = MappingProxyType({})
    """Free-form provider-specific settings; primarily used by tests and embedders."""

    @classmethod
    def from_env(cls) -> Config:
        """Build a config from process environment variables."""
        return cls(
            quote_currency=_env_str("TICKERKIT_QUOTE_CURRENCY", "USD").upper(),
            exchanges=_env_tuple("TICKERKIT_EXCHANGES", DEFAULT_EXCHANGES),
            perp_exchanges=_env_tuple("TICKERKIT_PERP_EXCHANGES", DEFAULT_PERP_EXCHANGES),
            http_timeout=_env_float("TICKERKIT_HTTP_TIMEOUT", 15.0),
            rate_limit_rpm=_env_int("TICKERKIT_RATE_LIMIT", 60),
            max_retries=_env_int("TICKERKIT_MAX_RETRIES", 3),
            retry_max_after=_env_float("TICKERKIT_MAX_RETRY_AFTER", 60.0),
            cache_enabled=_env_bool("TICKERKIT_CACHE_ENABLED", True),
            cache_ttl=_env_float("TICKERKIT_CACHE_TTL", 30.0),
            cache_path=_env_path("TICKERKIT_CACHE_PATH"),
            cache_max_entries=_env_int("TICKERKIT_CACHE_MAX_ENTRIES", 2000),
            cache_max_bytes=_env_int("TICKERKIT_CACHE_MAX_BYTES", 10 * 1024 * 1024),
            outlier_pct=_env_float("TICKERKIT_OUTLIER_PCT", 5.0),
            stale_seconds=_env_float("TICKERKIT_STALE_SECONDS", 120.0),
            orderbook_depth=_env_int("TICKERKIT_ORDERBOOK_DEPTH", 20),
            trades_limit=_env_int("TICKERKIT_TRADES_LIMIT", 100),
            ohlcv_limit=_env_int("TICKERKIT_OHLCV_LIMIT", 200),
            ohlcv_timeframes=_env_tuple("TICKERKIT_OHLCV_TIMEFRAMES", ("5m", "1h", "1d")),
            history_max_calls=_env_int("TICKERKIT_HISTORY_MAX_CALLS", 25),
            history_discrepancy_pct=_env_float("TICKERKIT_HISTORY_DISCREPANCY_PCT", 1.0),
            coingecko_api_key=_env_opt("TICKERKIT_COINGECKO_API_KEY"),
            coingecko_pro_api_key=_env_opt("TICKERKIT_COINGECKO_PRO_API_KEY"),
            cmc_api_key=_env_opt("TICKERKIT_CMC_API_KEY"),
        )

    def with_overrides(self, **kwargs: object) -> Config:
        """Return a copy with the given (non-``None``) fields replaced."""
        clean = {k: v for k, v in kwargs.items() if v is not None}
        if not clean:
            return self
        return replace(self, **clean)  # type: ignore[arg-type]

    def is_fiat_equivalent(self, quote: str) -> bool:
        """Whether ``quote`` may be read as the quote currency (stablecoin proxy)."""
        return quote.upper() in self.fiat_equivalent_stablecoins

    def outlier_threshold(self) -> Decimal:
        """Outlier cut-off as a fraction, e.g. ``0.05`` for 5%."""
        return Decimal(str(self.outlier_pct)) / Decimal(100)