"""Per-provider rate limiting, retry with jittered backoff, and an httpx wrapper.

Retry policy is deliberately narrow: only 429, 5xx and transport timeouts are retried.
A 400/401/403/404 is a caller or credential problem — retrying it just wastes quota.
"""

from __future__ import annotations

import asyncio
import random
import time
from types import TracebackType
from typing import Any, Self

import httpx

from .config import Config
from .errors import RateLimitExceeded

#: HTTP statuses worth another attempt.
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504})

#: Cap on how long a provider may park us via ``Retry-After``. A hostile or buggy value
#: must not turn one throttled call into an hours-long stall.
MAX_RETRY_AFTER = 60.0

#: Reminder appended to a throttling error when the provider suggests nothing better.
GENERIC_HINT = (
    "lower TICKERKIT_RATE_LIMIT, stagger requests, set the provider's API key "
    "for a higher quota, or rely on another source"
)


def _header_int(headers: httpx.Headers, *names: str) -> int | None:
    for name in names:
        raw = headers.get(name)
        if raw:
            try:
                return int(float(raw))
            except ValueError:
                continue
    return None


def parse_retry_after(
    headers: httpx.Headers, *, now: float | None = None, ceiling: float = MAX_RETRY_AFTER
) -> float | None:
    """Seconds to wait, from ``Retry-After`` (delta or HTTP-date) or rate-limit hints.

    Providers spell this half a dozen ways — CoinGecko sends ``x-ratelimit-reset`` as an
    epoch second, DefiLlama uses ``retry-after`` — so all the common spellings are read.
    """
    now = time.time() if now is None else now
    raw = headers.get("retry-after")
    if raw:
        try:
            return max(0.0, min(ceiling, float(raw)))
        except ValueError:
            try:  # HTTP-date form
                from email.utils import parsedate_to_datetime

                delta = parsedate_to_datetime(raw).timestamp() - now
                return max(0.0, min(ceiling, delta))
            except (TypeError, ValueError, IndexError):
                pass
    # CoinGecko/DefiLlama style: reset is an epoch timestamp or a seconds-from-now count.
    for name in ("x-ratelimit-reset", "ratelimit-reset", "x-rate-limit-reset"):
        raw = headers.get(name)
        if not raw:
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        # Values below ~1e9 cannot be epochs; treat them as relative seconds.
        delta = value - now if value > 1_000_000_000 else value
        return max(0.0, min(ceiling, delta))
    return None


def _limit_hint(source: str) -> str:
    """A remediation hint, provider-specific where we know the answer."""
    known = {
        "coingecko": (
            "set TICKERKIT_COINGECKO_API_KEY (Demo) or TICKERKIT_COINGECKO_PRO_API_KEY "
            "for 100+/min instead of the 25/min keyless IP limit"
        ),
        "coinmarketcap": "set TICKERKIT_CMC_API_KEY",
    }
    return known.get(source, GENERIC_HINT)


class RateLimiter:
    """A simple async token-bucket limiter, one instance per provider.

    ``rpm`` is requests-per-minute. When a provider documents its own stricter limit
    (CoinGecko Demo is 100/min; ccxt exposes ``exchange.rateLimit`` in ms), pass that
    instead and it takes precedence over the config default.
    """

    __slots__ = ("_interval", "_lock", "_next_allowed", "name", "rpm")

    def __init__(self, rpm: int, *, name: str = "default") -> None:
        if rpm <= 0:
            raise ValueError("rpm must be positive")
        self.name = name
        self.rpm = rpm
        self._interval = 60.0 / rpm
        self._lock = asyncio.Lock()
        self._next_allowed = 0.0

    async def acquire(self) -> None:
        """Block until the next request slot is free."""
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._interval

    def __enter__(self) -> Self:  # pragma: no cover - trivial
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:  # pragma: no cover - trivial
        return None


class RetryPolicy:
    """Exponential backoff with full jitter."""

    __slots__ = ("base_delay", "max_after", "max_delay", "max_retries")

    def __init__(
        self,
        *,
        max_retries: int = 3,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        max_after: float = MAX_RETRY_AFTER,
    ) -> None:
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.max_after = max_after

    @classmethod
    def from_config(cls, config: Config) -> RetryPolicy:
        return cls(
            max_retries=config.max_retries,
            base_delay=config.retry_base_delay,
            max_delay=config.retry_max_delay,
            max_after=config.retry_max_after,
        )

    def delay_for(self, attempt: int, *, rng: random.Random | None = None) -> float:
        """Backoff before ``attempt`` (0-based), jittered in ``[0, backoff)``."""
        source = rng or random
        ceiling = min(self.max_delay, self.base_delay * (2**attempt))
        return source.uniform(0.0, ceiling)


class HttpClient:
    """An ``httpx.AsyncClient`` wrapper applying rate limiting and retries.

    One of these is created per provider (``source`` names it). After a genuine throttle
    the client opens a **circuit breaker**: for ``cooldown`` seconds every call raises
    :class:`RateLimitExceeded` immediately instead of spending real quota. Without it a
    snapshot that fans out over eight capabilities would fire eight doomed requests at
    an already-exhausted source, each burning its own retry budget, purely to collect
    eight copies of the same 429.
    """

    __slots__ = (
        "_blocked_until",
        "_client",
        "_limiter",
        "_owns_client",
        "_policy",
        "_views",
        "source",
    )

    def __init__(
        self,
        *,
        config: Config | None = None,
        limiter: RateLimiter | None = None,
        retry: RetryPolicy | None = None,
        client: httpx.AsyncClient | None = None,
        source: str = "http",
        timeout: float | None = None,
    ) -> None:
        # `config` is only needed to derive what the caller did not supply; `view`
        # passes an explicit limiter and policy, so it has nothing to derive from.
        if config is None and (limiter is None or retry is None):
            raise ValueError("config is required unless limiter and retry are given")
        self._policy = retry or RetryPolicy.from_config(config)  # type: ignore[arg-type]
        if limiter is not None:
            self._limiter = limiter
        else:
            assert config is not None
            self._limiter = RateLimiter(config.rate_limit_rpm, name="http")
        self._owns_client = client is None
        self.source = source
        self._blocked_until = 0.0
        self._views: dict[str, HttpClient] = {}
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(config.http_timeout if config else (timeout or 15.0)),
            follow_redirects=True,
            headers={"User-Agent": "tickerkit/0.1 (+https://github.com/)"},
        )

    def view(self, source: str) -> HttpClient:
        """A per-source view sharing this client's transport, limiter and retry policy.

        Only the circuit breaker is private to the view, so one throttled provider never
        blocks the others. Ownership stays with the parent, so closing the parent closes
        the single underlying connection pool.

        Views are memoised per source. That matters once several snapshots are fetched
        concurrently against one client: they must agree on whether a source is already
        exhausted. Handing each caller a fresh view would give every one of them its own
        ``_blocked_until``, so a genuine 429 would be paid for again — retries and all —
        by each concurrent ticker instead of being short-circuited.
        """
        cached = self._views.get(source)
        if cached is None:
            cached = HttpClient(
                limiter=self._limiter,
                retry=self._policy,
                client=self._client,
                source=source,
            )
            self._views[source] = cached
        return cached

    @property
    def throttled_until(self) -> float:
        """Monotonic timestamp until which this client short-circuits every call."""
        return self._blocked_until

    @property
    def is_throttled(self) -> bool:
        return time.monotonic() < self._blocked_until

    def _suppressed(self) -> RateLimitExceeded:
        """The error raised in place of a request that must not be sent."""
        return RateLimitExceeded(
            self.source,
            f"{self.source} rate limit still in effect; request suppressed",
            retry_after=self._blocked_until - time.monotonic(),
            hint=_limit_hint(self.source),
            suppressed=True,
        )

    async def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        """GET ``url`` and decode JSON, with retries.

        Raises :class:`RateLimitExceeded` for a throttled source and
        ``httpx.HTTPStatusError`` for any other non-success status.
        """
        # Breaker first: never spend quota on a source we already know is exhausted.
        if self.is_throttled:
            raise self._suppressed()

        last_exc: httpx.HTTPError | None = None
        for attempt in range(self._policy.max_retries + 1):
            await self._limiter.acquire()
            # ...and again after the wait. Acquiring a rate-limit slot is itself a queue:
            # at 60 rpm a caller can sit here for a second, and by the time its turn
            # comes a sibling may have opened the breaker. Checking only on entry lets
            # that caller fire a request it already knows is doomed — which is how a
            # single real 429 turns into a whole batch of them, one retry budget per
            # queued task. Re-checking here is what makes a shared breaker actually save
            # quota rather than merely describe it.
            if self.is_throttled:
                raise self._suppressed()
            # Computed inside the handler and acted on out here: an exception raised
            # from within an `except` block is not caught by sibling handlers of the
            # same `try`, so signalling the retry in-band is the only correct option.
            throttle_delay: float | None = None
            try:
                response = await self._client.get(url, params=params, headers=headers)
                response.raise_for_status()
                return response.json()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status == 429:
                    throttle_delay = self._throttle_delay(exc, attempt)  # raises if spent
                elif status not in RETRYABLE_STATUS or attempt >= self._policy.max_retries:
                    raise
                else:
                    last_exc = exc
                    throttle_delay = self._policy.delay_for(attempt)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                if attempt >= self._policy.max_retries:
                    raise
                last_exc = exc
                throttle_delay = self._policy.delay_for(attempt)

            assert throttle_delay is not None  # every path above sets it or raises
            await asyncio.sleep(throttle_delay)
            # Deliberately *not* clearing `_blocked_until` here. Sleeping past the
            # cooldown expires it on its own — `_throttle_delay` only ever sets a window
            # no longer than the delay just slept. Clearing it explicitly was harmless
            # when one client owned the breaker, but the breaker is shared: a task
            # finishing its own retry wait would reset the suppression for every other
            # task still queued behind it, so a genuine 429 got re-paid in full once per
            # concurrent caller. Expiry is now the only thing that reopens the source.
        assert last_exc is not None  # pragma: no cover - loop always sets or raises
        raise last_exc

    def _throttle_delay(self, exc: httpx.HTTPStatusError, attempt: int) -> float:
        """How long to wait before retrying a 429, or raise if retries are spent.

        Honours the server's own ``Retry-After``/reset hint when it is longer than our
        backoff would be, and trips the breaker for the whole wait so sibling
        capabilities racing this provider short-circuit instead of joining the
        stampede and burning the remaining quota.
        """
        headers = exc.response.headers
        advertised = parse_retry_after(headers, ceiling=self._policy.max_after)
        if attempt >= self._policy.max_retries:
            cooldown = advertised if advertised is not None else self._policy.delay_for(attempt)
            # Always trip the breaker, advertised hint or not. A 429 with no Retry-After
            # is still a source telling us it is out of quota, and that fact is just as
            # true for the next caller as it is for this one — leaving the breaker shut
            # meant a silent server quietly handed every concurrent ticker its own full
            # retry budget. Gating on `advertised` here meant the protection applied only
            # to the servers polite enough to say how long to wait.
            self._blocked_until = time.monotonic() + min(cooldown, self._policy.max_after)
            raise RateLimitExceeded(
                self.source,
                f"{self.source} returned HTTP 429 after {attempt + 1} attempt(s)",
                retry_after=cooldown,
                limit=_header_int(headers, "x-ratelimit-limit", "ratelimit-limit",
                                  "x-rate-limit-limit"),
                remaining=_header_int(headers, "x-ratelimit-remaining", "ratelimit-remaining",
                                      "x-rate-limit-remaining"),
                hint=_limit_hint(self.source),
            ) from exc
        delay = self._policy.delay_for(attempt)
        if advertised is not None:
            delay = max(delay, min(advertised, self._policy.max_after))
        self._blocked_until = time.monotonic() + delay
        return delay

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()


__all__ = ["RETRYABLE_STATUS", "HttpClient", "RateLimiter", "RetryPolicy"]