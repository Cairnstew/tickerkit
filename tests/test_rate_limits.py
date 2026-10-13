"""Rate-limit identification: structured errors, honoured Retry-After, circuit breaker.

A throttle must be *identifiable* — a caller needs to know which source refused, why,
how long to wait and what to change. It must also not be *expensive*: a snapshot that
fans out over several capabilities should not fire every one of them at a source that
has already said no.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from tickerkit import Config, RateLimitExceeded, TickerKit
from tickerkit.capabilities import Capability
from tickerkit.http import MAX_RETRY_AFTER, HttpClient, parse_retry_after
from tickerkit.models import ProviderError
from tickerkit.providers.coinpaprika import CoinPaprikaProvider
from tickerkit.providers.coingecko import CoinGeckoProvider

URL = "https://example.test/data"


def _kit(*names: str) -> TickerKit:
    """A kit restricted to the named providers, so a 429 can be attributed cleanly."""
    chosen = {"coingecko": CoinGeckoProvider}
    return TickerKit(
        config=Config(max_retries=0, retry_max_after=0.01, cache_ttl=0),
        providers=tuple(chosen[n] for n in names),
    )


def _client(**kwargs: object) -> HttpClient:
    return HttpClient(
        config=Config(
            max_retries=1,
            retry_base_delay=0.01,
            retry_max_delay=0.01,
            # Keep the advertised-delay test honest without a minute-long suite.
            retry_max_after=0.5,
        ),
        **kwargs,  # type: ignore[arg-type]
    )


class TestRetryAfterParsing:
    def test_delta_seconds(self) -> None:
        assert parse_retry_after(httpx.Headers({"Retry-After": "30"})) == 30.0

    def test_http_date_form(self) -> None:
        headers = httpx.Headers({"Retry-After": "Wed, 21 Oct 2099 07:28:00 GMT"})
        # A date in the far future clamps rather than parking the caller for decades.
        assert parse_retry_after(headers) == MAX_RETRY_AFTER

    def test_epoch_reset_header(self) -> None:
        now = 1_000_000_000.0
        headers = httpx.Headers({"x-ratelimit-reset": str(now + 42)})
        assert parse_retry_after(headers, now=now) == pytest.approx(42, abs=1)

    def test_relative_reset_header(self) -> None:
        assert parse_retry_after(httpx.Headers({"x-ratelimit-reset": "45"})) == 45.0

    def test_absent_is_none(self) -> None:
        assert parse_retry_after(httpx.Headers()) is None

    def test_hostile_value_is_clamped(self) -> None:
        assert parse_retry_after(httpx.Headers({"Retry-After": "999999"})) == MAX_RETRY_AFTER


class TestThrottleIdentification:
    @pytest.mark.asyncio
    @respx.mock
    async def test_exhausted_429_raises_structured_error(self) -> None:
        route = respx.get(URL).mock(
            return_value=httpx.Response(
                429,
                headers={
                    "Retry-After": "30",
                    "X-RateLimit-Limit": "25",
                    "X-RateLimit-Remaining": "0",
                },
            )
        )
        client = HttpClient(
            config=Config(max_retries=1, retry_base_delay=0.01, retry_max_delay=0.01)
        )
        with pytest.raises(RateLimitExceeded) as excinfo:
            await client.get_json(URL)

        err = excinfo.value
        assert err.source == "http"
        assert err.limit == 25
        assert err.remaining == 0
        assert err.retry_after == pytest.approx(30, abs=1)
        assert err.hint  # a remedy is always offered
        assert not err.suppressed
        assert route.call_count == 2  # initial attempt + 1 retry, not more
        await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_retry_honours_server_delay_not_local_backoff(self) -> None:
        """A 429 asking for 30s must not be retried after our 0.01s jittered backoff.

        Retrying early is how a caller who is already over quota turns one throttled
        request into a burst that gets the whole IP range blocked.
        """
        route = respx.get(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "60"}),
                httpx.Response(200, json={"ok": True}),
            ]
        )
        client = _client()
        import time

        started = time.monotonic()
        result = await client.get_json(URL)
        elapsed = time.monotonic() - started
        assert result == {"ok": True}
        assert route.call_count == 2
        # The retry waited the server-advertised delay (clamped to 0.5s by config),
        # not the 0.01s local backoff. Retrying on local backoff alone is the bug:
        # it turns one over-quota request into a burst that gets the IP blocked.
        assert elapsed >= 0.5, "retry fired without honouring the advertised delay"
        await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_non_throttle_status_is_not_reclassified(self) -> None:
        respx.get(URL).mock(return_value=httpx.Response(404))
        client = _client()
        with pytest.raises(httpx.HTTPStatusError):
            await client.get_json(URL)
        await client.aclose()


class TestCircuitBreaker:
    @pytest.mark.asyncio
    @respx.mock
    async def test_second_call_is_suppressed_without_spending_quota(self) -> None:
        route = respx.get(URL).mock(
            return_value=httpx.Response(429, headers={"Retry-After": "60"})
        )
        client = _client(source="coingecko")

        with pytest.raises(RateLimitExceeded):
            await client.get_json(URL)
        spent = route.call_count

        for _ in range(4):  # sibling capabilities racing the same exhausted source
            with pytest.raises(RateLimitExceeded) as excinfo:
                await client.get_json(URL)
            assert excinfo.value.suppressed is True

        # The point of the breaker: no further requests left the process.
        assert route.call_count == spent

    @pytest.mark.asyncio
    @respx.mock
    async def test_breaker_is_per_source(self) -> None:
        """One throttled provider must not silence the others."""
        respx.get("https://a.test/x").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "60"})
        )
        respx.get("https://b.test/x").mock(return_value=httpx.Response(200, json={"ok": 1}))
        client = _client()
        throttled = client.view("a")
        healthy = client.view("b")

        with pytest.raises(RateLimitExceeded):
            await throttled.get_json("https://a.test/x")
        assert throttled.is_throttled, "a real 429 must open the breaker"

        assert await healthy.get_json("https://b.test/x") == {"ok": 1}
        assert not healthy.is_throttled, "one source's throttle must not block another"
        await client.aclose()


class TestSnapshotSurfacesThrottling:
    @pytest.mark.asyncio
    @respx.mock
    async def test_data_quality_reports_throttle_with_remedy(self) -> None:
        respx.get(url__startswith="https://api.coingecko.com").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "60"})
        )
        snap = await _kit("coingecko").get_ticker("BTC", cache_ttl=0)

        dq = snap.data_quality
        assert dq.throttled is True
        assert dq.rate_limited, "throttling must be reported in data_quality"
        limit = dq.rate_limited[0]
        assert limit.source == "coingecko"
        assert limit.retry_after is not None and limit.retry_after > 0
        assert "API_KEY" in (limit.remedy or ""), "must suggest the concrete fix"
        # It stays an error too — a snapshot must never hide a failed source.
        assert any(isinstance(e, ProviderError) for e in dq.errors)

    @pytest.mark.asyncio
    @respx.mock
    async def test_throttle_is_deduped_per_capability(self) -> None:
        respx.get(url__startswith="https://api.coingecko.com").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "60"})
        )
        snap = await _kit("coingecko").get_ticker("BTC", cache_ttl=0)
        keys = [(r.source, r.capability) for r in snap.data_quality.rate_limited]
        assert len(keys) == len(set(keys)), "duplicate throttle records"
        assert all(isinstance(c, Capability) for _, c in keys)

    @pytest.mark.asyncio
    @respx.mock
    async def test_unthrottled_snapshot_is_not_flagged(self) -> None:
        snap = await TickerKit(
            providers=(), config=Config(cache_ttl=0)
        ).get_ticker("NOTAREALCOIN")
        assert snap.data_quality.throttled is False
        assert snap.data_quality.rate_limited == ()

    def test_report_renders_the_remedy(self) -> None:
        from tickerkit.models import DataQuality, RateLimitInfo
        from tickerkit.render import quality_block

        class _S:
            data_quality = DataQuality(
                rate_limited=(
                    RateLimitInfo(
                        source="coingecko",
                        capability=Capability.AGG_MARKET,
                        message="HTTP 429",
                        retry_after=30.0,
                        limit=25,
                        remedy="set TICKERKIT_COINGECKO_API_KEY",
                    ),
                )
            )

        out = quality_block(_S())  # type: ignore[arg-type]
        assert "rate limit" in out
        assert "coingecko/agg_market" in out
        assert "TICKERKIT_COINGECKO_API_KEY" in out


class TestPaymentRequiredThrottle:
    """A 402 that advertises a retry window is a quota throttle, not a dead source.

    CoinPaprika's keyless tier answers 402 with ``retry-after`` and a body spelling out
    ``hard_limit`` (25k/month) / ``soft_limit`` (60/hour). Treating that as a hard failure
    meant the breaker never opened, so every concurrent capability independently spent
    its retry budget against a source that had already said "come back in an hour".
    """

    @pytest.mark.asyncio
    @respx.mock
    async def test_402_with_retry_window_raises_structured_error(self) -> None:
        route = respx.get(URL).mock(
            return_value=httpx.Response(
                402,
                headers={"Retry-After": "3351", "X-RateLimit-Limit": "60"},
                json={
                    "type": "payment_required",
                    "hard_limit": "25k requests per month",
                    "soft_limit": "60 requests per hour",
                },
            )
        )
        client = HttpClient(config=Config(max_retries=0, retry_max_after=3600.0))
        with pytest.raises(RateLimitExceeded) as excinfo:
            await client.get_json(URL)

        err = excinfo.value
        assert err.source == "http"
        # The advertised window is honoured, not replaced by a local backoff guess.
        assert err.retry_after == pytest.approx(3351, abs=1)
        assert err.hint  # a remedy is always offered
        assert not err.suppressed
        assert route.call_count == 1
        await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_402_opens_the_breaker_so_siblings_short_circuit(self) -> None:
        """The point of recognising a 402 as a throttle: no further requests leave."""
        route = respx.get(URL).mock(
            return_value=httpx.Response(402, headers={"Retry-After": "3351"})
        )
        client = _client(source="coinpaprika")

        with pytest.raises(RateLimitExceeded):
            await client.get_json(URL)
        assert client.is_throttled, "a quota 402 must open the breaker like a 429"
        spent = route.call_count

        for _ in range(4):  # sibling capabilities racing the same exhausted source
            with pytest.raises(RateLimitExceeded) as excinfo:
                await client.get_json(URL)
            assert excinfo.value.suppressed is True

        assert route.call_count == spent, "the breaker must stop the stampede"
        await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_bare_402_stays_a_hard_failure(self) -> None:
        """A 402 with no retry window is permanent — it must not be retried or throttled."""
        route = respx.get(URL).mock(
            return_value=httpx.Response(402, json={"type": "payment_required"})
        )
        client = _client()

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            await client.get_json(URL)
        assert excinfo.value.response.status_code == 402
        assert not client.is_throttled, "a bare 402 must not look throttled"
        assert route.call_count == 1, "a permanent 402 must not be retried"
        await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_402_is_reported_as_a_throttle_with_a_remedy(self) -> None:
        respx.get(url__startswith="https://api.coinpaprika.com").mock(
            return_value=httpx.Response(402, headers={"Retry-After": "3351"})
        )
        kit = TickerKit(
            config=Config(max_retries=0, retry_max_after=0.01, cache_ttl=0),
            providers=(CoinPaprikaProvider,),
        )
        snap = await kit.get_ticker("BTC", cache_ttl=0)

        dq = snap.data_quality
        assert dq.throttled is True, "a quota 402 must be reported as a throttle"
        assert dq.rate_limited, "and recorded in data_quality"
        assert dq.rate_limited[0].source == "coinpaprika"
        assert dq.rate_limited[0].retry_after is not None
        # The remedy must name the real constraint, not the generic hint.
        assert "60 requests/hour" in (dq.rate_limited[0].remedy or "")
        # It stays an error too — a snapshot must never hide a failed source.
        assert any(isinstance(e, ProviderError) for e in dq.errors)
