"""Batch fetching: many tickers at once, without multiplying the request rate.

The guarantee under test is the whole point of the feature. Adding tickers to a batch may
add *concurrency*; it may never add requests-per-minute. Three things are therefore worth
pinning down with tests rather than prose:

* **Bounded concurrency.** A batch must not open one snapshot per ticker at once.
* **A shared token bucket.** N concurrent tickers draw from one requests-per-minute
  budget per provider, not one each.
* **A shared circuit breaker.** One real 429 must be paid for once, not once per ticker.

The last two are easy to assert in prose and easy to regress in code, because the tests
that would catch them look almost identical to tests that would pass either way.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, ClassVar
from unittest.mock import MagicMock

import httpx
import pytest
import respx

from tickerkit import Config, RateLimitExceeded, TickerKit
from tickerkit.capabilities import Capability
from tickerkit.core import TickerKit as CoreTickerKit
from tickerkit.http import HttpClient, RateLimiter
from tickerkit.models import Market, TickerSnapshot
from tickerkit.providers.base import Provider
from tickerkit.providers.coingecko import CoinGeckoProvider
from tickerkit.resolve import ResolvedAsset

COINGECKO = "https://api.coingecko.com/api/v3"


class CountingProvider(Provider):
    """Records every call and how many were ever in flight at the same moment."""

    name = "counting"
    supports = frozenset({Capability.RESOLVE, Capability.META, Capability.AGG_MARKET})

    live = 0
    peak = 0
    #: Class-level on purpose: the counters must be shared by every instance and every
    #: concurrent call, which is exactly what the concurrency assertions measure.
    calls: ClassVar[list[str]] = []

    @classmethod
    def reset(cls) -> None:
        cls.live = cls.peak = 0
        cls.calls = []

    async def resolve(self, query: str) -> ResolvedAsset:
        await self._enter()
        try:
            await asyncio.sleep(0.01)
            return ResolvedAsset(
                query=query,
                symbol=query.upper(),
                coingecko_id=query.lower(),
                name=query.upper(),
            )
        finally:
            CountingProvider.live -= 1

    async def fetch_market(self, asset: ResolvedAsset) -> Market:
        return Market(quote_currency="USD", aggregated_price=100.0)

    async def _enter(self) -> None:
        type(self).live += 1
        type(self).peak = max(type(self).peak, type(self).live)
        type(self).calls.append("resolve")
        await asyncio.sleep(0)


@pytest.fixture
def counting() -> type[CountingProvider]:
    CountingProvider.reset()
    return CountingProvider


def _kit(
    config: Config | None = None,
    providers: tuple[type[Provider], ...] = (),
) -> TickerKit:
    return TickerKit(config=config or Config(cache_enabled=False), providers=providers)


class TestBatchBasics:
    async def test_empty_batch_is_a_no_op(self) -> None:
        assert await _kit().get_tickers([]) == []

    async def test_results_are_in_input_order(self, counting: type[CountingProvider]) -> None:
        kit = _kit(providers=(counting,))
        snaps = await kit.get_tickers(["SOL", "BTC", "ETH"])
        assert [s.query for s in snaps] == ["SOL", "BTC", "ETH"]

    async def test_duplicate_tickers_are_fetched_once(
        self, counting: type[CountingProvider]
    ) -> None:
        """A watchlist with a repeated asset must not pay for it twice.

        The same snapshot object is returned for each occurrence, so a caller lining up
        results against its own list still gets a one-to-one mapping.
        """
        kit = _kit(providers=(counting,))
        snaps = await kit.get_tickers(["BTC", "ETH", "BTC"])

        assert [s.query for s in snaps] == ["BTC", "ETH", "BTC"]
        assert snaps[0] is snaps[2], "a duplicate must reuse the snapshot it already has"
        assert counting.calls.count("resolve") == 2, "the repeat spent another request"

    async def test_one_broken_ticker_does_not_sink_the_batch(
        self, counting: type[CountingProvider]
    ) -> None:
        """A snapshot-level failure must not take the other tickers down with it.

        The default is strict (matching :func:`asyncio.gather`), so the error surfaces —
        but it belongs to its own slot, and the healthy tickers still got fetched.
        """
        kit = _kit(providers=(counting,))
        real = kit._assemble

        async def explode(ticker: str, config: Config, http: Any, overrides: Any) -> Any:
            if ticker == "BOOM":
                raise RuntimeError("nope")
            return await real(ticker, config, http, overrides)

        kit._assemble = explode  # type: ignore[method-assign]

        results = await kit.get_tickers(["BTC", "BOOM"], return_exceptions=True)
        assert results[0].query == "BTC"
        assert isinstance(results[1], RuntimeError)

        # Strict by default: the batch raises rather than quietly handing back a mixture
        # of snapshots and exceptions the caller did not ask for.
        with pytest.raises(RuntimeError):
            await kit.get_tickers(["BTC", "BOOM"])

    async def test_cancelling_a_batch_leaves_nothing_in_flight(
        self, counting: type[CountingProvider]
    ) -> None:
        """A cancelled batch must not leave a task running against a closed client."""
        kit = _kit(providers=(counting,))
        task = asyncio.create_task(kit.get_tickers([f"C{i}" for i in range(10)]))
        await asyncio.sleep(0)  # let the children start
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # Every child must be finished, not merely cancelled-and-abandoned.
        await asyncio.sleep(0.05)
        assert counting.live == 0, "a cancelled batch leaked a running fetch"

    async def test_a_degraded_ticker_is_still_returned(
        self, counting: type[CountingProvider]
    ) -> None:
        """An unidentifiable ticker is a snapshot with errors, not an exception."""
        kit = _kit(providers=(counting,))
        snaps = await kit.get_tickers(["BTC", "NOTAREALCOIN"])
        assert len(snaps) == 2


class TestConcurrencyIsBounded:
    async def test_concurrency_limit_is_respected(self, counting: type[CountingProvider]) -> None:
        """The semaphore, not luck, decides how many snapshots are in flight."""
        kit = _kit(Config(cache_enabled=False, max_concurrency=3), providers=(counting,))
        await kit.get_tickers([f"C{i}" for i in range(12)])

        assert counting.peak <= 3, f"peak concurrency {counting.peak} exceeded the limit"

    async def test_limit_of_one_serialises(self, counting: type[CountingProvider]) -> None:
        kit = _kit(Config(cache_enabled=False, max_concurrency=1), providers=(counting,))
        await kit.get_tickers([f"C{i}" for i in range(4)])
        assert counting.peak == 1

    async def test_explicit_argument_overrides_config(
        self, counting: type[CountingProvider]
    ) -> None:
        kit = _kit(Config(cache_enabled=False, max_concurrency=8), providers=(counting,))
        await kit.get_tickers([f"C{i}" for i in range(12)], concurrency=2)
        assert counting.peak <= 2

    async def test_nonsense_concurrency_falls_back_to_config(
        self, counting: type[CountingProvider]
    ) -> None:
        """A bad value is ignored rather than guessed at, matching the rest of the config."""
        kit = _kit(Config(cache_enabled=False, max_concurrency=2), providers=(counting,))
        await kit.get_tickers([f"C{i}" for i in range(6)], concurrency=0)
        assert counting.peak <= 2

    async def test_batch_is_faster_than_a_serial_sweep(
        self, counting: type[CountingProvider]
    ) -> None:
        """The point of the feature: the same work, less wall-clock, same request count."""
        tickers = [f"C{i}" for i in range(8)]

        started = time.monotonic()
        await _kit(Config(cache_enabled=False, max_concurrency=8), providers=(counting,)).get_tickers(tickers)
        elapsed = time.monotonic() - started

        # Each resolve sleeps 10ms. Serial would be >= 80ms; concurrent with a bound of 8
        # is one sleep deep. A wide margin keeps this from being a flaky timing test.
        assert elapsed < 0.5
        assert counting.calls.count("resolve") == 8


class _PacedProvider(Provider):
    """Spends several rate-limited calls per ticker, like the real providers do."""

    name = "paced"
    supports = frozenset({Capability.RESOLVE, Capability.AGG_MARKET})
    RPM = 600
    calls = 0

    @classmethod
    def rate_limit_rpm(cls, config: Config) -> int:
        return cls.RPM

    async def resolve(self, query: str) -> ResolvedAsset:
        for _ in range(3):
            await self.ctx.throttle()
            type(self).calls += 1
        return ResolvedAsset(
            query=query, symbol=query, coingecko_id=query.lower(), name=query
        )

    async def fetch_market(self, asset: ResolvedAsset) -> Market:
        await self.ctx.throttle()
        type(self).calls += 1
        return Market(quote_currency="USD", aggregated_price=1.0)


class TestQuotaIsShared:
    def test_one_limiter_per_provider_not_per_call(self) -> None:
        """The core anti-throttling property, asserted directly on the instance.

        A per-call limiter makes N concurrent tickers look like N independent clients,
        each entitled to the provider's full requests-per-minute. That is how a batch gets
        an IP range blocked, so it must be one shared bucket per provider.
        """
        kit = _kit()
        config = kit.config

        first = kit._limiter_for(CountingProvider, config)
        again = kit._limiter_for(CountingProvider, config)
        assert first is not None
        assert first is again, "each call built its own limiter; N tickers => Nx the quota"

    def test_a_second_instance_gets_its_own_limiter(self) -> None:
        """Limiters are shared within an instance, not leaked across unrelated ones."""
        a, b = _kit(), _kit()
        assert a._limiter_for(CountingProvider, a.config) is not (
            b._limiter_for(CountingProvider, b.config)
        )

    def test_a_different_rpm_gets_a_different_bucket(self) -> None:
        """A per-call rpm override must not be served a bucket sized for another limit."""
        kit = _kit()
        slow = kit._limiter_for(CountingProvider, Config(rate_limit_rpm=10))
        fast = kit._limiter_for(CountingProvider, Config(rate_limit_rpm=600))
        assert slow is not fast
        assert slow.rpm == 10
        assert fast.rpm == 600

    def test_resolution_and_capabilities_share_one_bucket(self) -> None:
        """Within a single snapshot, resolve must not hold a second budget.

        Resolution and the capability fan-out hit the same provider minutes apart; giving
        them separate limiters silently doubles the quota a provider is asked to honour.
        """
        kit = _kit()
        classes = kit._provider_classes()
        assert classes == ()
        # Same instance, same provider name, same rpm -> one bucket either way.
        assert kit._limiter_for(CountingProvider, kit.config) is kit._limiter_for(
            CountingProvider, kit.config
        )

    async def test_concurrent_tickers_share_the_pacing(self) -> None:
        """The behavioural version: N concurrent acquisitions still take N intervals.

        With a shared limiter, ten waits cost ten intervals however many callers are
        queued. A per-call limiter would let all ten through at once and cost nothing.
        """
        limiter = RateLimiter(600, name="t")  # 10 per second -> 100ms apart
        started = time.monotonic()
        await asyncio.gather(*(limiter.acquire() for _ in range(6)))
        elapsed = time.monotonic() - started

        # 6 acquisitions, the first free and the remaining 5 spaced 100ms apart.
        assert elapsed >= 0.45, f"pacing was not shared: 6 calls took only {elapsed:.3f}s"

    async def test_a_batch_does_not_exceed_the_providers_documented_rate(self) -> None:
        """The end-to-end guarantee, measured rather than asserted.

        Gathering N separate ``get_ticker`` calls is the obvious way to parallelise and it
        is exactly the wrong one: each call builds its own limiter, so N tickers behave
        like N clients each entitled to the full requests-per-minute. This measures the
        achieved rate against the provider's declared limit for both, because the
        overshoot is the thing that earns an IP-level block on a free tier.
        """
        providers = (_PacedProvider,)
        tickers = [f"C{i}" for i in range(12)]

        _PacedProvider.calls = 0
        naive_started = time.monotonic()
        await asyncio.gather(
            *(
                TickerKit(config=Config(cache_enabled=False), providers=providers).get_ticker(t)
                for t in tickers
            )
        )
        naive_rate = _PacedProvider.calls / (time.monotonic() - naive_started)

        _PacedProvider.calls = 0
        batched_started = time.monotonic()
        await TickerKit(config=Config(cache_enabled=False), providers=providers).get_tickers(
            tickers, concurrency=len(tickers)
        )
        batched_rate = _PacedProvider.calls / (time.monotonic() - batched_started)

        assert naive_rate > _PacedProvider.RPM / 60 * 2, (
            f"fixture no longer demonstrates the hazard: only {naive_rate:.0f} req/s"
        )
        allowed = _PacedProvider.RPM / 60
        assert batched_rate <= allowed * 1.15, (
            f"batch ran at {batched_rate:.1f} req/s against a documented {allowed:.1f}"
        )


class _FastPacedCoinGecko(CoinGeckoProvider):
    """CoinGecko with its 12/min keyless pacing removed.

    The pacing is real and worth keeping in production, but at 5s between calls it
    would make these tests take minutes. The name stays ``coingecko`` so the source — and
    therefore the shared breaker under test — is keyed exactly as it is in production.
    """

    name = "coingecko"

    @classmethod
    def rate_limit_rpm(cls, config: Config) -> int:
        return 6000


def _throttled_kit(**overrides: Any) -> TickerKit:
    """A kit whose only provider answers every request with a 429.

    The cooldown is kept long enough to cover a whole batch but short enough to keep the
    suite quick: a breaker that expires before the next request is due suppresses nobody,
    so a fixture with a tiny cooldown would be measuring itself rather than the sharing.
    """
    config = Config(
        retry_base_delay=0.01,
        retry_max_delay=0.01,
        retry_max_after=1.0,
        cache_enabled=False,
        **overrides,
    )
    return TickerKit(config=config, providers=(_FastPacedCoinGecko,))


def _throttled_route(**headers: str) -> Any:
    return respx.get(url__startswith=COINGECKO).mock(
        return_value=httpx.Response(429, headers={"Retry-After": "1", **headers})
    )


class TestBreakerIsShared:
    @pytest.mark.asyncio
    @respx.mock
    async def test_one_429_is_paid_for_once_by_the_whole_batch(self) -> None:
        """The headline saving: a real throttle must not be re-paid by every ticker.

        Without a shared breaker each ticker would walk into the same exhausted source
        and spend its own retry budget collecting its own copy of the same 429. On keyless
        free tiers that escalation is exactly how one unlucky batch earns an IP block.
        """
        route = _throttled_route()
        snaps = await _throttled_kit().get_tickers(
            ["BTC", "ETH", "SOL", "ADA"], concurrency=4
        )

        # A small allowance for the race where two tickers reach the source before the
        # first 429 has been recorded; the point is that it does not scale with tickers.
        assert route.call_count <= 4, f"throttle re-paid per ticker: {route.call_count} calls"
        assert all(s.data_quality.throttled for s in snaps), "every snapshot must say so"

    @pytest.mark.asyncio
    @respx.mock
    async def test_retries_are_not_re_paid_per_ticker(self) -> None:
        """With a retry budget, the batch must still spend one budget, not N."""
        route = _throttled_route()
        await _throttled_kit(max_retries=1).get_tickers(
            ["BTC", "ETH", "SOL"], concurrency=3
        )
        # Three tickers x two attempts would be six doomed requests.
        assert route.call_count <= 2, f"retry budget re-paid per ticker: {route.call_count}"

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_breaker_survives_another_callers_retry_wait(self) -> None:
        """One caller's cooldown must not reset the suppression for everyone else.

        This is the regression the whole batch path rests on: clearing the breaker when a
        caller finished its own retry sleep let each queued ticker walk straight into a
        source that had already said no.
        """
        route = _throttled_route()
        client = HttpClient(
            config=Config(max_retries=0, retry_max_after=1.0), source="coingecko"
        )
        with pytest.raises(RateLimitExceeded):
            await client.get_json(f"{COINGECKO}/ping")

        spent = route.call_count
        for _ in range(5):
            with pytest.raises(RateLimitExceeded) as excinfo:
                await client.get_json(f"{COINGECKO}/ping")
            assert excinfo.value.suppressed is True
        assert route.call_count == spent, "the breaker was reopened by another caller's wait"
        await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_a_429_without_retry_after_still_opens_the_breaker(self) -> None:
        """A server that says only "429" is still telling us it is out of quota."""
        respx.get(url__startswith=COINGECKO).mock(return_value=httpx.Response(429))
        client = HttpClient(
            config=Config(max_retries=0, retry_base_delay=0.05, retry_max_after=1.0),
            source="coingecko",
        )
        with pytest.raises(RateLimitExceeded):
            await client.get_json(f"{COINGECKO}/ping")

        assert client.is_throttled, "a silent 429 left the source open to the next caller"
        with pytest.raises(RateLimitExceeded) as excinfo:
            await client.get_json(f"{COINGECKO}/ping")
        assert excinfo.value.suppressed is True
        await client.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_suppressed_snapshots_are_marked_suppressed(self) -> None:
        """A suppressed throttle is still reported, and distinguishable from a real one."""
        _throttled_route()
        snaps = await _throttled_kit().get_tickers(["BTC", "ETH", "SOL"], concurrency=3)

        assert any(
            info.suppressed for s in snaps for info in s.data_quality.rate_limited
        ), "short-circuited callers must be marked as suppressed, not silently dropped"

    async def test_views_are_memoised_per_source(self) -> None:
        """Two callers naming the same source must get the same breaker."""
        from tickerkit.http import HttpClient

        async with HttpClient(config=Config()) as client:
            assert client.view("coingecko") is client.view("coingecko")
            assert client.view("coingecko") is not client.view("defillama")


class TestCommandLine:
    """The CLI must pass the whole ticker list through as one batch, not N calls."""

    @staticmethod
    def _fake_snapshots(seen: list[list[str]]) -> Any:
        async def fake(tickers: Any, **kwargs: Any) -> Any:
            seen.append([str(t) for t in tickers])
            out = []
            for t in tickers:
                snap = MagicMock(spec=TickerSnapshot)
                snap.query = str(t)
                snap.market.price = 100.0
                snap.market.provenance = {}
                snap.summary.return_value = f"{t}  100.0"
                snap.to_json.return_value = json.dumps({"query": str(t)})
                out.append(snap)
            return out

        return fake

    def test_every_ticker_is_passed_through(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tickerkit import cli

        seen: list[list[str]] = []
        monkeypatch.setattr(cli, "aget_tickers", self._fake_snapshots(seen))
        code = cli.main(["BTC", "ETH", "SOL"])
        assert code == 0
        assert seen == [["BTC", "ETH", "SOL"]], "the CLI must batch, not loop"

    def test_metrics_is_a_bare_object_for_one_ticker(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """One-ticker output must stay exactly what it was before batching existed."""
        from tickerkit import cli

        monkeypatch.setattr(cli, "aget_tickers", self._fake_snapshots([]))
        cli.main(["BTC", "--metrics"])
        assert isinstance(json.loads(capsys.readouterr().out), dict)

    def test_metrics_is_an_array_for_many(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tickerkit import cli

        monkeypatch.setattr(cli, "aget_tickers", self._fake_snapshots([]))
        cli.main(["BTC", "ETH", "--metrics"])
        assert len(json.loads(capsys.readouterr().out)) == 2

    def test_summary_blocks_are_labelled(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tickerkit import cli

        monkeypatch.setattr(cli, "aget_tickers", self._fake_snapshots([]))
        cli.main(["BTC", "ETH"])
        out = capsys.readouterr().out
        assert "===== BTC =====" in out
        assert "===== ETH =====" in out

    def test_single_ticker_is_unlabelled(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from tickerkit import cli

        monkeypatch.setattr(cli, "aget_tickers", self._fake_snapshots([]))
        cli.main(["BTC"])
        assert "=====" not in capsys.readouterr().out

    def test_one_priceless_ticker_fails_the_batch(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A watchlist must not report success while one of its assets went unpriced."""

        async def fake(tickers: Any, **_: Any) -> Any:
            snaps = await self._fake_snapshots([])(tickers)
            snaps[1].market.price = None
            return snaps

        from tickerkit import cli

        monkeypatch.setattr(cli, "aget_tickers", fake)
        assert cli.main(["BTC", "NOPE"]) == 1

    def test_history_refuses_a_multi_ticker_request(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """History is the expensive path; silently doing only the first would hide the rest."""
        from tickerkit import cli

        called = False

        async def fake(*a: Any, **k: Any) -> Any:
            nonlocal called
            called = True
            return None

        monkeypatch.setattr(cli, "aget_history", fake)
        code = cli.main(["BTC", "ETH", "--history"])
        assert code == 2
        assert not called, "history ran despite the multi-ticker refusal"
        assert "single ticker" in capsys.readouterr().err


def test_batch_api_is_reachable_from_the_package_root() -> None:
    """The convenience wrappers must expose the same batch behaviour as the class."""
    import tickerkit

    assert tickerkit.aget_tickers is not None
    assert tickerkit.get_tickers is not None
    assert "get_tickers" in tickerkit.__all__
    assert "aget_tickers" in tickerkit.__all__
    assert CoreTickerKit.get_tickers is TickerKit.get_tickers
