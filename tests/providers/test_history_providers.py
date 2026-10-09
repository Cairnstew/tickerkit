"""History adapters and the deep-pagination walk, replayed from recorded fixtures.

No network is touched: CoinGecko and DefiLlama answer through ``respx``; ccxt's pager is
driven by a fake exchange so the walk itself is tested rather than a live venue.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import ClassVar

import httpx
import pytest
import respx

from tickerkit.cache import MemoryCache
from tickerkit.capabilities import Capability
from tickerkit.config import Config
from tickerkit.history import build_coverage
from tickerkit.http import HttpClient
from tickerkit.models import HistoryCandle, HistoryRequest, HistorySeries
from tickerkit.providers.base import Provider, ProviderContext, provenance_for
from tickerkit.providers.ccxt_provider import (
    _candles_from_rows,
    _exchange_max_limit,
    _ohlcv_fetcher,
    _paginate_ohlcv,
)
from tickerkit.providers.coingecko import DEMO_ROOT, PRO_ROOT, CoinGeckoProvider
from tickerkit.providers.defillama import COINS_BASE_URL, DefiLlamaProvider
from tickerkit.resolve import ResolvedAsset

from ..conftest import load_fixture

DAY_MS = 86_400_000
START_MS = 1_704_067_200_000  # 2024-01-01T00:00:00Z


def _context(config: Config | None = None) -> tuple[Config, HttpClient, ProviderContext]:
    config = config or Config(cache_enabled=False, max_retries=0)
    http = HttpClient(config=config)
    return config, http, ProviderContext(config=config, http=http, cache=MemoryCache())


def _request(**kwargs: object) -> HistoryRequest:
    values: dict[str, object] = {"timeframe": "1d"}
    values.update(kwargs)
    return HistoryRequest(**values)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- coingecko


class TestCoinGeckoHistory:
    @pytest.mark.asyncio
    @respx.mock
    async def test_parses_close_only_series_and_marks_it_price_kind(self) -> None:
        # A Pro key is used because the recorded window (Jan 2024) is older than the
        # keyless 365-day limit; the keyless clamp is exercised in the next test.
        respx.get(f"{PRO_ROOT}/coins/bitcoin/market_chart/range").mock(
            return_value=httpx.Response(
                200, json=load_fixture("coingecko_market_chart_range.json")
            )
        )
        _config, http, ctx = _context(Config(cache_enabled=False, max_retries=0, coingecko_pro_api_key="k"))
        provider = CoinGeckoProvider(ctx)
        asset = ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")
        request = _request(
            start=datetime(2024, 1, 1, tzinfo=UTC), end=datetime(2024, 1, 5, tzinfo=UTC)
        )

        series = await provider.fetch_history(asset, request)

        assert len(series) == 1
        result = series[0]
        assert result.kind == "price"
        assert len(result.candles) == 5
        # Close-only: OHLC is never invented from the single price.
        assert result.candles[0].open is None
        assert result.candles[0].close == 42000.0
        assert result.coverage is not None and result.coverage.complete is True
        assert ctx.provenance["history:coingecko:1d"].capability is Capability.HISTORY
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_public_api_window_is_clamped_and_reported(self) -> None:
        """A pre-365-day request is clamped, not failed, and says it was truncated."""
        route = respx.get(f"{DEMO_ROOT}/coins/bitcoin/market_chart/range").mock(
            return_value=httpx.Response(
                200, json=load_fixture("coingecko_market_chart_range.json")
            )
        )
        _config, http, ctx = _context()
        provider = CoinGeckoProvider(ctx)
        asset = ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")
        request = _request(start=datetime(2017, 1, 1, tzinfo=UTC))

        series = await provider.fetch_history(asset, request)

        coverage = series[0].coverage
        assert coverage is not None
        assert coverage.truncated is True
        assert coverage.complete is False
        assert coverage.note is not None and "365 days" in coverage.note
        # The clamp is real: the outgoing `from` is not the requested 2017 start.
        assert "from=1483228800" not in str(route.calls[0].request.url)
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_error_envelope_shape_is_rejected(self) -> None:
        """``{"error": {"status": ...}}`` must raise, not flow into the price reader."""
        from tickerkit.errors import ProviderError

        respx.get(f"{DEMO_ROOT}/coins/bitcoin/market_chart/range").mock(
            return_value=httpx.Response(
                200,
                json={
                    "error": {
                        "status": {
                            "error_code": 10012,
                            "error_message": "Your request exceeds the allowed time range.",
                        }
                    }
                },
            )
        )
        _config, http, ctx = _context()
        provider = CoinGeckoProvider(ctx)
        asset = ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")

        with pytest.raises(ProviderError) as excinfo:
            await provider.fetch_history(asset, _request())
        assert "10012" in str(excinfo.value)
        await http.aclose()

    @pytest.mark.asyncio
    async def test_unresolved_asset_yields_nothing(self) -> None:
        _config, http, ctx = _context()
        provider = CoinGeckoProvider(ctx)
        assert await provider.fetch_history(ResolvedAsset(query="X", symbol="X"), _request()) == ()
        await http.aclose()


# --------------------------------------------------------------------------- defillama


class TestDefiLlamaHistory:
    @pytest.mark.asyncio
    @respx.mock
    async def test_parses_chart_points_using_the_gecko_join_key(self) -> None:
        route = respx.get(url__startswith=f"{COINS_BASE_URL}/chart/").mock(
            return_value=httpx.Response(200, json=load_fixture("defillama_chart_btc.json"))
        )
        _config, http, ctx = _context()
        provider = DefiLlamaProvider(ctx)
        asset = ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")

        series = await provider.fetch_history(asset, _request())

        assert len(series) == 1
        assert series[0].kind == "price"
        assert [c.close for c in series[0].candles] == [42000.0, 43000.0, 42500.0]
        # The join key is the CoinGecko id, encoded literally.
        assert "coingecko:bitcoin" in str(route.calls[0].request.url)
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_sub_hourly_timeframe_is_declined_with_a_reason(self) -> None:
        _config, http, ctx = _context()
        provider = DefiLlamaProvider(ctx)
        asset = ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")

        series = await provider.fetch_history(asset, _request(timeframe="5m"))

        assert series == ()
        assert any("history.5m" in m.field for m in ctx.missing)
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_unresolved_asset_is_never_guessed(self) -> None:
        _config, http, ctx = _context()
        provider = DefiLlamaProvider(ctx)
        assert await provider.fetch_history(ResolvedAsset(query="BTC", symbol="BTC"), _request()) == ()
        await http.aclose()

    @pytest.mark.asyncio
    @respx.mock
    async def test_empty_coins_object_is_missing_not_zero(self) -> None:
        respx.get(url__startswith=f"{COINS_BASE_URL}/chart/").mock(
            return_value=httpx.Response(200, json={"coins": {}})
        )
        _config, http, ctx = _context()
        provider = DefiLlamaProvider(ctx)
        asset = ResolvedAsset(query="BTC", symbol="BTC", coingecko_id="bitcoin")

        assert await provider.fetch_history(asset, _request()) == ()
        assert any("history.1d" in m.field for m in ctx.missing)
        await http.aclose()


# --------------------------------------------------------------------------- ccxt pager


class _FakeExchange:
    """Serves a fixed candle list, slicing by ``since`` like a real venue."""

    def __init__(self, rows: list[list[float]], *, ignore_since: bool = False) -> None:
        self.rows = rows
        self.ignore_since = ignore_since
        self.calls: list[int | None] = []

    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, since: int | None = None, limit: int | None = None
    ) -> list[list[float]]:
        self.calls.append(since)
        limit = limit or len(self.rows)
        if since is None or self.ignore_since:
            return self.rows[:limit]
        return [row for row in self.rows if row[0] >= since][:limit]


def _rows(count: int) -> list[list[float]]:
    return [
        [START_MS + i * DAY_MS, 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i, 10.0]
        for i in range(count)
    ]


class TestCcxtPager:
    @pytest.mark.asyncio
    async def test_walks_forward_until_the_window_is_covered(self) -> None:
        exchange = _FakeExchange(_rows(10))
        fetch = _ohlcv_fetcher(exchange, "BTC/USDT", "1d")
        rows, calls, truncated = await _paginate_ohlcv(
            fetch,
            interval_ms=DAY_MS,
            start_ms=START_MS,
            end_ms=START_MS + 9 * DAY_MS,
            page_limit=3,
            max_calls=10,
        )
        assert len(rows) == 10
        assert calls == 4  # three full pages of 3 plus the final 1
        assert truncated is False

    @pytest.mark.asyncio
    async def test_stops_and_flags_truncation_at_the_call_ceiling(self) -> None:
        exchange = _FakeExchange(_rows(10))
        fetch = _ohlcv_fetcher(exchange, "BTC/USDT", "1d")
        rows, calls, truncated = await _paginate_ohlcv(
            fetch,
            interval_ms=DAY_MS,
            start_ms=START_MS,
            end_ms=START_MS + 9 * DAY_MS,
            page_limit=3,
            max_calls=2,
        )
        assert calls == 2
        assert truncated is True
        assert len(rows) == 6

    @pytest.mark.asyncio
    async def test_a_venue_that_ignores_since_cannot_spin(self) -> None:
        """Kraken-style responses would otherwise loop forever re-requesting the top."""
        exchange = _FakeExchange(_rows(5), ignore_since=True)
        fetch = _ohlcv_fetcher(exchange, "BTC/USDT", "1d")
        rows, calls, _ = await _paginate_ohlcv(
            fetch,
            interval_ms=DAY_MS,
            start_ms=START_MS,
            end_ms=START_MS + 100 * DAY_MS,
            page_limit=3,
            max_calls=10,
        )
        assert calls == 2, "must stop as soon as a page adds nothing new"
        assert len(rows) == 3

    @pytest.mark.asyncio
    async def test_empty_page_ends_the_walk(self) -> None:
        exchange = _FakeExchange([])
        fetch = _ohlcv_fetcher(exchange, "BTC/USDT", "1d")
        rows, calls, truncated = await _paginate_ohlcv(
            fetch,
            interval_ms=DAY_MS,
            start_ms=START_MS,
            end_ms=START_MS + DAY_MS,
            page_limit=1000,
            max_calls=10,
        )
        assert rows == [] and calls == 1 and truncated is False

    def test_row_conversion_drops_malformed_rows(self) -> None:
        rows = [
            [START_MS, 1.0, 2.0, 0.5, 1.5, 10.0],
            [START_MS + DAY_MS, 1.0],  # too short
            [START_MS + 2 * DAY_MS, float("nan"), 2.0, 0.5, 1.5, 10.0],  # non-finite
        ]
        candles = _candles_from_rows(rows)
        assert len(candles) == 1
        assert candles[0].close == 1.5 and candles[0].is_ohlcv

    def test_max_limit_reads_the_exchange_advertisement(self) -> None:
        class _Ex:
            options: ClassVar[dict[str, object]] = {"fetchOHLCV": {"maxLimit": 300}}

        assert _exchange_max_limit(_Ex()) == 300
        assert _exchange_max_limit(object()) == 1000


# --------------------------------------------------------------------------- orchestration


class _FakeHistoryProvider(Provider):
    name = "fakehist"
    supports = frozenset({Capability.RESOLVE, Capability.HISTORY})

    async def resolve(self, query: str) -> ResolvedAsset | None:
        return ResolvedAsset(
            query=query, symbol="BTC", name="Bitcoin", coingecko_id="bitcoin"
        )

    async def fetch_history(
        self, asset: ResolvedAsset, request: HistoryRequest
    ) -> tuple[HistorySeries, ...]:
        candles = tuple(
            HistoryCandle(
                timestamp=datetime(2024, 1, d, tzinfo=UTC),
                close=float(42000 + d),
                open=float(42000 + d),
                high=float(42000 + d),
                low=float(42000 + d),
                volume=1.0,
            )
            for d in (1, 2, 3)
        )
        return (
            HistorySeries(
                source="fakehist:btc",
                symbol="BTC",
                timeframe=request.timeframe,
                quote_currency="USD",
                kind="ohlcv",
                candles=candles,
                coverage=build_coverage([c.timestamp for c in candles], request, calls=1),
                provenance=provenance_for(
                    source="fakehist:btc",
                    capability=Capability.HISTORY,
                    url="https://fake.test/btc",
                ),
            ),
        )


class TestHistoryOrchestration:
    @pytest.mark.asyncio
    async def test_get_history_resolves_then_fans_out(self) -> None:
        from tickerkit.core import TickerKit

        kit = TickerKit(providers=(_FakeHistoryProvider,), config=Config(cache_ttl=0))
        result = await kit.get_history("BTC", timeframe="1d")

        assert result.identity.coingecko_id == "bitcoin"
        assert len(result.series) == 1
        assert result.consensus is not None
        assert result.consensus.candles[0].close == 42001.0
        assert "fakehist" in result.data_quality.providers_used
        assert result.data_quality.provenance  # provenance flowed through

    @pytest.mark.asyncio
    async def test_no_provider_explains_the_empty_result(self) -> None:
        from tickerkit.core import TickerKit

        kit = TickerKit(providers=(), config=Config(cache_ttl=0))
        result = await kit.get_history("BTC", timeframe="1d")

        assert result.series == () and result.consensus is None
        assert any(m.field == "history" for m in result.data_quality.missing)

    @pytest.mark.asyncio
    async def test_incomplete_window_is_surfaced_in_quality(self) -> None:
        from tickerkit.core import TickerKit

        kit = TickerKit(providers=(_FakeHistoryProvider,), config=Config(cache_ttl=0))
        # The fake only has 2024-01-01..03; asking for a wider window must say so.
        result = await kit.get_history(
            "BTC",
            timeframe="1d",
            start=datetime(2023, 12, 1, tzinfo=UTC),
            end=datetime(2024, 1, 10, tzinfo=UTC),
        )
        assert any(m.field == "history.coverage" for m in result.data_quality.missing)

    @pytest.mark.asyncio
    async def test_bounded_complete_window_is_not_flagged(self) -> None:
        from tickerkit.core import TickerKit

        kit = TickerKit(providers=(_FakeHistoryProvider,), config=Config(cache_ttl=0))
        result = await kit.get_history(
            "BTC",
            timeframe="1d",
            start=datetime(2024, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 3, tzinfo=UTC),
        )
        assert not any(m.field == "history.coverage" for m in result.data_quality.missing)
