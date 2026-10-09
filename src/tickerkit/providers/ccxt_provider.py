"""ccxt provider: exchange-direct market data.

Docs (read before writing this adapter, per project rule):
  * Manual / Overview     — https://docs.ccxt.com/en/latest/exchanges.html
  * Manual / Markets      — https://docs.ccxt.com/#markets
  * Manual / Implicit API — https://docs.ccxt.com/#implicit-api
  * Manual / Error Handling — https://docs.ccxt.com/#error-handling

Facts this adapter depends on, all from the pages above:
  * Every exchange class implements the same *unified* public methods: ``fetch_ticker``,
    ``fetch_tickers``, ``fetch_order_book``, ``fetch_trades``, ``fetch_ohlcv``,
    ``load_markets``. Support is **not** uniform, so ``exchange.has[...]`` must be checked
    before every call — this adapter never assumes.
  * ``load_markets`` populates ``exchange.markets`` keyed by unified symbol such as
    ``"BTC/USDT"``; each market dict carries ``base``, ``quote``, ``type`` (spot/swap/
    future), ``active`` and ``spot/swap/future``.
  * ``exchange.rateLimit`` is the exchange's own documented request weight in
    **milliseconds**; tickerkit reads it at runtime instead of hardcoding limits.
  * Ticker dicts carry ``bid``/``ask``/``last``/``quoteVolume``/``baseVolume`` and a
    millisecond ``timestamp``.
  * ``fetch_order_book`` returns ``{"bids": [[price, amount], ...],
    "asks": [[price, amount], ...], "timestamp": ...}`` with bids descending.
  * ``fetch_ohlcv`` returns ``[[ts_ms, open, high, low, close, volume], ...]``.
  * ``fetch_trades`` returns ``[{"timestamp", "price", "amount", "side", "id", ...}, ...]``.
  * Derivatives are exposed through the ``has`` keys ``fetchFundingRate``,
    ``fetchOpenInterest``, ``fetchMarkPrice``, ``fetchIndexPrice``; funding rate is a
    fractional rate per the exchange's funding interval.

Only public, unauthenticated endpoints are used.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any, cast

import ccxt.async_support as ccxt_async

from ..capabilities import Capability
from ..errors import UnsupportedCapability
from ..history import build_coverage, interval_seconds, normalise_candles
from ..models import (
    Candle,
    DerivativesData,
    HistoryCandle,
    HistoryRequest,
    HistorySeries,
    OHLCVSeries,
    OrderBook,
    Trade,
    VenueTicker,
    as_utc,
    finite,
    utcnow,
)
from ..normalise import (
    book_levels,
    depth_within,
    epoch_ms_to_utc,
    imbalance_from,
    spread_from,
)
from ..resolve import MarketRef, ResolvedAsset
from .base import Provider, provenance_for

#: Maps a ccxt ``has`` key to the snapshot field it would populate, for reporting.
_CAP_FIELD: dict[str, str] = {
    "fetchFundingRate": "funding_rate",
    "fetchOpenInterest": "open_interest",
    "fetchMarkPrice": "mark_price",
    "fetchIndexPrice": "index_price",
}

#: Depth bands reported for every order book, as fractions of the mid.
DEPTH_BANDS: tuple[float, ...] = (0.005, 0.01, 0.02)

#: Quote currencies preferred when an exchange lists several pairs for the base asset.
_QUOTE_PREFERENCE: tuple[str, ...] = ("USD", "USDT", "USDC", "EUR", "BTC")

#: Quote preference for perpetuals.
#:
#: Linear, USDT-marginned contracts are preferred over inverse (USD:BTC) perps even when
#: the snapshot's quote currency is USD: a linear contract's ``contractSize`` is the base
#: amount, so open interest converts unambiguously, whereas an inverse contract's
#: ``contractSize`` is the *settled* fiat value and would need the inverse conversion.
_PERP_QUOTE_PREFERENCE: tuple[str, ...] = ("USDT", "USDC", "USD", "BTC")


class CcxtProvider(Provider):
    """Exchange-direct data via ``ccxt.async_support``."""

    name = "ccxt"
    supports = frozenset(
        {
            Capability.RESOLVE,
            Capability.VENUE_TICKER,
            Capability.ORDERBOOK,
            Capability.TRADES,
            Capability.OHLCV,
            Capability.HISTORY,
            Capability.DERIVATIVES,
        }
    )

    def __init__(self, context: Any) -> None:
        super().__init__(context)
        self._exchanges: dict[str, Any] = {}

    # -- lifecycle --------------------------------------------------------------------

    def _exchange(self, exchange_id: str) -> Any:
        """Instantiate (once) an async ccxt exchange with our timeout settings."""
        if exchange_id not in self._exchanges:
            if not hasattr(ccxt_async, exchange_id):
                raise UnsupportedCapability(f"ccxt has no exchange id {exchange_id!r}")
            cls = getattr(ccxt_async, exchange_id)
            self._exchanges[exchange_id] = cls(
                {
                    "enableRateLimit": True,
                    "timeout": int(self.ctx.config.http_timeout * 1000),
                }
            )
        return self._exchanges[exchange_id]

    async def aclose(self) -> None:
        for exchange in self._exchanges.values():
            await exchange.close()
        self._exchanges.clear()

    async def _load_markets(self, exchange_id: str) -> Any:
        exchange = self._exchange(exchange_id)
        await exchange.load_markets()
        return exchange

    # -- resolve ----------------------------------------------------------------------

    async def resolve(self, query: str) -> ResolvedAsset | None:
        """Discover tradable markets for the base symbol across the configured venues."""
        from ..resolve import normalise_query

        base = normalise_query(query)
        quote = self.ctx.config.quote_currency
        spot: list[MarketRef] = []
        derivative: list[MarketRef] = []
        notes: list[str] = []

        spot_ids = self.ctx.config.exchanges
        perp_ids = self.ctx.config.perp_exchanges
        await asyncio.gather(
            *(
                self._scan_exchange(eid, base, quote, spot, derivative, notes)
                for eid in spot_ids
            ),
            *(
                self._scan_exchange(
                    eid, base, quote, spot, derivative, notes, derivatives=True
                )
                for eid in perp_ids
            ),
            return_exceptions=True,
        )
        if not spot and not derivative:
            return None
        return ResolvedAsset(
            query=query,
            symbol=base,
            spot_markets=tuple(spot),
            derivative_markets=tuple(derivative),
            notes=tuple(notes),
        )

    async def _scan_exchange(
        self,
        exchange_id: str,
        base: str,
        quote: str,
        spot: list[MarketRef],
        derivative: list[MarketRef],
        notes: list[str],
        *,
        derivatives: bool = False,
    ) -> None:
        """Find the best market for ``base`` on one exchange. Never raises."""
        try:
            exchange = await self._load_markets(exchange_id)
        except Exception as exc:
            notes.append(f"{exchange_id}: markets unavailable ({type(exc).__name__})")
            return

        candidates = [
            MarketRef(
                exchange=exchange_id,
                symbol=symbol,
                quote_currency=str(market.get("quote") or "").upper(),
                kind=str(market.get("type") or "spot"),
            )
            for symbol, market in exchange.markets.items()
            if str(market.get("base") or "").upper() == base
            and market.get("active", True)
        ]
        wanted = {"swap", "future"} if derivatives else {"spot"}
        candidates = [c for c in candidates if c.kind in wanted]
        if not candidates:
            kind_label = "perp" if derivatives else "spot"
            notes.append(f"{exchange_id}: no {kind_label} {base}/{quote} market")
            return

        bucket = derivative if derivatives else spot
        bucket.extend(_rank_markets(candidates, derivatives, quote))
        picked = _rank_markets(candidates, derivatives, quote)[0]
        notes.append(f"{exchange_id}: selected {picked.symbol} ({picked.kind})")

    # -- helpers ----------------------------------------------------------------------

    def _market_for(self, asset: ResolvedAsset, exchange_id: str) -> MarketRef | None:
        refs = asset.markets_for(exchange_id)
        return _prefer_quote(refs, self.ctx.config.quote_currency)

    def _deriv_for(self, asset: ResolvedAsset, exchange_id: str) -> MarketRef | None:
        """Prefer the perpetual; fall back to the first dated contract."""
        refs = asset.derivatives_for(exchange_id)
        if not refs:
            return None
        swaps = [r for r in refs if r.kind == "swap"]
        return swaps[0] if swaps else refs[0]

    async def _deepest_market(
        self,
        exchange: Any,
        asset: ResolvedAsset,
        exchange_id: str,
        timeframe: str,
        start_ms: int,
    ) -> tuple[MarketRef | None, int]:
        """Pick the market whose history goes back furthest.

        Quote preference alone is not enough for history: Binance now lists a spot
        ``BTC/USD`` fiat pair with a far shorter history than ``BTC/USDT``, and picking
        the better-quoted pair would silently return a short series for a deep request.
        Each candidate is probed from ``start_ms`` and the one whose oldest candle is
        earliest wins; ties fall back to the best-quoted of those. The probe cost is
        counted in the returned series' ``calls``.
        """
        best = self._market_for(asset, exchange_id)
        candidates = _rank_markets(
            list(asset.markets_for(exchange_id)), False, self.ctx.config.quote_currency
        )
        if best is None or len(candidates) <= 1:
            return best, 0
        oldest_by: dict[str, int] = {}
        probe_calls = 0
        for candidate in candidates:
            try:
                rows = cast(
                    "list[list[float]]",
                    await exchange.fetch_ohlcv(
                        candidate.symbol, timeframe, since=start_ms, limit=3
                    ),
                )
            except Exception:
                continue
            probe_calls += 1
            if rows and len(rows) >= 1:
                oldest_by[candidate.symbol] = int(rows[0][0])
        if not oldest_by:
            return best, probe_calls
        deepest = min(oldest_by, key=oldest_by.__getitem__)
        for candidate in candidates:
            if candidate.symbol == deepest:
                return candidate, probe_calls
        return best, probe_calls

    def _supports(self, exchange: Any, key: str) -> bool:
        """Whether ``exchange.has[key]`` means the venue genuinely implements it.

        ccxt uses the string ``"emulated"`` to mark a method that only *synthesises* a
        plausible response (bybit's ``fetchFundingRate`` returns a constant 0.0001).
        Reporting such a value would be fabricating data, so it is treated as
        unsupported and recorded in ``data_quality.missing`` instead.
        """
        value = exchange.has.get(key)
        if isinstance(value, str):
            return value != "emulated"
        return bool(value)

    # -- venue tickers ----------------------------------------------------------------

    async def fetch_venue_tickers(self, asset: ResolvedAsset) -> tuple[VenueTicker, ...]:
        results = await asyncio.gather(
            *(self._fetch_ticker_one(eid, asset) for eid in self.ctx.config.exchanges),
            return_exceptions=True,
        )
        out: list[VenueTicker] = []
        for exchange_id, result in zip(self.ctx.config.exchanges, results, strict=True):
            if isinstance(result, BaseException):
                self.ctx.note_error(self.name, Capability.VENUE_TICKER, result)
                continue
            if result is None:
                self.ctx.note_missing(
                    f"venues.{exchange_id}",
                    f"no active {self.ctx.config.quote_currency} market for {asset.symbol}",
                )
                continue
            out.append(result)
        return tuple(out)

    async def _fetch_ticker_one(self, exchange_id: str, asset: ResolvedAsset) -> VenueTicker | None:
        market = self._market_for(asset, exchange_id)
        if market is None:
            return None
        exchange = await self._load_markets(exchange_id)
        if not self._supports(exchange, "fetchTicker"):
            raise UnsupportedCapability(f"{exchange_id} does not support fetchTicker")

        raw = cast(dict[str, Any], await exchange.fetch_ticker(market.symbol))
        bid = finite(raw.get("bid"))
        ask = finite(raw.get("ask"))
        spread, spread_bps = spread_from(bid, ask)
        timestamp = epoch_ms_to_utc(raw.get("timestamp"))
        notes: tuple[str, ...] = ()
        quoted_in_stablecoin = (
            self.ctx.config.is_fiat_equivalent(market.quote_currency)
            and market.quote_currency != self.ctx.config.quote_currency
        )
        if quoted_in_stablecoin:
            notes = (
                f"quoted in {market.quote_currency}; read as "
                f"{self.ctx.config.quote_currency} via the configured "
                "stablecoin assumption",
            )

        prov = provenance_for(
            source=f"ccxt:{exchange_id}:fetch_ticker",
            capability=Capability.VENUE_TICKER,
            exchange_timestamp=timestamp,
            url=_exchange_url(exchange),
            notes=notes,
        )
        self.ctx.record(f"venue:{exchange_id}", prov)

        return VenueTicker(
            exchange=exchange_id,
            symbol=market.symbol,
            quote_currency=self.ctx.config.quote_currency,
            bid=bid,
            ask=ask,
            last=finite(raw.get("last")),
            spread=spread,
            spread_bps=spread_bps,
            volume_24h_base=finite(raw.get("baseVolume")),
            volume_24h_quote=finite(raw.get("quoteVolume")),
            timestamp=timestamp,
            provenance=prov,
        )

    # -- order book -------------------------------------------------------------------

    async def fetch_orderbooks(self, asset: ResolvedAsset) -> tuple[OrderBook, ...]:
        results = await asyncio.gather(
            *(self._fetch_book_one(eid, asset) for eid in self.ctx.config.exchanges),
            return_exceptions=True,
        )
        out: list[OrderBook] = []
        for _exchange_id, result in zip(self.ctx.config.exchanges, results, strict=True):
            if isinstance(result, BaseException):
                self.ctx.note_error(self.name, Capability.ORDERBOOK, result)
                continue
            if result is not None:
                out.append(result)
        return tuple(out)

    async def _fetch_book_one(self, exchange_id: str, asset: ResolvedAsset) -> OrderBook | None:
        market = self._market_for(asset, exchange_id)
        if market is None:
            return None
        exchange = await self._load_markets(exchange_id)
        if not self._supports(exchange, "fetchOrderBook"):
            raise UnsupportedCapability(f"{exchange_id} does not support fetchOrderBook")

        raw = cast(dict[str, Any], await exchange.fetch_order_book(
            market.symbol, limit=self.ctx.config.orderbook_depth
        ))
        bids = book_levels(raw.get("bids"), ascending=False)
        asks = book_levels(raw.get("asks"), ascending=True)
        if not bids or not asks:
            self.ctx.note_missing(f"orderbook.{exchange_id}", "empty book returned by exchange")
            return None

        best_bid = bids[0].price
        best_ask = asks[0].price
        spread, spread_bps = spread_from(best_bid, best_ask)
        mid = (best_bid + best_ask) / 2 if best_ask >= best_bid else None
        bid_depth = sum(lvl.amount for lvl in bids)
        ask_depth = sum(lvl.amount for lvl in asks)
        timestamp = epoch_ms_to_utc(raw.get("timestamp"))

        prov = provenance_for(
            source=f"ccxt:{exchange_id}:fetch_order_book",
            capability=Capability.ORDERBOOK,
            exchange_timestamp=timestamp,
            url=_exchange_url(exchange),
        )
        self.ctx.record(f"orderbook:{exchange_id}", prov)

        return OrderBook(
            exchange=exchange_id,
            symbol=market.symbol,
            quote_currency=self.ctx.config.quote_currency,
            bids=bids,
            asks=asks,
            best_bid=best_bid,
            best_ask=best_ask,
            mid=mid,
            spread=spread,
            spread_bps=spread_bps,
            bid_depth_base=bid_depth,
            ask_depth_base=ask_depth,
            depth_quote_within=depth_within(bids, asks, DEPTH_BANDS, mid),
            imbalance=imbalance_from(bid_depth, ask_depth),
            timestamp=timestamp,
            provenance=prov,
        )

    # -- trades -----------------------------------------------------------------------

    async def fetch_trades(self, asset: ResolvedAsset) -> tuple[Trade, ...]:
        results = await asyncio.gather(
            *(self._fetch_trades_one(eid, asset) for eid in self.ctx.config.exchanges),
            return_exceptions=True,
        )
        out: list[Trade] = []
        for _exchange_id, result in zip(self.ctx.config.exchanges, results, strict=True):
            if isinstance(result, BaseException):
                self.ctx.note_error(self.name, Capability.TRADES, result)
                continue
            out.extend(result or ())
        out.sort(key=lambda t: t.timestamp, reverse=True)
        return tuple(out[: self.ctx.config.trades_limit])

    async def _fetch_trades_one(self, exchange_id: str, asset: ResolvedAsset) -> tuple[Trade, ...]:
        market = self._market_for(asset, exchange_id)
        if market is None:
            return ()
        exchange = await self._load_markets(exchange_id)
        if not self._supports(exchange, "fetchTrades"):
            raise UnsupportedCapability(f"{exchange_id} does not support fetchTrades")

        raw = cast(list[dict[str, Any]], await exchange.fetch_trades(
            market.symbol, limit=self.ctx.config.trades_limit
        ))
        trades: list[Trade] = []
        for row in raw:
            ts = epoch_ms_to_utc(row.get("timestamp"))
            price = finite(row.get("price"))
            amount = finite(row.get("amount"))
            if ts is None or price is None or amount is None:
                continue
            trades.append(
                Trade(
                    exchange=exchange_id,
                    symbol=market.symbol,
                    timestamp=ts,
                    price=price,
                    amount=amount,
                    side=row.get("side"),
                    trade_id=str(row["id"]) if row.get("id") is not None else None,
                )
            )
        return tuple(trades)

    # -- ohlcv ------------------------------------------------------------------------

    async def fetch_ohlcv(self, asset: ResolvedAsset) -> tuple[OHLCVSeries, ...]:
        results = await asyncio.gather(
            *(
                self._fetch_ohlcv_one(exchange_id, asset, timeframe)
                for exchange_id in self.ctx.config.exchanges
                for timeframe in self.ctx.config.ohlcv_timeframes
            ),
            return_exceptions=True,
        )
        # Flatten (exchange, timeframe) pairs in the same order as the gather() above.
        pairs = [
            (exchange_id, timeframe)
            for exchange_id in self.ctx.config.exchanges
            for timeframe in self.ctx.config.ohlcv_timeframes
        ]
        out: list[OHLCVSeries] = []
        for (_exchange_id, _timeframe), result in zip(pairs, results, strict=True):
            if isinstance(result, BaseException):
                self.ctx.note_error(self.name, Capability.OHLCV, result)
                continue
            if result is not None:
                out.append(result)
        return tuple(out)

    async def _fetch_ohlcv_one(
        self, exchange_id: str, asset: ResolvedAsset, timeframe: str
    ) -> OHLCVSeries | None:
        market = self._market_for(asset, exchange_id)
        if market is None:
            return None
        exchange = await self._load_markets(exchange_id)
        if not self._supports(exchange, "fetchOHLCV"):
            raise UnsupportedCapability(f"{exchange_id} does not support fetchOHLCV")

        raw = cast(list[list[float]], await exchange.fetch_ohlcv(
            market.symbol, timeframe, limit=self.ctx.config.ohlcv_limit
        ))
        candles: list[Candle] = []
        for row in raw:
            if len(row) < 6:
                continue
            ts = epoch_ms_to_utc(row[0])
            o, h, low_, c, v = (finite(x) for x in row[1:6])
            if ts is None or o is None or h is None or low_ is None or c is None or v is None:
                continue
            candles.append(
                Candle(timestamp=ts, open=o, high=h, low=low_, close=c, volume=v)
            )
        if not candles:
            return None

        prov = provenance_for(
            source=f"ccxt:{exchange_id}:fetch_ohlcv",
            capability=Capability.OHLCV,
            exchange_timestamp=candles[-1].timestamp,
            url=_exchange_url(exchange),
            notes=(f"timeframe={timeframe}",),
        )
        self.ctx.record(f"ohlcv:{exchange_id}:{timeframe}", prov)
        return OHLCVSeries(
            exchange=exchange_id,
            symbol=market.symbol,
            timeframe=timeframe,
            quote_currency=self.ctx.config.quote_currency,
            candles=tuple(candles),
            provenance=prov,
        )

    # -- deep history -----------------------------------------------------------------

    async def fetch_history(
        self, asset: ResolvedAsset, request: HistoryRequest
    ) -> tuple[HistorySeries, ...]:
        """Walk each venue back to the requested start, paginating as needed.

        Unlike :meth:`fetch_ohlcv` (a single recent page used to decorate a snapshot),
        this pages forward from ``request.start`` until the window is covered, the venue
        runs out of data, or ``history_max_calls`` is hit — in which case the series is
        marked truncated rather than silently short.
        """
        results = await asyncio.gather(
            *(self._fetch_history_one(eid, asset, request) for eid in self.ctx.config.exchanges),
            return_exceptions=True,
        )
        out: list[HistorySeries] = []
        for _exchange_id, result in zip(self.ctx.config.exchanges, results, strict=True):
            if isinstance(result, BaseException):
                self.ctx.note_error(self.name, Capability.HISTORY, result)
                continue
            if result is not None:
                out.append(result)
        return tuple(out)

    async def _fetch_history_one(
        self, exchange_id: str, asset: ResolvedAsset, request: HistoryRequest
    ) -> HistorySeries | None:
        market = self._market_for(asset, exchange_id)
        if market is None:
            return None
        exchange = await self._load_markets(exchange_id)
        if not self._supports(exchange, "fetchOHLCV"):
            raise UnsupportedCapability(f"{exchange_id} does not support fetchOHLCV")

        interval = interval_seconds(request.timeframe)
        if interval is None:
            raise UnsupportedCapability(f"unsupported timeframe {request.timeframe!r}")

        page_limit = _exchange_max_limit(exchange)
        if request.limit is not None and request.limit > 0:
            page_limit = min(page_limit, request.limit)
        max_calls = max(1, self.ctx.config.history_max_calls)
        end_ms = _epoch_ms(request.end)
        if end_ms is None:
            end_ms = int(utcnow().timestamp() * 1000)
        start_ms = _epoch_ms(request.start)

        probe_calls = 0
        if start_ms is not None:
            # Prefer the deepest market, not the best-quoted one. Binance lists a new
            # BTC/USD fiat pair whose history starts after the USDT pair's, so the exact-
            # quote preference would silently return a short series for a deep request.
            market, probe_calls = await self._deepest_market(
                exchange, asset, exchange_id, request.timeframe, start_ms
            )
        assert market is not None  # guarded at the top of this method
        fetch = _ohlcv_fetcher(exchange, market.symbol, request.timeframe)

        if start_ms is None:
            # No start: the conventional reading is "the most recent `limit` candles".
            raw = cast(
                "list[list[float]]",
                await exchange.fetch_ohlcv(
                    market.symbol, request.timeframe, limit=page_limit
                ),
            )
            calls, truncated = 1, False
        else:
            raw, calls, truncated = await _paginate_ohlcv(
                fetch,
                interval_ms=interval * 1000,
                start_ms=start_ms,
                end_ms=end_ms,
                page_limit=page_limit,
                max_calls=max_calls,
            )
        walk_calls = calls  # before depth-probe costs are folded in
        calls += probe_calls
        candles = normalise_candles(_candles_from_rows(raw), request)
        note: str | None = None
        if request.start is not None and raw:
            start = as_utc(request.start)
            tolerance = timedelta(seconds=interval * 2)
            stamps = [int(row[0]) for row in raw if len(row) >= 6]
            earliest = epoch_ms_to_utc(min(stamps)) if stamps else None
            newest = epoch_ms_to_utc(max(stamps)) if stamps else None
            end_dt = epoch_ms_to_utc(end_ms)
            if earliest is not None and earliest > start + tolerance:
                at_recent_edge = (
                    newest is not None and end_dt is not None and newest >= end_dt - tolerance
                )
                if walk_calls <= 1 and at_recent_edge:
                    # Kraken returns only its most recent ~720 bars whatever ``since``
                    # says. That is a silent wrong-range answer, so it is labelled as
                    # truncation: coverage must not read as "we simply asked for less".
                    truncated = True
                    note = (
                        f"{exchange_id} ignored 'since' and returned only its most recent "
                        f"window (earliest {earliest.isoformat()})"
                    )
                else:
                    # Reached the venue's genuine history floor (e.g. the asset listed
                    # late). Not a cap, so not truncated — but still not the full window,
                    # which ``complete=False`` already conveys.
                    note = (
                        f"{exchange_id} earliest available candle is {earliest.isoformat()}, "
                        f"later than the requested start {start.isoformat()}"
                    )

        prov = provenance_for(
            source=f"ccxt:{exchange_id}:fetch_ohlcv",
            capability=Capability.HISTORY,
            exchange_timestamp=candles[-1].timestamp if candles else None,
            url=_exchange_url(exchange),
            notes=(f"timeframe={request.timeframe}", f"calls={calls}"),
        )
        self.ctx.record(f"history:{exchange_id}:{request.timeframe}", prov)
        return HistorySeries(
            source=f"ccxt:{exchange_id}",
            symbol=market.symbol,
            timeframe=request.timeframe,
            quote_currency=self.ctx.config.quote_currency,
            kind="ohlcv",
            candles=candles,
            coverage=build_coverage(
                [c.timestamp for c in candles],
                request,
                calls=calls,
                truncated=truncated,
                note=note,
            ),
            provenance=prov,
        )

    # -- derivatives ------------------------------------------------------------------

    async def fetch_derivatives(self, asset: ResolvedAsset) -> tuple[DerivativesData, ...]:
        results = await asyncio.gather(
            *(self._fetch_derivs_one(eid, asset) for eid in self.ctx.config.perp_exchanges),
            return_exceptions=True,
        )
        out: list[DerivativesData] = []
        for _exchange_id, result in zip(self.ctx.config.perp_exchanges, results, strict=True):
            if isinstance(result, BaseException):
                self.ctx.note_error(self.name, Capability.DERIVATIVES, result)
                continue
            if result is not None:
                out.append(result)
        return tuple(out)

    async def _fetch_derivs_one(
        self, exchange_id: str, asset: ResolvedAsset
    ) -> DerivativesData | None:
        market = self._deriv_for(asset, exchange_id)
        if market is None:
            return None
        exchange = await self._load_markets(exchange_id)
        if not self._supports(exchange, "swap") and not self._supports(exchange, "future"):
            raise UnsupportedCapability(f"{exchange_id} has no derivative markets")

        funding: float | None = None
        funding_interval: float | None = None
        oi_contracts: float | None = None
        mark: float | None = None
        index: float | None = None
        timestamp = None
        used: list[str] = []

        if self._supports(exchange, "fetchFundingRate"):
            try:
                raw = cast(dict[str, Any], await exchange.fetch_funding_rate(market.symbol))
                funding = finite(raw.get("fundingRate"))
                funding_interval = _hours_from_ms(finite(raw.get("fundingInterval")))
                timestamp = epoch_ms_to_utc(raw.get("timestamp")) or timestamp
                used.append("fetchFundingRate")
            except Exception as exc:
                self.ctx.note_missing(
                    f"derivatives.{exchange_id}.funding_rate",
                    f"fetchFundingRate failed: {type(exc).__name__}",
                )

        if self._supports(exchange, "fetchOpenInterest"):
            try:
                raw = cast(dict[str, Any], await exchange.fetch_open_interest(market.symbol))
                oi_contracts = finite(
                    raw.get("openInterestAmount")
                ) or finite(raw.get("openInterest"))
                timestamp = epoch_ms_to_utc(raw.get("timestamp")) or timestamp
                used.append("fetchOpenInterest")
            except Exception as exc:
                self.ctx.note_missing(
                    f"derivatives.{exchange_id}.open_interest",
                    f"fetchOpenInterest failed: {type(exc).__name__}",
                )

        if self._supports(exchange, "fetchMarkPrice"):
            try:
                raw = cast(dict[str, Any], await exchange.fetch_mark_price(market.symbol))
                mark = finite(raw.get("markPrice"))
                timestamp = epoch_ms_to_utc(raw.get("timestamp")) or timestamp
                used.append("fetchMarkPrice")
            except Exception as exc:
                self.ctx.note_missing(
                    f"derivatives.{exchange_id}.mark_price",
                    f"fetchMarkPrice failed: {type(exc).__name__}",
                )

        if self._supports(exchange, "fetchIndexPrice"):
            try:
                raw = cast(dict[str, Any], await exchange.fetch_index_price(market.symbol))
                index = finite(raw.get("indexPrice"))
                timestamp = epoch_ms_to_utc(raw.get("timestamp")) or timestamp
                used.append("fetchIndexPrice")
            except Exception as exc:
                self.ctx.note_missing(
                    f"derivatives.{exchange_id}.index_price",
                    f"fetchIndexPrice failed: {type(exc).__name__}",
                )

        if not any(x is not None for x in (funding, oi_contracts, mark, index)):
            raise UnsupportedCapability(
                f"{exchange_id} exposes no genuinely-supported derivative sub-capability "
                "(emulated ccxt methods are excluded as fabricated data)"
            )
        for key in ("fetchFundingRate", "fetchOpenInterest", "fetchMarkPrice", "fetchIndexPrice"):
            if not self._supports(exchange, key):
                self.ctx.note_missing(
                    f"derivatives.{exchange_id}.{_CAP_FIELD[key]}",
                    f"{exchange_id} does not implement {key} "
                    "(ccxt reports it as emulated, i.e. a synthetic value)",
                )

        prov = provenance_for(
            source=f"ccxt:{exchange_id}:derivatives",
            capability=Capability.DERIVATIVES,
            exchange_timestamp=timestamp,
            url=_exchange_url(exchange),
            notes=tuple(f"used={used}"),
        )
        self.ctx.record(f"derivatives:{exchange_id}", prov)

        # For the linear markets we select, contractSize counts base units, so notional
        # converts directly. An inverse market is never selected by _rank_markets, but we
        # still guard: if contractSize looks like a fiat amount (>= 1000) we decline to
        # convert rather than emit a number with the wrong unit.
        contract_size = finite(exchange.markets.get(market.symbol, {}).get("contractSize")) or 1.0
        if contract_size >= 1000:
            self.ctx.note_missing(
                f"derivatives.{exchange_id}.open_interest_base",
                f"contractSize={contract_size} indicates an inverse contract; "
                "not converted to avoid a wrong-unit figure",
            )
            oi_base = None
            oi_quote_value = oi_contracts * contract_size if oi_contracts is not None else None
        else:
            oi_base = oi_contracts * contract_size if oi_contracts is not None else None
            oi_quote_value = (
                oi_base * mark if oi_base is not None and mark is not None else None
            )
        return DerivativesData(
            exchange=exchange_id,
            symbol=market.symbol,
            quote_currency=self.ctx.config.quote_currency,
            contract_type=market.kind,
            funding_rate=funding,
            funding_rate_pct=funding * 100 if funding is not None else None,
            funding_interval_hours=funding_interval,
            open_interest_contracts=oi_contracts,
            open_interest_base=oi_base,
            open_interest_quote=oi_quote_value,
            mark_price=mark,
            index_price=index,
            timestamp=timestamp,
            provenance=prov,
        )


def _exchange_url(exchange: Any) -> str | None:
    """Best-effort API base URL for provenance.

    ccxt 4.x exposes ``exchange.urls`` as a nested dict (``{"api": {"public": ...}}``)
    and has no flat ``url`` attribute.
    """
    urls: Any = getattr(exchange, "urls", None)
    if not isinstance(urls, dict):
        return None
    entries = cast("dict[object, object]", urls)
    api = entries.get("api")
    if isinstance(api, dict):
        api_entries = cast("dict[object, object]", api)
        for key in ("public", "private"):
            value = api_entries.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _hours_from_ms(value: float | None) -> float | None:
    if value is None or value <= 0:
        return None
    return value / 3_600_000.0


def _rank_markets(
    candidates: list[MarketRef], derivatives: bool, quote: str
) -> list[MarketRef]:
    """Order candidate markets best-first.

    Perpetuals beat dated futures (a dated future's price converges to spot and its
    funding/OI are not the perp market's), then the preferred quote currency, then the
    symbol — so selection is deterministic rather than dependent on dict ordering.
    """
    preferences = _PERP_QUOTE_PREFERENCE if derivatives else _QUOTE_PREFERENCE

    def rank(ref: MarketRef) -> tuple[int, int, str]:
        # swap (perpetual) first, then future, then anything else.
        kind_rank = {"swap": 0, "future": 1}.get(ref.kind, 2)
        quote_rank = len(preferences)
        normalised = ref.quote_currency.upper()
        if not derivatives and normalised == quote.upper():
            # Spot: an exact quote-currency match is always the best choice.
            quote_rank = -1
        elif normalised in preferences:
            quote_rank = preferences.index(normalised)
        return (kind_rank, quote_rank, ref.symbol)

    return sorted(candidates, key=rank)


def _prefer_quote(refs: tuple[MarketRef, ...], quote: str) -> MarketRef | None:
    """Choose the market whose quote currency best matches the target."""
    if not refs:
        return None
    upper_quote = quote.upper()
    for candidate in refs:
        if candidate.quote_currency.upper() == upper_quote:
            return candidate
    for preferred in _QUOTE_PREFERENCE:
        for candidate in refs:
            if candidate.quote_currency.upper() == preferred:
                return candidate
    return refs[0]


# --------------------------------------------------------------------------- history

#: Candles requested per call when a venue does not advertise its own ceiling.
_DEFAULT_OHLCV_PAGE = 1000

#: ``async (since_ms, limit) -> rows`` — the shape :func:`_paginate_ohlcv` walks.
OHLCVFetcher = Callable[[int | None, int], Awaitable[list[list[float]]]]


def _ohlcv_fetcher(exchange: Any, symbol: str, timeframe: str) -> OHLCVFetcher:
    """Bind a ccxt exchange's ``fetch_ohlcv`` into the pager's callable shape."""

    async def fetch(since: int | None, limit: int) -> list[list[float]]:
        rows = await exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        return cast("list[list[float]]", rows)

    return fetch


async def _paginate_ohlcv(
    fetch: OHLCVFetcher,
    *,
    interval_ms: int,
    start_ms: int,
    end_ms: int,
    page_limit: int,
    max_calls: int,
) -> tuple[list[list[float]], int, bool]:
    """Walk ``fetch`` forward from ``start_ms`` until ``end_ms`` or ``max_calls``.

    Returns ``(rows, calls, truncated)``. Rows are deduplicated by timestamp; the walk
    advances by ``interval_ms`` past the newest candle seen, which is what every ccxt
    exchange expects. ``truncated`` is true only when the loop stopped *because it ran out
    of calls*, which is the caller's cue that the window is not fully covered.
    """
    rows: list[list[float]] = []
    seen: set[int] = set()
    calls = 0
    truncated = False
    cursor: int | None = start_ms
    while calls < max_calls:
        page = await fetch(cursor, page_limit)
        calls += 1
        if not page:
            break
        newest: int | None = None
        added = False
        for row in page:
            if len(row) < 6:
                continue
            timestamp = int(row[0])
            newest = timestamp if newest is None else max(newest, timestamp)
            if timestamp not in seen:
                seen.add(timestamp)
                rows.append(row)
                added = True
        if not added or newest is None:
            break  # the venue returned no progress; stop rather than spin
        next_cursor = newest + interval_ms
        if next_cursor > end_ms:
            break
        cursor = next_cursor
    else:
        truncated = True
    return rows, calls, truncated


def _candles_from_rows(raw: list[list[float]]) -> list[HistoryCandle]:
    """Convert ccxt ``[[ts, o, h, l, c, v], ...]`` rows into history candles."""
    out: list[HistoryCandle] = []
    for row in raw:
        if len(row) < 6:
            continue
        ts = epoch_ms_to_utc(row[0])
        o, h, low_, c, v = (finite(x) for x in row[1:6])
        if ts is None or o is None or h is None or low_ is None or c is None or v is None:
            continue
        out.append(HistoryCandle(timestamp=ts, close=c, open=o, high=h, low=low_, volume=v))
    return out


def _exchange_max_limit(exchange: Any) -> int:
    """A venue's advertised per-request candle ceiling, defaulting to 1000."""
    options = getattr(exchange, "options", None)
    if isinstance(options, dict):
        entry = cast("dict[str, Any]", options).get("fetchOHLCV")
        if isinstance(entry, dict):
            limit = cast("dict[str, Any]", entry).get("maxLimit")
            if isinstance(limit, int) and limit > 0:
                return limit
    return _DEFAULT_OHLCV_PAGE


def _epoch_ms(value: Any) -> int | None:
    """Aware-UTC datetime to epoch milliseconds, or ``None``."""
    return int(as_utc(value).timestamp() * 1000) if value is not None else None


__all__ = ["DEPTH_BANDS", "CcxtProvider"]
