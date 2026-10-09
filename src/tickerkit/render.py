"""Clean, composable output functions.

Two audiences:

* **Humans** — aligned plain-text sections and tables. No pandas, no colour codes, so
  they paste cleanly into a terminal, a log, an issue or a code review.
* **Pipelines** — :func:`flat_metrics` returns one flat ``dict`` of scalars per asset
  (one row per asset, appendable straight into a DataFrame or a CSV), and
  :func:`to_records` returns normalised row lists for the tabular sections.

Every formatter degrades predictably: ``None`` renders as ``-`` (or ``n/a``), never as
``0`` or ``None``, so an absent value can never be mistaken for a real measurement.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Protocol

from .models import (
    DataQuality,
    HistoryCoverage,
    HistoryResult,
    TickerSnapshot,
    utcnow,
)

#: Rendered in place of a missing value.
NA = "-"


class QualityCarrier(Protocol):
    """Anything with a ``data_quality`` block: a snapshot or a history result.

    The two quality blocks render identically, so the shared formatters accept either
    rather than being duplicated per result type.
    """

    @property
    def data_quality(self) -> DataQuality: ...


# --------------------------------------------------------------------------- primitives


def fmt_price(value: float | None, quote: str | None = None, *, sig: int = 6) -> str:
    """A price with enough significant digits to stay useful across magnitudes.

    ``fmt_price(0.000003925)`` -> ``"3.925e-06"`` rather than ``"0.00000"``.
    """
    if value is None:
        return NA
    # "g" keeps small magnitudes readable (3.925e-06) while "," groups large ones
    # (81,795.5). The "#" flag is deliberately absent: it would preserve trailing zeros.
    body = f"{value:,.{sig}g}"
    return f"{body} {quote}" if quote else body


def fmt_number(value: float | None, *, digits: int = 2) -> str:
    """A fixed-point number with thousands separators."""
    return NA if value is None else f"{value:,.{digits}f}"


def fmt_compact(value: float | None, *, prefix: str = "") -> str:
    """A compact magnitude for market caps and volumes: ``1.64T``, ``43.61B``, ``-``."""
    if value is None:
        return NA
    sign = "-" if value < 0 else ""
    magnitude = abs(value)
    for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if magnitude >= cutoff:
            return f"{sign}{prefix}{magnitude / cutoff:,.2f}{suffix}"
    return f"{sign}{prefix}{magnitude:,.2f}"


def fmt_pct(value: float | None, *, signed: bool = True, digits: int = 2) -> str:
    """A percentage, optionally signed. ``None`` -> ``-``."""
    if value is None:
        return NA
    return f"{value:+.{digits}f}%" if signed else f"{value:.{digits}f}%"


def fmt_bps(value: float | None, *, digits: int = 2) -> str:
    return NA if value is None else f"{value:.{digits}f}bps"


def fmt_age(timestamp: datetime | None, *, now: datetime | None = None) -> str:
    """How long ago a timestamp was: ``3s``, ``4m``, ``2.1h``, ``3d``."""
    if timestamp is None:
        return NA
    reference = now or utcnow()
    seconds = (reference - timestamp).total_seconds()
    if seconds < 0:
        return "in the future"
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    if seconds < 172_800:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86_400:.1f}d"


def fmt_time(timestamp: datetime | None, *, short: bool = False) -> str:
    """An ISO-8601 UTC timestamp, or ``-``."""
    if timestamp is None:
        return NA
    text = timestamp.astimezone(UTC).isoformat()
    return text[11:19] if short else text


def plural(count: int, singular: str, suffix: str = "s") -> str:
    """``3 venues`` / ``1 venue``, for tidy counters."""
    return f"{count} {singular}{'' if count == 1 else suffix}"


def table(headers: list[str], rows: list[list[str]], *, indent: str = "") -> str:
    """An aligned plain-text table. Column widths follow the rendered cells."""
    if not rows:
        return f"{indent}(none)"
    widths: list[int] = []
    for i, header in enumerate(headers):
        column: list[list[str]] = [[str(header)]]
        for row in rows:
            column.append([str(row[i])])
        widest = 0
        for cell in column:
            widest = max(widest, len(cell[0]))
        widths.append(widest)
    lines: list[str] = []
    head: list[str] = []
    rule: list[str] = []
    for i, header in enumerate(headers):
        head.append(str(header).ljust(widths[i]))
        rule.append("-" * widths[i])
    lines.append(indent + "  ".join(head).rstrip())
    lines.append(indent + "  ".join(rule).rstrip())
    for row in rows:
        cells: list[str] = []
        for i, cell in enumerate(row):
            cells.append(str(cell).ljust(widths[i]))
        lines.append(indent + "  ".join(cells).rstrip())
    return "\n".join(lines)


def _flag(*, stale: bool = False, outlier: bool = False) -> str:
    marks: list[str] = []
    if outlier:
        marks.append("OUTLIER")
    if stale:
        marks.append("STALE")
    return ",".join(marks)


# --------------------------------------------------------------------------- sections


def summary_block(snap: TickerSnapshot, *, indent: str = "") -> str:
    """Headline: identity, consensus price, coverage counters, reconciliation method."""
    market = snap.market
    dq = snap.data_quality
    lines = [
        f"{snap.identity.symbol}  {snap.identity.name or NA}"
        f"  ({snap.identity.coingecko_id or 'unresolved'})",
        f"price        {fmt_price(market.price, market.quote_currency)}",
    ]
    if market.consensus_method:
        lines.append(f"consensus    {market.consensus_method}")
    if market.change_24h_pct is not None:
        lines.append(f"24h change   {fmt_pct(market.change_24h_pct)}")
    if market.market_cap is not None:
        lines.append(
            f"market cap   {fmt_compact(market.market_cap)}"
            f"   fdv {fmt_compact(market.fdv)}"
            f"   vol24h {fmt_compact(market.volume_24h)}"
        )
    if market.circulating_supply is not None:
        lines.append(
            f"supply       {fmt_compact(market.circulating_supply)} circulating"
            f"   max {fmt_compact(market.max_supply)}"
        )
    if market.ath is not None:
        lines.append(
            f"ath / atl    {fmt_compact(market.ath)} @ {fmt_time(market.ath_date, short=True)}"
            f"   /   {fmt_compact(market.atl)} @ {fmt_time(market.atl_date, short=True)}"
        )
    lines.append(
        f"coverage     {plural(len(snap.venues), 'venue')},"
        f" {plural(len(snap.orderbooks), 'book')},"
        f" {plural(len(snap.trades), 'trade')},"
        f" {plural(len(snap.ohlcv), 'ohlcv series', suffix='')},"
        f" {plural(len(snap.derivatives), 'derivative')}"
    )
    if snap.onchain is not None and snap.onchain.tvl is not None:
        lines.append(
            f"tvl          {fmt_compact(snap.onchain.tvl)}"
            f"   ({snap.onchain.defillama_slug})"
        )
    errors = len(dq.errors)
    lines.append(f"quality      {'no errors' if errors == 0 else plural(errors, 'error')}")
    return "\n".join(indent + line for line in lines)


def venues_table(snap: TickerSnapshot, *, indent: str = "") -> str:
    """Per-venue bid/ask/mid with spread, deviation and provenance flags."""
    headers = ["venue", "symbol", "bid", "ask", "mid", "spread", "dev%", "age", "flags"]
    rows = [
        [
            v.exchange,
            v.symbol,
            fmt_price(v.bid),
            fmt_price(v.ask),
            fmt_price(v.mid),
            fmt_bps(v.spread_bps),
            fmt_pct(v.deviation_pct, signed=True),
            fmt_age(v.timestamp),
            _flag(stale=v.is_stale, outlier=v.is_outlier),
        ]
        for v in snap.venues
    ]
    return table(headers, rows, indent=indent)


def orderbook_table(snap: TickerSnapshot, *, indent: str = "") -> str:
    """Top-of-book and depth per venue."""
    headers = ["venue", "best bid", "best ask", "mid", "spread", "bid depth", "ask depth", "imb"]
    rows = [
        [
            b.exchange,
            fmt_price(b.best_bid),
            fmt_price(b.best_ask),
            fmt_price(b.mid),
            fmt_bps(b.spread_bps),
            fmt_number(b.bid_depth_base, digits=2),
            fmt_number(b.ask_depth_base, digits=2),
            fmt_pct(None if b.imbalance is None else b.imbalance * 100, signed=True),
        ]
        for b in snap.orderbooks
    ]
    return table(headers, rows, indent=indent)


def derivatives_table(snap: TickerSnapshot, *, indent: str = "") -> str:
    """Funding, open interest and mark/index per venue.

    Unsupported columns print ``-``; they are never inferred from other fields.
    """
    headers = [
        "venue", "symbol", "funding", "OI (quote)", "mark", "index", "age", "source"
    ]
    rows = [
        [
            d.exchange,
            d.symbol,
            fmt_pct(d.funding_rate_pct, digits=4),
            fmt_compact(d.open_interest_quote),
            fmt_price(d.mark_price),
            fmt_price(d.index_price),
            fmt_age(d.timestamp),
            d.provenance.source if d.provenance else NA,
        ]
        for d in snap.derivatives
    ]
    return table(headers, rows, indent=indent)


def onchain_block(snap: TickerSnapshot, *, indent: str = "", top: int = 5) -> str:
    """TVL plus the largest chain breakdown."""
    if snap.onchain is None:
        return f"{indent}onchain      (not applicable)"
    o = snap.onchain
    lines = [
        f"onchain      {fmt_compact(o.tvl)}"
        f"  ({o.defillama_slug}, {o.category or 'uncategorised'})"
    ]
    for chain in o.chains[:top]:
        lines.append(f"             {chain.chain:<20} {fmt_compact(chain.tvl)}")
    if len(o.chains) > top:
        lines.append(f"             ... and {len(o.chains) - top} more chains")
    return "\n".join(indent + line for line in lines)


def provenance_table(snap: QualityCarrier, *, indent: str = "") -> str:
    """Every field group with its source and age — the audit trail."""
    now = utcnow()
    headers = ["group", "source", "fetched (UTC)", "exchange timestamp", "cached"]
    rows = [
        [
            key,
            prov.source,
            fmt_time(prov.fetched_at),
            fmt_time(prov.exchange_timestamp),
            "yes" if prov.cached else "no",
        ]
        for key, prov in sorted(snap.data_quality.provenance.items())
    ]
    body = table(headers, rows, indent=indent)
    if rows:
        newest = max(p.fetched_at for p in snap.data_quality.provenance.values())
        body += f"\n{indent}newest fetch is {fmt_age(newest, now=now)} old"
    return body


def quality_block(snap: QualityCarrier, *, indent: str = "") -> str:
    """Errors, discrepancies, ambiguities and missing-field explanations."""
    dq = snap.data_quality
    lines: list[str] = []
    for lim in dq.rate_limited:
        detail = [f"{lim.source}/{lim.capability}"]
        if lim.limit is not None:
            detail.append(f"quota {lim.limit}")
        if lim.retry_after is not None:
            detail.append(f"retry in {lim.retry_after:.0f}s")
        if lim.suppressed:
            detail.append("call suppressed, not sent")
        lines.append("rate limit   " + ": ".join(detail))
        if lim.remedy:
            lines.append(f"             remedy: {lim.remedy}")
    for err in dq.errors:
        # Throttles already have a dedicated, richer line above.
        if err.exception_type == "RateLimitExceeded":
            continue
        lines.append(f"error        [{err.provider}/{err.capability}] "
                     f"{err.exception_type}: {err.message}")
    for skipped in dq.providers_skipped:
        lines.append(f"skipped      {skipped.provider}: {skipped.reason}")
    for disc in dq.discrepancies:
        lines.append(f"discrepancy  [{disc.kind}] {disc.detail}")
    for flag in dq.excluded_outliers:
        lines.append(f"outlier      {flag} excluded from the consensus")
    for flag in dq.stale_flags:
        lines.append(f"stale        {flag}")
    for amb in dq.ambiguities[:10]:
        lines.append(
            f"ambiguous    {amb.symbol} -> {amb.coingecko_id}"
            f" ({amb.name}) mcap {fmt_compact(amb.market_cap)}"
        )
    if len(dq.ambiguities) > 10:
        lines.append(f"ambiguous    ... and {len(dq.ambiguities) - 10} more")
    for item in dq.missing:
        lines.append(f"missing      {item.field}: {item.reason}")
    if not lines:
        lines.append("quality      clean: no errors, discrepancies or missing fields")
    return "\n".join(indent + line for line in lines)


# --------------------------------------------------------------------------- composites


def render_report(
    snap: TickerSnapshot,
    *,
    sections: tuple[str, ...] = ("summary", "venues", "orderbook", "derivatives",
                                  "onchain", "quality", "provenance"),
    indent: str = "",
    provenance_top: int | None = None,
) -> str:
    """A complete human-readable report.

    ``sections`` selects which blocks appear, in the given order; an empty tuple renders
    only the summary.
    """
    builders: dict[str, Any] = {
        "summary": lambda: summary_block(snap, indent=indent),
        "venues": lambda: venues_table(snap, indent=indent),
        "orderbook": lambda: orderbook_table(snap, indent=indent),
        "derivatives": lambda: derivatives_table(snap, indent=indent),
        "onchain": lambda: onchain_block(snap, indent=indent),
        "quality": lambda: quality_block(snap, indent=indent),
    }
    titles = {
        "summary": "SUMMARY",
        "venues": "VENUES",
        "orderbook": "ORDER BOOKS",
        "derivatives": "DERIVATIVES",
        "onchain": "ONCHAIN",
        "quality": "DATA QUALITY",
        "provenance": "PROVENANCE",
    }
    parts: list[str] = []
    for name in sections:
        if name == "provenance":
            block = provenance_table(snap, indent=indent)
            if provenance_top is not None:
                lines = block.splitlines()
                block = "\n".join(lines[: provenance_top + 2])
        elif name in builders:
            block = builders[name]()
        else:
            raise ValueError(f"unknown section {name!r}; choose from {sorted(titles)}")
        parts.append(f"{indent}--- {titles[name]} ---\n{block}")
    return "\n".join(parts)


def render_markdown(snap: TickerSnapshot, *, sections: tuple[str, ...] | None = None) -> str:
    """The report as a fenced markdown document, suitable for a PR or an issue."""
    wanted = sections or ("summary", "venues", "derivatives", "quality")
    body = render_report(snap, sections=wanted)
    title = f"# {snap.identity.symbol} snapshot ({snap.generated_at.isoformat()})"
    return f"{title}\n\n```\n{body}\n```"


# --------------------------------------------------------------------------- history


def coverage_text(coverage: HistoryCoverage | None) -> str:
    """A compact completeness verdict: ``complete``, ``truncated``, ``2 gap(s)`` …"""
    if coverage is None:
        return NA
    bits: list[str] = ["complete" if coverage.complete else "incomplete"]
    if coverage.truncated:
        bits.append("truncated")
    if coverage.gaps:
        bits.append(f"{len(coverage.gaps)} gap(s)")
    if coverage.missing:
        bits.append(f"{coverage.missing} missing")
    return ", ".join(bits)


def history_summary_block(result: HistoryResult, *, indent: str = "") -> str:
    """Headline for a history result: window, available span, consensus, coverage."""
    request = result.request
    first, last = result.available_range()
    lines = [
        f"{result.identity.symbol}  {result.identity.name or NA}"
        f"  ({result.identity.coingecko_id or 'unresolved'})",
        f"timeframe    {request.timeframe}",
        f"requested    {fmt_time(request.start)} .. {fmt_time(request.end)}"
        + (f"  limit {request.limit}" if request.limit is not None else ""),
        f"available    {fmt_time(first)} .. {fmt_time(last)}",
    ]
    consensus = result.consensus
    if consensus is not None:
        lines.append(
            f"consensus    {consensus.method}(n={len(consensus.inputs)})"
            f"  {plural(len(consensus.candles), 'candle')}"
        )
        lines.append(f"coverage     {coverage_text(consensus.coverage)}")
    else:
        lines.append("consensus    -  (no OHLCV source answered)")
    lines.append(
        f"series       {plural(len(result.series), 'source')}:"
        f" {len(result.ohlcv_series)} ohlcv, {len(result.reference_series)} price"
    )
    errors = len(result.data_quality.errors)
    lines.append(f"quality      {'no errors' if errors == 0 else plural(errors, 'error')}")
    return "\n".join(indent + line for line in lines)


def history_sources_table(result: HistoryResult, *, indent: str = "") -> str:
    """One row per source: what it returned and how complete it is."""
    headers = ["source", "kind", "candles", "first", "last", "coverage", "calls", "note"]
    rows: list[list[str]] = []
    for series in result.series:
        coverage = series.coverage
        rows.append(
            [
                series.source,
                series.kind,
                str(len(series.candles)),
                fmt_time(coverage.first if coverage else None),
                fmt_time(coverage.last if coverage else None),
                coverage_text(coverage),
                str(coverage.calls) if coverage else NA,
                (coverage.note or "") if coverage else "",
            ]
        )
    return table(headers, rows, indent=indent)


def history_consensus_table(
    result: HistoryResult, *, head: int = 3, tail: int = 3, indent: str = ""
) -> str:
    """A windowed view of the consensus — first and last few intervals, not thousands."""
    if result.consensus is None or not result.consensus.candles:
        return f"{indent}(no consensus series)"
    candles = result.consensus.candles
    headers = ["timestamp", "close", "n", "sources", "spread%"]
    rows = [
        [fmt_time(c.timestamp), fmt_price(c.close), str(c.n), ",".join(c.sources),
         fmt_pct(c.spread_pct)]
        for c in candles
    ]
    if len(rows) > head + tail:
        rows = [
            *rows[:head],
            ["...", f"({len(candles)} intervals)", "", "", ""],
            *rows[-tail:],
        ]
    body = table(headers, rows, indent=indent)
    worst = max(
        (c.spread_pct for c in candles if c.spread_pct is not None), default=None
    )
    if worst is not None:
        body += f"\n{indent}widest cross-source spread: {fmt_pct(worst, signed=False)}"
    return body


def render_history_report(
    result: HistoryResult,
    *,
    sections: tuple[str, ...] = ("summary", "sources", "consensus", "quality", "provenance"),
    indent: str = "",
) -> str:
    """A complete human-readable history report."""
    builders: dict[str, Any] = {
        "summary": lambda: history_summary_block(result, indent=indent),
        "sources": lambda: history_sources_table(result, indent=indent),
        "consensus": lambda: history_consensus_table(result, indent=indent),
        "quality": lambda: quality_block(result, indent=indent),
    }
    titles = {
        "summary": "HISTORY",
        "sources": "SOURCES",
        "consensus": "CONSENSUS",
        "quality": "DATA QUALITY",
        "provenance": "PROVENANCE",
    }
    parts: list[str] = []
    for name in sections:
        if name == "provenance":
            block = provenance_table(result, indent=indent)
        elif name in builders:
            block = builders[name]()
        else:
            raise ValueError(f"unknown section {name!r}; choose from {sorted(titles)}")
        parts.append(f"{indent}--- {titles[name]} ---\n{block}")
    return "\n".join(parts)


def render_history_markdown(
    result: HistoryResult, *, sections: tuple[str, ...] | None = None
) -> str:
    """The history report as a fenced markdown document."""
    wanted = sections or ("summary", "sources", "consensus", "quality")
    body = render_history_report(result, sections=wanted)
    title = (
        f"# {result.identity.symbol} history ({result.request.timeframe}) "
        f"({result.generated_at.isoformat()})"
    )
    return f"{title}\n\n```\n{body}\n```"


def history_to_records(result: HistoryResult) -> dict[str, list[dict[str, Any]]]:
    """Normalised row lists for history — the analogue of :func:`to_records`."""
    history: list[dict[str, Any]] = []
    for series in result.series:
        for candle in series.candles:
            history.append(
                {
                    "source": series.source,
                    "symbol": series.symbol,
                    "timeframe": series.timeframe,
                    "kind": series.kind,
                    "timestamp": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                }
            )
    consensus: list[dict[str, Any]] = []
    if result.consensus is not None:
        for candle in result.consensus.candles:
            consensus.append(
                {
                    "timestamp": candle.timestamp,
                    "open": candle.open,
                    "high": candle.high,
                    "low": candle.low,
                    "close": candle.close,
                    "volume": candle.volume,
                    "sources": ",".join(candle.sources),
                    "n": candle.n,
                    "spread_pct": candle.spread_pct,
                }
            )
    return {"history": history, "consensus": consensus}


def history_flat_metrics(result: HistoryResult) -> dict[str, Any]:
    """One flat row per history request, for a pipeline.

    Reports completeness explicitly (``complete``/``truncated``/``missing``) so a
    downstream job can reject a short series without re-reading the coverage objects.
    """
    consensus = result.consensus
    coverage = consensus.coverage if consensus else None
    first, last = result.available_range()
    return {
        "symbol": result.identity.symbol,
        "name": result.identity.name,
        "coingecko_id": result.identity.coingecko_id,
        "timeframe": result.request.timeframe,
        "requested_start": result.request.start,
        "requested_end": result.request.end,
        "limit": result.request.limit,
        "series_count": len(result.series),
        "ohlcv_series_count": len(result.ohlcv_series),
        "reference_series_count": len(result.reference_series),
        "consensus_method": consensus.method if consensus else None,
        "consensus_inputs": len(consensus.inputs) if consensus else 0,
        "consensus_candles": len(consensus.candles) if consensus else 0,
        "first_candle": first,
        "last_candle": last,
        "coverage_complete": coverage.complete if coverage else None,
        "coverage_truncated": coverage.truncated if coverage else None,
        "coverage_missing": coverage.missing if coverage else None,
        "sources": ",".join(s.source for s in result.series),
        "error_count": len(result.data_quality.errors),
        "missing_count": len(result.data_quality.missing),
        "discrepancy_count": len(result.data_quality.discrepancies),
        "generated_at": result.generated_at,
    }


# --------------------------------------------------------------------------- pipelines


def to_records(snap: TickerSnapshot) -> dict[str, list[dict[str, Any]]]:
    """Normalised row lists for the tabular sections — no pandas required.

    Keys mirror :meth:`TickerSnapshot.to_dataframes`, but these are plain dicts so a
    pipeline can consume them without importing pandas.
    """
    records: dict[str, list[dict[str, Any]]] = {}

    records["venues"] = [
        {
            "exchange": v.exchange,
            "symbol": v.symbol,
            "quote_currency": v.quote_currency,
            "bid": v.bid,
            "ask": v.ask,
            "mid": v.mid,
            "last": v.last,
            "spread": v.spread,
            "spread_bps": v.spread_bps,
            "volume_24h_quote": v.volume_24h_quote,
            "timestamp": v.timestamp,
            "is_stale": v.is_stale,
            "is_outlier": v.is_outlier,
            "deviation_pct": v.deviation_pct,
            "source": v.provenance.source if v.provenance else None,
            "fetched_at": v.provenance.fetched_at if v.provenance else None,
        }
        for v in snap.venues
    ]

    records["orderbooks"] = [
        {
            "exchange": b.exchange,
            "symbol": b.symbol,
            "best_bid": b.best_bid,
            "best_ask": b.best_ask,
            "mid": b.mid,
            "spread_bps": b.spread_bps,
            "bid_depth_base": b.bid_depth_base,
            "ask_depth_base": b.ask_depth_base,
            "imbalance": b.imbalance,
            "levels_bid": len(b.bids),
            "levels_ask": len(b.asks),
            "timestamp": b.timestamp,
        }
        for b in snap.orderbooks
    ]

    records["trades"] = [
        {
            "exchange": t.exchange,
            "symbol": t.symbol,
            "timestamp": t.timestamp,
            "price": t.price,
            "amount": t.amount,
            "side": t.side,
            "trade_id": t.trade_id,
        }
        for t in snap.trades
    ]

    ohlcv: list[dict[str, Any]] = []
    for series in snap.ohlcv:
        for candle in series.candles:
            ohlcv.append({
                "exchange": series.exchange,
                "symbol": series.symbol,
                "timeframe": series.timeframe,
                "timestamp": candle.timestamp,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume,
            })
    records["ohlcv"] = ohlcv

    records["derivatives"] = [
        {
            "exchange": d.exchange,
            "symbol": d.symbol,
            "contract_type": d.contract_type,
            "funding_rate": d.funding_rate,
            "funding_rate_pct": d.funding_rate_pct,
            "open_interest_quote": d.open_interest_quote,
            "mark_price": d.mark_price,
            "index_price": d.index_price,
            "basis": d.basis,
            "timestamp": d.timestamp,
        }
        for d in snap.derivatives
    ]
    return records


def flat_metrics(snap: TickerSnapshot) -> dict[str, Any]:
    """One flat row per asset, for appending to a DataFrame or a CSV.

    Keys are stable and snake_case. Absent measurements are ``None`` (JSON ``null``),
    never ``0``. Error counts are included so a downstream job can filter on degraded
    snapshots without re-reading the snapshot.
    """
    m = snap.market
    dq = snap.data_quality
    prices = [v.mid for v in snap.venues if v.mid is not None]
    spreads = [v.spread_bps for v in snap.venues if v.spread_bps is not None]
    funding = [d.funding_rate for d in snap.derivatives if d.funding_rate is not None]
    oi = [d.open_interest_quote for d in snap.derivatives if d.open_interest_quote is not None]
    agg_prov = m.provenance.get("aggregated")

    return {
        "symbol": snap.identity.symbol,
        "name": snap.identity.name,
        "coingecko_id": snap.identity.coingecko_id,
        "quote_currency": m.quote_currency,
        "price": m.price,
        "consensus_price": m.consensus_price,
        "aggregated_price": m.aggregated_price,
        "consensus_inputs": m.consensus_inputs,
        "consensus_method": m.consensus_method,
        "change_24h_pct": m.change_24h_pct,
        "volume_24h": m.volume_24h,
        "market_cap": m.market_cap,
        "fdv": m.fdv,
        "circulating_supply": m.circulating_supply,
        "max_supply": m.max_supply,
        "ath": m.ath,
        "atl": m.atl,
        "venue_count": len(snap.venues),
        "min_venue_price": min(prices) if prices else None,
        "max_venue_price": max(prices) if prices else None,
        "median_spread_bps": (sorted(spreads)[len(spreads) // 2] if spreads else None),
        "outlier_count": len(dq.excluded_outliers),
        "stale_count": len(dq.stale_flags),
        "derivative_count": len(snap.derivatives),
        "mean_funding_rate": (sum(funding) / len(funding) if funding else None),
        "total_open_interest_quote": (sum(oi) if oi else None),
        "tvl": snap.onchain.tvl if snap.onchain else None,
        "defillama_slug": snap.onchain.defillama_slug if snap.onchain else None,
        "error_count": len(dq.errors),
        "missing_count": len(dq.missing),
        "ambiguity_count": len(dq.ambiguities),
        "providers_used": ",".join(dq.providers_used),
        "price_source": agg_prov.source if agg_prov else None,
        "price_fetched_at": agg_prov.fetched_at if agg_prov else None,
        "generated_at": snap.generated_at,
    }


def snapshot_to_frame_row(snap: TickerSnapshot) -> dict[str, Any]:
    """Alias for :func:`flat_metrics`, named for the DataFrame use case."""
    return flat_metrics(snap)


__all__ = [
    "NA",
    "QualityCarrier",
    "coverage_text",
    "derivatives_table",
    "flat_metrics",
    "fmt_age",
    "fmt_bps",
    "fmt_compact",
    "fmt_number",
    "fmt_pct",
    "fmt_price",
    "fmt_time",
    "history_consensus_table",
    "history_flat_metrics",
    "history_sources_table",
    "history_summary_block",
    "history_to_records",
    "onchain_block",
    "orderbook_table",
    "plural",
    "provenance_table",
    "quality_block",
    "render_history_markdown",
    "render_history_report",
    "render_markdown",
    "render_report",
    "snapshot_to_frame_row",
    "summary_block",
    "table",
    "to_records",
    "venues_table",
]