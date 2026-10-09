"""Deep-history assembly: coverage accounting, consensus and cross-source checks.

The snapshot path answers "what is it worth right now?". This module answers "what did it
do, over this window, and can I trust the shape of the answer?". It is deliberately the
analogue of :mod:`tickerkit.reconcile`, one level down the time axis:

* **Coverage** — a series states the window it actually covers, the intervals it is
  missing and whether a provider cap cut the walk short. A history answer that silently
  stops in 2021 is worse than one that says so.
* **Consensus** — where several exchanges returned bars for the same interval, the
  per-interval median close becomes the reference series.
* **Outlier trimming** — a venue whose closes deviate more than
  ``history_discrepancy_pct`` from the running median is dropped from the consensus and
  recorded, never silently averaged in.
* **Cross-source check** — close-only references (CoinGecko, DefiLlama) are compared with
  the consensus at the intervals they share, and the divergence is recorded.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from itertools import pairwise

from .config import Config
from .models import (
    CandleGap,
    ConsensusCandle,
    Discrepancy,
    HistoryCandle,
    HistoryConsensus,
    HistoryCoverage,
    HistoryRequest,
    HistorySeries,
    as_utc,
)
from .normalise import TIMEFRAME_SECONDS


def interval_seconds(timeframe: str) -> int | None:
    """Seconds per interval for a ccxt timeframe, or ``None`` if unknown."""
    return TIMEFRAME_SECONDS.get(timeframe)


def floor_to_interval(moment: datetime, seconds: int) -> datetime:
    """Floor a timestamp onto its interval boundary (UTC).

    Sources disagree about alignment: exchange daily bars open at 00:00 UTC, but
    DefiLlama returns price points at roughly-24h offsets from whatever ``start`` was
    given. Flooring lets a price point be compared with a bar for the same day rather
    than landing beside it.
    """
    utc = as_utc(moment)
    epoch = int(utc.timestamp())
    return datetime.fromtimestamp(epoch - epoch % seconds, tz=UTC)


def find_gaps(timestamps: Sequence[datetime], interval: int) -> tuple[CandleGap, ...]:
    """Interior stretches where one or more expected intervals have no candle.

    Only gaps *between* candles are reported; a short tail or head is a coverage fact,
    not a gap, and is handled by :func:`build_coverage`.
    """
    if interval <= 0 or len(timestamps) < 2:
        return ()
    ordered = sorted(set(timestamps))
    gaps: list[CandleGap] = []
    tolerance = interval * 1.5
    for previous, current in pairwise(ordered):
        delta = (current - previous).total_seconds()
        if delta <= tolerance:
            continue
        missing = max(1, round(delta / interval) - 1)
        gaps.append(CandleGap(start=previous, end=current, missing=missing))
    return tuple(gaps)


def expected_intervals(request: HistoryRequest, interval: int | None) -> int | None:
    """Number of intervals a fully-populated requested window would contain."""
    if interval is None or interval <= 0:
        return None
    if request.start is None or request.end is None:
        return None
    span = (as_utc(request.end) - as_utc(request.start)).total_seconds()
    if span < 0:
        return None
    return int(span // interval) + 1


def build_coverage(
    timestamps: Sequence[datetime],
    request: HistoryRequest,
    *,
    calls: int = 0,
    truncated: bool = False,
    note: str | None = None,
) -> HistoryCoverage:
    """Summarise how much of the requested window a series actually covers.

    ``complete`` is deliberately conservative: it is true only when the window is bounded
    and every interval in it is present, or (for an unbounded "give me as much as you
    have" request) when the walk was not cut short and left no interior gaps.
    """
    interval = interval_seconds(request.timeframe)
    ordered = sorted(set(map(as_utc, timestamps)))
    first = ordered[0] if ordered else None
    last = ordered[-1] if ordered else None
    gaps = find_gaps(ordered, interval) if interval else ()
    expected = expected_intervals(request, interval)

    missing: int | None = None
    if expected is not None:
        missing = max(0, expected - len(ordered))
    elif gaps:
        missing = sum(g.missing for g in gaps)

    bounded = request.start is not None and request.end is not None
    complete = False
    if first is not None and last is not None and not truncated:
        if bounded and interval is not None:
            start = as_utc(request.start) if request.start is not None else None
            end = as_utc(request.end) if request.end is not None else None
            if start is not None and end is not None:
                tolerance = timedelta(seconds=interval)
                complete = (
                    start - tolerance <= first <= start + tolerance
                    and end - tolerance <= last <= end + tolerance
                    and not gaps
                )
        elif not bounded:
            complete = not gaps

    return HistoryCoverage(
        requested_start=request.start,
        requested_end=request.end,
        first=first,
        last=last,
        candles=len(ordered),
        expected=expected,
        missing=missing,
        gaps=gaps,
        truncated=truncated,
        calls=calls,
        complete=complete,
        note=note,
    )


def normalise_candles(
    candles: Sequence[HistoryCandle], request: HistoryRequest, *, align: bool = False
) -> tuple[HistoryCandle, ...]:
    """Sort, deduplicate (last write wins) and window-trim a set of candles.

    ``align`` floors timestamps onto the interval boundary, which is what a price-only
    source needs so its points line up with exchange bars. It is never applied to genuine
    OHLC bars, whose own alignment is authoritative.
    """
    interval = interval_seconds(request.timeframe)
    by_ts: dict[datetime, HistoryCandle] = {}
    for candle in candles:
        timestamp = as_utc(candle.timestamp)
        if align and interval:
            timestamp = floor_to_interval(timestamp, interval)
        if request.start is not None and timestamp < as_utc(request.start):
            continue
        if request.end is not None and timestamp > as_utc(request.end):
            continue
        by_ts[timestamp] = replace(candle, timestamp=timestamp)

    ordered = tuple(by_ts[ts] for ts in sorted(by_ts))
    if request.limit is not None and len(ordered) > request.limit:
        # Keep the most recent `limit` points; that is the conventional reading.
        ordered = ordered[-request.limit :]
    return ordered


def build_history_consensus(
    series: Sequence[HistorySeries], config: Config, request: HistoryRequest
) -> tuple[HistoryConsensus | None, tuple[Discrepancy, ...]]:
    """Merge OHLCV series into a per-interval median series.

    Returns the consensus (or ``None`` when no OHLCV source answered) and any discrepancies
    raised while building it. A single source still produces a consensus, so downstream
    code has one shape to read regardless of how many venues replied.
    """
    sources = [s for s in series if s.kind == "ohlcv" and s.candles]
    if not sources:
        return None, ()

    # Per-timestamp map of source name -> candle.
    by_ts: dict[datetime, dict[str, HistoryCandle]] = {}
    for source in sources:
        for candle in source.candles:
            by_ts.setdefault(as_utc(candle.timestamp), {})[source.source] = candle

    names = {s.source for s in sources}
    threshold = config.history_discrepancy_pct
    initial = {
        ts: statistics.median([c.close for c in members.values()])
        for ts, members in by_ts.items()
    }
    deviations = _deviations_from(initial, by_ts, names)

    keep = set(names)
    excluded: tuple[str, ...] = ()
    method = "single_source" if len(sources) == 1 else "median"
    if len(sources) >= 3:
        # With three or more venues a large mean deviation is evidence about that venue,
        # not about the market; with two it is ambiguous, so nothing is dropped.
        candidates = {name for name, dev in deviations.items() if dev > threshold}
        survivors = names - candidates
        if candidates and survivors:
            keep, excluded = survivors, tuple(sorted(candidates))
            method = "median_outlier_trimmed"

    candles: list[ConsensusCandle] = []
    for timestamp in sorted(by_ts):
        members = {name: candle for name, candle in by_ts[timestamp].items() if name in keep}
        if not members:
            continue
        closes = [c.close for c in members.values()]
        centre = statistics.median(closes)
        spread = None
        if len(closes) > 1 and centre > 0:
            spread = (max(closes) - min(closes)) / centre * 100.0
        candles.append(
            ConsensusCandle(
                timestamp=timestamp,
                close=centre,
                open=_median([c.open for c in members.values()]),
                high=_median([c.high for c in members.values()]),
                low=_median([c.low for c in members.values()]),
                volume=_median([c.volume for c in members.values()]),
                sources=tuple(sorted(members)),
                n=len(members),
                spread_pct=spread,
            )
        )

    # Report deviations against the *final* consensus, so a number in the output matches
    # the series a reader can actually see.
    final = {c.timestamp: c.close for c in candles}
    deviations = _deviations_from(final, by_ts, names)

    discrepancies: list[Discrepancy] = [
        Discrepancy(
            kind="history_source_outlier",
            detail=(
                f"{name} deviates {deviations.get(name, 0.0):.2f}% from the history median "
                f"and was excluded (threshold {threshold}%)"
            ),
            left_source=name,
            right_source="history_median",
            deviation_pct=deviations.get(name),
        )
        for name in excluded
    ]

    consensus = HistoryConsensus(
        timeframe=request.timeframe,
        quote_currency=config.quote_currency,
        method=method,
        inputs=tuple(sorted(keep)),
        candles=tuple(candles),
        deviations_pct=deviations,
        coverage=build_coverage(
            [c.timestamp for c in candles],
            request,
            calls=sum(s.coverage.calls for s in sources if s.coverage),
            truncated=any(s.coverage.truncated for s in sources if s.coverage),
            note=f"median of {len(keep)} source(s)",
        ),
    )
    return consensus, tuple(discrepancies)


def cross_check_references(
    consensus: HistoryConsensus | None,
    references: Sequence[HistorySeries],
    threshold_pct: float,
) -> tuple[Discrepancy, ...]:
    """Compare close-only references with the consensus at shared intervals.

    This is the history analogue of comparing an aggregator price with the venue
    consensus: CoinGecko and DefiLlama never *build* the series, they *audit* it.
    """
    if consensus is None or not consensus.candles:
        return ()
    reference_close = {c.timestamp: c.close for c in consensus.candles}
    out: list[Discrepancy] = []
    for source in references:
        deviations: list[float] = []
        for candle in source.candles:
            centre = reference_close.get(as_utc(candle.timestamp))
            if centre is None or centre <= 0:
                continue
            deviations.append(abs(candle.close - centre) / centre * 100.0)
        if not deviations:
            continue
        mean_dev = sum(deviations) / len(deviations)
        if mean_dev <= threshold_pct:
            continue
        out.append(
            Discrepancy(
                kind="history_reference_vs_consensus",
                detail=(
                    f"{source.source} close deviates {mean_dev:.2f}% on average from the "
                    f"consensus over {len(deviations)} shared interval(s)"
                ),
                left_source=source.source,
                right_source=f"consensus({consensus.method})",
                deviation_pct=mean_dev,
            )
        )
    return tuple(out)


# --------------------------------------------------------------------------- internals


def _deviations_from(
    reference: dict[datetime, float],
    by_ts: dict[datetime, dict[str, HistoryCandle]],
    names: set[str],
) -> dict[str, float]:
    """Mean absolute percent deviation of each source's close from a reference series."""
    totals = dict.fromkeys(names, 0.0)
    counts = dict.fromkeys(names, 0)
    for timestamp, members in by_ts.items():
        centre = reference.get(timestamp)
        if centre is None or centre <= 0:
            continue
        for name, candle in members.items():
            if name not in totals:
                continue
            totals[name] += abs(candle.close - centre) / centre * 100.0
            counts[name] += 1
    return {name: (totals[name] / counts[name] if counts[name] else 0.0) for name in names}


def _median(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return statistics.median(present) if present else None


__all__ = [
    "build_coverage",
    "build_history_consensus",
    "cross_check_references",
    "expected_intervals",
    "find_gaps",
    "floor_to_interval",
    "interval_seconds",
    "normalise_candles",
]
