"""Reconciliation.

Turns many venue prices into one defensible consensus number, and records everything
needed to audit how that number was reached:

* **Consensus** — median across venues, with gross outliers removed first.
* **Outlier rejection** — a venue is excluded if it deviates from the median by more than
  ``outlier_pct``, or if its robust z-score (median absolute deviation) exceeds
  ``outlier_mad_z``. The z-score criterion only applies from ``min_venues_for_mad``
  venues up, because with three venues that agree to five significant figures the MAD is
  far too small to estimate dispersion and would reject ordinary venues. Every exclusion
  is recorded in ``DataQuality.excluded_outliers``.
* **Deviation flags** — any remaining venue deviating by more than the threshold is
  flagged on its :class:`~tickerkit.models.VenueTicker`.
* **Staleness** — data older than ``stale_seconds`` is flagged and named.
* **Cross-source check** — the aggregator price is compared with the venue consensus and
  the divergence recorded.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Any

from .config import Config
from .models import Discrepancy, TickerSnapshot, VenueTicker, finite, jsonify


@dataclass(frozen=True, slots=True)
class Consensus:
    """The outcome of combining venue prices."""

    price: float | None
    method: str
    inputs: int
    excluded: tuple[str, ...]
    deviations_pct: dict[str, float]

    @property
    def method_detail(self) -> str:
        return f"{self.method}(n={self.inputs}, excluded={len(self.excluded)})"


def _venue_price(ticker: VenueTicker) -> float | None:
    """Best available price for a venue: mid, else last, else one side of the book."""
    return finite(ticker.mid) or finite(ticker.last) or finite(ticker.bid) or finite(ticker.ask)


def median_absolute_deviation(values: list[float], centre: float) -> float:
    """MAD scaled to be comparable with a standard deviation."""
    if not values:
        return 0.0
    return statistics.median(abs(v - centre) for v in values)


def robust_z_scores(values: list[float], centre: float) -> list[float]:
    """Modified z-scores: ``0.6745 * (x - median) / MAD``.

    Falls back to a plain deviation when MAD is zero, which happens when venues agree
    exactly — in that case nothing is an outlier, which is the desired behaviour.
    """
    if not values:
        return []
    mad = median_absolute_deviation(values, centre)
    if mad <= 0:
        return [0.0 for _ in values]
    scale = mad * 1.4826
    return [0.6745 * (v - centre) / scale for v in values]


def _is_gross_outlier(value: float, centre: float, threshold: Decimal) -> bool:
    if centre <= 0:
        return False
    deviation = abs(value - centre) / centre
    return deviation > float(threshold)


def build_consensus(tickers: tuple[VenueTicker, ...], config: Config) -> Consensus:
    """Compute the consensus price across venues.

    A venue is excluded when *either* criterion fires:

    1. **Gross deviation** — it sits more than ``outlier_pct`` away from the median.
    2. **Robust z-score** — its median-absolute-deviation z-score exceeds
       ``outlier_mad_z``. This only applies when MAD is non-zero.

    The two are unioned rather than intersected on purpose. A robust z-score alone has
    almost no power against a *single* outlier among few venues: with prices
    ``[100, 100, 5000]`` the MAD is zero, so every z-score collapses to 0 and the rogue
    venue survives — precisely the case this function exists to catch. Requiring both
    would therefore silently disable outlier rejection in the most common scenario.
    """
    priced: list[tuple[str, float]] = []
    for ticker in tickers:
        price = _venue_price(ticker)
        if price is not None and price > 0:
            priced.append((ticker.exchange, price))

    if not priced:
        return Consensus(price=None, method="median", inputs=0, excluded=(), deviations_pct={})

    threshold = config.outlier_threshold()
    values = [p for _, p in priced]
    centre = statistics.median(values)

    # The MAD criterion needs enough venues to estimate dispersion at all. With three
    # venues 0.003% apart the MAD is ~1e-5 of the price, so a single point sitting a
    # hair above the median scores z > 3.5 and gets excluded even though it agrees to
    # five significant figures. Below the floor the gross percentage test carries the
    # decision on its own.
    use_statistical = len(priced) >= config.min_venues_for_mad
    z_scores = robust_z_scores(values, centre) if use_statistical else []
    mad_nonzero = use_statistical and median_absolute_deviation(values, centre) > 0

    keep: list[tuple[str, float]] = []
    excluded: list[str] = []
    for index, (exchange, price) in enumerate(priced):
        gross = _is_gross_outlier(price, centre, threshold)
        z = z_scores[index] if z_scores else 0.0
        statistical = mad_nonzero and abs(z) > config.outlier_mad_z
        if gross or statistical:
            excluded.append(exchange)
        else:
            keep.append((exchange, price))

    if not keep:  # every venue looked anomalous; refuse to trim and use all of them
        keep, excluded = list(priced), []

    method = "median"
    if excluded:
        method = "median_outlier_trimmed"
        # Recompute the centre from survivors so the consensus is not dragged by them.
        centre = statistics.median([p for _, p in keep])

    deviations = {
        exchange: (price - centre) / centre * 100.0 if centre > 0 else 0.0
        for exchange, price in keep
    }
    return Consensus(
        price=centre,
        method=method,
        inputs=len(keep),
        excluded=tuple(excluded),
        deviations_pct=deviations,
    )


def mark_venues(
    tickers: tuple[VenueTicker, ...], consensus: Consensus, config: Config, now: datetime
) -> tuple[VenueTicker, ...]:
    """Attach outlier/deviation/staleness flags to each venue ticker."""
    threshold_pct = config.outlier_pct
    cutoff = now - timedelta(seconds=config.stale_seconds)
    flagged: list[VenueTicker] = []
    for ticker in tickers:
        deviation = consensus.deviations_pct.get(ticker.exchange)
        is_outlier = ticker.exchange in consensus.excluded
        is_deviant = deviation is not None and abs(deviation) > threshold_pct
        stale = ticker.timestamp is not None and ticker.timestamp < cutoff
        flagged.append(
            replace(
                ticker,
                is_outlier=is_outlier or is_deviant,
                deviation_pct=deviation,
                deviation_bps=(deviation * 100.0) if deviation is not None else None,
                is_stale=stale,
            )
        )
    return tuple(flagged)


def cross_check(
    consensus: Consensus,
    aggregated_price: float | None,
    aggregated_source: str,
    threshold_pct: float,
) -> tuple[Discrepancy, ...]:
    """Compare the aggregator price with the venue consensus."""
    if consensus.price is None or not aggregated_price or aggregated_price <= 0:
        return ()
    deviation = (aggregated_price - consensus.price) / consensus.price * 100.0
    if abs(deviation) <= threshold_pct:
        return ()
    return (
        Discrepancy(
            kind="aggregator_vs_consensus",
            detail=(
                f"{aggregated_source} price deviates {deviation:+.3f}% from the "
                f"median of {consensus.inputs} venue(s)"
            ),
            left_source=aggregated_source,
            left_value=aggregated_price,
            right_source=f"consensus({consensus.method})",
            right_value=consensus.price,
            deviation_pct=deviation,
        ),
    )


def stale_flags(tickers: tuple[VenueTicker, ...], config: Config, now: datetime) -> tuple[str, ...]:
    """Names of venues whose data is older than the configured staleness threshold."""
    cutoff = now - timedelta(seconds=config.stale_seconds)
    return tuple(
        f"{t.exchange} (age={_age_text(t.timestamp, now)})"
        for t in tickers
        if t.timestamp is not None and t.timestamp < cutoff
    )


def _age_text(timestamp: datetime | None, now: datetime) -> str:
    if timestamp is None:
        return "unknown"
    seconds = (now - timestamp).total_seconds()
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def apply_consensus(
    snapshot: TickerSnapshot, config: Config, now: datetime
) -> TickerSnapshot:
    """Fold reconciliation results back into the snapshot.

    Returns a new snapshot with consensus price/method filled in, venue flags applied,
    discrepancies recorded, and the excluded-outlier / stale-flag lists populated.
    """
    consensus = build_consensus(snapshot.venues, config)
    venues = mark_venues(snapshot.venues, consensus, config, now)

    aggregated_source = _aggregated_source(snapshot)
    discrepancies = list(
        cross_check(
            consensus,
            snapshot.market.aggregated_price,
            aggregated_source,
            config.outlier_pct,
        )
    )
    for exchange in consensus.excluded:
        discrepancies.append(
            Discrepancy(
                kind="outlier_excluded",
                detail=(
                    f"{exchange} deviated more than {config.outlier_pct}% from the median "
                    "and was excluded from the consensus"
                ),
                left_source=exchange,
            )
        )

    from dataclasses import replace as _replace

    quality = _replace(
        snapshot.data_quality,
        discrepancies=tuple(discrepancies),
        excluded_outliers=consensus.excluded,
        stale_flags=stale_flags(venues, config, now),
    )
    market = _replace(
        snapshot.market,
        consensus_price=consensus.price,
        consensus_method=consensus.method_detail if consensus.price is not None else None,
        consensus_inputs=consensus.inputs,
    )
    return _replace(snapshot, venues=venues, market=market, data_quality=quality)


def _aggregated_source(snapshot: TickerSnapshot) -> str:
    prov = snapshot.market.provenance.get("aggregated")
    return prov.source if prov else "aggregator"


def summarise(snapshot: TickerSnapshot) -> dict[str, Any]:
    """A compact view of the reconciliation outcome, handy for the CLI and tests."""
    dq = snapshot.data_quality
    return {
        "price": snapshot.market.consensus_price,
        "method": snapshot.market.consensus_method,
        "inputs": snapshot.market.consensus_inputs,
        "excluded_outliers": list(dq.excluded_outliers),
        "stale": list(dq.stale_flags),
        "discrepancies": [jsonify(d) for d in dq.discrepancies],
        "errors": len(dq.errors),
    }


__all__ = [
    "Consensus",
    "apply_consensus",
    "build_consensus",
    "cross_check",
    "mark_venues",
    "median_absolute_deviation",
    "robust_z_scores",
    "stale_flags",
    "summarise",
]