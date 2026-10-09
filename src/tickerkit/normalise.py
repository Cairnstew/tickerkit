"""Normalisation helpers.

Everything funnels through here so the canonical schema stays consistent: UTC-aware
timestamps, finite floats, and quote currency stated explicitly on every group.
"""

from __future__ import annotations

import html
import re
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any, cast

from .models import BookLevel, DepthBand, as_utc, finite

#: Block-level tags become whitespace so adjacent words do not merge.
_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|br|li|ul|ol|div|tr|td|th|table|h[1-6]|blockquote|pre)\b[^>]*>",
    re.IGNORECASE,
)
#: Any remaining tag (inline markup, attributes, self-closing) is removed outright.
_TAG_RE = re.compile(r"<[^>]+>")

#: ccxt timeframe -> approximate duration in seconds. Used only for ordering/coverage.
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14_400,
    "6h": 21_600,
    "12h": 43_200,
    "1d": 86_400,
    "3d": 259_200,
    "1w": 604_800,
    "1M": 2_629_800,
}


def epoch_ms_to_utc(value: object) -> datetime | None:
    """Convert a millisecond epoch (ccxt, CoinGecko charts) to aware UTC."""
    ms = finite(value)
    if ms is None or ms <= 0:
        return None
    try:
        return datetime.fromtimestamp(ms / 1000.0, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def epoch_s_to_utc(value: object) -> datetime | None:
    """Convert a second epoch (DefiLlama, CoinGecko history) to aware UTC."""
    secs = finite(value)
    if secs is None or secs <= 0:
        return None
    try:
        return datetime.fromtimestamp(secs, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def parse_iso(value: object) -> datetime | None:
    """Parse an ISO-8601 string (``Z`` included) into aware UTC."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().replace("Z", "+00:00")
    try:
        return as_utc(datetime.fromisoformat(text))
    except ValueError:
        return None


def parse_timestamp(value: object, *, unit: str = "ms") -> datetime | None:
    """Parse a numeric epoch in the given ``unit`` (``"ms"`` or ``"s"``)."""
    return epoch_ms_to_utc(value) if unit == "ms" else epoch_s_to_utc(value)


def _is_pair(value: object) -> bool:
    """Whether ``value`` looks like a ccxt ``[price, amount]`` row."""
    return isinstance(value, (list, tuple))


def book_levels(raw: object, *, ascending: bool) -> tuple[BookLevel, ...]:
    """Convert ccxt's ``[[price, amount], ...]`` into finite :class:`BookLevel` tuples.

    Bids must descend, asks must ascend; ccxt already guarantees this, but we sort so a
    single odd exchange cannot corrupt derived stats.
    """
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    levels: list[BookLevel] = []
    source = cast("Sequence[object]", raw)
    for entry in source:
        # The guard is a plain bool predicate so the loop variable stays `object`;
        # an isinstance() narrowing here would leak list[Unknown] into the types.
        if not _is_pair(entry):
            continue
        pair = cast("tuple[object, ...]", entry)
        if len(pair) < 2:
            continue
        price = finite(pair[0])
        amount = finite(pair[1])
        # A zero-amount level is not a real level; it would distort depth stats.
        if price is None or amount is None or price <= 0 or amount <= 0:
            continue
        levels.append(BookLevel(price=price, amount=amount))
    levels.sort(key=lambda lvl: lvl.price, reverse=not ascending)
    return tuple(levels)


def depth_within(
    bids: Sequence[BookLevel],
    asks: Sequence[BookLevel],
    bands: Sequence[float],
    mid: float | None,
) -> tuple[DepthBand, ...]:
    """Cumulative depth on both sides within ``pct`` of ``mid``.

    ``pct`` values are fractions of the mid (0.005 == 0.5%).
    """
    if mid is None or mid <= 0:
        return ()
    out: list[DepthBand] = []
    for pct in bands:
        lower = mid * (1.0 - pct)
        upper = mid * (1.0 + pct)
        bid_levels = [lvl for lvl in bids if lower <= lvl.price <= upper]
        ask_levels = [lvl for lvl in asks if lower <= lvl.price <= upper]
        bid_base = sum(lvl.amount for lvl in bid_levels)
        ask_base = sum(lvl.amount for lvl in ask_levels)
        out.append(
            DepthBand(
                pct=pct,
                bid_base=bid_base,
                ask_base=ask_base,
                bid_quote=bid_base * mid,
                ask_quote=ask_base * mid,
            )
        )
    return tuple(out)


def imbalance_from(bid_base: float | None, ask_base: float | None) -> float | None:
    """(bid - ask) / (bid + ask); ``None`` when both sides are empty."""
    if bid_base is None or ask_base is None:
        return None
    total = bid_base + ask_base
    if total <= 0:
        return None
    return (bid_base - ask_base) / total


def spread_from(
    best_bid: float | None, best_ask: float | None
) -> tuple[float | None, float | None]:
    """Return ``(absolute_spread, spread_in_bps)``."""
    if best_bid is None or best_ask is None or best_bid <= 0 or best_ask <= 0:
        return None, None
    if best_ask < best_bid:
        # Crossed/locked book: the derived numbers would be meaningless.
        return None, None
    mid = (best_bid + best_ask) / 2
    spread = best_ask - best_bid
    return spread, (spread / mid * 10_000 if mid > 0 else None)


def percent_of(value: float | None, reference: float | None) -> float | None:
    """``value`` as a percentage of ``reference``; ``None`` if either is unusable."""
    if value is None or reference is None or reference == 0:
        return None
    return (value / reference - 1.0) * 100.0


def strip_html(text: object, *, limit: int | None = None) -> str | None:
    """Best-effort plain text from CoinGecko's HTML description. Never raises.

    Descriptions contain arbitrary inline markup (``<b>``, ``<a href>``, ``<li>`` …), so
    tags are removed generically rather than matched against a fixed list. Block-level
    tags become spaces so words do not run together; entities are decoded afterwards.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    out = _BLOCK_TAG_RE.sub(" ", text)
    out = _TAG_RE.sub("", out)
    out = html.unescape(out)
    out = " ".join(out.split())
    if limit is not None:
        out = out[:limit].rstrip()
    return out or None


def first_link(value: Any) -> str | None:
    """CoinGecko link fields are sometimes ``str`` and sometimes ``list[str]``."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        candidates = cast("Sequence[object]", value)
        for item in candidates:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return None


def link_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        entries = cast("Sequence[object]", value)
        return tuple(e.strip() for e in entries if isinstance(e, str) and e.strip())
    return ()


__all__ = [
    "TIMEFRAME_SECONDS",
    "book_levels",
    "depth_within",
    "epoch_ms_to_utc",
    "epoch_s_to_utc",
    "first_link",
    "imbalance_from",
    "link_tuple",
    "parse_iso",
    "parse_timestamp",
    "percent_of",
    "spread_from",
    "strip_html",
]