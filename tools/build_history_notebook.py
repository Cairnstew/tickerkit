#!/usr/bin/env python3
"""Generate notebook/history.ipynb — the deep-history lookup/testing notebook.

Same pattern as tools/build_notebook.py: the notebook is committed, but built from a
script so cells stay diffable and the file is reproducible. ``(kind, source)`` cells;
code cells are executed in order by ``tools/validate_notebook.py``.

    uv run python tools/build_history_notebook.py
    uv run python tools/validate_notebook.py notebook/history.ipynb
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebook" / "history.ipynb"


def md(source: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(source)}


def code(source: str) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _lines(source),
    }


def _lines(text: str) -> list[str]:
    """Notebook sources are line lists with trailing newlines, as nbformat expects."""
    text = text.strip("\n")
    return [line + "\n" for line in text.split("\n")[:-1]] + [text.split("\n")[-1]]


CELLS: list[dict[str, Any]] = [
    md("""
# tickerkit — history: deep lookup and the tests behind it

Two halves, deliberately separated:

1. **Live historical lookup** (sections 1-3) — `aget_history` against the real free
   APIs: a deep backfill, the coverage contract that makes a short series impossible to
   mistake for a full one, and the cross-referencing that audits it.
2. **The tests behind it** (section 4) — the same contracts exercised offline, so the
   behaviour is specified even when the network is not there. This is exactly what
   `tools/validate_notebook.py notebook/history.ipynb` runs, end to end.

**Everything here hits live free APIs.** No key and no account: CoinGecko is IP-rate
limited and may 429 under probing, and CoinPaprika has a monthly budget. A reported rate
limit is the library behaving correctly.
"""),
    code("""
import asyncio
import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta

import pandas as pd

import tickerkit
from tickerkit.history import (
    build_coverage,
    build_history_consensus,
    cross_check_references,
)
from tickerkit.models import HistoryCandle, HistoryRequest, HistorySeries
from tickerkit.providers.ccxt_provider import _ohlcv_fetcher, _paginate_ohlcv
from tickerkit.render import history_flat_metrics, render_history_report

pd.set_option("display.max_columns", 60)
pd.set_option("display.width", 180)
print(f"tickerkit {tickerkit.__version__}")
"""),
    md("""
## 1. The history API

`aget_history(ticker, timeframe=..., start=..., end=..., limit=...)` mirrors the snapshot
flow one level down the time axis: the ticker is resolved first, then every
history-capable source answers the same window concurrently.

The result carries one `HistorySeries` per source, a merged `consensus`, and the same
`data_quality` block as a snapshot.

This first call uses a **recent** window (120 days) so every source — including
CoinGecko's 365-day free limit — can serve it and the cross-check is full-strength.
"""),
    code("""
recent = (datetime.now(UTC) - timedelta(days=120)).date().isoformat()
hist = await tickerkit.aget_history("BTC", timeframe="1d", start=recent)

print(render_history_report(hist, sections=("summary", "sources")))
"""),
    code("""
consensus = hist.consensus
print(f"method     : {consensus.method} over {consensus.inputs}")
print(f"candles    : {len(consensus.candles)}")
print("deviations :", {k: round(v, 4) for k, v in consensus.deviations_pct.items()})
print()
for discrepancy in hist.data_quality.discrepancies:
    print(f"discrepancy [{discrepancy.kind}] {discrepancy.detail}")
"""),
    md("""
### Machine shapes

`history_flat_metrics` is the pipeline row — one dict per request, with the completeness
verdict stated explicitly (`coverage_complete`, `coverage_truncated`, `coverage_missing`)
so a downstream job can reject a short series without re-reading the coverage objects.
"""),
    code("""
frames = hist.to_dataframes()
print("history frames:", sorted(frames))
frames["consensus"].tail(3)
"""),
    code("""
pd.Series({k: v for k, v in history_flat_metrics(hist).items()})
"""),
    md("""
## 2. Depth is what you ask for

The `start` argument is what buys depth. Without it, a provider returns only its most
recent page(s), capped by `limit` — handy for a quick recent tail:

- no `start`, `limit=8` → the last 8 hourly bars from every venue
- `exchanges=["coinbase"]` → depth selection and consensus restricted to one venue
"""),
    code("""
hourly = await tickerkit.aget_history("BTC", timeframe="1h", limit=8)
print(render_history_report(hourly, sections=("summary", "sources")))
"""),
    code("""
cb = await tickerkit.aget_history("BTC", timeframe="1d", limit=5, exchanges=["coinbase"])
for series in cb.series:
    coverage = series.coverage
    print(f"{series.source:20} {series.kind:5} {len(series.candles)} candles  "
          f"complete={coverage.complete}  calls={coverage.calls}")
"""),
    md("""
## 3. The coverage contract

A deep request is where the accountability matters. `start="2021-01-01"` asks for three
years of daily bars. Look at what each source says about itself:

- **ccxt:coinbase / ccxt:binance** — complete; Binance's earliest available candle note
  appears only when asked for data before its pair existed.
- **ccxt:kraken** — *truncated*, and the note says why: Kraken ignores `since` and
  returns only its most recent ~720 bars. That is a silent wrong-range answer elsewhere;
  here it is labelled.
- **coingecko** — *missing*, because the free tier hard-limits queries to the past 365
  days; the clamp is stated, not hidden.
- **defillama** — pages in 500-point walks and reports its genuine interior gaps.
"""),
    code("""
deep = await tickerkit.aget_history(
    "BTC", timeframe="1d", start="2021-01-01", end="2024-01-01"
)
print(render_history_report(deep, sections=("summary", "sources")))
"""),
    code("""
for item in deep.data_quality.missing:
    if item.field.startswith("history"):
        print(f"missing  {item.field}: {item.reason}")
print()
print("consensus coverage:",
      deep.consensus.coverage.complete, "| candles:", len(deep.consensus.candles))
"""),
    code("""
# Sub-hourly is where DefiLlama draws the line (charts start at 1h). The decline is a
# reasoned missing record, not a crash.
sub = await tickerkit.aget_history("BTC", timeframe="5m", limit=8)
for series in sub.series:
    print(f"{series.source:20} {series.kind:5} {len(series.candles)} candles")
for item in sub.data_quality.missing:
    if item.field.startswith("history"):
        print(f"missing  {item.field}: {item.reason}")
"""),
    md("""
## 4. How the history layer is tested offline

The CI suite never touches the network: adapter tests replay recorded fixtures through
`respx`, and the pure logic is specified directly. Everything below runs in this
notebook without a single request — it is exactly what
`pytest tests/test_history.py tests/providers/test_history_providers.py` covers.

### 4a. The coverage invariants

`HistoryCoverage.complete` is deliberately conservative. A full window is complete; a
missing interior candle, and a walk cut short by a cap, are not.
"""),
    code("""
request = HistoryRequest(
    timeframe="1d",
    start=datetime(2024, 1, 1, tzinfo=UTC),
    end=datetime(2024, 1, 5, tzinfo=UTC),
)
full_ts = [datetime(2024, 1, day, tzinfo=UTC) for day in (1, 2, 3, 4, 5)]

assert build_coverage(full_ts, request).complete is True  # every interval present

broken = [t for t in full_ts if t.day != 3]
coverage = build_coverage(broken, request)
assert coverage.complete is False and coverage.missing == 1 and len(coverage.gaps) == 1

truncated = build_coverage(full_ts, request, truncated=True)
assert truncated.complete is False and truncated.truncated is True

print("coverage invariants hold: complete / gap / truncated")
print("gap:", coverage.gaps[0].start.date(), "->", coverage.gaps[0].end.date(),
      f"({coverage.gaps[0].missing} missing)")
"""),
    md("""
### 4b. The pagination walk

`_paginate_ohlcv` walks a venue forward from `start_ms` in pages. It stops when the
window is covered, when a page adds nothing new (a venue that ignores `since` must not
spin forever), or when `max_calls` is exhausted — and *only then* does it report
truncation.
"""),
    code("""
class FakeVenue:
    \"\"\"A minimal venue: serves rows at or after `since`, like a real klines API.\"\"\"

    def __init__(self, rows, ignore_since=False):
        self.rows = rows
        self.ignore_since = ignore_since

    async def fetch_ohlcv(self, symbol, timeframe, since=None, limit=None):
        limit = limit or len(self.rows)
        if since is None or self.ignore_since:
            return self.rows[:limit]
        return [r for r in self.rows if r[0] >= since][:limit]


DAY_MS = 86_400_000
start_ms = 1_704_067_200_000  # 2024-01-01T00:00:00Z
rows = [[start_ms + i * DAY_MS, 100.0 + i, 101.0 + i, 99.0 + i, 100.0 + i, 1.0]
        for i in range(10)]

for max_calls in (2, 25):
    got, calls, truncated = await _paginate_ohlcv(
        _ohlcv_fetcher(FakeVenue(rows), "BTC/USDT", "1d"),
        interval_ms=DAY_MS, start_ms=start_ms, end_ms=start_ms + 9 * DAY_MS,
        page_limit=3, max_calls=max_calls,
    )
    print(f"max_calls={max_calls:>2}: rows={len(got):>2} calls={calls} truncated={truncated}")

spin, calls, _ = await _paginate_ohlcv(
    _ohlcv_fetcher(FakeVenue(rows, ignore_since=True), "BTC/USDT", "1d"),
    interval_ms=DAY_MS, start_ms=start_ms, end_ms=start_ms + 100 * DAY_MS,
    page_limit=3, max_calls=10,
)
print(f"since-ignoring venue: rows={len(spin)} calls={calls} (stopped on no progress)")
"""),
    md("""
### 4c. The consensus rules

OHLCV venues merge into a per-interval median. A venue whose closes deviate by more than
`history_discrepancy_pct` (default 1%) is excluded — never averaged in — and a close-only
`kind="price"` series is never a consensus input.
"""),
    code("""
def ohlcv_series(source, days, price):
    return HistorySeries(
        source=source, symbol="BTC", timeframe="1d", quote_currency="USD", kind="ohlcv",
        candles=tuple(
            HistoryCandle(timestamp=datetime(2024, 1, d, tzinfo=UTC), close=price,
                          open=price, high=price, low=price, volume=1.0)
            for d in days
        ),
        coverage=build_coverage(
            [datetime(2024, 1, d, tzinfo=UTC) for d in days], HistoryRequest(timeframe="1d")
        ),
    )


config = tickerkit.Config()
days = (1, 2, 3)
consensus, discrepancies = build_history_consensus(
    [ohlcv_series("ccxt:binance", days, 100.0),
     ohlcv_series("ccxt:okx", days, 100.0),
     ohlcv_series("ccxt:bybit", days, 150.0)],  # 50% high: a wrong feed
    config, HistoryRequest(timeframe="1d"),
)
assert consensus.method == "median_outlier_trimmed"
assert "ccxt:bybit" not in consensus.inputs and consensus.candles[0].close == 100.0
print("consensus:", consensus.method, "->", consensus.inputs)
print("recorded :", [(d.kind, d.left_source) for d in discrepancies])
"""),
    code("""
price_only = HistorySeries(
    source="defillama:bitcoin", symbol="BTC", timeframe="1d", quote_currency="USD",
    kind="price",
    candles=(HistoryCandle(timestamp=datetime(2024, 1, 1, tzinfo=UTC), close=100.0),),
)
consensus2, _ = build_history_consensus(
    [ohlcv_series("ccxt:binance", (1,), 100.0), price_only],
    config, HistoryRequest(timeframe="1d"),
)
assert consensus2.inputs == ("ccxt:binance",)
print("price-only series excluded from consensus:", consensus2.inputs)
"""),
    md("""
### 4d. Auditing the consensus

Close-only aggregators never build the series — they audit it. A reference agreeing with
the consensus is silent; one deviating systematically is recorded as a discrepancy.
"""),
    code("""
consistent, _ = build_history_consensus([ohlcv_series("ccxt:coinbase", days, 100.0)],
                                        config, HistoryRequest(timeframe="1d"))


def price_series(source, close):
    return HistorySeries(
        source=source, symbol="BTC", timeframe="1d", quote_currency="USD", kind="price",
        candles=tuple(HistoryCandle(timestamp=datetime(2024, 1, d, tzinfo=UTC), close=close)
                      for d in days),
    )


agreeing = price_series("coingecko:bitcoin", 100.1)
assert cross_check_references(consistent, [agreeing], 1.0) == ()

diverging = price_series("defillama:bitcoin", 120.0)
reported = cross_check_references(consistent, [diverging], 1.0)
assert len(reported) == 1 and reported[0].kind == "history_reference_vs_consensus"
print("agreeing reference:", cross_check_references(consistent, [agreeing], 1.0))
print("diverging reference:", [(d.kind, round(d.deviation_pct, 2)) for d in reported])
"""),
    md("""
### 4e. The real suite

Everything above is codified in the two history test files. Run them — still offline,
via `respx` fixtures plus the same fake venue used here.
"""),
    code("""
result = subprocess.run(
    [sys.executable, "-m", "pytest", "-q",
     "tests/test_history.py", "tests/providers/test_history_providers.py"],
    capture_output=True, text=True,
)
print(result.stdout.strip()[-800:])
print("pytest exit:", result.returncode)
"""),
    md("""
## 5. The live/offline boundary

- **Offline (section 4)** is the contract: deterministic, fast, and CI-safe. `respx`
  replays recorded fixtures, and the pure logic is asserted directly.
- **Live (sections 1-3)** is the smoke test the offline suite cannot do: real pagination,
  real venue behaviour (Kraken's `since`, DefiLlama's 500-point pages), real free-tier
  limits (CoinGecko's 365 days). Treat every number here as a demonstration of the
  plumbing, not market data to trade on.

Going further: `examples/history.py` turns the live half into an executable pipeline
(`--csv`, `--json`, `--report`), and the README's *Deep history* section documents the
two config knobs — `TICKERKIT_HISTORY_MAX_CALLS` (pagination ceiling per series,
default 25) and `TICKERKIT_HISTORY_DISCREPANCY_PCT` (disagreement threshold, default 1.0).
"""),
    md("""
## 6. The disk cache

Every provider call is TTL-cached. By default the cache is in-memory and dies with the
process; pointing `TICKERKIT_CACHE_PATH` at a file swaps in a git-trackable SQLite
cache instead — bounded (2000 entries / 10 MiB default, then LRU), auto-vacuumed, with
canonical JSON payloads so identical data is identical bytes. You can commit the file
and replay history without re-hitting the rate-limited APIs.
"""),
    code("""
import tempfile
from pathlib import Path

from tickerkit import QueryCache

# QueryCache is the user-facing store: git-trackable, and a drop-in TickerKit cache
# (it speaks the Cache protocol). Exercise it on a scratch file so validate_notebook
# never touches the tracked tickerkit_cache.sqlite3 in the repo.
scratch = Path(tempfile.gettempdir()) / "tickerkit_demo_cache.sqlite3"
qc = QueryCache(scratch)
await qc.clear()
print("stats :", {k: qc.stats()[k] for k in ("entries", "bytes", "max_entries", "max_bytes")})

# save the `hist` HistoryResult from section 1 under its natural key (ttl=-1: immortal)
await qc.save_history("BTC", "1d", start=recent, history=hist, ttl=-1)
replay = await qc.load_history("BTC", "1d", start=recent)
print("replay :", type(replay).__name__, "with", len(replay.consensus.candles), "consensus candles")
print(render_history_report(replay, sections=("summary",))[:240])
qc.close()
"""),
]


def build() -> dict[str, Any]:
    return {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> int:
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK.write_text(json.dumps(build(), indent=1) + "\n")
    code_cells = sum(1 for c in CELLS if c["cell_type"] == "code")
    print(f"wrote {NOTEBOOK} ({len(CELLS)} cells, {code_cells} code)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())