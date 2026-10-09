#!/usr/bin/env python3
"""Generate notebook/pipeline.ipynb.

The notebook is committed to the repo, but building it from a script rather than by
hand means the cells stay diffable and the file is reproducible. Each cell below is
``(kind, source)``; markdown cells are the prose, code cells are executed in order by
``validate.py`` to prove the notebook actually runs.

    uv run python tools/build_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NOTEBOOK = Path(__file__).resolve().parent.parent / "notebook" / "pipeline.ipynb"


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
# tickerkit — pipeline walkthrough

An executable tour of the pipeline: fetch snapshots, reshape them for analysis, and
check the data-quality machinery that makes the numbers trustworthy.

**Everything here hits live free APIs.** There is no key and no account, so CoinGecko's
keyless tier is IP-rate-limited (429s under repeated probing) and CoinPaprika's public
tier has a monthly budget. If a cell reports a rate limit, that is the library behaving
correctly — see the *Rate limits* section near the bottom rather than treating it as a
failure.

## Contents

1. [Setup](#1-setup)
2. [Fetch one snapshot](#2-fetch-one-snapshot)
3. [Human-readable output](#3-human-readable-output)
4. [Pipeline shapes](#4-pipeline-shapes)
5. [DataFrames](#5-dataframes)
6. [Batch collection](#6-batch-collection)
7. [Quality: how to trust a number](#7-quality-how-to-trust-a-number)
8. [Rate limits and degradation](#8-rate-limits-and-degradation)
9. [Overriding a single provider](#9-overriding-a-single-provider)
10. [Deep history with cross-referencing](#10-deep-history-with-cross-referencing)
"""),
    code("""
import asyncio
import json

import pandas as pd

import tickerkit
from tickerkit import TickerKit
from tickerkit.render import (
    flat_metrics,
    render_markdown,
    render_report,
    to_records,
)

pd.set_option("display.max_columns", 50)
pd.set_option("display.width", 160)
print(f"tickerkit {tickerkit.__version__}")
"""),
    md("""
## 1. Setup

The public surface is three functions:

- `get_ticker(symbol)` / `await aget_ticker(symbol)` — one snapshot
- `TickerKit` — a reusable instance (keeps its own cache and limiter)
- the `tickerkit` CLI — see the end

A snapshot is a frozen dataclass. Nothing mutates after construction, so it is safe to
hold on to and compare.
"""),
    md("""
## 2. Fetch one snapshot

`aget_ticker` is the async entry point. It resolves the ticker, then runs every declared
provider capability concurrently, so one slow venue does not serialise the rest.
"""),
    code("""
snap = await tickerkit.aget_ticker("BTC")

print(snap.identity.symbol, "-", snap.identity.name)
print("quote currency:", snap.market.quote_currency)
print("price        :", snap.market.price)
print("market cap   :", snap.market.market_cap)
print("venues       :", len(snap.venues))
print("order books  :", len(snap.orderbooks))
print("trades       :", len(snap.trades))
print("ohlcv series :", len(snap.ohlcv))
print("derivatives  :", len(snap.derivatives))
"""),
    md("""
### Per-venue detail

Each venue keeps its own provenance. `mid` is computed from bid/ask, `deviation_pct` is
its distance from the consensus, and `is_stale` / `is_outlier` are set by the reconcile
layer rather than guessed.
"""),
    code("""
pd.DataFrame(
    [
        {
            "venue": v.exchange,
            "symbol": v.symbol,
            "bid": v.bid,
            "ask": v.ask,
            "mid": v.mid,
            "spread_bps": v.spread_bps,
            "deviation_pct": v.deviation_pct,
            "is_stale": v.is_stale,
            "is_outlier": v.is_outlier,
            "source": v.provenance.source if v.provenance else None,
        }
        for v in snap.venues
    ]
)
"""),
    md("""
### The consensus

The consensus price is a median across venues after outliers are excluded, so a single
bad venue cannot move it. `market.price` prefers the consensus and falls back to the
aggregator price when no venue answered — it is never fabricated.
"""),
    code("""
market = snap.market
print("consensus method :", market.consensus_method)
print("consensus price :", market.consensus_price)
print("consensus inputs:", market.consensus_inputs)
print("aggregated price:", market.aggregated_price, "(independent aggregator)")
print("reported price  :", market.price, "(consensus if present, else aggregator)")
print()
print("discrepancies   :", [(d.kind, d.detail) for d in snap.data_quality.discrepancies])
"""),
    md("""
## 3. Human-readable output

The render layer needs no pandas and no colour, so its output is safe to paste into a
terminal, a PR, or an issue.

Sections are composable — pass `sections=` to choose which ones you want.
"""),
    code("""
print(render_report(snap, sections=("summary", "quality")))
"""),
    code("""
print(render_report(snap, sections=("venues", "derivatives")))
"""),
    code("""
# The whole thing as a fenced markdown document.
print(render_markdown(snap, sections=("summary",)))
"""),
    md("""
## 4. Pipeline shapes

Two shapes built for machines rather than humans.

- `flat_metrics(snap)` — one flat dict of scalars per asset. Append one per ticker and
  you have a tidy DataFrame.
- `to_records(snap)` — normalised row lists for the per-venue / per-derivatives tables.

**The important convention:** absent measurements are `None` (`null` in JSON), never `0`.
A missing measurement and a real zero are different facts, and a pipeline that cannot
tell them apart will quietly average garbage.
"""),
    code("""
row = flat_metrics(snap)
pd.Series({k: v for k, v in row.items()})
"""),
    code("""
records = to_records(snap)
print("record groups:", sorted(records))
for name, rows in records.items():
    print(f"  {name}: {len(rows)} rows")
"""),
    code("""
# Absent measurements stay None so pandas gives NaN, not a fabricated 0.
pd.DataFrame(records["derivatives"])
"""),
    md("""
## 5. DataFrames

`snapshot.to_dataframes()` gives the same tables with provenance columns attached. Pandas
is imported lazily inside the method, so the core path stays light.
"""),
    code("""
frames = snap.to_dataframes()
print(sorted(frames))
for name, frame in frames.items():
    print(f"{name:14} {frame.shape[0]:>5} rows x {frame.shape[1]:>2} cols")
"""),
    code("""
frames["venues"]
"""),
    md("""
## 6. Batch collection

Collecting several tickers is the realistic case, and it is where rate limits bite:
CoinGecko's keyless tier is shared per IP. Spacing requests keeps a sweep inside it.

The helper below is the one to reuse — it is what `examples/pipeline.py` uses too.
"""),
    code("""
async def collect(tickers, delay=6.0, **kwargs):
    \"\"\"One snapshot per ticker, spaced out to respect free-tier rate limits.\"\"\"
    out = []
    for i, ticker in enumerate(tickers):
        if i:
            await asyncio.sleep(delay)
        out.append(await tickerkit.aget_ticker(ticker, **kwargs))
    return out


snaps = await collect(["BTC", "ETH", "SOL"], delay=6.0)
print(f"collected {len(snaps)} snapshots")
"""),
    code("""
table = pd.DataFrame([flat_metrics(s) for s in snaps])
table[["symbol", "price", "change_24h_pct", "market_cap", "venue_count",
       "error_count", "missing_count"]]
"""),
    md("""
Note the `error_count` and `missing_count` columns. They travel with every row so a
downstream job can filter degraded snapshots without re-reading the snapshot objects —
and `price_source` records *which* provider that price came from, which matters once
more than one aggregator is contributing.
"""),
    code("""
table[["symbol", "price", "price_source", "providers_used", "tvl"]]
"""),
    md("""
## 7. Quality: how to trust a number

Every snapshot carries a `data_quality` block. This is the part that distinguishes a
measured value from an absent one, and a stale number from a fresh one.

Checks worth wiring into a real pipeline:

- `dq.ok` / `dq.errors` — did any provider fail?
- `dq.throttled` / `dq.rate_limited` — was anything rate limited (and what is the fix)?
- `dq.excluded_outliers` — was any venue dropped from the consensus, and why?
- `dq.stale_flags` — is any number too old to trust?
- `dq.discrepancies` — do independent aggregators disagree?
- `dq.missing` — why is each `None` a `None`?
"""),
    code("""
dq = snap.data_quality

print("ok        :", dq.ok)
print("errors    :", len(dq.errors))
print("throttled :", dq.throttled)
print("outliers  :", dq.excluded_outliers)
print("stale     :", dq.stale_flags)
print("discrep.  :", [d.kind for d in dq.discrepancies])
print()
print("providers used   :", dq.providers_used)
print("providers skipped:", [(p.provider, p.reason) for p in dq.providers_skipped])
"""),
    code("""
for item in dq.missing:
    print(f"missing  {item.field}: {item.reason}")
"""),
    code("""
for ambiguity in dq.ambiguities[:5]:
    print(f"ambiguous  {ambiguity.symbol} -> {ambiguity.coingecko_id} ({ambiguity.name})")
if len(dq.ambiguities) > 5:
    print(f"... and {len(dq.ambiguities) - 5} more")
"""),
    md("""
### Symbol ambiguity is normal

Crypto has dozens of tokens sharing a ticker. The rule is: highest market cap wins, and
every other plausible match is recorded as an `Ambiguity` so you can see what else the
caller might have meant. Nothing is silently dropped.
"""),
    code("""
amb = dq.ambiguities
if not amb:
    print("no ambiguities for this snapshot")

rows = pd.DataFrame(
    [
        {
            "symbol": a.symbol,
            "candidate_id": a.coingecko_id,
            "name": a.name,
            "market_cap": a.market_cap,
        }
        for a in amb
    ]
)
if not rows.empty:
    rows.sort_values("market_cap", ascending=False, na_position="last")
"""),
    md("""
### Provenance for every field group

No number in a snapshot is unattributable. `market.provenance` says where each group of
fields came from, with the URL and fetch time.
"""),
    code("""
for key, prov in snap.market.provenance.items():
    print(f"{key:12} {prov.source}")
    print(f"{'':12} url={prov.url}")
    print(f"{'':12} fetched_at={prov.fetched_at} cached={prov.cached}")
    for note in prov.notes:
        print(f"{'':12} note: {note}")
"""),
    md("""
## 8. Rate limits and degradation

This is the part that makes a free-API pipeline survivable.

A `429` is not just an error — it is recorded as a *structured* throttle that says which
source refused, how long to wait, and what to change:

```text
rate limit   coingecko/resolve: retry in 60s
             remedy: set TICKERKIT_COINGECKO_API_KEY (Demo) or
             TICKERKIT_COINGECKO_PRO_API_KEY for 100+/min instead of the 25/min
             keyless IP limit
```

Two behaviours make that survivable:

1. **`Retry-After` is honoured**, capped by `TICKERKIT_MAX_RETRY_AFTER` (default 60s).
   Retrying a 429 on local backoff alone is what escalates one over-quota request into
   an IP-level block.
2. **A per-source circuit breaker** short-circuits that provider's remaining calls
   instead of firing doomed requests. Breakers are per source, so one exhausted
   provider never blocks the others.

The redundancy matters most: CoinPaprika is keyless with a *monthly* budget, so when
CoinGecko throttles, price and market cap still arrive.
"""),
    code("""
limited = dq.rate_limited
if not limited:
    print("this snapshot was not rate limited")
else:
    for item in limited:
        print(f"source    : {item.source} (capability {item.capability})")
        print(f"message   : {item.message}")
        print(f"retry in  : {item.retry_after}")
        print(f"quota     : {item.limit} (remaining {item.remaining})")
        print(f"remedy    : {item.remedy}")
        print(f"suppressed: {item.suppressed}")
"""),
    md("""
### Proving the degradation

Run the identical pipeline with CoinGecko removed from the registry and compare. This is
the scenario a throttled keyless tier puts you in — the question is whether the snapshot
still carries a price and a market cap, or collapses to nothing.

CoinGecko is excluded by constructing a `TickerKit` with an explicit provider list rather
than by mocking it: `respx` cannot pass unmatched requests through to the real network in
this environment (it returns an empty 200), so a mocked demo would silently stub out
CoinPaprika and the exchanges too, and prove nothing.
"""),
    code("""
from tickerkit.providers import ccxt_provider, coinpaprika, defillama

without_coingecko = TickerKit(
    providers=(coinpaprika.CoinPaprikaProvider, ccxt_provider.CcxtProvider,
               defillama.DefiLlamaProvider),
    config=tickerkit.Config(cache_ttl=0),
)
degraded = await without_coingecko.get_ticker("BTC")
"""),
    code("""
print("symbol      :", degraded.identity.symbol, "|", degraded.identity.name)
print("price       :", degraded.market.price)
print("market cap  :", degraded.market.market_cap)
print("24h change  :", degraded.market.change_24h_pct)
print("price from  :", {k: v.source for k, v in degraded.market.provenance.items()})
print("providers   :", degraded.data_quality.providers_used)
"""),
    md("""
Compare the two rows. The price survives on the exchanges either way; what CoinGecko was
actually contributing is the market cap and 24h change, and CoinPaprika supplies those
too — because its quota is monthly, not per-minute.

`identity.coingecko_id` stays `None` when CoinGecko is absent. That is deliberate: a
snapshot must not claim to know what an asset *is* on the strength of a source that never
answered.
"""),
    md("""
## 9. Why the test suite can run offline

The test suite runs 230 tests without touching the network, because `respx` replays
recorded fixtures from `tests/fixtures/` for every provider adapter. That is the right
tool for deterministic assertions, but note the boundary: `respx` intercepts *all*
traffic, so it is for offline fixtures — not for simulating one live source failing
while the others stay live. For that, exclude the provider from the registry instead, as
the previous cell did.
"""),
    code("""
# Same snapshot, CoinGecko present this time: market cap and identity come back.
with_coingecko = await tickerkit.aget_ticker("BTC")

comparison = pd.DataFrame(
    [
        {
            "sources": "without coingecko",
            "coingecko_id": degraded.identity.coingecko_id,
            "price": degraded.market.price,
            "market_cap": degraded.market.market_cap,
            "change_24h_pct": degraded.market.change_24h_pct,
        },
        {
            "sources": "with coingecko",
            "coingecko_id": with_coingecko.identity.coingecko_id,
            "price": with_coingecko.market.price,
            "market_cap": with_coingecko.market.market_cap,
            "change_24h_pct": with_coingecko.market.change_24h_pct,
        },
    ]
)
comparison
"""),
    md("""
## 10. Deep history with cross-referencing

`aget_history(ticker, timeframe=..., start=..., end=...)` mirrors the snapshot flow one
level down the time axis: the ticker is resolved first, then every history-capable source
is asked for the same window concurrently.

Depth demands accountability, so a history result differs from a snapshot:

- Each source's **coverage** says how much of the window it actually returned: whether a
  pagination cap cut the walk short, or a venue simply does not reach that far back. A
  series that silently stops in 2021 would be worse than one that says so.
- OHLCV venues (the exchanges) merge into a **consensus**: a per-interval median over
  agreeing sources, after a venue whose closes deviate too far is excluded and recorded.
- Close-only aggregators (CoinGecko, DefiLlama) never build the series — they **audit** it
  against the consensus, and a systematic deviation is reported as a discrepancy.

`start` is the argument that buys depth. Omit it, and a provider returns only its most
recent page(s).
"""),
    code("""
history = await tickerkit.aget_history(
    "BTC", timeframe="1d", start="2026-08-01", end="2026-10-01"
)
print(tickerkit.render_history_report(history, sections=("summary", "sources", "quality")))
"""),
    code("""
consensus = history.consensus
if consensus is not None:
    print(f"method    : {consensus.method} over {consensus.inputs}")
    print(f"candles   : {len(consensus.candles)}")
    print("deviations:", {k: round(v, 4) for k, v in consensus.deviations_pct.items()})
for discrepancy in history.data_quality.discrepancies:
    print(f"discrepancy [{discrepancy.kind}] {discrepancy.detail}")
"""),
    code("""
frames = history.to_dataframes()
print("history frames:", sorted(frames))
frames["consensus"].tail(3)
"""),
    code("""
# One flat row per history request, with completeness stated explicitly.
row = tickerkit.history_flat_metrics(history)
pd.Series({k: v for k, v in row.items()})
"""),
    md("""
The same honesty rule applies one level down: `history.coverage.complete` and
`coverage.truncated` tell a pipeline whether the gap between the first and last candle is
data that exists elsewhere or data this source could not provide — the difference between
"merge in a deeper source" and "re-run the walk".
"""),
    md("""
## Going further

- **Serialise** — `snap.to_json()` gives the full document, including provenance and
  every quality record.
- **CLI** — the same pipeline without writing any code:
  `tickerkit BTC ETH SOL --report`, `--markdown`, `--metrics`, or history via
  `tickerkit BTC --history --timeframe 1d --start 2024-01-01 --end 2024-06-01`.
- **Overriding a resolution** — when the ticker is ambiguous, name the asset:
  `await tickerkit.aget_ticker("BTC", coingecko_id="bitcoin")`.
- **Writing CSV** — `examples/pipeline.py` has a ready-made `to_csv` over `flat_metrics`
  rows.
"""),
    code("""
payload = json.loads(snap.to_json())
print("top-level keys:", sorted(payload))
print("data_quality  :", sorted(payload["data_quality"]))
"""),
    md("""
---

Every adapter cites the official docs it was written against; see the *Documentation
sources* section of the README. All figures in this notebook come from live free APIs at
run time — treat them as a demonstration of the plumbing, not as market data to trade on.
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