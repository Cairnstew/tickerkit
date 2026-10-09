# tickerkit

![CI](https://github.com/Cairnstew/tickerkit/actions/workflows/ci.yml/badge.svg)

One call, one complete, provenance-tracked snapshot of everything obtainable for free
about a crypto asset.

```python
import asyncio, tickerkit

snap = asyncio.run(tickerkit.aget_ticker("BTC"))

snap.market.price                 # consensus price (median across venues)
snap.to_dict()                    # plain dict, JSON-serialisable
snap.to_json()                    # str
snap.to_dataframes()              # dict[str, pandas.DataFrame]
snap.data_quality.provenance      # where every number came from, and when
```

```console
$ tickerkit BTC --json
$ tickerkit ETH --frames ohlcv venues
$ tickerkit SOL --show-errors
```

- **Free tiers only.** No paid key is required for the core path. Optional keys live in
  environment variables and light up extra providers when present.
- **Partial-failure tolerant.** A provider that explodes is recorded in
  `data_quality.errors`; the snapshot still returns everything that worked.
- **Nothing is fabricated.** Missing data is `None` plus an explanation. ccxt's
  *emulated* methods (synthetic values) are excluded on purpose.
- **Everything is traceable.** Every field group carries `source`, `fetched_at` and the
  provider's own timestamp where it reports one.

---

## Install (Python package)

tickerkit is a normal Python package, published on PyPI and installable from this repo.

```console
# from PyPI (once released — see Publishing below)
pip install tickerkit            # or: uv add tickerkit

# from this repository (any revision)
pip install "git+https://github.com/Cairnstew/tickerkit.git@main"
uv add "git+https://github.com/Cairnstew/tickerkit.git@main"

# editable for development
uv sync                          # or: pip install -e ".[dev]"
```

Python 3.12+. Runtime dependencies: `ccxt` (MIT), `httpx`, `pandas`. The `dev` extra
adds test/lint/notebook tooling.

> **AGPL note:** `cryptofeed` is archived and AGPL-licensed, so it is deliberately *not*
> used. Everything here is MIT/BSD/Apache.

---

## Nix (run it without installing Python)

The flake builds a **hermetic** CLI straight from `uv.lock` (via uv2nix — ccxt is not in
nixpkgs, so the lockfile's wheels are fetched by hash into the Nix store). No Python,
pip, or uv needed on the machine that runs it:

```console
# local checkout
nix run .#default -- BTC                # hermetic CLI:  nix run .#default -- BTC --report
nix run .#tickerkit -- ETH --summary
nix build .#default                     # -> result/bin/tickerkit

# remote, straight from GitHub — works on any Nix machine with no checkout at all
nix run github:Cairnstew/tickerkit -- SOL --summary
nix run github:Cairnstew/tickerkit#tickerkit -- XRP --report

# the other app recipes
nix run .#test              # offline test suite (uv on the host)
nix run .#notebook          # execute notebook/pipeline.ipynb
nix run .#history-notebook  # execute notebook/history.ipynb
nix flake check             # ruff, sandboxed
```

The `default`/`tickerkit` app runs the exact same CLI as the Python package — including
`--history`, `--json`, `--frames`, and the git-trackable `QueryCache` — so
`nix run github:Cairnstew/tickerkit -- BTC --report` is a one-liner ticker report.

---

## Development environment (Nix)

A `flake.nix` also provides a reproducible dev shell. This is the recommended path on
NixOS because it fixes a real trap: `uv` installs PyPI wheels for numpy and pandas that
link against `libz.so.1` and `libstdc++.so.6` by soname, and NixOS has no `/lib` — so
`import pandas` fails with `libz.so.1: cannot open shared object file`. The shell puts
those libraries on `LD_LIBRARY_PATH` for you.

```bash
nix develop          # python 3.12, uv, ruff, basedpyright, and the runtime libs
                     # creates .venv via `uv sync` on first run
nix run .#test       # the offline test suite
nix run .#notebook   # execute every notebook cell
nix flake check      # ruff, sandboxed
```

`ccxt` is not packaged in nixpkgs, so the Python *dependencies* still come from `uv` and
`uv.lock`; the flake supplies the interpreter, tooling, and library paths around them.
The dev shell prepends `.venv/bin` to `PATH`, so `python`, `pytest`, `jupyter` and `ruff`
are the project's own.

The plain `uv` workflow still works if you already have the libraries available:

```bash
uv sync
uv run pytest -m "not live"
```

---

## Quickstart

```python
import asyncio
import tickerkit

async def main() -> None:
    snap = await tickerkit.aget_ticker("BTC/USDT")

    print(snap.identity.name, snap.identity.coingecko_id)
    print("price   ", snap.market.price, snap.market.quote_currency)
    print("venues  ", [(v.exchange, v.mid) for v in snap.venues])
    print("consensus", snap.market.consensus_method)

    prov = snap.market.provenance["aggregated"]
    print("from    ", prov.source, prov.fetched_at)

    for err in snap.data_quality.errors:
        print("tolerated:", err.provider, err.capability, err.message)

asyncio.run(main())
```

Synchronous helper, for scripts and notebooks:

```python
snap = tickerkit.get_ticker("ETH")          # refuses to run inside an event loop
snap = asyncio.run(tickerkit.aget_ticker("ETH"))   # use this inside one
```

### Overrides

```python
snap = await tickerkit.aget_ticker(
    "BTC",
    coingecko_id="bitcoin",        # force the asset
    exchanges=["kraken"],          # restrict spot venues
    perp_exchanges=["binance"],    # restrict derivative venues
    quote_currency="EUR",
    outlier_pct=2.0,               # tighter outlier rejection
    stale_seconds=30,              # stricter staleness
)
```

### Pandas views

```python
frames = snap.to_dataframes()
# venues, trades, ohlcv, orderbook, orderbook_depth, derivatives
frames["ohlcv"].tail()
```

### Output functions

`tickerkit.render` holds the presentation layer, split by audience. It imports no
pandas, so these work in a minimal environment and in CI.

**Humans** — aligned plain-text, no colour, safe to paste anywhere:

```python
from tickerkit.render import render_report, summary_block, venues_table, quality_block

print(render_report(snap))          # every section
print(summary_block(snap))          # headline only
print(venues_table(snap))           # bid/ask/mid/spread/deviation/flags
print(quality_block(snap))          # errors, discrepancies, ambiguities, missing fields
```

Sections are composable and ordered by you:

```python
render_report(snap, sections=("quality", "summary"))
render_report(snap, sections=("summary", "venues"), indent="  ")
render_markdown(snap)               # fenced block for a PR or an issue
```

Formatters degrade predictably — `None` becomes `-`, never `0`:

| helper | `None` | `81795.5` | `0.000003925` |
|---|---|---|---|
| `fmt_price` | `-` | `81,795.5` | `3.925e-06` |
| `fmt_compact` | `-` | `81.80K` | `0.00` |
| `fmt_pct` | `-` | — | — |
| `fmt_age` | `-` | — | — |

**Pipelines** — shapes built for machines, not eyes:

```python
from tickerkit.render import flat_metrics, to_records

flat_metrics(snap)      # one flat dict of scalars per asset -> one DataFrame row
to_records(snap)        # {"venues": [...], "trades": [...], "ohlcv": [...], ...}
```

`flat_metrics` includes `venue_count`, `median_spread_bps`, `outlier_count`,
`stale_count`, `mean_funding_rate`, `error_count`, `price_source` and
`price_fetched_at`, so a downstream job can filter on degraded snapshots without
re-reading the snapshot. Missing values are `None` (JSON `null`), never `0`.

```console
$ tickerkit BTC --metrics        # one flat JSON row
$ tickerkit BTC ETH --csv        # via examples/pipeline.py
```

---

## Caching

Every provider call goes through a TTL cache so repeated runs don't burn free-tier quota.
Three implementations share one async interface (`tickerkit.cache.Cache`):

| Implementation | When it's used | Persistence |
|---|---|---|
| `MemoryCache` | default (unless overridden) | per-process, dies with the process |
| `SQLiteCache` | `TICKERKIT_CACHE_PATH` set, or passed to `TickerKit(cache=...)` | one file, survives restarts, git-trackable |
| `NullCache` | `TICKERKIT_CACHE_ENABLED=0` | stores nothing |

### Disk cache (git-trackable)

Setting `TICKERKIT_CACHE_PATH=tickerkit_cache.sqlite3` swaps in a SQLite-backed cache
that can be committed to git:

- **One file, no sidecars** — `journal_mode=DELETE` means no `-wal`/`-shm` siblings to
  forget to commit.
- **Bounded, then LRU** — `TICKERKIT_CACHE_MAX_ENTRIES` (default 2000) and
  `TICKERKIT_CACHE_MAX_BYTES` (default 10 MiB). Every write evicts expired entries
  first, then the least-recently-used; `auto_vacuum` shrinks the file as rows go, so
  the committed artifact stays small. A just-written entry survives its own write —
  DefiLlama's full protocols list (~9 MB) refreshes without trashing the rest of the
  cache.
- **Deterministic bytes** — payloads are canonical JSON (`sort_keys`, compact
  separators), so identical data is identical bytes and re-committing a stable cache
  produces no churn. Non-JSON values fall back to pickle (tagged per row); only point
  the file at a cache you control.
- **TTL** — positive seconds (normal entries, e.g. the default 30 s snapshot TTL),
  zero expires immediately, **negative means never** — store a deep history series
  with `ttl=-1` and it persists for offline replay.
- **Lifetime values**: use `tickerkit cache prune` before committing to drop expired
  rows, or `vacuum` to reclaim pages after a big clear.

```
TICKERKIT_CACHE_PATH=tickerkit_cache.sqlite3 tickerkit BTC --summary   # seeds the file
tickerkit cache stats      # entries / bytes / bounds / expiry breakdown
tickerkit cache prune      # evict expired + over-budget rows
tickerkit cache vacuum     # reclaim file pages (commit a tight artifact)
tickerkit cache clear      # empty it
```

Programmatically: `TickerKit(config=Config(cache_path=Path("tickerkit_cache.sqlite3")))`,
or `TickerKit(cache=SQLiteCache("tickerkit_cache.sqlite3", limits=CacheLimits(...)))`.
Reads bump an entry's recency, so hot data survives eviction.

### Save & load your own queries (`QueryCache`)

`QueryCache` is the user-facing piece: point it at a file in your repo and it saves and
loads **whole fetches by their ticker key** — a `TickerSnapshot` or `HistoryResult` you
care about, replayable later without re-fetching or burning rate-limit quota. It is the
same git-trackable SQLite file, so committing it makes those queries reproducible
offline. It also speaks the `Cache` protocol, so the same object can be handed to
`TickerKit` for automatic provider caching.

```python
from tickerkit import QueryCache

async with QueryCache("tickerkit_cache.sqlite3") as qc:
    snapshot = await tickerkit.aget_ticker("BTC")
    await qc.save_snapshot("BTC", snapshot)                 # ttl=-1: keep forever
    later = await qc.load_snapshot("BTC")
    later.summary()                                         # a real TickerSnapshot again

    hist = await tickerkit.aget_history("BTC", timeframe="1d",
                                        start="2021-01-01", end="2024-01-01")
    await qc.save_history("BTC", "1d", start="2021-01-01", end="2024-01-01",
                          history=hist)
    replay = await qc.load_history("BTC", "1d", start="2021-01-01", end="2024-01-01")
    replay.consensus.candles[-1]                            # a real HistoryResult again
```

Saving accepts either the model object or its `to_dict()`/`to_json()` form. Both
`TickerSnapshot` and `HistoryResult` gained the inverse constructors too, so a JSON file
you kept can become an object again:

```python
snapshot = TickerSnapshot.from_json(open("btc_good_day.json").read())
hist     = HistoryResult.from_json(open("btc_2021_2024.json").read())
```

---

## Configuration

> Deep history (see below) is exercised in `examples/history.py` and in the notebook's
> section 10.

## Deep history

For windows rather than snapshots, `aget_history` mirrors the snapshot flow one level
down the time axis — same resolution, then every history-capable source answers the same
window concurrently. It is accountability-first: a series states how much of the window
it actually covered, and diverging sources are reported, never averaged.

```python
result = await tickerkit.aget_history("BTC", timeframe="1d",
                                      start="2017-01-01", end="2024-01-01")
result.consensus.method                    # "median" / "median_outlier_trimmed"
result.consensus.candles[-1]               # ConsensusCandle(timestamp, close, sources, n)
result.series                              # one HistorySeries per source
```

What you get per source — the **coverage** — is the part that makes depth trustworthy:

- `complete` — the requested window is fully covered with no gaps.
- `truncated` — a pagination cap, or a venue that ignores `since` (Kraken returns only
  its most recent ~720 bars whatever you ask), stopped the walk before the requested
  start. Truncation is never silent.
- `gaps` / `missing` — intervals the source genuinely does not have.

How sources combine:

- **OHLCV venues** (exchanges) merge into a per-interval median consensus. A venue whose
  closes deviate by more than `history_discrepancy_pct` (default `1.0%`) from the running
  median is excluded and recorded as a `history_source_outlier` discrepancy.
- **Close-only aggregators** (CoinGecko `market_chart/range`, DefiLlama `chart`) never
  build the series — they audit it, and a systematic deviation from the consensus is
  recorded as a `history_reference_vs_consensus` discrepancy.
- CoinGecko's free tier serves only the past 365 days; further-back requests are clamped
  and reported truncated (a Pro key lifts this). DefiLlama reaches ~2015, so it is the
  pre-listing cross-check.
- A venue's best-quoted pair is not necessarily its deepest: Binance now lists a spot
  `BTC/USD` pair with a far shorter history than `BTC/USDT`, so tickerkit probes the
  candidates and picks the one whose history goes back furthest.

`start` is what buys depth: without it a provider returns only its most recent page(s)
(capped by `limit`). `start`/`end` accept a `datetime`, an ISO-8601 string, or an epoch
(in seconds or milliseconds).

```python
print(tickerkit.render_history_report(result))      # summary + sources + consensus
result.to_dataframes()                              # {"history": long form, "consensus": ...}
tickerkit.history_flat_metrics(result)              # one pipeline row, with completeness flags
```

The same thing on the command line:

```bash
tickerkit BTC --history --timeframe 1d --start 2017-01-01 --end 2024-01-01 --report
tickerkit BTC --history --timeframe 1d --limit 30 --metrics
```

Config knobs: `TICKERKIT_HISTORY_MAX_CALLS` (default `25`) caps upstream requests spent
paginating one series, and `TICKERKIT_HISTORY_DISCREPANCY_PCT` (default `1.0`) sets the
disagreement threshold.

Everything is environment-driven; nothing is hardcoded, and no secret is ever stored in
the repo.

| Variable | Default | Meaning |
|---|---|---|
| `TICKERKIT_QUOTE_CURRENCY` | `USD` | Target quote for every normalised number |
| `TICKERKIT_EXCHANGES` | `binance,kraken,coinbase` | ccxt spot venues |
| `TICKERKIT_PERP_EXCHANGES` | `binance,bybit,okx` | ccxt derivative venues |
| `TICKERKIT_HTTP_TIMEOUT` | `15` | Per-request timeout, seconds |
| `TICKERKIT_RATE_LIMIT` | `60` | Global safety-net requests/minute |
| `TICKERKIT_CACHE_TTL` | `30` | Snapshot cache TTL, seconds |
| `TICKERKIT_CACHE_ENABLED` | `1` | Set `0` to bypass the cache |
| `TICKERKIT_CACHE_PATH` | — | SQLite cache file; set to persist across runs (git-trackable) |
| `TICKERKIT_CACHE_MAX_ENTRIES` | `2000` | Disk-cache entry cap before LRU eviction |
| `TICKERKIT_CACHE_MAX_BYTES` | `10485760` | Disk-cache payload cap (10 MiB) before LRU eviction |
| `TICKERKIT_MAX_RETRIES` | `3` | Retries for 429/5xx/timeouts only |
| `TICKERKIT_OUTLIER_PCT` | `5.0` | Outlier threshold, percent |
| `TICKERKIT_STALE_SECONDS` | `120` | Staleness threshold |
| `TICKERKIT_ORDERBOOK_DEPTH` | `20` | Levels per side |
| `TICKERKIT_OHLCV_TIMEFRAMES` | `5m,1h,1d` | Candle timeframes |
| `TICKERKIT_COINGECKO_API_KEY` | — | Optional CoinGecko **Demo** key |
| `TICKERKIT_COINGECKO_PRO_API_KEY` | — | Optional CoinGecko **Pro** key |
| `TICKERKIT_CMC_API_KEY` | — | Enables the optional CoinMarketCap source |

A Demo key raises the CoinGecko throttle from 25/min (keyless, IP-shared) to the
documented 100/min.

---

## Provider / capability matrix

| Provider | `resolve` | `meta` | `agg_market` | `venue_ticker` | `orderbook` | `trades` | `ohlcv` | `history` | `derivatives` | `onchain` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `ccxt` (binance, kraken, coinbase, …) | ✅ | – | – | ✅ | ✅ | ✅ | ✅ | ✅ deep | ✅ `has[]`-gated | – |
| `coingecko` | ✅ | ✅ | ✅ | – | – | – | – | ✅ ≤365d | – | – |
| `coinpaprika` | ✅ | ✅ | ✅ | – | – | – | – | – | – | – |
| `defillama` | ✅ | – | – | – | – | – | – | ✅ ~2015 | – | ✅ |
| `coinmarketcap` | ✅ | ✅ | ✅ | – | – | – | – | – | – | – |
| **Requires a key** | no | no | no | no | no | no | no | **no** (CoinGecko Pro lifts the 365d cap) | no | **CMC only** |

Per-exchange `derivatives` support varies and is decided at runtime by
`exchange.has[...]`. ccxt marks some methods as `"emulated"` — meaning they *synthesise*
a value rather than fetch one (bybit's `fetchFundingRate` returns a constant `0.0001`).
tickerkit treats emulated as unsupported and records why, because publishing a
synthesised number as if it were real would be fabricating data.

---

## Output schema

Top level is a frozen, slotted `dataclass` (`tickerkit.models.TickerSnapshot`).

```
TickerSnapshot
├── identity      symbol, name, coingecko_id, categories, links, description,
│                 contract_address, asset_platform_id, provenance
├── market        quote_currency, consensus_price, consensus_method, consensus_inputs,
│                 aggregated_price, change_24h(_pct), high_24h, low_24h, volume_24h,
│                 market_cap, fdv, circulating/total/max_supply,
│                 ath(+date,change), atl(+date,change), provenance: dict[str, Provenance]
├── venues[]      exchange, symbol, quote_currency, bid, ask, last, spread, spread_bps,
│                 volume_24h_base/quote, timestamp, mid, is_stale, is_outlier,
│                 deviation_pct/_bps, provenance
├── orderbooks[]  exchange, symbol, bids/asks[BookLevel], best_bid, best_ask, mid,
│                 spread(_bps), depth within 0.5%/1%/2% [DepthBand], imbalance,
│                 timestamp, provenance
├── trades[]      exchange, symbol, timestamp, price, amount, side, trade_id
├── ohlcv[]       exchange, symbol, timeframe, candles[Candle], provenance
├── derivatives[] exchange, symbol, contract_type, funding_rate(_pct),
│                 funding_interval_hours, open_interest_{contracts,base,quote},
│                 mark_price, index_price, basis, timestamp, provenance
├── onchain       defillama_slug, name, category, tvl, chains[ChainTVL], url, provenance
└── data_quality  provenance: dict[str, Provenance]
                  errors[]           provider, capability, exception_type, message
                  ambiguities[]      symbol, coingecko_id, name, market_cap
                  discrepancies[]    kind, detail, left/right source + value, deviation_pct
                  stale_flags[]      "kraken (age=4m)"
                  excluded_outliers[]
                  missing[]          field, reason
                  providers_used[] / providers_skipped[] (provider, reason)
```

`Provenance` = `source`, `capability`, `fetched_at` (UTC), `exchange_timestamp`,
`url`, `cached`, `notes`.

A deep-history request returns `tickerkit.models.HistoryResult` in the same shape:

```
HistoryResult
├── identity      (same as a snapshot)
├── request       timeframe, start, end, limit
├── series[]      source, symbol, timeframe, quote_currency, kind ("ohlcv" | "price"),
│                 candles[HistoryCandle], coverage, provenance
├── consensus     (optional) timeframe, quote_currency, method, inputs,
│                 candles[ConsensusCandle{timestamp, o/h/l/c/volume, sources, n, spread_pct}],
│                 deviations_pct, coverage
└── data_quality  (same block as a snapshot)

HistoryCandle    timestamp, close, open?, high?, low?, volume?
                 # open/high/low are None for close-only sources; never back-filled.
HistoryCoverage  requested_start/end, first, last, candles, expected, missing,
                 gaps[CandleGap{start, end, missing}], truncated, calls, complete, note
```

### Conventions

- **Timestamps** are timezone-aware UTC throughout. Both `fetched_at` (when tickerkit
  retrieved it) and `exchange_timestamp` (the provider's own clock) are recorded.
- **Quote currency** is stated on every group. A market quoted in USDT is read as USD
  only via the explicit, configurable stablecoin assumption
  (`Config.fiat_equivalent_stablecoins`), and that assumption is written into the
  group's provenance notes — never applied silently.
- **Precision.** Prices and volumes are IEEE-754 doubles (~1e-16 relative error).
  `Decimal` is not used: ccxt and pandas are float64-native, so converting at the
  boundary would advertise a precision the data does not have. Every value that crosses
  into JSON passes through `finite()`, so `NaN`/`inf` become `null` rather than
  invalid JSON.
- **Missing is `None`, never `0`.** Any `None` that matters has a matching
  `data_quality.missing` entry explaining it.

---

## Symbol resolution

`get_ticker("BTC")` resolves to `(a)` a CoinGecko id, `(b)` per-exchange market symbols,
`(c)` a DefiLlama slug where one applies.

Ambiguity is the normal case in crypto — dozens of tokens share a symbol. The rule is:

1. Ask CoinGecko `/coins/markets?symbols=…&include_tokens=all`.
2. **Highest market cap wins.** A token with no market cap sorts *last*; it is never
   treated as zero.
3. Every other plausible match is recorded in `data_quality.ambiguities`.
4. DefiLlama joins on `gecko_id`, and only falls back to a symbol match when that match
   is **unique** — three protocols sharing `BTC` is not a resolution.

```python
snap = await tickerkit.aget_ticker("BTC")
for a in snap.data_quality.ambiguities:
    print(a.symbol, a.coingecko_id, a.name, a.market_cap)
```

---

## Accuracy and reconciliation

- **Consensus** — median across venues.
- **Outlier exclusion** — a venue is excluded if it deviates more than
  `outlier_pct` from the median **or** its robust z-score (median absolute deviation)
  exceeds `outlier_mad_z`. The two criteria are unioned deliberately: with prices
  `[100, 100, 5000]` the MAD is zero, so a z-score-only test would collapse every score
  to 0 and let a rogue venue straight through. Every exclusion is recorded in
  `data_quality.excluded_outliers` and as a `discrepancy`.
- **Deviation flags** — surviving venues more than `outlier_pct` from consensus are
  flagged on their own record (`is_outlier`, `deviation_pct`, `deviation_bps`).
- **Staleness** — data older than `stale_seconds` is flagged in
  `data_quality.stale_flags`. A venue that publishes *no* timestamp is never claimed
  stale; absence of data is not evidence of old data.
- **Cross-source check** — the aggregator price (CoinGecko/CMC) is compared with the
  venue consensus and any divergence over `outlier_pct` is recorded as
  `aggregator_vs_consensus`.

```python
from tickerkit.reconcile import summarise
summarise(snap)
# {'price': 81795.9, 'method': 'median(n=3)', 'inputs': 3, 'excluded_outliers': [],
#  'stale': [], 'discrepancies': [], 'errors': 0}
```

---

## Reliability

- **Per-provider rate limiters.** CoinGecko documents 100 calls/min on the Demo plan;
  keyless is IP-shared and far tighter, so tickerkit throttles to 25/min until a key is
  present. CoinPaprika's ceiling is a *monthly* budget (20,000 calls), so it is
  self-throttled to a gentle 30/min rather than spending the month in a minute. ccxt
  manages its own request weight internally.
- **Throttling is identified, not just retried.** A 429 becomes a structured
  `RateLimitExceeded` carrying the source, the advertised `Retry-After`, the quota
  headers, and a concrete remedy ("set `TICKERKIT_COINGECKO_API_KEY`"). These surface on
  the snapshot as `data_quality.rate_limited` / `.throttled`, so a pipeline can react
  without string-matching exception names:

  ```console
  rate limit   coingecko/resolve: retry in 60s
               remedy: set TICKERKIT_COINGECKO_API_KEY (Demo) or ..._PRO_API_KEY
               for 100+/min instead of the 25/min keyless IP limit
  ```

  `Retry-After` (delta-seconds, HTTP-date, and CoinGecko's epoch `x-ratelimit-reset`) is
  honoured on retry, capped by `TICKERKIT_MAX_RETRY_AFTER` (default 60s) so a hostile
  value cannot park a caller for hours. Retrying a 429 on local backoff alone is what
  turns one over-quota request into an IP-level block.
- **A per-source circuit breaker.** Once a source is genuinely throttled, its remaining
  capabilities short-circuit instead of firing doomed requests — a snapshot fans out
  over ~8 calls per provider, and without this a throttled source would receive eight
  more 429s purely to record eight copies of the same failure. Breakers are per source,
  so CoinGecko being exhausted never blocks DefiLlama or the exchanges.
- **Redundant keyless sources.** CoinPaprika (keyless, 20k calls/mo) carries price and
  market cap when CoinGecko's keyless tier throttles, so a rate-limited snapshot
  degrades instead of emptying.
- **Retry policy** is narrow on purpose: only `408/425/429/5xx` and transport timeouts
  retry, with exponential backoff and full jitter. A `400/401/403/404` is a caller or
  credential problem — retrying it would only burn quota.
- **Timeouts** are per-request and configurable.
- **TTL cache**, in-memory by default behind a pluggable `Cache` protocol:

  ```python
  class Cache(Protocol):
      async def get(self, key: str) -> Any | None: ...
      async def set(self, key: str, value: Any, ttl: float) -> None: ...
      async def clear(self) -> None: ...
  ```

  ```python
  snap = await tickerkit.aget_ticker("BTC", cache=MyRedisCache())
  ```

- **Partial-failure tolerance.** Each `(provider, capability)` pair runs concurrently and
  independently. A failure becomes a `ProviderError` record; it never propagates.

---

## Verified against live APIs

Exercised against the real free endpoints with `exchanges=binance,kraken,coinbase`:

| Ticker | Resolves to | Notes |
|---|---|---|
| `BTC` `ETH` `XRP` `LINK` | majors | 3 venues, 3 perps, 1800 candles |
| `SOL` | `solana` | $6.2B **chain** TVL; also matches `solana-farm`, never guessed at |
| `DOGE` `ADA` `LTC` | majors | full coverage |
| `UNI` | `uniswap` | $3.95B parent aggregate across v1–v4 |
| `AAVE` | `aave` | $18.4B parent aggregate, not the `aave-v2` row ($109M) |
| `LDO` | `lido-dao` | $24.3B, per-chain breakdown |
| `ARB` | `arbitrum` | L2, chain TVL |
| `PEPE` | `pepe` | 20 symbol collisions recorded as ambiguities |
| `WBTC` | `wrapped-bitcoin` | 2 venues, 36 ambiguities, no perps |
| `USDT` | `tether` | 48 ambiguities (scam clones) |
| `NOTAREALCOIN` | — | no price; returns a valid snapshot with the reason |

Under CoinGecko's keyless IP rate limit a snapshot degrades honestly rather than
breaking: the throttle is recorded in `data_quality.rate_limited` with its remedy, the
identity may stay unresolved, `identity.coingecko_id` gets a `missing` explanation, and
no protocol TVL is guessed. Price and market cap still arrive, from CoinPaprika and the
exchanges. Set `TICKERKIT_COINGECKO_API_KEY` to lift the throttle to the documented
100 calls/min.

**CoinPaprika caveats.** Its public tier is a *monthly* budget rather than a per-minute
one, and its `ratelimit-reset` header is seconds remaining in that window (observed
~1.9M) — not an epoch and not a short cooldown, so it is deliberately not read as a
per-request delay. Its coin ids are `{symbol}-{slug}` (`btc-bitcoin`) and are unrelated
to CoinGecko's; it therefore never claims a `coingecko_id`, because a fabricated one
would make DefiLlama attach the wrong protocol's TVL. The endpoint does not publish
circulating supply, FDV, or 24h high/low, so those are recorded as `missing` rather
than reported as `0`.

## Adding a provider

1. Subclass `Provider`, declare `supports`, implement only what you claim:

   ```python
   from tickerkit.providers.base import Provider
   from tickerkit.providers import register
   from tickerkit.capabilities import Capability

   class MyProvider(Provider):
       name = "mine"
       supports = frozenset({Capability.META, Capability.AGG_MARKET})

       @classmethod
       def available(cls, config):
           if not config.extra.get("MY_KEY"):
               return False, "set MY_KEY to enable"
           return True, None

       @classmethod
       def rate_limit_rpm(cls, config):
           return 60          # optional; ccxt manages its own

       async def fetch_identity(self, asset):
           data = await self.ctx.http.get_json("https://api.example/coin")
           self.ctx.record("identity", provenance_for(
               source="mine:/coin", capability=Capability.META, url="https://api.example/coin"))
           return Identity(symbol=asset.symbol, name=data["name"])

   register(MyProvider)
   ```

2. That's it. The orchestrator discovers capabilities from the registry; no changes to
   `core.py` are needed.

Inside an adapter, `self.ctx` gives you `config`, `http` (rate-limited, retrying
`get_json`), `cache`, `limiter`, plus `throttle()`, `record()`, `note_missing()` and
`note_error()`. Always `await self.ctx.throttle()` before a request if you declare
`rate_limit_rpm`, and cite the provider's official doc URL in a comment at the top of the
module — every adapter here does.

---

## CLI

```console
tickerkit BTC                          # short digest
tickerkit BTC --json                   # full JSON snapshot
tickerkit BTC --report                 # full human-readable report
tickerkit BTC --markdown               # the report as a fenced markdown block
tickerkit BTC --metrics                # one flat JSON row (pipeline shape)
tickerkit ETH --frames ohlcv venues    # DataFrame views
tickerkit SOL --show-errors            # tolerated errors, missing data, ambiguities
tickerkit BTC --exchanges kraken --quote EUR
tickerkit BTC --history --timeframe 1d --start 2017-01-01 --end 2024-01-01
tickerkit BTC --history --report       # full history report (sources + consensus + quality)
tickerkit BTC --history --metrics      # one flat history row with completeness flags
tickerkit BTC --history --frames       # history + consensus DataFrames
tickerkit cache stats                  # disk-cache size / bounds / expiry (prune, vacuum, clear)
```

Exit code is `0` when the object was produced (a snapshot with a price, or a history with
at least one series), `1` otherwise.

---

## Development

```console
uv run ruff check src tests           # lint
uv run basedpyright                   # types, strict
uv run pytest                         # tests (no network)
uv run pytest -m live                 # opt-in live smoke test
```

Tests never touch the network: provider adapters are exercised against recorded JSON
fixtures in `tests/fixtures/` via `respx`. The only live tests are in `tests/test_live.py`,
marked `live` and deselected by default.

Examples:

| Script | Shows |
|---|---|
| `examples/print_json.py` | the full JSON snapshot |
| `examples/dataframes.py` | the pandas views, notebook-style |
| `examples/pipeline.py` | `flat_metrics` rows, CSV output, batch collection |

### Notebook

`notebook/pipeline.ipynb` is an executable walkthrough: fetch a snapshot, reshape it for
analysis, read the quality block, and see what happens when an aggregator is removed from
the registry. It hits the live free APIs, so a rate limit in its output is the library
working as intended rather than a broken cell.

The notebook is generated, so edit the cells in `tools/build_notebook.py` rather than the
`.ipynb`:

```bash
uv sync                                    # installs jupyterlab + ipykernel (dev group)
uv run python tools/install_kernel.py       # register the kernelspec Zed looks for
uv run python tools/build_notebook.py       # regenerate notebook/pipeline.ipynb
uv run python tools/validate_notebook.py    # execute every cell, fail on any error
---

`notebook/history.ipynb` is the deep-history companion: live historical lookup and the
offline tests behind it. Its first half exercises `aget_history` (deep backfill, the
coverage contract, the exchange-vs-aggregator cross-check), its second half specifies the
same contracts offline — coverage invariants, the pagination walk against a fake venue,
consensus with a rogue venue, and the real pytest files via subprocess. It is generated
the same way:

```bash
uv run python tools/build_history_notebook.py
uv run python tools/validate_notebook.py notebook/history.ipynb
```

`nix run .#notebook` validates the pipeline notebook and `nix run .#history-notebook`
the history notebook.

### Opening it in Zed

Zed discovers notebook kernels through `jupyter kernelspec list`, so a notebook with no
registered kernel just reports "no kernel". `tools/install_kernel.py` writes a spec whose
`argv` names `.venv/bin/python` **by absolute path**. The venv's own `python3` spec is not
enough: its `argv` is a bare `python`, resolved through `PATH`, so which interpreter
actually runs depends on how the editor was launched — and a cell can end up importing a
different `tickerkit`, or none, than the one you are testing.

`.zed/settings.json` pins `jupyter.kernel.name` to `tickerkit` and points the Python LSP
at the same `.venv`. Re-run `install_kernel.py` after any `uv sync` that recreates the
virtualenv.

`jupyter lab` works too if you prefer a browser, but the validator and the kernel spec are
what keep the notebook honest either way.

`validate_notebook.py` exists because a committed-but-never-run notebook rots silently as
APIs drift. It runs each cell in one shared namespace with top-level `await` allowed, the
way a kernel would.

## Documentation sources

Every adapter cites the official docs it was written against:

- CoinGecko — <https://docs.coingecko.com/reference/coins-markets>,
  [/coins/{id}](https://docs.coingecko.com/demo/reference/coins-id),
  [authentication](https://docs.coingecko.com/demo/reference/authentication),
  [errors & rate limits](https://docs.coingecko.com/docs/errors-and-rate-limits)
- ccxt — <https://docs.ccxt.com/en/latest/exchanges.html>
- DefiLlama — <https://api-docs.defillama.com/llms-free.txt>
- CoinMarketCap — <https://coinmarketcap.com/api/documentation/pro-api-reference>
- CoinPaprika — <https://coinpaprika.com/api/>, plus the `/search/`, `/tickers/{id}`,
  `/coins/{id}` and `/global` endpoints

## Publishing to PyPI

The repo ships everything needed to publish the package (a `hatchling` build, a GitHub
Actions **CI** workflow that lints + builds artefacts, and a **Publish** workflow using
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/)).

One-time setup:

1. Register the `tickerkit` project on PyPI (provisional or final).
2. In this GitHub repo: *Settings → Environments → New environment → `publishing`*,
   then configure the PyPI publisher for that environment (no token is stored).

Then releasing is just a tag:

```console
git tag v0.1.0 && git push origin v0.1.0
```

The `publish.yml` workflow verifies the tag matches `pyproject.toml`'s version, builds
`uv build`, and publishes. To preview artefacts locally at any time:

```console
uv build                # -> dist/tickerkit-0.1.0-py3-none-any.whl + sdist
```

## Licence

MIT.