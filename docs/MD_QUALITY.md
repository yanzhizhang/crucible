# Market-data quality standard

A trading day of market data may feed bars, factors or a backtest only after it passes this
standard. The checks live in `src/quarry/quality.py`, the thresholds in
`cfg/md_quality.toml` (the same file the praxis C++ live checker is meant to read), and the
runner is `research/md_quality/run.py`, which writes
`/work/crucible_data/quality/date=<d>/source=<s>/checks.parquet` and a `verdict.json`.

## Grades

| grade | meaning | downstream |
|---|---|---|
| PASS | within every threshold | used |
| WARN | measurable defect, small enough to use with care | used; the report says what |
| FAIL | defect large enough to bias results | **refused**, unless run with an explicit override that is logged |
| SKIP | specified here but not implemented yet, or its input is absent | listed in the report so the gap is visible |

A day's verdict is the worst grade among its checks. SKIP never improves or worsens it.

## Timestamps used

* `exchange_ts`: the exchange's own event time (vendor field `time`).
* `arrival_ts`: when our capture received it (vendor `spiderTs`). This is the
  **point-in-time** clock: a feature may only use what had arrived.
* both are int64 nanoseconds since the epoch, **UTC**. Local exchange time is +08:00.
* `latency = arrival_ts - exchange_ts`.

## Checks

### 1. Completeness

| check | measures | WARN | FAIL | why |
|---|---|---|---|---|
| `completeness.stream_present` | each of order / transaction / quotation / index decoded | -- | stream absent | a missing stream silently turns every dependent check into nothing |
| `completeness.minutes_missing` | continuous-session minutes (09:30-11:30, 13:00-14:57 local) without a non-empty dump file | 1 | 1 | a missing minute is missing for every symbol |
| `completeness.book_levels_truncated` | snapshots whose book side carried more than 10 levels | 1 | -- | the decoder keeps 10; more means a schema change |
| `completeness.trade_symbols_missing_frac` | share of stocks with positive end-of-day snapshot volume but no trade records | 1e-4 | 1e-3 | the snapshot volume is the exchange's own count |

### 2. Sequence integrity

Orders and trades share one sequence space per channel, so contiguity is checked on the
**merged** stream. Checking either stream alone reads a clean capture as ~50% loss.

| check | measures | WARN | FAIL |
|---|---|---|---|
| `sequence.missing_frac` | ids missing from the merged per-channel sequence / span | 1e-9 | 1e-5 |
| `sequence.duplicates` | repeated ids in the merged sequence | 1 | 100 |

Which vendor column carries the shared sequence is set per exchange in `[sequence].seq_col`.

### 3. Timestamps

| check | measures | WARN | FAIL | why |
|---|---|---|---|---|
| `timestamps.missing` | records with a zero/negative timestamp | -- | 1 | |
| `timestamps.negative_latency_frac` | share of records arriving more than 5 ms *before* their exchange time | 1e-7 | 1e-4 | the capture clock is behind the exchange: not jitter, a sync fault |
| `timestamps.latency_p99_ms` | 99th percentile latency, per stream | order/trade 500, quote 2500 | 2000 / 5000 | a slow capture shifts every point-in-time decision |
| `timestamps.drift_ms` | range of the 15-minute median latency over the day | 50 | 500 | a walking clock |
| `timestamps.resolution_ms` | coarsest unit every exchange timestamp is a multiple of | order/trade 10 | 1000 | bounds what the stream can order |

Latency quantiles come from a log-binned histogram (bins ~0.5% wide), so a full day is
processed in constant memory.

A small negative latency is tolerated (5 ms) because the capture clock and the exchange clock
are different machines; the first 20260615 profile showed snapshot latencies down to -2.4 ms.
Snapshot `time` is floored to whole seconds, so up to ~1 s of apparent quote latency is format,
not delay -- hence the looser quotation thresholds.

Event **ordering** inside a second must use the channel sequence, never timestamps.

### 4. Internal consistency

| check | measures | WARN | FAIL | why |
|---|---|---|---|---|
| `consistency.volume_mismatch_frac` | share of stocks whose last snapshot `total_volume` != sum of their trade volumes | 1e-4 | 1e-3 | the exchange's accounting against the tick stream |
| `consistency.amount_mismatch_frac` | same for amount (relative tolerance 1e-6 + 1 CNY) | 1e-4 | 1e-3 | |
| `consistency.crossed_book_frac` | continuous-session stock snapshots with bid1 >= ask1 > 0 | 1e-6 | 1e-4 | |
| `consistency.price_outside_limits` | stock trades outside the day's limit prices | -- | 1 | |
| `consistency.off_tick_prices` | stock trades not on the 0.01 grid | -- | 1 | a price-scaling bug |
| `consistency.book_vs_mbo_match` | order book rebuilt from orders+trades vs snapshot top-N | -- | -- | **SKIP** until the stage-7 replay kernel exists |

### 5. Multi-source consistency

| check | reference | status |
|---|---|---|
| `multisource.daily_totals_vs_wind` | Wind `w.wsd` daily volume/amount must match exactly | **SKIP** until the Wind terminal is logged in |
| 1-minute OHLCV vs Wind `w.wsi` | tolerance table (to be set from the first comparison) | planned |
| XTP / XTP-X quote recordings vs feitu | same checks, plus which source arrives first | planned (praxis md) |
| QX TDF (intranet) | `prototype-compare` style row match | planned |

## Measured results

See the "Results" section appended by each run below.

## Results

### Multi-source: TDX official end-of-day package (2026-09-29)

Per-stock daily volume and amount summed from our feitu trade stream vs the TDX official
end-of-day package (fetched through the `a-stock-data` skill, `research/astock/fetch.py`):

| day | stocks | volume differs | amount differs | traded per TDX, absent from capture |
|---|---:|---:|---:|---:|
| 20260615 | 5,189 | 0 | 0 | 0 |
| 20260805 | 5,199 | 0 | 0 | 0 |
| 20260921 | 5,209 | 0 | 0 | 0 |

The trade stream is complete on all three days, including 20260921, whose FAIL comes only from
missing *order*-stream minutes (10:28-11:17). Trade-derived products (bars, flow factors) are
therefore sound on that day; order-book reconstruction is not.

### Verdicts

| day | verdict | driver |
|---|---|---|
| 20260615 | WARN | open-burst capture backlog (SSE trades up to 77 s late 09:30-09:45); steady-state drift 70-130 ms |
| 20260805 | WARN | same pattern |
| 20260921 | FAIL | order stream: 4 minute files missing, merged-sequence loss 0.6-0.9% |
