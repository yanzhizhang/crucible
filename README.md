# crucible

Where the ore gets tested.

Python research layer for a Chinese A-share and futures quant system. crucible
is a **consumer**, not a producer: factor values are computed in C++ (`prism`)
and dumped as Parquet `factor_frame` files. Python never re-implements a
production factor — it evaluates, screens, models, and sizes.

Works at daily frequency and at 3-second microstructure frequency, from either
Parquet dumps or raw exchange tick streams.

## Packages

| package    | role        | what it owns                                                           |
| ---------- | ----------- | ---------------------------------------------------------------------- |
| `crucible` | shared core | polars/pandas bridge, invariant errors, determinism helpers            |
| `quarry`   | data access | raw exchange standard, DuckDB/Parquet IO, `factor_frame` + fingerprint |
| `almanac`  | alignment   | trading calendar, 3s/1min slot grids, PIT universe, masks, adjustment  |
| `forge`    | transforms  | research-only ops: winsorize, neutralize, zscore, rank, rolling        |
| `horizon`  | labels      | forward returns, vol-normalized and ternary targets                    |
| `assay`    | evaluation  | IC, quantiles, decay, turnover; Sharpe/Calmar/drawdown; overfit check  |
| `sieve`    | screening   | correlation, orthogonalization, factor pool with admission log         |
| `kiln`     | models      | purged walk-forward, LightGBM, leakage canary, noise benchmark         |
| `ballast`  | portfolio   | scores→weights, constraints, covariance, index-futures hedging         |
| `toll`     | costs       | stamp duty, commission, square-root impact, analytic deadband          |
| `ledger`   | accounting  | position accounting and the C++ parity gate                            |
| `herald`   | reporting   | self-contained HTML tearsheets                                         |

Each subpackage depends only on `crucible` plus packages **earlier in the build
order** — never later. The graph is a DAG, so any prefix can be split into its
own repository.

## Invariants

1. **Point-in-time.** Features read only data at or before `t`; labels only
   after. `merge` + `shift` is banned — `horizon` joins on an explicit slot
   index, so a gap yields null rather than a plausibly wrong row.
2. **No factor reimplementation.** `forge/` holds transforms. Factors belong to
   prism; `research/` is a stand-in producer, not production.
3. **Parity gate.** `ballast` + `toll` PnL must reconcile against the C++
   backtester. A mismatch raises `ParityError`.
4. **Determinism.** Fixed seeds; same inputs, byte-identical outputs. No
   wall-clock in any computation.
5. **Cross-sectional grouping.** All symbols at `t` are ONE sample. Random
   K-fold is unsupported; use purged walk-forward with embargo.

## `quarry.raw` — the data contract

Field names, types, scaling and enums follow the **exchange** specs (SSE 行情网关
BINARY IS120 v0.61; SZSE Binary 行情), not any vendor encoding. Vendor feeds are
adapters onto this standard.

**Venue scaling differs and is never a module constant:**

| field    | SSE            | SZSE           | if crossed          |
| -------- | -------------- | -------------- | ------------------- |
| price    | `N13(5)` → 1e5 | `N13(4)` → 1e4 | 10× price error     |
| quantity | `N15(3)` → 1e3 | `N15(2)` → 1e2 | 10× size error      |
| amount   | `N16(2)` → 1e2 | `N18(4)` → 1e4 | 100× notional error |

**Cancels arrive on different streams.** SSE cancels are _order_ records
(`ExecType='4'` 删除委托订单, vs `'0'` 新增委托订单; IS120 v0.61); SZSE cancels are
_trade_ records (`ExecType='4'` Cancelled, vs `'F'` Trade; SZSE Binary 行情
Ver1.17). `RecordType` normalises both. (`updateType=2` / `tradeType=2` are the
vendor capnp re-encoding of the same flags — see `quarry.feitu`.)

**Sequence is the replay order, not timestamp.** `(ChannelNo, ApplSeqNum)` is
contiguous within a channel. Orders and trades **share one sequence space**, so
gap detection must use `merged_sequence_gaps` — a single stream reports the
other stream's records as loss.

**Timestamps** are `exchange_ts ≤ broker_ts ≤ arrival_ts`. Features gate on
`arrival_ts`; `PIT_TIMESTAMP` names it. Gating on exchange time assumes zero
wire latency.

`from_vendor_float_price()` is the only sanctioned crossing point from vendor
`Float64` back to exchange integers, and it rounds — truncation loses a tick.

## Frames

polars-native through IO, alignment, features and labels. pandas is downstream:
`kiln`, `ballast` and `herald` convert once at their own edge for `statsmodels` /
`lightgbm` / `matplotlib`. Public functions accept either flavor and return the
flavor they were given.

## Install

```bash
uv sync --extra dev
```

On a mainland network use a local mirror — the default index times out on the
large wheels:

```bash
uv sync --extra dev --default-index https://pypi.tuna.tsinghua.edu.cn/simple
```

Optional extras (`opt` cvxpy, `dl` torch, `mining` gplearn, `agentic` rdagent)
plug into existing seams — see [docs/EXTRAS.md](docs/EXTRAS.md).

## Storage

Parquet partitioned `date=YYYYMMDD/symbol=NNNNNN/`. DuckDB registers views over
the tree rather than loading frames; hive key types are detected per dataset and
pinned to VARCHAR, so `symbol=000001` stays a string. `load_ticks` refuses an
unbounded full-day read unless passed `allow_full_day=True`; `iter_ticks`
streams in bounded batches.

## `research/` — prism stand-in

Not core, not production. Plays the producer role: emits a `factor_frame` that
the pipeline consumes through the same loader and fingerprint gate it would use
against a real C++ dump. Survivors must be ported to prism before trading.

### Daily pipeline

```bash
python research/fetch_ashare.py 80     # daily bars, stdlib only
python research/build_store.py         # -> data/store + fingerprint sidecar
python research/run_study.py           # -> data/out
```

36 factors: classical A-share, WorldQuant Alpha101, Guotai Junan Alpha191,
qlib Alpha158-style rolling features.

### Industry classification

```bash
python research/fetch_industry.py      # one dated snapshot -> data/raw/industry/
```

**Wind and CITIC are licensed and intranet-only.** Nothing external reproduces
them, so this fetches Eastmoney's board classification as a stand-in and writes
it under `scheme=em`. Measured 2026-09-08: 5567 of 5911 listed symbols across
**128 industries**; the 344 left out are PT/退/ST shells the source itself does
not classify, excluded rather than given a placeholder industry. `almanac.Classification` is
scheme-agnostic: the real `citic`/`wind` change history drops into the same
loader at levels 1..N and nothing downstream moves.

The endpoint gives **current** membership only, so a snapshot is never written
as history. Each run appends one dated file and
`almanac.classification_from_snapshots` folds the accumulated series into
effective-dated intervals — a reclassification is dated at the snapshot that
first saw it, which is an upper bound on the true date and never early.
Snapshot cadence *is* the resolution of the history. Until a real change
history is available, `Classification.at()` refuses any date before the first
snapshot rather than back-casting today's labels.

### Tick / microstructure pipeline

Decodes raw exchange tick dumps, reconstructs the limit order book by
market-by-order replay, and builds a 3-second panel.

```bash
python research/probe_feitu.py <dumps>          # verify the feed contract
python research/probe_mbo_linkage.py <o> <t>    # confirm order<->trade keys
python research/build_tick_panel.py ...         # flow panel
python research/build_mbo_panel.py ...          # order-book panel
python research/run_tick_study.py panel.parquet --mbo mbo.parquet
```

**MBO replay** (`research/mbo_book.py`) merges both tick streams, sorts by
`(channelId, seqId)`, and applies venue-specific order keys — `orderId` on SSE,
`seqId` on SZSE. Using the wrong key resolves ~0% of trade references and yields
a book that is silently wrong. Replay is per channel with a warmup prefix so
the book is populated before snapshots are taken.

**Factors** (`research/micro_factors.py`):

- _flow_ (18) — order-flow imbalance, trade-flow imbalance, cancel ratio/skew,
  trade intensity, realized vol, Roll implied spread, Kyle's lambda, Amihud,
  VPIN, momentum/reversal, order-size imbalance, market-order ratio
- _book_ (13) — microprice tilt, L1/L5 imbalance, relative spread, order-count
  imbalance, resting order size per side, book slope, consumption, microprice
  momentum/reversal, spread z-score, imbalance persistence

`micro_tilt` is not independent of `imb_l1` (`microprice − mid ≡ (spread/2)·imb_l1`);
after cross-sectional standardisation they rank identically.

## The research workflow this encodes

The package boundaries follow the standard systematic-equity research pipeline,
one stage per package:

```
hypothesis -> data -> feature -> label -> single-factor eval -> screening
    -> combination -> portfolio -> costs -> backtest -> monitor -> retire
     quarry   almanac  forge   horizon    assay        sieve
                                            kiln    ballast  toll  ledger  herald
```

Three ideas drive most of the design decisions:

**Decay vs turnover is the decisive pair.** A factor that turns over faster than
its alpha decays pays costs for nothing. This kills more candidates than any
significance test, which is why `assay` returns both from one call.

**Multiple testing is the central statistical problem.** The conventional
|t| > 2 hurdle is far too weak given how many factors get tried; Harvey–Liu–Zhu
argue for |t| > 3. `sieve`'s admission log records what each candidate was
tested against and on what sample, so the count of attempts is recoverable.

**Breadth beats strength.** The fundamental law, `IR ≈ IC × √breadth`, means a
modest IC across thousands of names beats a strong IC across dozens.

`CLAUDE.md` covers this in full — the pipeline stage by stage, the
researcher/PM split and what transfers between them, A-share specifics (T+1,
price limits, stamp duty, futures hedging), where the field is moving, and a
reading list. Written for someone learning the researcher side.

## Development

```bash
uv run pytest -q          # 245 tests
uv run ruff check .
uv run ruff format .
uv run mypy
```

Lint config is in `ruff.toml`, not `pyproject.toml`, so a subpackage keeps a
usable config if split out.
