# PM reproduction -- findings per stage and the intranet runbook

What is known about the PM's high-frequency pipeline, what we reproduced locally, and what can
only be settled on the intranet box (10.11.1.97, PM results under `/work/prod`). Live status:
the overview page (`tools/crucible_ui.sh`, http://localhost:2718).

## Stage 3 -- samplerR (per-stock order-size thresholds)

### What PM's file looks like (from the 20260430 sample)

`hstats/sod/HS300/<ZZAC|ZZCR>/20260430/samplerR.parquet`: shape (300, 17) -- `sid` (float64, the
6-digit code as a number: 000001 -> 1.0) plus `{as, ps, ab, pb} x {q50, q90, q95, q98}`.
First sids 1, 2, 63, 100, 157 = CSI 300 constituents in code order. Row sid=1 (000001):

| | q50 | q90 | q95 | q98 |
|---|---:|---:|---:|---:|
| as | 799.5 | 6,981 | 15,377 | 33,638 |
| ps | 648 | 5,194 | ... | ... |
| ab | ... | 4,565 | 8,432.5 | 18,218 |
| pb | 557 | 4,389 | 8,684 | 19,490.5 |

### Hypothesis and evidence

`a`/`p` = active (aggressor) / passive (resting) side of each trade, `s`/`b` = sell / buy;
the size measured is **the order's total filled shares for the day** (fills summed per order id).

- Per-trade volume is ruled out by construction: `as` and `pb` would be the *same* trades
  (every sell-initiated trade has one active seller and one passive buyer) and so equal; PM's
  `as` and `pb` clearly differ (799.5 vs 557 at q50).
- Magnitudes fit per-order totals: 000001 active-sell on 20260615 (different day) is
  700 / 4,900 / 10,000 / 24,100 per order vs 500 / 3,000 / 5,300 / 10,664 per trade.
- Our universe, taken from the CSI official file dated 2026-08-31, reproduces PM's first sids.

**Open:** PM values are frequently half-integers (799.5, 8,432.5, 19,490.5); ours never are.
Candidates: odd-lot remainders in PM's order sums, a different quantile method (midpoint /
averaging), or a multi-day window. Needs a same-day comparison.

### What we compute (`research/pm_factors/sampler_r.py`)

Three variants per trade day, PM column layout, into
`/work/crucible_data/store/pm_factors/sampler_r/variant=<v>/date=<D>/samplerR.parquet`:

| variant | size measured | trades used |
|---|---|---|
| `order_all` | per-order filled total (main hypothesis) | every trade with a direction |
| `order_cont` | per-order filled total | continuous session only (auction trades carry a vendor-assigned direction but have no real aggressor) |
| `trade_all` | per-trade volume (control; should *not* match) | every trade with a direction |

`D` is the *trade* day: compare PM `sod/<next trading day>` (or `hstats/<D>`) against it.
Quantiles: DuckDB `quantile_cont` (linear interpolation = numpy/pandas default). Each variant
also carries `<role>.n` (sample sizes) for diagnosing the half-integer question.
Runtime 6-21 s per day and variant, 3-5 GB peak (DuckDB, spills to `/work/crucible_data/duckdb_tmp`).

### Result on 97 (20260429, 2026-10-16..20): a random sample, not a deterministic statistic

Data: TickRepository archive (172.21.2.23, `/mnt/nas/pica_data_hdd_1/TickRepository/hdfs_data/`),
trades for 0416-0429 and orders for 0428-0429, decoded with `decode_feitu_day.py`. PM's
`sod/20260430` is identical to `HS300/<fam>/20260429` -- **sod/D is built from day D-1** (proven).

Two layouts: **ZZCR** one row per stock, `{as, ps, ab, pb} x {q50, q90, q95, q98}`; **ZZAC** 48 rows
per stock (`rec_epoch` = every five minutes of 0429, 09:35 .. 15:00), `{as, bs} x {q5 .. q98}`.

What the diagnostics showed, in order:

1. Per-order totals are the right quantity (median PM/ours 1.0-1.08, rank corr 0.91-0.97);
   per-trade volume is 1.4-2.1x off (`diag_sampler_r.py`). But never exact.
2. No window / quantile method / rounding reproduces it (`sampler_r_grid.py`: best 0.5 % exact);
   longer pools fit better (w10 0.31 at 1 day -> 0.46 at 10 days).
3. ZZAC does not follow same-day accumulation and ZZCR is none of ZZAC's epochs
   (`diag_sampler_ac.py`); its five-minute steps are not an EWMA of slot quantiles (R^2 0.01-0.04)
   and often move **against** the new slot (`diag_sampler_update.py`). Only randomness does that.
4. Treating PM's value as the quantile of a random K-sample of orders and scoring its percentile
   in simulated K-samples (`sampler_r_mc.py`): consistent, with a one-sided bias using filled
   totals (mean PIT 0.57-0.69), largely removed by **submitted order size** (SZSE: the order's
   own volume, key `seq_id`; SSE: aggressor fills + published rest, key `order_id`).

| target | best fit | ks (0.02 = perfect here) | mean PIT | in 90 % band |
| --- | --- | --- | --- | --- |
| ZZCR close | submitted, today, continuous session, **K = 100** | 0.038 | 0.495 | 0.875 |
| ZZAC 11:30 | submitted, today (from the open), **K = 40** | 0.054 | 0.533 | 0.889 |
| ZZAC 15:00 | submitted, today, K = 70 | 0.139 | 0.608 | 0.795 |

Reading: samplerR is a **reservoir sample** of submitted order sizes ("Algorithm R" is the textbook
name) -- about 100 orders for ZZCR, about 40 for ZZAC -- and the thresholds are its quantiles.
Exact values are therefore not reproducible (they depend on PM's random state and stream order);
the procedure is. Open: ZZAC at the close is still biased upwards with a uniform same-day pool --
consistent with a sample that favours recent orders (a decaying / biased reservoir), to be tested.
For Stage 4, PM's own samplerR files are the size thresholds, so factor formulas can be matched
without this noise.

## Intranet runbook (10.11.1.97)

**Short version -- one command** (`research/intranet_run.py`): check first, then run everything
whose inputs exist, then bring back one tarball of reports and logs (no PM data in it):

```
git clone -b feat/pm-repro https://github.com/yanzhizhang/crucible.git && cd crucible
conda env create -f environment.yml && conda activate crucible
python research/intranet_run.py --preflight-only
python research/intranet_run.py --feitu-root '<dump dir with {date}>'   # omit if raw is decoded
# bring back: data/intranet_run/<timestamp>.tar.gz
```

It runs, per step (each its own log, failures do not stop the rest): inventory, the model probe
(stage 5), decoding + quality gate for missing days, catalog, PM exports, bars, samplerR,
candidates for all 14 families, and the comparisons (samplerR per variant, bars vs `1min_src`
root and p1..p10, `rank_candidates.py` per family). The numbered steps below are the same
things by hand.

1. Environment: `conda env create -f environment.yml` (generated from pyproject), clone crucible.
2. Inventory PM's tree (schema baseline for every comparison):
   `python research/pm_inventory.py --root /work/prod --out data/pm_manifest`
3. Raw market data for 20260429/30: the manifest tells whether `mp/` is decodable; otherwise
   use the feitu v3 dumps for those days with `research/decode_feitu_day.py`, then
   `research/md_quality/run.py` and `research/catalog.py` (paths in both are `/work/crucible_data`).
4. Stage 3: `python research/pm_factors/sampler_r.py --dates 20260429` then, per variant:
   `python research/compare_pm.py --pm /work/prod/hstats/sod/HS300/ZZAC/20260430/samplerR.parquet --ours /work/crucible_data/store/pm_factors/sampler_r/variant=order_all/date=20260429/samplerR.parquet --keys sid --atol 0.5 --out data/reports/stage3/order_all.parquet`
   Also compare `ZZCR` (same layout; are the two families' samplerR identical?) and the
   `hstats/HS300/<fam>/20260429` copies.
5. Stage 2: `research/build_bars_from_l2.py --date 20260429`, then `compare_pm.py` against
   `1min_src/20260430.nc` and every `p1..p10` pass.
6. Stage 4 (candidate formulas for 14 families, see the section below):
   1. Export each family's samplerS and **read the printout**: which dates the `D` dimension
      holds, what `I` looks like (positions 0..n-1 or times), what the value variable is called.
      ```
      for F in ZZUG ZZDS ZZQI ZZVC; do
        python research/compare_pm.py --pm /work/prod/hstats/sod/HS300/$F/20260430/samplerS.nc \
          --export data/pm_export/${F}_sod_20260430.parquet
      done
      ```
   2. Every date in `D` needs its raw data (step 3) and, for ZZVC/ZZUG, the **previous day's**
      samplerR (the size tiers use it): `python research/pm_factors/sampler_r.py --dates <D-1>,<D>`.
   3. `python research/pm_factors/candidates.py --dates <the D dates>` (all four families,
      ~5 min per day, 6-7 GB peak).
   4. Per family:
      `python research/pm_factors/rank_candidates.py --family ZZUG --pm data/pm_export/ZZUG_sod_20260430.parquet --out data/reports/stage4/ZZUG.parquet`
      If the PM file has no date dimension, add `--dates <our day>`. If a dimension has an
      unexpected name, pass `--s/--d/--i/--v`.
   5. Bring back the four `data/reports/stage4/*.parquet` and the printed verdicts. What to look
      for: a family-verdict row with `exact_scaled` near 1.0 = formula found (and `ratio` gives the
      unit); high `within` with low `exact` = right quantity, PM normalises it; nothing above
      ~0.5 = none of the candidates, the column goes to the question list.

## Stage 2 -- 1-minute bars

Built for 20260615 and 20260805 (20260921 refused by the quality gate). Validated internally:
end-of-day volume identity exact for every stock, amount relative error 2e-15, close = last
snapshot price; externally: daily totals equal the TDX official package for every stock.
Comparison with Wind `w.wsi` waits for the Wind terminal; with PM `1min_src` for step 5 above.

### PM raw market data (`/data/prod/mp/20260430`) -- decoded

The only raw day on 97 is the PM's own `.mp` dump: `{SH,SZ}_{HS300,ZZ500,ZZ1000}_{order,trade,
snap,index}.mp`, each a zstd-compressed msgpack stream of arrays (timestamps float epoch seconds
UTC: position 1 = local arrival, 2 = exchange time). `research/decode_pm_mp.py` writes them into
the feitu raw layout (`kind=order|transaction|quotation|index/date=D/`, manifest
`source=pm_mp`), so every builder reads them unchanged; the field mapping is in its docstring.
`--check` confirmed the snapshot ask-volume order and the trade / order categories on 20260430
(263M rows, SZ cancels 17.36M). The quality gate FAILs this day by design (constituents only):
build bars with `--allow-fail`.

### Against PM `1min_src` (20260430, `research/compare_bars_pm.py`)

`1min_src` is `x(S=1800, D, I=257, V=10)`: labels 09:15-11:30 and 13:00-15:00, V = volumeTotal,
dollarVolumeTotal, tradesTotal, lastTradePrice, askPrice, askSize, bidPrice, bidSize, close,
midPriceLastValid; the index file has S = 6 (10000000 + code) and V = cumVol, cumTo,
lastTradePrice. The script builds each field several ways (clock exchange/local, minute edge
inclusive/exclusive, whole day / continuous session only, book vs trades) and scores every
candidate per cell.

All rules read snapshots by **exchange time**, a snapshot exactly on the label belonging to that
minute, with PM's session closes: the 11:30 label reads the last snapshot at or before 13:00:00 (on
SZSE the one stamped 13:00:00, which already holds the first afternoon trades) and 15:00 reads the
day's last snapshot (SSE publishes it after the close auction). **Every field of the stock and
the index cube is reproduced** -- score on 20260430 = exact share
where both sides have a value / share of cells where both are empty or both filled.

| field | rule (candidate) | exact / nan_agree |
| --- | --- | --- |
| askPrice / askSize / bidPrice / bidSize / lastTradePrice | last snapshot **inside** the minute, no carry-forward -- a minute without a snapshot is empty (SZSE pre-open, a quiet stock) (`exch/incl_sess/book_nocarry`) | 1.0000 / 1.0000 |
| midPriceLastValid | same snapshot; both sides -> mid, one side (limit up / down) -> that side, neither (close auction, 0 / 0) -> 0 (`mid_side_nocarry`) | 1.0000 / 1.0000 |
| volumeTotal / dollarVolumeTotal / tradesTotal | snapshot totals minus the opening-auction trades (before 09:30:00); empty through 09:30, 13:00 = 11:30, 14:58-15:00 held at 14:57 (close auction not counted) (`snap_ex_auction_day`) | 1.0000 / 1.0000 |
| close | snapshot price of the last minute the day volume grew, carried forward, empty through 09:30; a stock that never trades (suspended: 601718, 601003, 600958, 300352 on 20260430) takes the snapshot price, 0 (`snaplast_ffill_px`) | 1.0000 / 1.0000 |
| index cumVol / cumTo | index cumulative minus its value at 09:30, same whole-day shape as the stock totals (`exch/incl_sess/index_ex_open_day`) | 1.0000 / 1.0000 |
| index lastTradePrice | last index record inside the minute, no carry-forward (`exch/incl_sess/index_nocarry`) | 1.0000 / 1.0000 |

How the totals were found: trades never explain them -- no constant cutoff delay gets past ~60 %
and 12 % of PM values are no prefix sum of continuous trades (`--implied-cut`). Snapshots do:
`PM - total_volume(last snapshot <= label)` is one constant per stock for all 1800 stocks and all
three totals, and that constant is the opening-auction trade total (trades before 09:30:00
exchange time) for 100 % of stocks on both venues (`--snap-base`). The earlier base -- the
snapshot at 09:30:00 -- already holds the first continuous trades on 99.6 % of SZSE and 17 % of
SSE stocks, which is where the 59 % came from. Candidate `exch/incl/snap_ex_auction`: 99.72 % on all three totals,
and every remaining miss is the 11:30 label (SZ 863 stocks, SH 325; every other hour 100 %), always
a few trades short. `--snap-match 11:30` settled it: PM's 11:30 value equals its own 13:00 value
for 100 % of stocks, i.e. the last snapshot at or before 13:00:00 exchange time -- on SZSE the
snapshot stamped exactly 13:00:00, which already holds the first afternoon trades (the morning
close alone fixed only ~170 of the ~1200 misses). Candidate `exch/incl/snap_ex_auction_1300`:
**100 %** on all three totals on 97; 14:57 needs no such rule.

## Stage 4 -- candidate formulas (14 families)

The formulas are unknown; `samplerS` gives only column names and slot counts. So each column
gets several candidates (`research/pm_factors/candidates.py`, the list is in its docstring) and
`research/pm_factors/rank_candidates.py` scores all of them against the PM cube: exact match,
exact up to a unit (`ratio` = 0.01 means the PM counts lots), rank correlation, and per-stock
time-series correlation (survives any per-stock normalisation). Pair columns are tried as
imbalance, difference and share. For positional `I`, every slot grid of that length is tried
(236 = 09:31-14:57 minus the first or the last minute), each shifted by one minute either way.

Self-test (a known candidate planted as a fake PM file, codes as `000001.SZ`, dates as datetimes,
volumes in lots, one column z-scored per stock): the tool recovers the planted candidate, the
unit, and the slot grid (up to the equivalent one-minute-shifted grid), and still points the
z-scored column at the right formula (`within` = 1.0).

Built locally for 20260615 and 20260805 (300 stocks). ZZUG / ZZDS / ZZQI / ZZVC in
`candidates.py`, the other ten (HL MS GW XC TS QO WA AL SQ CR) in `candidates_more.py` -- each
module's docstring lists the candidates. 3-6 candidates per family; ZZAC has only a samplerR
(stage 3). ZZCR's `vr` needs the previous session's bars (null locally).
Locally the size tiers use the same day's samplerR (no consecutive day here); on 97 they use D-1.

Data facts found on the way:

* per-level order counts (`bid_no` / `ask_no`) are **empty on both venues** in the feitu dumps;
  SSE snapshots carry whole-book order counts (`total_buy_no` / `total_sell_no`), SZSE none;
* SZSE's 14:57:00 snapshot is already in close-auction mode (resting totals read 0) -- snapshot
  families stop just before it;
* the resting book rebuilt from the order and trade streams (`mboclose`) matches SZSE's own
  totals (median ratio 0.9995, 92-99 % of stocks within 1 %); on SSE it reads 3-6 % high
  (unexplained; SSE snapshots publish the totals directly, so that candidate matters only on
  SZSE).

## Stage 4c -- samplerS rolling medians (ZZQO, ZZAL, ZZMS, ZZDS ...), 2026-11

Each samplerS cell is the **median over the last 22 trading days, day D included,** of a per-(stock, 5-minute bar)
daily value. N is even, so the median is the mean of the two middle day values: `window_probe.py`'s rank line puts PM
at position 0.50 between our 11th and 12th values for 95-97 % of ZZQO (SZSE) and ZZAL cells. Other windows (20, 21, 23, D
excluded) widen ZZMS's error from +-0.5 % to +-3-5 %. Bars are right-closed and labelled by their end (09:35 .. 11:30,
13:05 .. 15:00). SSE ticks are timed by spider_ts - 25 ms, SZSE ticks by exchange time, and the opening auction is
left out of the trade flows. Columns live in `basics.py` (`--convention pm`); `window_probe.py` scores them.

| column | best definition | vs PM |
| --- | --- | --- |
| ZZDS h / l / c | every fill up to 14:57 exchange time; c = last fill by (time, seq_id) | exact |
| ZZXC q / n | the bar's continuous-session trade volume / amount (`vol`, `amt`) | exact 0.97 over 16 PM days (both exchanges) |
| ZZTS q | the bar's buy-initiated amount by the BS flag (`b_amt`) | exact 0.965 |
| ZZQO q | new-order volume from 09:30 (`ordvolrc_sum`): SZSE net of the order's own aggressive fills, SSE as published (the SSE feed already publishes the remainder) | SZ p50 -0.0002, exact 0.37; SH p50 -0.7 %, one-sided |
| ZZAL an / bn | resting amount (price x volume summed over our replayed book) at the last snapshot strictly before the bar end (`book.aamt_q1`) | SZ +-0.06 %, exact 0.22; SH exact 0.07 |
| ZZMS q / n | day-cumulative volume / amount, opening auction included (`cum_vol_oa`, `cvol_x13`) | p50 0, +-0.5 %, exact 0 |

What is settled and what is not:

* **PM's `.mp` is the feitu stream itself** (`mp_diff.py`, 20260430): every SSE add / delete and SZSE order is the same,
  and PM's local stamp equals feitu's spider_ts within float rounding. On SSE, "the larger order number is the
  aggressor" equals the BS flag 100 %. PM had 140 fewer SSE trades (0.0002 % of volume) and 9 % more SSE snapshots
  (2 % SZSE). Remaining gaps are PM's processing rules, not its data (beyond those snapshots).
* ZZQO SH: PM's day values sit between "as published" (PM below our lower middle value on many cells) and "net of
  all own fills" (above our upper middle value). A fixed fraction (0.4 centres p50) only spreads the error to +-1 %,
  so whatever PM removes is not proportional to the order's own fills. Open.
* ZZMS: PM lies between our two middle day values for 85-89 % of cells, but anywhere from 0.2 to 0.9 across the gap,
  and never equals one of our day values. So the statistic is right, but PM's daily value differs from ours by a
  roughly symmetric +-0.5 %. Neither the cut (snapshot at / before the bar end, by exchange or receive time, PM's 1min_src
  13:00 / 14:57 rules) nor the auctions change that, and a running sum of per-bar medians is about 10 % off. Treated as
  reproduced to +-0.5 %.
* Flows from snapshot differences (PM's 1min_src way, `snap_*`) score 0.05-0.08 on ZZXC / ZZTS against 0.97 for the
  trade flows: samplerS flows come from the trade stream, not the snapshots.
* ZZAL SH: our replay equals the feed's own resting totals at that snapshot for only 40-47 % of cells (`book_check.py`,
  within 0.1 % for 80-89 %), and PM sees snapshots feitu does not store. Open.

## PM's strategy (execution) -- context for translating it later

Capital: 10M CNY net = 9M base position in CSI 300 constituents + 1M cash for the T0 buys (user, 2026-11-07). The
base is a target, rolled daily: sht-demo `shtcommon/py/ced/base_position.py` spreads x (the base's value at T's close,
the day's profit rolled in) over the index's T-close weights into `<T+1>/base_position_{v1|v2}_<product>.csv`
(pos / pos_adj); the actual holding is only moved toward that target each day, not set to it. PM sends one prediction per CSI 300 snapshot; what chrono itself logs (`trk`
lines) is read by sht-demo `chrono/tools/pnl_report.py` (fills, fees, PnL) and ends there. The untimestamped lines in
224's `/tmp/zyq/sht-logs/*.log` are PM's strategy deciding whether to trade, not its factor model: `qlive/catchRun.cpp`
(srcEpoch / recEpoch / sid) and `sol/alg.cpp` (fa = the prediction again; pos, pp_, executp, asq, lthr, hs, liid,
winsdQty, asf; emsMktVol, emsOurVol). `research/pm_factors/pm_signals.py` stores them in `pm_signals/state/date=D.parquet`.
Seen so far: hs ~ half a tick over the price; winsdQty = emsMktVol x 0.17647 (3/17) exactly; asf = 0.8; lthr < 0 and
grows with hs.

## Stage 5 -- PM's tk5m LightGBM models (`/data/prod/models/tk5m`, read on 97 2026-11-09)

The full parameter list, machine-readable: `research/pm_factors/pm_lgb_params.json`.

* All 17 month folders, 202509 .. 202612 plus `202601.updated` (dated 2026-09-04), hold
  `lgb/livev1_fit{1,2,3}/model_s{1,2,3,4}.txt` as regular files: 204 files, ~13 MB each. (An earlier reading of
  "three months only" was wrong.) By their column ranges and split counts (`pm_lgb_meta.py`) there are three
  generations: A = 202509-202604, B = `202601.updated` + 202605-202608, C = 202609-202612. November trades on C.
  `/data/prod/models.zip` and 2.23's `prod.tar.gz` of 2026-11-09 hold the same files (md5 84da910c, 27 entries).
  That tarball ends 2026-11-03 and adds nothing that 97's `/data/prod` lacks.
* The header is plain LightGBM v3: `objective=regression`, `num_class=1`, `max_feature_idx=208` (209 columns),
  `feature_names=Column_0 ..` (generic, so the column names come from `lgb_fn_v1_209_cte.csv`), and
  `feature_infos` = each column's training [min:max].
* Every file has 2000 trees, although `num_iterations` is 3000 and `early_stopping_round` is 0, so the file
  was probably saved at a chosen iteration. `metric` is `custom`.
* With the seed lines excluded, the parameters block is identical in all 36 files (hash c61ef6fa).
  * Seeds by fit, the same in every month and every s: fit1 and fit3 use 18467 / 26500 / 15724 / 6334 / 41 / 19169
    (bagging / feature_fraction / extra / drop / data_random / objective); fit2 uses 400 / 30056 / 12879 / 17869 / 175 / 16083.
  * fit1 and fit3 also share their `feature_infos` (the same training rows) but are different files (all 36 file md5s
    differ): with `deterministic` 0 and 70 threads, a rerun with the same seeds gives different trees.
  * s1..s4 have different `feature_infos` in every month, so each was trained on its own rows (a time-of-day segment
    or a horizon), not a seed ensemble.
  * The empty-valued parameters, the per-fit seeds and the md5s are in `pm_lgb_params.json`. Every file's 209
    training ranges and split counts are written by `pm_lgb_meta.py` to `data/reports/pm_lgb_meta.csv` on 97.
* Parameters: `gbdt`, `learning_rate` 0.03, `num_leaves` 64, `max_depth` 6, `min_data_in_leaf` 20,
  `bagging_fraction` 0.9 (`bagging_freq` 2), `feature_fraction` 0.7, `lambda_l1` 0.5, `lambda_l2` 1,
  **`max_bin` 128**, `min_data_in_bin` 3, `bin_construct_sample_cnt` 200000, `use_missing` 1,
  `zero_as_missing` 0, `boost_from_average` 1.
  * `max_bin` 128: a column has at most ~127 split thresholds, roughly a 1/128 quantile grid of its training
    values. This is the grid the fingerprint's `q_*` estimates (the probe's docstring said ~255).
  * `use_missing` with `zero_as_missing` 0: a NaN takes each split's learned default branch and 0 is a real
    value, so `ic_score`'s "column missing" baseline is the model's own missing path.

## PM's runS::initialize() (C++ screenshot from Zhang Yiqing, 2026-11-09)

`runS` is PM's strategy plugin inside chrono: 224's chrono config (`/tmp/o/executor/<D>/run_s.json5`, copy on
97 at `/work/crucible_data/chrono_cfg/20261110/`) loads `libcatchRun.so` as `strategy_type "runS"` (`strategy_id
runS_a`, capital 10M CNY, base position `base_position_v1_hs300.csv`, one XTP account). Its `params_path` is
`.../zhangyiqing/workspace/chrono/release/chrono_runtime/cfg/sim_example/config2_HS300.json` (not readable by
our account). Chrono's risk rules there: trading window 930-1130 / 1300-1455, price cage 2 % each way,
400 orders/s; fees 1.5 bp commission, 5 bp stamp on sells, 0.641 bp transfer.
`initialize()` logs "waiting for attach", reserves `time_slots` of 6000 x 240 (~ all
A-shares x 240 one-minute slots; reserve only), then sleeps 2 s. It reads that JSON config
(`params_path()`) with keys including `date` and `NumLevels`, hands the whole config to `qTrader(dateS, cfg)`,
and sizes the book message `FBL` to `NumLevels` levels; `resize(n, 10)` fills with the value 10. The level count
is not shown, probably 10 for L2. Execution-side code, so its value for the factors is uncertain. The one lead
is that PM's book is likely 10 levels deep, while our book candidates (`lit_factors`) stop at 5.

## Stage 6 -- ic_score rounds (score candidates through PM's own models)

`research/pm_factors/ic_score.py` fills the 209 columns with candidate formulas and keeps whichever raise the
per-mark Spearman between our model's prediction and PM's own prediction (`pm_signals`). Days 20261104-06,
240 random marks. `name_map.py` prints the assignment as a family / stem / leaf tree.

| round | change | IC | columns |
|---|---|---|---|
| R5-R7 | scan, residual both ways | 0.5945 -> 0.5993 | |
| R8 | `--drop-gap 1` (stop dropping off-scale picks between rounds) | 0.6242 | |
| R12 | literature + order-flow candidates (`lit_factors`, `tick_flow`) | 0.6351 | 37 |
| R13 | ZZQF/WJ/DW/OP/RI/KP/HI/HO families vs the new candidates | 0.6629 | 75 |
| R14 | `cal:` (quantile-map onto PM's training CDF) on off-scale families | 0.6641 | 75 |
| R15 | top-60-gain columns, with `cal:` | 0.6717 | 81 (49.9 % of gain) |
| R16 | unmapped families + 10-level book candidates | 0.6791 | 100 (54.7 % of gain) |
| R17 | book imbalance and OFI at every depth (levels 1..k and level k alone, k = 1..10; `ebb4df5`) | 0.6806 | 102 |
| R18 | scoring only: R17's formulas on DataYesSharesL2 (`CRUCIBLE_SOURCE=dysl2`, what 224 runs on) instead of feitu | 0.6805 | 102 |
| R19 | dysl2, scan of the 107 unmapped columns (each tries every candidate, 10-level book included) | 0.6900 | 144 (61.8 % of gain) |

R17-R19 use 660 marks (t 172-173). R18 shows the two sources score the same (0.6806 feitu vs 0.6805 dysl2), so
rounds from R19 on run on dysl2. The out-of-sample check on 20261109 was skipped in R18 and R19 (no state file).

R19 per family (mapped / columns, gain): ZZRO 5/6 12.2 %, ZZQF 19/21 6.1 %, ZZZL 6/9 5.9 %, ZZWA 8/11 5.4 %,
ZZGH 1/2 4.6 %, ZZEU 2/2 3.9 %, ZZPE 4/5 3.8 %, ZZAL 5/8 2.9 %, ZZOP 3/4 2.8 %, ZZSF 1/2 2.5 %, ZZEL 0/2 2.4 %,
ZZAG 1/6 2.3 %, ZZEG 0/2 2.1 %, ZZSA 2/4 2.1 %, ZZXU 2/2 2.0 %. Largest single gains in R19's scan (t_gain): `zzie4.c`
<- passive order net amount at or inside the best quote over 60 s (12.9), `zzlk` <- amount share of orders over
200k CNY over 120 s (8.0), `zzlk2` <- `lpek50` (7.8), `zzyj4.a` <- `simba20` (7.2), `zzxt3` <- CED's ZZAL `an`
(6.8), `zzfe2` <- trade-sign autocorrelation over 60 s (6.4).

* An assignment is a stand-in, not a recovered formula: the IC rewards any candidate that carries similar
  information (`zzro2.a` once picked a spread). Only stage 1-4c results are cell-by-cell matches against PM's own
  files; candidates also pass the fingerprint gate first (>= 95 % inside PM's training range, close quantiles).
* Sibling stems share formulas (ZZQF: the `q.c2` / `c3` / `c4` leaves match `-return_open_to_1000_log`, and the
  `q.c1` leaves match bid depth or small-order net buying), so a stem looks like one template with parameter
  variants.
* ZZRO is relative to the market (`dm:` returns, t 10-16). At R16 its biggest leaf, `zzro2.a` (5.3 % of gain), was
  unmapped; R19 maps 5 of its 6 columns.
* At R16 no column was mapped in ZZEL, ZZAG, ZZEG, ZZSA, ZZBT, ZZPP or ZZQO (~14 % of gain together). After R19,
  ZZEL and ZZEG still have none, ZZAG has 1 of 6 and ZZSA 2 of 4 (ZZBT / ZZPP / ZZQO are not in R19's top-15 line).

## Stage 4b -- PM's day-end factor state (`/data/prod/h/HS300/p1_intrading_dcp/<D>.jz`)

53 days (2023-01-16 .. 2023-02-28, 2026-04-29, 2026-09-01 .. 2026-10-13), plus `sod/` copies (sod/D =
day D-1). Format: gzip JSON `{sid: {key: float}}`, 300 stocks x 7532 keys, same key set for every
stock. It is the factor engines' state at the close, not the model features: price lists,
intraday EMAs (`<p>.sum / .t / .wgt`, value = sum / wgt), histograms (`<p>.c.<bin>`)
and a few scalars. Families present include ZZGH (model rank 4-5), ZZBA, ZZXU, ZZPK, ZZUP, ZZEG,
ZZAB2, ZZEB, ZZBD, ZZQJ, ZZVM, ZZXO, ZZUI, ZZHJ, ZZFB, ZZJB, CDB, CDC; ZZRO / ZZEU are not.

**PM's live feed loses SSE trades** (user-confirmed 2026-10-22; our feitu archive is complete, SZSE
is complete on both sides). Trade-volume keys therefore match exactly on SZSE and 32-92 % on SSE,
varying by day; SZSE (and 2026-04-29, which looks like an offline replay of the same data -- its
slow clock equals our local arrival to the millisecond) are the reference.

Confirmed on 9 days (0429, 0922-0930, 1008, 1009; `research/pm_factors/dcp_multiday.py`, rebuilt by
`research/pm_factors/dcp_build.py`):

| key | rule | holds |
| --- | --- | --- |
| `zzpk.high/low/open`, `zzvx.high/low`, `zzxu.high/low` | the day's high / low / open | 1.0 every day |
| `zzmh.hm` | highest snapshot last price over 09:30-14:57 exchange time (not the trade high: matches it on only ~50 %) | 1.0 SZSE, 0.995-1.0 SSE on live days (0429: 0.95-0.98) |
| `zzxu.last`, `zzxo.last`, `cdc.lp` | the day's last price | ~1.0 |
| `zzqj.v/tpv`, `zzvm.c.v/tpv` | volume / amount of all trades from 09:30:00 exchange time, close auction included | SZSE 1.0 (0929 0.964); SSE short by PM's lost trades |
| `zzxo.r.0..29`, `zzui.r.0..9` | last snapshot **received** (local arrival) at or before 14:57:00 - (n-1-k) x 10 s | 1.0 both exchanges on live days (0429: 0.97); by exchange time: SSE ~1.0, SZSE 0.94 |
| fast clock (`zzba.p.t`, `zzeb.*.t`, `zzpk.p.t`, `zzup.*.t`, `zzxu.*.t`, ...) | local arrival, trading-clock seconds (09:30 = 0, lunch removed), of the last snapshot received before PM's process stopped (15:02:28-15:02:30 local) | 1.0 within 10 ms; `dcp_build --stop auto` takes the stop from the latest `zzba.p.t` (0429 SSE: 129 s off -- its SSE snapshot arrival times differ from PM's) |
| slow clock (`zzbd.*EmaT.t`, `zzqj.r.t`, `zztx.r.t`, `zzvm.c.re.t`) | local arrival of the stock's last trade | 0429: exact; live days PM is 3-7 ms earlier, earlier even than our first close-auction trade: PM's trade feed is faster than feitu's capture, so not reproducible to the ms |
| `zzgh.nb` | same-day only (fitting nb[D] = a imb[D] + (1-a) nb[D-1] gives a = 0.993-1.006); closest: continuous-session amount imbalance | median rel. error 0.67 % on SZSE -- not exact yet |

The clocks are "last updated" stamps, not inputs. Histograms grow 0-46 counts per day.

**The 26 weighted averages are intraday EMAs, not multi-day ones.** `research/pm_factors/diag_ema.py`
fitted each day's `wgt` as `a wgt_{D-1} + c w_D` on 7 consecutive day pairs (SZSE only, trimmed):
nothing fits, `a` is ~0 or negative and moves 0.36-0.99 between days. An earlier reading of
"weights keep 0.88-0.99 of their value day to day" was an EMA saturating to a similar weight every
day. 24 of the 26 run on a clock (20 on the snapshot clock -- `zzbd.pEmaS` --, 4 on the trade clock
-- `zzbd.pEmaT`); `zzui.r` (a ~ 1.0) and `zzhj.p` do carry the previous day.
`research/pm_factors/diag_iema.py` fits each as `wgt = sum_i w_i exp(-(T - t_i) / tau)` over that
day's snapshots or trades, ranking (updates, clock, weight) by how constant tau is across SZSE
stocks, then matches `sum / wgt` against candidate values. Result (3 days): weight 1 per update and
`tau ln 2` a round half-life on the trading clock -- 10 s `zzmh.r`; 30 s `zzvx.*`, `zzcb.r`, `zzqj.r`
(trades); 100 s `zzeb.p/r`, `zzbd.pEmaS`, `zzbd.pEmaT` / `zztx.r` (trades); 200 s `zzbd.rEmaS`,
`zzbd.rEmaT` (trades); 300 s `zzpk.p`, `zzxu.p/a`, `zzab2.r`, `zzup.pf`; 600 s `zzup.ps`. Values exact:
price (`zzpk.p`, `zzab2.r`, `zzup.pf`, `zzxu.p`), log price (`zzbd.pEmaS/T`), mid (`zzeb.p`).
`research/pm_factors/dcp_ema.py` scans round half-lives x time source x update set x weight and a
wide value list. Solved so far (weight 1 per update received, local arrival on the trading clock;
weights within 1e-4 for every snapshot EMA, 79-90 % of stocks for trade EMAs):

| prefix | updates | half-life | value |
| --- | --- | --- | --- |
| `zzpk.p`, `zzab2.r`, `zzup.pf`, `zzxu.p` | snapshots | 300 s | price |
| `zzup.ps` | snapshots | 600 s | price |
| `zzxu.a` | snapshots | 300 s | day range in price units (high - low) |
| `zzvx.trEmaD/O`, `zzvx.trDEmaD/O` | snapshots | 30 s | day range in price units (D/O variants equal at the close) |
| `zzvx.rEmaD/O` | snapshots | 30 s | 1 (constant) |
| `zzbd.pEmaS` / `zzbd.pEmaT` | snapshots / trades | 100 s | log price |
| `zzeb.p` | snapshots | 100 s | mid |
| `zzbd.rEmaS` / `zzbd.rEmaT` | snapshots / trades | 200 s | log return against the previous update |
| `zzqj.r` | trades | 30 s | the day's VWAP so far (PM's value / its own `zzqj.tpv / zzqj.v` = 1.00001 median, 89-99 % within 1e-4 on both exchanges; the per-trade running VWAP is being confirmed) |

Open: `zzba.p/r` (price / range, ~3 h, but PM's range is 1.09x ours and no weight fits -- multi-day
carry?), `zzcb.r` (ranks against the range at -0.92), `zzmh.r` (~4e-4, a monotone function of the distance below
the running snapshot high: rank 0.999999), `zzeb.r` (~0.15 for all stocks), `zztx.r` (mostly negative). Still open as well: the exact `nb` definition,
the EMAs' tau / weight / value, the histograms' bins.

**Per-slot profiles (`<p>.c.<bin>`)** -- intraday time grids of the 14220 s continuous session
(`research/pm_factors/diag_profile.py`, `diag_slots.py`):

| prefix | grid | content | holds |
| --- | --- | --- | --- |
| `zzfb` | 711 x 20 s | the snapshot's own `total_amount` (open auction included) of the last snapshot received before each slot end | 1.0 both exchanges |
| `zzgh` / `zzgh2` | 948 x 15 s / 1422 x 10 s | per-slot amount imbalance (buy - sell) / (buy + sell) binned at -0.6 / -0.2 / 0.2 / 0.6 into 0..4, local time | 0.997 / 0.983 |
| `zzjb` / `zzjb2` | 948 x 15 s / 1422 x 10 s | 0 when the slot's price is 3.04 % or more below the previous close, else 4 | 0.998 / 0.9986 (always-4 baseline 0.927) |
| `zzhj` | 474 | entries 0..15: the mid 30 s before each quarter-hour end (k x 900 + 870 s); the 16th at 14:56:30 | bars 0..14 ~0.997; last point being confirmed |


## Stage 1 -- CED in crucible (`research/ced`, `research/run_ced.py`)

A port of shtcommon's `ced` (the PM's common equity data): the same SQL against Wind / JYDB /
朝阳永续 and the same Barra delivery files, the same rules (limit rounding, new-listing band
windows, ex-right reference price on the total-share `_DIF` basis, share switches, dividend
aggregation, index-weight drift model), but written as one date-partitioned Parquet table per
dataset (`store/ced/<dataset>/date=D/`) instead of per-day `.xr` files. Live and hist SOD /
index weights are separate datasets (CED wrote both to one file). Each module's docstring
lists its columns; `research/ced/__init__.py` has the table of datasets.

Differences from CED, all deliberate:

* concept boards count from the day **after** listing (`list_dt < d`, like members'
  `entry_dt < d`). CED's code used `<=` after an SQL bound of `< end`, so a board listed on d
  counted inside a range but not on a single day; production files are the single-day case.
* Wind industry: one query per range sliced per day (CED queried every day) -- same rows.
* null instead of the `-1` integer sentinels and the `'None'` industry codes the `.xr` files had.
* check results also stored as Parquet (`store/ced/_checks/<check>/date=D/`), not only logged.

Offline tests (`tests/test_ced.py`) drive the SQL -> Parquet paths with a fake database.
Database URLs: environment (`CRUCIBLE_WIND_URL` / `_JY_URL` / `_ZY_URL`) first, else built from `research/ced/db.py`.

Trading calendar: `python research/run_ced.py calendar` caches Wind `ASHARECALENDAR` to
`store/ced/calendar/exchange=SSE.parquet`; with `CRUCIBLE_TRADING_DAYS` pointing at it,
`almanac.TradingCalendar` uses that list instead of `exchange_calendars` (identical slot grids,
checked); `research/intranet_run.py` does this by itself.

Research rebuild over a range: `python research/run_ced.py all-hist --start S --end E`.

**hist is the default basis.** hist SOD takes its listing and every field from the
AShareEODPrices row of that day, `ret_o_pc` is open / preclose from EOD, and in hist mode no
dataset is written past `MAX(TRADE_DT)` of AShareEODPrices (`--live` selects the pre-open basis
and skips the cut). Database addresses and names are fixed in `research/ced/db.py` (WindDB / Zyyx2.0_GDS on 172.24.2.4
since the 2026-10 migration, JYDB on 10.14.3.20;
`tds_version=7.0`); the account is filled in there **on 97 only** --
the committed `DB_USER` / `DB_PASSWORD` stay empty.

Against production (`/data/share/CED`, 20260401-20260430, `tools/ced_vs_share.sh START END`):
Barra identical; EOD identical except one `ret_o_pc` cell (the EOD rule above); hist SOD is the
closer basis; the residual free-share (~23 per day) and index-weight differences are Wind rows
revised after production ran (their OPDATE is later). Treated as reproduced.

### The PM's research CED (`/data/prod/CEDxr4d`, 20260401-20260430)

Produced from our CED by the user for the PM, re-cut into one variable `x(S, D, I, V)` per file
(`.nc`): `daily/{T}_{SOD,EOD}.nc`, `index/{T}.nc`, `barra/{T}_{exp,cov,fret,rate,stats}.nc`.
Stock keys are bare numbers in daily / barra (`'1'` = 000001) and Wind codes in index; I is
09:00:00 for SOD / index weights and 15:00:00 for EOD / Barra.

`sod/` is the PM's start-of-day cut and deliberately **one day behind**: `sod/.../{T}_*.nc` holds
D = T-1 (on the morning of T the latest EOD / Barra is yesterday's). Not a definition change.
`research/compare_ced.py --ext .nc` reads this layout and compares each file with our dataset
on the file's own D.
