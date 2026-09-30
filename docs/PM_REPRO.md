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

## Intranet runbook (10.11.1.97)

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
6. Stage 4 (ZZUG, ZZDS, ZZQI, ZZVC -- candidate formulas, see the section below):
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

## Stage 4 -- candidate formulas (ZZUG, ZZDS, ZZQI, ZZVC)

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

Built locally for 20260615 and 20260805 (300 stocks; ZZVC 5 candidates, ZZUG 4, ZZQI 4, ZZDS 4).
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
