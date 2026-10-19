r"""Stage 3: is PM's samplerR a random sample of orders (reservoir sampling, "Algorithm R")? Estimate K and the memory.

Evidence so far (docs/PM_REPRO.md, stage 3): per-order filled totals are the right quantity, but
PM's values are never exactly equal to any deterministic window / quantile method, and ZZAC's
five-minute path jumps at random -- sometimes against the new slot's own orders. A random sample
of size K from a pool of past orders explains all of it. If so, PM's value is one draw from the
distribution of "quantile of a K-sample", so its percentile (PIT) inside many simulated
K-samples should be uniform across stocks. Wrong K shows up at once: K too large -> PIT piles at
0 and 1; K too small -> PIT bunches around 0.5.

Pools (``--models``): ``today`` (the trade day up to the epoch), ``decayM`` (earlier days weighted
exp(-days_back / M)), ``poolN`` (the last N days, equal weight). Samples are drawn with
replacement (K is small against the pool), quantiles linear. Targets: ZZAC's as / bs columns at
``--epochs`` and, with ``--cr``, ZZCR's 16 columns at the close.

Score per (target, pool, K): ``ks`` (max distance of the PIT CDF from uniform; lower is better),
``cover90`` (share of PM values inside the simulated 5-95 % band; 0.90 if right) and ``mean_pit``
(0.5 if unbiased).

    python research/pm_factors/sampler_r_mc.py --date 20260429 \
        --ac /data/prod/hstats/sod/HS300/ZZAC/20260430/samplerR.parquet \
        --cr /data/prod/hstats/sod/HS300/ZZCR/20260430/samplerR.parquet
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import diag_sampler_ac as A  # ZZAC reader
import sampler_r as R  # catalog, universe, codes

KS = (50, 100, 200, 500, 1000, 2000)
TZ = dt.timezone(dt.timedelta(hours=8))


def totals(con: duckdb.DuckDBPyConnection, days: list[str], today: str, end_ns: int | None,
           size: str = "filled") -> pl.DataFrame:
    """Per (day, stock, role, order) size; earlier days whole, ``today`` up to ``end_ns``.

    ``filled``: the order's filled total. ``submitted``: the order's submitted quantity, rebuilt per
    venue from the order stream (``raw_order``): SZSE publishes the original order (key ``seq_id``);
    SSE publishes only what rests after the immediate fills (key ``order_id``) and nothing for an
    order filled on arrival, so there it is the order's aggressor fills plus the published rest.
    """
    dl = ", ".join(str(int(d)) for d in days)
    cut = f"AND (r.date <> {int(today)} OR r.time <= {end_ns})" if end_ns else ""
    roles = f"""
      WITH t AS (
        SELECT r.date, r.symbol_id, r.market_id, r.dir, r.volume, r.buy_seq_id, r.sell_seq_id
        FROM raw_transaction r JOIN uni u USING (symbol_id, market_id)
        WHERE r.date IN ({dl}) AND r.trade_type = 1 AND r.dir IN ({R.BUY}, {R.SELL}) {cut}),
      f AS (
        SELECT date, symbol_id, market_id, CASE dir WHEN {R.BUY} THEN 'ab' ELSE 'as' END AS role,
               CASE dir WHEN {R.BUY} THEN buy_seq_id ELSE sell_seq_id END AS oid, sum(volume) AS v
        FROM t GROUP BY ALL
        UNION ALL
        SELECT date, symbol_id, market_id, CASE dir WHEN {R.BUY} THEN 'ps' ELSE 'pb' END,
               CASE dir WHEN {R.BUY} THEN sell_seq_id ELSE buy_seq_id END, sum(volume)
        FROM t GROUP BY ALL)"""
    if size == "filled":
        return con.sql(roles + " SELECT date, symbol_id, role, oid, v FROM f").pl()
    ocut = f"AND (o.date <> {int(today)} OR o.time <= {end_ns})" if end_ns else ""
    return con.sql(roles + f""",
      aggr AS (SELECT date, symbol_id, oid, sum(v) AS af FROM f WHERE role IN ('as', 'ab') GROUP BY ALL),
      ord AS (
        SELECT o.date, o.symbol_id, CASE o.market_id WHEN 3553 THEN o.order_id ELSE o.seq_id END AS oid,
               max(o.volume) AS ov
        FROM raw_order o JOIN uni u USING (symbol_id, market_id)
        WHERE o.date IN ({dl}) AND (o.market_id = 3554 OR o.update_type <> 2) {ocut}
        GROUP BY ALL)
      SELECT f.date, f.symbol_id, f.role, f.oid,
             CASE f.market_id WHEN 3553 THEN coalesce(ord.ov, 0) + coalesce(aggr.af, 0)
                              ELSE coalesce(ord.ov, f.v) END AS v
      FROM f LEFT JOIN ord USING (date, symbol_id, oid) LEFT JOIN aggr USING (date, symbol_id, oid)""").pl()


def weights(back: np.ndarray, model: str) -> np.ndarray | None:
    """Return sampling weights for rows ``back`` days before the trade day (None = empty pool)."""
    if model == "today":
        w = (back == 0).astype(float)
    elif model.startswith("decay"):
        w = np.exp(-back / float(model[5:]))
    elif model.startswith("pool"):
        w = (back < int(model[4:])).astype(float)
    else:
        raise ValueError(model)
    s = w.sum()
    return w / s if s > 0 else None


def pits(task: tuple) -> list[tuple]:
    """For one (stock, role) pool: PIT of each PM value under every (model, K)."""
    v, back, targets, models, seed, sims = task  # targets: list of (target, column, level, pm)
    rng = np.random.default_rng(seed)
    out = []
    for model in models:
        w = weights(back, model)
        if w is None:
            continue
        cdf = np.cumsum(w)
        for k in KS:
            idx = np.minimum(np.searchsorted(cdf, rng.random((sims, k))), len(v) - 1)
            levels = sorted({t[2] for t in targets})
            q = np.quantile(v[idx], levels, axis=1, method="linear")  # (levels, sims)
            qmap = dict(zip(levels, q, strict=True))
            for target, column, level, pm in targets:
                s = qmap[level]
                lo, eq = (s < pm - 0.25).mean(), (np.abs(s - pm) <= 0.25).mean()
                out.append((target, model, k, column, lo + rng.random() * eq, float(np.quantile(s, 0.05)) <= pm <= float(np.quantile(s, 0.95))))
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True, help="trade day of the PM rows (sod/<D+1> holds day D)")
    ap.add_argument("--ac", type=Path, help="ZZAC samplerR (48 epochs x as/bs)")
    ap.add_argument("--cr", type=Path, help="ZZCR samplerR (as/ps/ab/pb at the close)")
    ap.add_argument("--epochs", default="11:30,15:00", help="ZZAC epochs to test")
    ap.add_argument("--models", default="today,decay1,decay3,decay10,pool10")
    ap.add_argument("--size", default="filled", choices=("filled", "submitted"),
                    help="order size: filled total, or submitted quantity (needs the order stream decoded)")
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--workers", type=int, default=min(64, os.cpu_count() or 1))
    ap.add_argument("--index", default="000300")
    a = ap.parse_args()
    con = duckdb.connect(str(R.CATALOG), read_only=True)
    days = [str(d) for d in con.sql(f"SELECT DISTINCT date FROM raw_transaction WHERE date <= {int(a.date)} "
                                    "ORDER BY date DESC LIMIT 11").pl()["date"]]
    back_of = {d: i for i, d in enumerate(days)}
    print(f"trade days in the pool (newest first): {days}; size = {a.size}")
    uni = R.ced_universe(a.index, a.date)
    con.register("uni", uni if uni is not None else R.universe(a.index))
    models = a.models.split(",")
    d0 = dt.date(int(a.date[:4]), int(a.date[4:6]), int(a.date[6:]))
    jobs = []  # (label, end_ns, targets long frame with sid / role / column / level / pm)
    if a.ac:
        ac = A.ac_long(a.ac)
        for e in a.epochs.split(","):
            end = int(dt.datetime(d0.year, d0.month, d0.day, int(e[:2]), int(e[3:]), tzinfo=TZ).timestamp()) * 10**9
            t = ac.filter(pl.col("hhmm") == e).with_columns(
                pl.col("column").str.slice(0, 2).replace({"bs": "ab"}).alias("role"),
                (pl.col("column").str.slice(4).cast(pl.Int64) / 100).alias("level"))
            jobs.append((f"ZZAC {e}", end, t))
    if a.cr:
        cr = pl.read_parquet(a.cr)
        cols = [c for c in cr.columns if c[:3] in ("as.", "ps.", "ab.", "pb.")]
        t = cr.unpivot(index="sid", on=cols, variable_name="column", value_name="pm").with_columns(
            pl.col("column").str.slice(0, 2).alias("role"), (pl.col("column").str.slice(4).cast(pl.Int64) / 100).alias("level"))
        jobs.append(("ZZCR close", None, t))
    rows = []
    for label, end, t in jobs:
        tot = totals(con, days, a.date, end, a.size).with_columns(
            pl.col("date").cast(pl.String).replace_strict(back_of).alias("back"))
        t = t.filter(pl.col("pm") > 0)
        tasks = []
        for (sym, role), g in tot.group_by("symbol_id", "role"):
            tg = t.filter((pl.col("sid") == float(sym)) & (pl.col("role") == role))
            if tg.is_empty():
                continue
            targets = [(label, c, lv, p) for c, lv, p in tg.select("column", "level", "pm").iter_rows()]
            tasks.append((g["v"].to_numpy().astype(float), g["back"].to_numpy().astype(float), targets, models,
                          zlib.crc32(f"{label}|{sym}|{role}".encode()), a.sims))
        print(f"{label}: {len(tasks)} stock x role pools, {sum(len(x[2]) for x in tasks)} PM values")
        with ProcessPoolExecutor(a.workers) as ex:
            for r in ex.map(pits, tasks, chunksize=8):
                rows += r
    df = pl.DataFrame(rows, schema={"target": pl.String, "model": pl.String, "K": pl.Int64, "column": pl.String,
                                    "pit": pl.Float64, "in90": pl.Boolean}, orient="row")
    grid = np.linspace(0, 1, 101)
    agg = df.group_by("target", "model", "K").agg(
        pl.len().alias("n"), pl.col("in90").mean().alias("cover90"), pl.col("pit").mean().alias("mean_pit"),
        pl.col("pit").alias("pits"))
    ks = [float(np.max(np.abs(np.searchsorted(np.sort(np.asarray(p)), grid, side="right") / max(len(p), 1) - grid)))
          for p in agg["pits"].to_list()]  # max distance of the PIT CDF from the uniform CDF
    score = agg.drop("pits").with_columns(pl.Series("ks", ks))
    out = Path("data/reports/stage3")
    out.mkdir(parents=True, exist_ok=True)
    score.write_csv(out / f"sampler_r_mc_{a.date}.csv")
    with pl.Config(tbl_rows=80, tbl_width_chars=200, float_precision=4):
        for tg in score["target"].unique().sort():
            s = score.filter(pl.col("target") == tg).sort("ks")
            print(f"== {tg}: best (pool, K) by ks -- uniform PIT means ks near 0 and cover90 near 0.90")
            print(s.head(10))
            print(s.pivot(on="K", index="model", values="ks", sort_columns=True).sort("model"))
    print(f"full table: {out / f'sampler_r_mc_{a.date}.csv'}")


if __name__ == "__main__":
    main()
