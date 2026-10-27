r"""Stage 5g: what PM's start-of-day stats (``stats/sod/YYYYMMDD.xr``: msr, mb, vr per stock) are.

Each file holds three numbers per stock of PM's 1800-stock universe, computed from the previous trading
day (its ``D`` coordinate). Unlike a model feature, these are given values, so a candidate is checked
directly: per stock, from our ticks of that day and the CED store, and compared across stocks by
Spearman correlation and by the spread of log(PM / candidate) -- an exact match has |rho| ~ 1 and a
log-ratio IQR ~ 0 (the median ratio is then the unit factor).

* snapshots (continuous session): relative / absolute spread, top-of-book and 5-level depth in shares
  and yuan, mean squared and summed 1- / 5-minute mid log returns;
* trades: volume, amount, trade count, mean trade size, per-minute volume, open / close auction share,
  squared daily return and squared log range;
* CED (history up to that day): 20 / 60-day volatility squared, 20-day turnover / average volume /
  amount, and the day's volume / amount over those.

    python research/pm_factors/sod_stats_match.py --dates 20260923,20260924,20260929,20260930,20261009
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dcp_build as B  # catalog
import dcp_ema as E  # tick loading

CONT = (34200, 53820)  # continuous session in exchange seconds: 09:30 to the 14:57 close auction


def tick_stats(con: duckdb.DuckDBPyConnection, d: int, sids: list[int], chunk: int = 200) -> pl.DataFrame:
    """Per stock: the day's snapshot and trade statistics (one row per sid)."""
    out = []
    for i in range(0, len(sids), chunk):
        up = E.load(con, d, sids[i:i + chunk], 86400.0)
        f, t = up["fast"], up["slow"]
        if f.is_empty() or t.is_empty():
            continue
        cont = (pl.col("s") >= CONT[0]) & (pl.col("s") < CONT[1])
        fq = f.filter(cont & (pl.col("b1") > 0) & (pl.col("a1") > 0)).with_columns(((pl.col("a1") + pl.col("b1")) / 2).alias("mid"))
        snap = fq.group_by("sid").agg(
            ((pl.col("a1") - pl.col("b1")) / pl.col("mid")).mean().alias("spr_rel"),
            ((pl.col("a1") - pl.col("b1")) / pl.col("mid")).median().alias("spr_rel_med"),
            (pl.col("a1") - pl.col("b1")).mean().alias("spr_abs"),
            ((pl.col("bv1") + pl.col("av1")) / 2).mean().alias("depth1"),
            (pl.col("bv1") + pl.col("av1")).mean().alias("depth1_sum"),
            pl.col("bv1").mean().alias("bid1_vol"),
            ((pl.col("bv5") + pl.col("av5")) / 2).mean().alias("depth5"),
            ((pl.col("bv1") * pl.col("b1") + pl.col("av1") * pl.col("a1")) / 2).mean().alias("depth1_amt"),
            pl.col("pc").max().alias("pc"))
        for w in (60, 300):  # mid log returns on a w-second grid: last mid of each bucket
            r = (fq.with_columns((pl.col("s") // w).alias("bk")).group_by("sid", "bk").agg(pl.col("mid").last())
                 .sort("sid", "bk").with_columns(pl.col("mid").log().diff().over("sid").alias("r")).drop_nulls("r"))
            snap = snap.join(r.group_by("sid").agg((pl.col("r") ** 2).mean().alias(f"msr{w // 60}m"),
                                                    (pl.col("r") ** 2).sum().alias(f"rv{w // 60}m")), on="sid", how="left")
        tr = t.group_by("sid").agg(
            pl.col("vol").sum().alias("vol_day"), pl.col("amt").sum().alias("amt_day"), pl.len().alias("ntrades"),
            pl.col("vol").filter(cont).sum().alias("vol_cont"),
            (pl.col("vol").filter(pl.col("s") < CONT[0]).sum() / pl.col("vol").sum()).alias("open_share"),
            (pl.col("vol").filter(pl.col("s") >= CONT[1]).sum() / pl.col("vol").sum()).alias("close_share"),
            pl.col("price").last().alias("last"), pl.col("price").max().alias("hi"), pl.col("price").min().alias("lo"))
        tr = tr.with_columns((pl.col("vol_day") / pl.col("ntrades")).alias("trade_size"), (pl.col("amt_day") / pl.col("ntrades")).alias("trade_amt"),
                             (pl.col("vol_cont") / 237).alias("vol_per_min"))
        out.append(snap.join(tr, on="sid", how="full", coalesce=True))
    df = pl.concat(out, how="diagonal_relaxed")
    return df.with_columns((pl.col("last") / pl.col("pc")).log().pow(2).alias("ret_d2"), (pl.col("hi") / pl.col("lo")).log().pow(2).alias("hl2")
                           ).drop("last", "hi", "lo", "pc")


def score(x: np.ndarray, c: np.ndarray) -> tuple[float, int, float, float]:
    """Spearman rho, pairs, median PM / candidate and the IQR of log(PM / candidate) over the stocks with both."""
    m = np.isfinite(x) & np.isfinite(c)
    if m.sum() < 30:
        return np.nan, int(m.sum()), np.nan, np.nan
    rx = pl.Series(x[m]).rank("average").to_numpy()
    rc = pl.Series(c[m]).rank("average").to_numpy()
    rho = float(np.corrcoef(rx, rc)[0, 1])
    p = m & (x > 0) & (c > 0)
    if p.sum() < 30:
        return rho, int(m.sum()), np.nan, np.nan
    lr = np.log(x[p] / c[p])
    q1, q3 = np.quantile(lr, [0.25, 0.75])
    return rho, int(m.sum()), float(np.exp(np.median(lr))), float(q3 - q1)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True, help="sod file dates; each file's statistics are of the trading day before")
    ap.add_argument("--sod-dir", type=Path, default=Path("/data/s/stats/sod"))
    ap.add_argument("--ced", action=argparse.BooleanOptionalAction, default=True, help="also CED daily candidates (ced_match.daily_candidates)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage5"))
    args = ap.parse_args()
    t0 = time.perf_counter()
    from compare_pm import load_pm  # the one netCDF reader

    con = duckdb.connect(str(B.CATALOG), read_only=True)
    CM = None
    if args.ced:
        import ced_match as CM
    rows = []
    for T in [x.strip() for x in args.dates.split(",") if x.strip()]:
        f = args.sod_dir / f"{T}.xr"
        if not f.exists():
            print(f"skip {T}: no {f}")
            continue
        long = load_pm(f)  # S, D, I, V, x
        D = str(long["D"][0])
        pm = (long.pivot(on="V", index="S", values="x", aggregate_function="first")
              .rename({"S": "sid"}).with_columns(pl.col("sid").cast(pl.Int64)))
        sids = pm["sid"].to_list()
        st = tick_stats(con, int(D), sids)
        if st.is_empty():
            print(f"skip {T}: no ticks of {D} in {B.CATALOG}")
            continue
        if CM is not None and (dc := CM.daily_candidates(T, sids)) is not None:
            dc = dc.select("sid", "vol20", "vol60", "turn20", "adv20", "avgvol20", "freeshare", "range20").with_columns(
                (pl.col("vol20") ** 2).alias("vol20_2"), (pl.col("vol60") ** 2).alias("vol60_2"))
            st = st.join(dc, on="sid", how="left").with_columns(
                (pl.col("vol_day") / pl.col("avgvol20")).alias("vol_over_avg20"), (pl.col("amt_day") / pl.col("adv20")).alias("amt_over_adv20"),
                (pl.col("vol_day") / pl.col("freeshare")).alias("turnover_day"))
        j = pm.join(st, on="sid", how="left")
        cands = [c for c in st.columns if c != "sid"]
        for v in [c for c in pm.columns if c != "sid"]:
            x = j[v].to_numpy().astype(float)
            for c in cands:
                rho, n, ratio, iqr = score(x, j[c].to_numpy().astype(float))
                rows.append({"date": T, "day": D, "stat": v, "candidate": c, "rho": rho, "n": n, "ratio": ratio, "logratio_iqr": iqr})
        print(f"{T} (stats of {D}): {pm.height} stocks, ticks for {st.height} ({time.perf_counter() - t0:.0f}s)")
    if not rows:
        return
    res = pl.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    res.write_csv(args.out / "sod_stats_match.csv")
    agg = (res.group_by("stat", "candidate").agg(pl.col("rho").mean(), pl.col("n").mean().round(0), pl.col("ratio").median(),
                                                  pl.col("logratio_iqr").median(), pl.len().alias("days"))
           .with_columns(pl.col("rho").abs().alias("abs_rho")).sort("stat", "abs_rho", descending=[False, True]))
    with pl.Config(tbl_rows=60, tbl_cols=7, tbl_width_chars=150, float_precision=4):
        print(f"\n== PM sod stats vs candidates, mean over days: exact = |rho| ~ 1 and logratio_iqr ~ 0 (ratio = unit factor); top {args.top} per stat")
        print(agg.group_by("stat", maintain_order=True).head(args.top).drop("abs_rho"))
    print(f"table: {args.out / 'sod_stats_match.csv'} ({time.perf_counter() - t0:.0f}s)")


if __name__ == "__main__":
    main()
