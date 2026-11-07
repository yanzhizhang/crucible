r"""PM's daily stats file (``/tmp/s/stsfile/stats/sod/<T>.xr``: msr / mb / vr per stock, built from day T-1) against
candidates computed from our snapshots of that day.

The trading side loads it every morning (``qlive/trd.cpp``) and its ``hs`` looks like msr, so these are PM's own
normalisers. Candidates, over the continuous-session snapshots (09:30 - 14:57 exchange time):

* msr: mean of (ask1 - bid1) / (ask1 + bid1) (half spread over mid), its median, the mean relative spread, its half;
* mb: mean of (bid1 vol + ask1 vol) / 2, of bid1 / ask1 vol, of min(bid1, ask1), medians;
* vr: mean volume between snapshots over mean level-1 depth, mean of the per-snapshot ratio, day volume over
  (snapshots x depth), mid move std over msr.

Per candidate: median of PM / ours and the share of stocks within 1e-3 / 1e-2 of PM.

    python research/pm_factors/stats_sod_check.py --file /work/crucible_data/pm_stsfile/stats/sod/20261105.xr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog
import dcp_ema as E  # snapshot loading


def candidates(g: pl.DataFrame) -> dict[str, float]:
    """One stock-day of snapshots -> every candidate value."""
    g = g.filter((pl.col("t_exch") > 0) & (pl.col("t_exch") < 14220))
    b1, a1, bv, av = (g[c].to_numpy().astype(float) for c in ("b1", "a1", "bv1", "av1"))
    cv = g["cv"].to_numpy().astype(float)
    two = (b1 > 0) & (a1 > 0) & (a1 > b1)
    if two.sum() < 20:
        return {}
    hs = (a1[two] - b1[two]) / (a1[two] + b1[two])
    dep = (bv + av) / 2
    dv = np.clip(np.diff(cv, prepend=cv[0]), 0, None)
    lm = np.log((a1[two] + b1[two]) / 2)
    out = {"msr.mean_hs": hs.mean(), "msr.median_hs": np.median(hs), "msr.mean_rs": 2 * hs.mean(), "msr.mean_hs_all": np.nanmean(np.where(two, (a1 - b1) / (a1 + b1), np.nan)),
           "mb.mean_dep": dep.mean(), "mb.median_dep": np.median(dep), "mb.mean_bid": bv.mean(), "mb.mean_ask": av.mean(),
           "mb.mean_min": np.minimum(bv, av).mean(), "mb.mean_sum": (bv + av).mean()}
    with np.errstate(divide="ignore", invalid="ignore"):
        out |= {"vr.dv_over_dep": dv.mean() / dep.mean(), "vr.mean_ratio": np.nanmean(np.where(dep > 0, dv / dep, np.nan)),
                "vr.dv_over_sum": dv.mean() / (bv + av).mean(), "vr.day_over": dv.sum() / dep.sum(),
                "vr.mstd_over_msr": np.std(np.diff(lm)) / hs.mean(), "vr.mabs_over_msr": np.mean(np.abs(np.diff(lm))) / hs.mean(),
                "vr.dv_over_dep_moved": dv[dv > 0].mean() / dep.mean() if (dv > 0).any() else np.nan}
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", type=Path, required=True)
    a = ap.parse_args()
    ds = xr.open_dataset(a.file)
    day = str(ds["D"].values[0])
    x = ds["x"].squeeze()
    pm = {name: dict(zip((int(s) for s in ds["S"].values), x.sel(V=name).values, strict=True)) for name in ds["V"].values}
    sids = sorted(pm[next(iter(pm))])
    print(f"{a.file.name}: {len(sids)} stocks, data of {day}, columns {list(pm)}")
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    rows = []
    for i in range(0, len(sids), 300):
        fast = E.load(con, int(day), sids[i:i + 300], 54150.005)["fast"]
        for g in fast.partition_by("sid"):
            c = candidates(g)
            if c:
                rows.append({"sid": int(g["sid"][0]), **c})
    con.close()
    ours = pl.DataFrame(rows)
    print(f"ours: {ours.height} stocks with snapshots on {day}")
    for col in [c for c in ours.columns if c != "sid"]:
        name = col.split(".")[0]
        y = np.array([pm[name].get(s, np.nan) for s in ours["sid"].to_list()])
        v = ours[col].to_numpy()
        m = np.isfinite(y) & np.isfinite(v) & (v != 0)
        r = y[m] / v[m]
        print(f"  {col:22s} PM/ours median {np.median(r):.4f} p10/p90 {np.percentile(r, 10):.4f}/{np.percentile(r, 90):.4f}  "
              f"within 1e-3 {np.mean(np.abs(r - 1) < 1e-3):.3f}  1e-2 {np.mean(np.abs(r - 1) < 1e-2):.3f}  n {m.sum()}")


if __name__ == "__main__":
    main()
