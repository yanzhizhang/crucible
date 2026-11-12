r"""Which per-stock daily input is PM using that we are not? The per-stock part of the residual (PM's rank beyond what ours
explains, as residual_breakdown) against every daily candidate of ced_match.daily_candidates: returns / volatility / range
over n days, adv / turnover / caps, price level and tick size over close, the price band, multi-day close EMAs, and the
Barra exposures (industry dummies, styles) and stats. Spearman across stocks, whole day and first hour, top by |rho|.

    python research/pm_factors/static_resid.py --pred data/reports/stage6_r35_20260430/pred.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return float("nan")
    rx = pl.Series(x[m]).rank("average").to_numpy()
    ry = pl.Series(y[m]).rank("average").to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    import ced_match as CM

    d = pl.read_parquet(a.pred).filter(pl.col("ours").is_finite() & pl.col("pm").is_finite())
    g = ["date", "mark"]
    d = d.with_columns(((pl.col(c).rank() - 0.5) / pl.len() - 0.5).over(g).alias(f"r_{c}") for c in ("ours", "pm"))
    d = d.with_columns((pl.col("r_pm") - (pl.col("r_pm") * pl.col("r_ours")).sum().over(g) / (pl.col("r_ours") ** 2).sum().over(g)
                        * pl.col("r_ours")).alias("res"))
    rows, dummies = [], []
    for (day,), dd in d.group_by("date"):
        sd = dd.group_by("sid").agg(pl.col("res").mean().alias("res_day"), pl.col("res").filter(pl.col("mark") < 1800).mean().alias("res_30m"),
                                    pl.col("res").filter(pl.col("mark") < 3600).mean().alias("res_1h"))
        dc = CM.daily_candidates(str(day), sd["sid"].to_list())
        if dc is None:
            print(f"{day}: no daily candidates")
            continue
        j = sd.join(dc, on="sid", how="inner")
        print(f"{day}: {sd.height} stocks with a residual, {j.height} with daily candidates, {len(dc.columns) - 1} candidates")
        for c in dc.columns:
            if c == "sid" or not dc.schema[c].is_numeric():
                continue
            x = j[c].to_numpy().astype(float)
            fx = x[np.isfinite(x)]
            if fx.size and np.isin(fx, (0.0, 1.0)).all():  # a 0 / 1 dummy (industry, stored 1 / NaN): mean residual inside minus outside, as a t
                x = np.nan_to_num(x, nan=0.0)
                r = j["res_day"].to_numpy()
                ins, out = r[(x == 1) & np.isfinite(r)], r[(x == 0) & np.isfinite(r)]
                if ins.size >= 3 and out.size >= 3:
                    tv = (ins.mean() - out.mean()) / np.sqrt(ins.var(ddof=1) / ins.size + out.var(ddof=1) / out.size)
                    dummies.append({"date": day, "dummy": c, "stocks": int(ins.size), "res_in": float(ins.mean()), "res_out": float(out.mean()), "t": float(tv)})
                continue
            rows.append({"date": day, "candidate": c, "rho_day": spearman(x, j["res_day"].to_numpy()),
                         "rho_30m": spearman(x, j["res_30m"].to_numpy()), "rho_1h": spearman(x, j["res_1h"].to_numpy()),
                         "finite": float(np.isfinite(x).mean())})
    t = pl.DataFrame(rows).filter(pl.col("rho_day").is_not_nan()).sort(pl.col("rho_day").abs(), descending=True)
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=3, fmt_str_lengths=50):
        print(f"\n== daily candidates against the per-stock residual (top {a.top} by |rho_day|; ~0.1 is noise for ~300 stocks)")
        print(t.head(a.top))
        print("\n== top by |rho_30m| (the open)")
        print(t.filter(pl.col("rho_30m").is_not_nan()).sort(pl.col("rho_30m").abs(), descending=True).head(15))
        if dummies:
            print("\n== 0 / 1 dummies (industries): mean residual inside vs outside, Welch t; |t| > 2.5 is worth a look")
            print(pl.DataFrame(dummies).sort(pl.col("t").abs(), descending=True).head(30))


if __name__ == "__main__":
    main()
