r"""Does 224's stale seed explain where PM's live prediction parts from ours?

224's PM process starts every day from one old state file soft-linked in place of yesterday's (the user, 2026-11-12), so
any of its factors that carry state from the previous day see that old day, not yesterday. For each day of ``--pred``
(``ic_score --dump-pred`` rows: date, sid, mark, ours, pm) this takes the per-stock residual (PM's rank beyond what ours
explains, as ``residual_breakdown``), whole day and first hour, and correlates it across stocks with how far the seed's
value of every state key is from the true previous day's (``--dcp`` dir): log ratio for positive keys, else a difference
scaled by the true value. Keys whose staleness tracks the residual are the ones PM's live process carries across days.

    python research/pm_factors/fake_seed.py --seed /work/crucible_data/seed_224/X.jz --dcp /work/crucible_data/seed_probe/HS300 \
        --pred data/reports/stage6_r23_implied_224/pred.parquet data/reports/stage6_r25_ic_1111/pred.parquet
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # noqa: E402 -- state file reader


def residuals(pred: pl.DataFrame) -> pl.DataFrame:
    """Per date and stock: mean residual over the day and over the first hour."""
    d = pred.filter(pl.col("ours").is_finite() & pl.col("pm").is_finite())
    g = ["date", "mark"]
    d = d.with_columns(((pl.col(c).rank() - 0.5) / pl.len() - 0.5).over(g).alias(f"r_{c}") for c in ("ours", "pm"))
    d = d.with_columns((pl.col("r_pm") - (pl.col("r_pm") * pl.col("r_ours")).sum().over(g) / (pl.col("r_ours") ** 2).sum().over(g)
                        * pl.col("r_ours")).alias("res"))
    return d.group_by("date", "sid").agg(pl.col("res").mean().alias("res_day"), pl.col("res").filter(pl.col("mark") < 3600).mean().alias("res_1h"))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 30:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[m])), np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=Path, required=True)
    ap.add_argument("--dcp", type=Path, required=True)
    ap.add_argument("--pred", type=Path, nargs="+", required=True)
    ap.add_argument("--top", type=int, default=40)
    a = ap.parse_args()
    seed = B.load_state(a.seed).rename({"pm": "seed"})
    print(f"seed {a.seed}: {seed['sid'].n_unique()} stocks, {seed['key'].n_unique()} keys")
    res = pl.concat([residuals(pl.read_parquet(p)) for p in a.pred], how="vertical_relaxed").unique(["date", "sid"], keep="last")
    days = sorted(str(x) for x in res["date"].unique().to_list())
    files = sorted(p.stem for p in a.dcp.glob("*.jz") if re.fullmatch(r"\d{8}", p.stem))
    rows = []
    for d in days:
        prev = [x for x in files if x < d]
        if not prev:
            print(f"{d}: no true previous state in {a.dcp}")
            continue
        true = B.load_state(a.dcp / f"{prev[-1]}.jz").rename({"pm": "true"})
        j = seed.join(true, on=["sid", "key"], how="inner")
        same = (j["seed"] == j["true"]).mean()
        print(f"{d}: true previous {prev[-1]}; seed == true previous for {same:.3f} of {j.height:,} values")
        j = j.with_columns(pl.when((pl.col("seed") > 0) & (pl.col("true") > 0)).then((pl.col("seed") / pl.col("true")).log())
                           .otherwise((pl.col("seed") - pl.col("true")) / (pl.col("true").abs() + 1e-12)).alias("stale"))
        r = res.filter(pl.col("date").cast(pl.String) == d)
        w = j.select("sid", "key", "stale").join(r.select("sid", "res_day", "res_1h"), on="sid")
        for (k,), g in w.group_by("key"):
            s = g["stale"].to_numpy()
            if np.nanstd(s) == 0:
                continue
            rows.append({"date": d, "key": k, "rho_day": spearman(s, g["res_day"].to_numpy()), "rho_1h": spearman(s, g["res_1h"].to_numpy()),
                         "stale_med": float(np.nanmedian(np.abs(s)))})
    if not rows:
        return
    t = pl.DataFrame(rows)
    agg = t.group_by("key").agg(pl.col("rho_day").mean(), pl.col("rho_1h").mean(), pl.col("rho_day").min().alias("rho_day_min"),
                                pl.col("rho_day").max().alias("rho_day_max"), pl.col("stale_med").median(), pl.len().alias("days"))
    agg = agg.with_columns(pl.col("key").str.extract(r"^([^.]+)", 1).alias("stem")).sort(pl.col("rho_day").abs(), descending=True)
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=3):
        print(f"\n== state keys whose staleness tracks the residual (mean over {len(days)} days, top {a.top} by |rho_day|)")
        print(agg.head(a.top))
        fam = agg.group_by("stem").agg(pl.col("rho_day").abs().max().alias("max_abs_rho_day"), pl.col("rho_1h").abs().max().alias("max_abs_rho_1h"),
                                       pl.len().alias("keys")).sort("max_abs_rho_day", descending=True)
        print("\n== per family: strongest key")
        print(fam.head(30))
    null = agg["rho_day"].abs().drop_nans()
    print(f"\n|rho_day| over all {null.len()} keys: median {null.median():.3f}, p90 {null.quantile(0.9):.3f} (about 0.06 would be noise for 300 stocks)")


if __name__ == "__main__":
    main()
