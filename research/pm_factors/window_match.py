r"""Is a samplerS cell a rolling quantile of a basic over the last N trading days? Scored cell by cell.

The unchanged-cell share of the profile families decays with the day gap like a rolling median over ~10-14 days
(``same_gap.py``). For each PM day d and column, every basic of the matching table (m5 for bar families, d1 for daily
ones) is stacked over the trading days [d-N+1, d] (``incl``) or [d-N, d-1] (``excl``), reduced with a NaN-aware
quantile (numpy linear: an even window's median is the mean of the two middle days), and compared with PM:
``exact`` = share of cells equal to 1e-6 after scaling by the median ratio, ``ok3`` the same at 1e-3, ``spearman``.

Only windows fully covered by basics days are scored. PM files come from the ``same_gap.py`` cache.

    python research/pm_factors/window_match.py --families ZZQO,ZZTS,ZZXC,ZZAL,ZZMS,ZZDS --days 20261020,20261021,20261022,20261023,20261026,20261027
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basic_match import BASICS
from rank_candidates import pm_frame

WINDOWS = (5, 10, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 28, 30, 32, 35, 40, 45, 50, 60)
LEVELS = (0.5,)  # the median: confirmed on ZZTS / ZZXC (ratio 1.000, 25-30% bit-exact at N = 20, rising with N)


def trading_days(hstats: Path, fam: str) -> list[str]:
    """SSE trading days from the CED calendar; PM's own day folders only start where its history starts."""
    pm = sorted(p.name for p in (hstats / fam).glob("20*") if p.is_dir())
    try:
        from ced.calendar import calendar
        return calendar.trade_days("20200101", max(pm))
    except Exception as e:  # no CED store: PM days, which caps the windows at PM's own history
        print(f"CED calendar unavailable ({type(e).__name__}: {e}); trading days = PM folders from {pm[0]}")
        return pm


def quantile_sorted(w: np.ndarray, nf: np.ndarray, q: float) -> np.ndarray:
    """numpy-'linear' quantile over axis 0 of an already sorted stack whose NaNs sit at the end (nf finite values each)."""
    pos = (nf - 1) * q
    lo = np.floor(pos).astype(np.int64).clip(0)
    hi = np.minimum(lo + 1, np.maximum(nf - 1, 0))
    f = pos - lo
    a = np.take_along_axis(w, lo[None], axis=0)[0]
    b = np.take_along_axis(w, hi[None], axis=0)[0]
    out = a + (b - a) * f
    return np.where(nf > 0, np.where(f == 0, a, out), np.nan)


def scores(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per column of X: ratio, exact (1e-6), ok3 (1e-3), spearman with y; NaN cells skipped."""
    k = X.shape[1]
    ratio, ex, ok3, rho = (np.full(k, np.nan) for _ in range(4))
    ry = pl.Series(y).rank().to_numpy()
    for c in range(k):
        x = X[:, c]
        m = np.isfinite(y) & np.isfinite(x) & (x != 0)
        if m.sum() < 100:
            continue
        r = np.median(y[m] / x[m])
        d = np.abs(y[m] - r * x[m])
        ratio[c], ex[c], ok3[c] = r, np.mean(d <= 1e-6 * np.abs(y[m]) + 1e-12), np.mean(d <= 1e-3 * np.abs(y[m]) + 1e-12)
        rx = pl.Series(np.where(m, x, np.nan)).rank().to_numpy()
        rho[c] = float(np.corrcoef(ry[m], rx[m])[0, 1])
    return ratio, ex, ok3, rho


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--families", default="ZZQO,ZZTS,ZZXC,ZZAL,ZZMS,ZZDS")
    ap.add_argument("--days", required=True, help="PM days to score")
    ap.add_argument("--hstats", type=Path, default=Path("/data/prod/hstats/HS300"))
    ap.add_argument("--cache", type=Path, default=Path("data/pm_export_all"))
    ap.add_argument("--basics", type=Path, default=BASICS)
    ap.add_argument("--max-cells", type=int, default=8000)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4/window_match.csv"))
    a = ap.parse_args()
    have = {p.name.split("=")[1] for p in a.basics.glob("date=*")}
    rng = np.random.default_rng(0)
    rows = []
    for fam in a.families.split(","):
        tdays = trading_days(a.hstats, fam)
        for d in a.days.split(","):
            f = a.cache / f"{fam}_{d}.parquet"
            if not f.exists() or d not in tdays or not (a.hstats / fam / d).is_dir():
                print(f"{fam} {d}: no PM file / not a PM day")
                continue
            pm, vals, kind = pm_frame(f, None, None, None, None, None)
            table = "d1" if kind == "daily" else "m5"
            keys = ["code"] if table == "d1" else ["code", "minute"]
            i = tdays.index(d)
            hist = tdays[max(0, i - max(WINDOWS)): i + 1]  # trading days up to d, oldest first
            hist = [x for x in hist if x in have]
            cover, gap = 0, "none"  # consecutive trading days back from d that basics hold, and the first one missing
            for x in reversed(tdays[: i + 1]):
                if x not in have:
                    gap = x
                    break
                cover += 1
            tabs = {}
            for x in hist:
                t = pl.read_parquet(a.basics / f"date={x}" / f"{table}.parquet").rename({"symbol_id": "code"})
                tabs[x] = t.with_columns(pl.col("code").cast(pl.Int64), pl.col("minute").cast(pl.Int64))
            names = [c for c in tabs[d].columns if c not in ("code", "minute")]
            for V in sorted(pm["V"].unique().to_list()):
                cells = pm.filter(pl.col("V") == V).select(*keys, pl.col(vals[0]).alias("y")).filter(pl.col("y").is_finite())
                if cells.height > a.max_cells:
                    cells = cells[np.sort(rng.choice(cells.height, a.max_cells, replace=False))]
                # stack[day, cell, basic] over the history, aligned to the PM cells
                stack = np.stack([cells.select(keys).join(tabs[x].select(*keys, *[c for c in names if c in tabs[x].columns]), on=keys, how="left")
                                  .select([pl.col(c).cast(pl.Float64) if c in tabs[x].columns else pl.lit(None, pl.Float64).alias(c) for c in names])
                                  .to_numpy() for x in hist if x in tabs])
                y = cells["y"].to_numpy()
                for N in WINDOWS:
                    for incl in (True, False):
                        need = N if incl else N + 1
                        if need > cover:
                            continue
                        w = np.sort(stack[-N:] if incl else stack[-N - 1:-1], axis=0)  # NaN sorts last
                        nf = np.isfinite(w).sum(axis=0)  # finite days per (cell, basic)
                        for q in LEVELS:
                            X = quantile_sorted(w, nf, q)
                            ratio, ex, ok3, rho = scores(y, X)
                            best = int(np.nanargmax(np.nan_to_num(ok3, nan=-1)))
                            rows.append({"file": fam, "V": V, "day": d, "N": N, "incl": incl, "q": q, "basic": names[best], "ratio": ratio[best],
                                         "exact": ex[best], "ok3": ok3[best], "spearman": rho[best]})
            print(f"{fam} {d}: {kind}, basics cover {cover} trading days (first missing {gap})", flush=True)
    if not rows:
        print("nothing scored")
        return
    t = pl.DataFrame(rows)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    t.write_csv(a.out)
    agg = (t.group_by("file", "V", "N", "incl", "q", "basic").agg(pl.len().alias("days"), pl.col("exact").mean(), pl.col("ok3").mean(),
                                                                 pl.col("ratio").median(), pl.col("spearman").mean())
           .sort("ok3", descending=True).group_by("file", "V", maintain_order=True).head(3).sort("file", "V", "ok3", descending=[False, False, True]))
    with pl.Config(tbl_rows=200, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print("== rolling quantile over the last N trading days (incl = day d in the window): best 3 per PM column by ok3 (1e-3), exact = 1e-6")
        print(agg)


if __name__ == "__main__":
    main()
