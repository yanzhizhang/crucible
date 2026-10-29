r"""Stage 4: are the size-like samplerS columns (ZZUG, ZZUGSS, ZZVC e/f, ZZQI) PM's own samplerR quantiles? PM against PM.

ZZUG's a/b ~1100 and c/d ~4100 per stock, steady through the day; ZZVC's e/f ~500 in multiples of 1/3; ZZUGSS's m = 2000,
s = 300 -- all look like order-size thresholds, which is what samplerR holds (ZZAC: 48 five-minute rows per stock of
``as`` / ``bs`` x q5..q98; ZZCR: one row of ``as ps ab pb`` x q50..q98). Each target column is scored against every samplerR
column of the same PM day and of the next one, at the matching five-minute mark (bar shifts -5 .. +5 for the per-minute
files): share of equal cells (1e-6 relative), median ratio, rank correlation.

    python research/pm_factors/sampler_pm.py --date 20261009 --hstats /data/prod/hstats/HS300 --exports data/pm_export
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rank_candidates import pm_frame

TARGETS = {"ZZUG": ("a", "b", "c", "d", "ab", "cd"), "ZZUGSS": ("a", "b", "c", "d", "ab", "cd", "m", "s"),
           "ZZVC": ("e", "f", "ef"), "ZZQI": ("a", "b", "c", "d", "m", "s")}


def ac_long(path: Path) -> pl.DataFrame:
    """ZZAC samplerR -> (code, minute, col, r)."""
    d = pl.read_parquet(path)
    d = d.with_columns(pl.from_epoch(pl.col("rec_epoch").cast(pl.Int64), "s").dt.replace_time_zone("UTC").dt.convert_time_zone("Asia/Shanghai").alias("ts"))
    d = d.with_columns((pl.col("ts").dt.hour().cast(pl.Int64) * 60 + pl.col("ts").dt.minute()).alias("minute"), pl.col("sid").round(0).cast(pl.Int64).alias("code"))
    cols = [c for c in d.columns if c[:3] in ("as.", "bs.")]
    return d.unpivot(index=["code", "minute"], on=cols, variable_name="col", value_name="r")


def cr_long(path: Path) -> pl.DataFrame:
    """ZZCR samplerR -> (code, col, r)."""
    d = pl.read_parquet(path).with_columns(pl.col("sid").round(0).cast(pl.Int64).alias("code"))
    cols = [c for c in d.columns if "." in c]
    return d.unpivot(index=["code"], on=cols, variable_name="col", value_name="r")


def score(j: pl.DataFrame) -> pl.DataFrame:
    """Agreement of x (target) with r (samplerR) per (V, col)."""
    x, r = pl.col("x"), pl.col("r")
    j = j.filter(x.is_finite() & r.is_finite())
    return j.group_by("V", "col").agg(pl.len().alias("n"), ((x - r).abs() <= 1e-6 * x.abs() + 1e-9).mean().alias("exact"),
                                      (x / r).filter(r != 0).median().alias("ratio"), pl.corr(x, r, method="spearman").alias("rho"))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--hstats", type=Path, default=Path("/data/prod/hstats/HS300"))
    ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4"))
    a = ap.parse_args()
    d = a.date
    pm_days = sorted(p.name for p in (a.hstats / "ZZAC").iterdir() if p.name.isdigit())
    nxt = [x for x in pm_days if x > d]
    srcs = [("AC", d, a.hstats / "ZZAC" / d / "samplerR.parquet"), ("CR", d, a.hstats / "ZZCR" / d / "samplerR.parquet")]
    if nxt:
        srcs += [("AC", nxt[0], a.hstats / "ZZAC" / nxt[0] / "samplerR.parquet"), ("CR", nxt[0], a.hstats / "ZZCR" / nxt[0] / "samplerR.parquet")]
    rows = []
    for fam, vs in TARGETS.items():
        f = a.exports / f"{fam}_{d}.parquet"
        if not f.exists():
            continue
        pm, vals, kind = pm_frame(f, None, None, None, None, None)
        pm = pm.filter(pl.col("V").is_in(vs)).rename({vals[0]: "x"}).select("code", "minute", "V", "x")
        for kind_r, day_r, path in srcs:
            if not path.exists():
                continue
            if kind_r == "CR":
                j = pm.join(cr_long(path), on="code")
                rows.append(score(j).with_columns(pl.lit(fam).alias("file"), pl.lit(f"ZZCR/{day_r}").alias("source"), pl.lit(0).alias("shift")))
                continue
            ac = ac_long(path)
            for sh in (0, -1, 1, -5, 5) if kind == "time" else (0,):
                if kind == "time":
                    j = pm.with_columns(pl.col("minute") + sh).join(ac, on=["code", "minute"])
                else:  # a daily file against ZZAC's last epoch
                    j = pm.drop("minute").join(ac.filter(pl.col("minute") == ac["minute"].max()).drop("minute"), on="code")
                if j.height:
                    rows.append(score(j).with_columns(pl.lit(fam).alias("file"), pl.lit(f"ZZAC/{day_r}").alias("source"), pl.lit(sh).alias("shift")))
    if not rows:
        print("nothing to compare")
        return
    t = pl.concat(rows).sort("file", "V", "exact", "rho", descending=[False, False, True, True])
    t.write_csv(a.out / f"sampler_pm_{d}.csv")
    best = t.group_by("file", "V", maintain_order=True).head(3)
    with pl.Config(tbl_rows=200, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print(f"== {d}: samplerS size-like columns against PM's samplerR quantiles (exact = share of equal cells)")
        print(best.select("file", "V", "source", "col", "shift", "n", "exact", "ratio", "rho"))


if __name__ == "__main__":
    main()
