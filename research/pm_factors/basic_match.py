r"""Stage 4: which basic tick quantity each PM sampler column is, up to a unit -- scored over every column of ``basics``.

For each PM export (``data/pm_export/<FILE>_<date>.parquet``) and column ``V``: joined to the basics table of its
granularity (1-minute / 5-minute / daily) on (stock, date, bar), at bar shifts 0 / +step / -step and against our
day D and D-1 (the ``sod`` convention), every basic column is scored as ``rank_candidates`` does: ``exact``
(equal to 1e-6), ``ratio`` (median PM / ours), ``exact_scaled`` (equal after dividing by the ratio), ``spearman``
and ``within`` (per-stock time-series correlation). Summed over the days given; the best rows per (file, V).

    python research/pm_factors/basic_match.py --dates 20260923,20260924 --out data/reports/stage4/basic_match.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rank_candidates import pm_frame, score

BASICS = Path("/work/crucible_data/store/pm_factors/basics")  # basics.OUT, without importing the DuckDB side


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
    ap.add_argument("--basics", type=Path, default=BASICS)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4/basic_match.csv"))
    a = ap.parse_args()
    dates = [x for x in a.dates.split(",") if x]
    days = sorted(p.name.split("=")[1] for p in a.basics.glob("date=*"))
    rows = []
    for d in dates:
        for f in sorted(a.exports.glob(f"*_{d}.parquet")):
            pm, vals, kind = pm_frame(f, None, None, None, None, None)
            if "date" not in pm.columns:
                pm = pm.with_columns(pl.lit(int(d)).alias("date"))
            step = 1
            if kind == "time":
                step = int(pm.select(pl.col("minute").unique().sort().diff().drop_nulls().min()).item() or 1)
                table = "m5" if step >= 5 else "m1"
            elif kind == "daily":
                table = "d1"
            else:
                print(f"{f.name}: I kind {kind} not handled")
                continue
            prev = [x for x in days if x < d]
            for off, our_day in ((0, d), (-1, prev[-1] if prev else None)):
                if our_day is None or not (a.basics / f"date={our_day}" / f"{table}.parquet").exists():
                    continue
                ours = pl.read_parquet(a.basics / f"date={our_day}" / f"{table}.parquet").with_columns(pl.lit(int(d)).alias("date"))
                ours = ours.rename({"symbol_id": "code"}).with_columns(pl.col("code").cast(pl.Int64), pl.col("minute").cast(pl.Int64))
                basics = [c for c in ours.columns if c not in ("code", "minute", "date")]
                for sh in ((0, step, -step) if kind == "time" else (0,)):
                    p = pm.with_columns(pl.col("minute") + sh) if kind == "time" else pm
                    keys = ["code", "date", "minute"] if kind == "time" else ["code", "date"]
                    j = p.join(ours if kind == "time" else ours.drop("minute"), on=keys)
                    if j.height == 0:
                        continue
                    for c in vals:
                        for bcol in basics:
                            jj = j.select(pl.col(c), pl.col(bcol).alias("ours"), "code", "date", pl.col("V"))
                            r = score(jj.with_columns(pl.lit(bcol).alias("basic")), c, ["V", "basic"], within=kind == "time")
                            rows.append(r.with_columns(pl.lit(f.stem.rsplit("_", 1)[0]).alias("file"), pl.lit(c).alias("value_col"),
                                                       pl.lit(sh).alias("shift"), pl.lit(off).alias("dayoff"), pl.lit(d).alias("day")))
            print(f"{d} {f.stem.rsplit('_', 1)[0]}: {kind}, {len(vals)} value columns, {pm['V'].n_unique()} V")
    if not rows:
        print("nothing scored")
        return
    t = pl.concat(rows, how="diagonal_relaxed")
    agg = (t.group_by("file", "value_col", "V", "basic", "shift", "dayoff").agg(
                pl.len().alias("days"), pl.col("n").sum(), pl.col("exact").mean(), pl.col("exact_scaled").mean(),
                pl.col("ratio").median(), pl.col("spearman").mean(), pl.col("within").mean())
           .with_columns(pl.max_horizontal(pl.col("spearman").abs(), pl.col("within").abs().fill_null(0)).alias("score"))
           .sort("file", "V", "exact_scaled", "score", descending=[False, False, True, True]))
    agg.write_csv(a.out)
    best = agg.group_by("file", "value_col", "V", maintain_order=True).head(a.top)
    with pl.Config(tbl_rows=400, tbl_cols=-1, tbl_width_chars=220, float_precision=3):
        print(f"\n== PM column -> basic quantity, best {a.top} per column over {len(dates)} day(s) (exact_scaled 1.0 = that basic up to a unit)")
        print(best.select("file", "V", "basic", "shift", "dayoff", "days", "exact_scaled", "exact", "ratio", "spearman", "within"))
    print(f"table: {a.out}")


if __name__ == "__main__":
    main()
