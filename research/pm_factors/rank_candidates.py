r"""Stage 4: score every candidate formula against the PM's samplerS, column by column.

Run on the intranet after exporting the PM file with ``research/compare_pm.py --export`` (the
only place xarray is allowed) and building our candidates for the same days with
``research/pm_factors/candidates.py``::

    python research/compare_pm.py --pm /work/prod/hstats/sod/HS300/ZZUG/20260430/samplerS.nc \\
        --export data/pm_export/ZZUG_sod_20260430.parquet
    python research/pm_factors/rank_candidates.py --family ZZUG \\
        --pm data/pm_export/ZZUG_sod_20260430.parquet --out data/reports/stage4/ZZUG.parquet

The PM cube is ``(S, D, I, V) -> value``. Keys are normalised before joining: ``S`` -> the
6-digit code as an int (numbers or strings like ``000001.SZ``), ``D`` -> yyyymmdd,
``I`` -> minute of day. ``I`` is either a time (used as is) or a position ``0..n-1``, in which
case every slot grid of length ``n`` in :data:`GRIDS` is tried, each shifted by -1/0/+1 minute
(snapshot-state vs flow conventions differ by one bar).

Per (PM value column, V, candidate, our variant, grid, shift) it reports:

* ``n``            matched cells
* ``exact``        share equal to 1e-6 relative -- a formula match, units included
* ``ratio``        median PM / ours -- 0.01 means the PM counts lots, 1e4 CNY-in-10k, ...
* ``exact_scaled`` share equal to 1e-4 after dividing by ``ratio`` (same formula, other unit)
* ``spearman``     rank correlation over all cells
* ``within``       mean per-stock time-series correlation (1-minute families): immune to any
  per-stock normalisation (z-scoring, dividing by the stock's own average), so a normalised
  PM column still points at the right raw formula

The best rows per PM column are printed; the whole table goes to ``--out``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

DATA = Path("/work/crucible_data")
CAND = DATA / "store" / "pm_factors" / "candidates"

_C237 = list(range(571, 691)) + list(range(781, 898))  # 09:31-11:30, 13:01-14:57 (right labels)
_C240 = list(range(571, 691)) + list(range(781, 901))  # + 14:58-15:00
_F48 = list(range(575, 691, 5)) + list(range(785, 901, 5))  # 5-minute right labels
GRIDS: dict[str, list[int]] = {
    "236_drop_last": _C237[:-1],
    "236_drop_first": _C237[1:],
    "237_continuous": _C237,
    "238_plus_close": [*_C237, 900],
    "240_full": _C240,
    "241_open_own": [570, *_C240],
    "48_5min": _F48,
}


def _find(cols: list[str], name: str | None, default: str) -> str | None:
    if name:
        return name
    for c in cols:
        if c.lower() == default.lower():
            return c
    return None


def _code(e: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    if dtype.is_numeric():
        return e.cast(pl.Float64).round(0).cast(pl.Int64)
    return e.cast(pl.String).str.extract(r"(\d{6})").cast(pl.Int64)


def _date(e: pl.Expr, dtype: pl.DataType) -> pl.Expr:
    if dtype.is_temporal():
        return e.dt.strftime("%Y%m%d").cast(pl.Int64)
    if dtype.is_numeric():
        return e.cast(pl.Int64)
    return e.cast(pl.String).str.replace_all(r"\D", "").str.slice(0, 8).cast(pl.Int64)


def pm_frame(path: Path, s: str | None, d: str | None, i: str | None, v: str | None,
             values: list[str] | None) -> tuple[pl.DataFrame, list[str], str]:
    """PM export -> (code, date?, I, V, value columns); returns frame, value columns, I kind."""
    pm = pl.read_parquet(path)
    cols = pm.columns
    s, d, i, v = _find(cols, s, "S"), _find(cols, d, "D"), _find(cols, i, "I"), _find(cols, v, "V")
    if s is None or v is None:
        raise SystemExit(f"cannot find S / V among {cols}; pass --s/--v")
    dims = [c for c in (s, d, i, v) if c]
    vals = values or [c for c, t in pm.schema.items() if c not in dims and t.is_numeric()]
    out = pm.select(
        _code(pl.col(s), pm.schema[s]).alias("code"),
        *([_date(pl.col(d), pm.schema[d]).alias("date")] if d else []),
        *([pl.col(i).alias("I")] if i else [pl.lit(0).alias("I")]),
        pl.col(v).cast(pl.String).str.strip_chars().alias("V"),
        *[pl.col(c).cast(pl.Float64) for c in vals],
    )
    it = out.schema["I"]
    if it.is_temporal():
        out = out.with_columns((pl.col("I").dt.hour().cast(pl.Int32) * 60 + pl.col("I").dt.minute()).alias("minute"))
        kind = "time"
    elif out["I"].n_unique() == 1:
        out = out.with_columns(pl.lit(-1).alias("minute"))
        kind = "daily"
    elif it.is_integer() and out["I"].min() >= 900 and out["I"].max() <= 1500:
        out = out.with_columns(((pl.col("I") // 100) * 60 + pl.col("I") % 100).alias("minute"))
        kind = "hhmm"
    else:
        kind = "position"
    return out, vals, kind


def score(j: pl.DataFrame, value: str, by: list[str], within: bool) -> pl.DataFrame:
    """Agreement metrics of PM ``value`` vs our ``ours`` per ``by`` group."""
    a, b = pl.col(value), pl.col("ours")
    j = j.filter(a.is_not_null() & b.is_not_null() & a.is_finite() & b.is_finite())
    base = j.group_by(by).agg(
        pl.len().alias("n"),
        ((a - b).abs() <= 1e-6 * a.abs().clip(1e-12, None) + 1e-12).mean().alias("exact"),
        (a / b).filter(b != 0).median().alias("ratio"),
        pl.corr(a, b, method="spearman").alias("spearman"),
        pl.corr(a, b).alias("pearson"),
    )
    base = base.join(
        j.join(base.select(*by, "ratio"), on=by).group_by(by).agg(
            ((a - b * pl.col("ratio")).abs() <= 1e-4 * a.abs().clip(1e-12, None) + 1e-12)
            .mean().alias("exact_scaled")),
        on=by, how="left")
    if within:
        ws = (j.group_by(*by, "code", "date").agg(pl.corr(a, b).alias("c"))
              .group_by(by).agg(pl.col("c").filter(pl.col("c").is_finite()).mean().alias("within")))
        base = base.join(ws, on=by, how="left")
    else:
        base = base.with_columns(pl.lit(None, pl.Float64).alias("within"))
    return base


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", required=True)
    ap.add_argument("--pm", type=Path, required=True, help="export from compare_pm.py --export")
    ap.add_argument("--ours", type=Path, help="default: store/pm_factors/candidates/family=<F>")
    ap.add_argument("--s"), ap.add_argument("--d"), ap.add_argument("--i"), ap.add_argument("--v")
    ap.add_argument("--values", help="PM value columns, comma-separated (default: all numeric)")
    ap.add_argument("--dates", help="our dates to use when the PM file has no date dimension")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()

    fam = a.family.upper()
    pm, vals, kind = pm_frame(a.pm, a.s, a.d, a.i, a.v, a.values.split(",") if a.values else None)
    root = a.ours or CAND / f"family={fam}"
    ours = pl.read_parquet(root / "date=*" / "part.parquet", hive_partitioning=False).select(
        pl.col("sid").round(0).cast(pl.Int64).alias("code"), "date",
        pl.col("minute").cast(pl.Int64), "cand", pl.col("V").alias("VV"), pl.col("value").alias("ours"),
    ).with_columns(pl.col("VV").str.split("@").list.first().alias("V"))
    has_date = "date" in pm.columns
    if not has_date:
        if not a.dates:
            raise SystemExit("PM file has no date dimension: pass --dates (our day(s) to compare)")
        ours = ours.filter(pl.col("date").is_in([int(x) for x in a.dates.split(",")]))
        pm = pm.join(ours.select("date").unique(), how="cross")

    pv, ov = set(pm["V"].unique()), set(ours["V"].unique())
    print(f"PM {a.pm.name}: {pm.height:,} cells, value columns {vals}, I = {kind} "
          f"({pm['I'].n_unique()} slots), stocks {pm['code'].n_unique()}, "
          f"dates {sorted(pm['date'].unique().to_list())[:8]}")
    print(f"ours {fam}: dates {sorted(ours['date'].unique().to_list())}, candidates "
          f"{sorted(ours['cand'].unique().to_list())}")
    print(f"PM columns without a candidate: {sorted(pv - ov) or 'none'}; "
          f"candidate columns not in PM: {sorted(ov - pv) or 'none'}")
    ours = ours.filter(pl.col("V").is_in(list(pv)))

    if kind == "position":
        n = pm["I"].n_unique()
        grids = {g: m for g, m in GRIDS.items() if len(m) == n}
        if not grids:
            raise SystemExit(f"no slot grid of length {n}; add one to GRIDS (have "
                             f"{ {g: len(m) for g, m in GRIDS.items()} })")
        maps = []
        for g, m in grids.items():
            for sh in (-1, 0, 1):
                idx = pl.DataFrame({"I": list(range(n)), "minute": [x + sh for x in m]},
                                   schema={"I": pm.schema["I"], "minute": pl.Int64})
                maps.append((g, sh, idx))
    else:
        maps = [(kind, 0, None)]

    by = ["value_col", "V", "cand", "VV", "grid", "shift"]
    res = []
    for g, sh, idx in maps:
        p = pm if idx is None else pm.join(idx, on="I")
        keys = ["code", "date", "minute", "V"] if kind != "daily" else ["code", "date", "V"]
        j = p.join(ours if kind != "daily" else ours.drop("minute"), on=keys)
        if j.height == 0:
            continue
        for c in vals:
            r = score(j.with_columns(pl.lit(c).alias("value_col"), pl.lit(g).alias("grid"),
                                     pl.lit(sh).alias("shift")), c, by, within=kind != "daily")
            res.append(r)
    if not res:
        raise SystemExit("nothing joined: check stock codes / dates / slots printed above")
    # a formula match (>= 90 % of cells equal up to a unit) first; below that the match rates are
    # noise (a normalised PM column), so rank by correlation
    table = pl.concat(res).with_columns(
        pl.max_horizontal(pl.col("spearman").abs(), pl.col("within").abs().fill_null(0)).alias("score"),
        (pl.col("exact_scaled") >= 0.9).alias("match"),
    ).sort(["value_col", "V", "match", "exact", "score"], descending=[False, False, True, True, True])

    # whole-family verdict: per candidate and slot grid, the best variant of every PM column,
    # averaged over the columns -- single columns tie (e.g. the >=q98 tier is in several schemes)
    fam_rank = (table.group_by("value_col", "cand", "grid", "shift", "V")
                .agg(pl.col("exact_scaled").max(), pl.col("score").max())
                .group_by("value_col", "cand", "grid", "shift")
                .agg(pl.len().alias("columns"), pl.col("exact_scaled").mean(), pl.col("score").mean())
                .sort("value_col", "exact_scaled", "score", descending=[False, True, True]))
    # per PM column: each candidate variant at its own best grid / shift
    per_v = table.group_by("value_col", "V", "cand", "VV", maintain_order=True).first()
    with pl.Config(tbl_rows=40, tbl_width_chars=200, float_precision=4):
        print("\nfamily verdict (mean over PM columns of the best variant; 1.0 exact_scaled = formula "
              "reproduced up to a unit):")
        print(fam_rank.group_by("value_col", maintain_order=True).head(8))
        print(f"\ntop {a.top} per PM column (formula match first, then exact, then correlation):")
        print(per_v.group_by("value_col", "V", maintain_order=True).head(a.top)
              .select("value_col", "V", "cand", "VV", "grid", "shift", "n", "exact", "exact_scaled",
                      "ratio", "spearman", "within"))
        print("\nnote: grids that differ only by a one-minute shift at the ends score almost alike "
              "(e.g. 236_drop_first/-1 vs 236_drop_last/0); decide those on 'exact'.")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        table.write_parquet(a.out)
        print(f"\nfull table: {a.out} ({table.height:,} rows)")


if __name__ == "__main__":
    main()
