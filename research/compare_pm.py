r"""Compare one of our reproduced tables against the matching PM result, cell by cell.

Intranet only (reads PM's netCDF with xarray, one of two files allowed to). Both sides are
turned into long polars frames keyed by ``--keys``; our columns are renamed with ``--map``
(``ours=pm``). For every compared column it reports the exact-match rate, the rate within
``--atol``/``--rtol``, the max absolute error and a few mismatching rows; key coverage on
each side is reported too, so a join that silently matched nothing cannot pass.

Examples::

    # 1-minute bars vs PM 1min_src (and each pass p1..p10 via --pm .../p3/20260430.nc)
    python research/compare_pm.py --pm /work/prod/1min_src/20260430.nc \\
        --ours /work/crucible_data/store/bars/1min/date=20260430/part.parquet \\
        --keys symbol,ts --out data/reports/stage2/1min_src_root.parquet

    # dump a PM file to long parquet (samplerS -> pm_factors/rank_candidates.py)
    python research/compare_pm.py --pm /work/prod/hstats/sod/HS300/ZZUG/20260430/samplerS.nc \\
        --export data/pm_export/ZZUG_sod_20260430.parquet

    # samplerS of one family: PM (S, D, I, V) cube vs our long frame
    python research/compare_pm.py --pm /work/prod/hstats/sod/HS300/ZZAL/20260430/samplerS.nc \\
        --ours data/pm_factors/zzal_20260430.parquet --keys S,I,V
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def load_pm(path: Path) -> pl.DataFrame:
    """PM file -> long polars frame. netCDF variables become columns over their dims."""
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    import xarray as xr

    with xr.open_dataset(path) as ds:
        df = ds.to_dataframe().reset_index()
    out = pl.from_pandas(df)
    # fixed-width byte strings (NC_CHAR) arrive as bytes; decode for joins
    return out.with_columns(
        [pl.col(c).cast(pl.Binary).cast(pl.String) for c, t in out.schema.items() if t == pl.Binary]
    )


def compare(
    pm: pl.DataFrame,
    ours: pl.DataFrame,
    keys: list[str],
    *,
    atol: float,
    rtol: float,
    sample: int = 5,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Per-column agreement table and key-coverage counts."""
    for k in keys:
        if pm.schema[k] != ours.schema[k]:
            ours = ours.with_columns(pl.col(k).cast(pm.schema[k]))
    j = pm.join(ours, on=keys, how="full", suffix="_ours", coalesce=True)
    cov = {
        "pm_rows": pm.height,
        "ours_rows": ours.height,
        "matched_keys": pm.join(ours.select(keys), on=keys, how="semi").height,
        "only_pm": pm.join(ours.select(keys), on=keys, how="anti").height,
        "only_ours": ours.join(pm.select(keys), on=keys, how="anti").height,
    }
    rows = []
    for c in [c for c in ours.columns if c not in keys and c in pm.columns]:
        a, b = pl.col(c), pl.col(f"{c}_ours")
        both = j.filter(a.is_not_null() & b.is_not_null())
        n = both.height
        if n == 0:
            rows.append({"column": c, "n": 0})
            continue
        numeric = both.schema[c].is_numeric() and both.schema[f"{c}_ours"].is_numeric()
        if numeric:
            err = (a.cast(pl.Float64) - b.cast(pl.Float64)).abs()
            st = both.select(
                (a.cast(pl.Float64) == b.cast(pl.Float64)).mean().alias("exact"),
                (err <= atol + rtol * a.cast(pl.Float64).abs()).mean().alias("close"),
                err.max().alias("max_abs_err"),
            ).row(0, named=True)
            bad = both.filter(err > atol + rtol * a.cast(pl.Float64).abs())
        else:
            st = {"exact": both.select((a == b).mean()).item(), "close": None, "max_abs_err": None}
            bad = both.filter(a != b)
        rows.append(
            {
                "column": c,
                "n": n,
                **st,
                "n_bad": bad.height,
                "sample": json.dumps(
                    bad.select(*keys, c, f"{c}_ours").head(sample).rows(), default=str
                ),
            }
        )
    return pl.DataFrame(rows), cov


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pm", type=Path, required=True)
    ap.add_argument("--export", type=Path,
                    help="only write the PM file as a long parquet (for pm_factors/rank_candidates.py)")
    ap.add_argument("--ours", type=Path)
    ap.add_argument("--keys", help="comma-separated key columns (PM names)")
    ap.add_argument("--map", default="", help="ours=pm renames, comma-separated")
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.export:
        pm = load_pm(a.pm)
        a.export.parent.mkdir(parents=True, exist_ok=True)
        pm.write_parquet(a.export)
        print(f"{a.pm} -> {a.export}: {pm.height:,} rows")
        print(pm.schema)
        for c, t in pm.schema.items():
            if not t.is_float():
                print(f"  {c}: {pm[c].n_unique()} values, first {pm[c].unique().sort().head(5).to_list()}")
        return
    if a.ours is None or a.keys is None:
        ap.error("--ours and --keys are required unless --export is given")
    ours = pl.read_parquet(a.ours)
    if a.map:
        ours = ours.rename(dict(kv.split("=", 1) for kv in a.map.split(",")))
    table, cov = compare(load_pm(a.pm), ours, a.keys.split(","), atol=a.atol, rtol=a.rtol)
    print(json.dumps(cov))
    with pl.Config(tbl_rows=200, tbl_width_chars=200):
        print(table.drop("sample", strict=False))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        table.with_columns(
            pl.lit(json.dumps(cov)).alias("coverage"), pl.lit(str(a.pm)).alias("pm_file")
        ).write_parquet(a.out)


if __name__ == "__main__":
    main()
