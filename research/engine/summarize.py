r"""Fold the per-day engine-vs-PM tables (verify.py --csv) into one: mean share per key over the days, keys not at 1.0 only.

    python research/engine/summarize.py data/reports/engine/verify_*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl


def main() -> None:
    """CLI entry point."""
    files = [Path(x) for x in sys.argv[1:] if Path(x).exists()]
    if not files:
        print("no engine tables")
        return
    t = pl.concat([pl.read_csv(f).with_columns(pl.lit(f.stem).alias("day")) for f in files], how="diagonal_relaxed")
    agg = (t.group_by("key").agg(pl.len().alias("days"), pl.col("all").mean(), pl.col("SZSE").mean(), pl.col("all_1e4").mean(),
                                 pl.col("SZSE_1e4").mean(), pl.col("mre").median())
           .filter((pl.col("all") < 0.999) | (pl.col("SZSE") < 0.999) | (pl.col("all_1e4") < 0.999))
           .with_columns(pl.col("key").str.split("|").list.first().alias("base"))
           .sort("base", "all", descending=[False, True]).drop("base"))
    with pl.Config(tbl_rows=200, tbl_width_chars=160, float_precision=4):
        print(f"== engine vs PM over {len(files)} day(s): mean share equal at 1e-6 / 1e-4 (all, SZSE), keys not at 1.0; variants under their key")
        print(agg)


if __name__ == "__main__":
    main()
