r"""gen_compare.csv: the columns flagged as moved, one row each (months moved, largest median-split shift, span ratio range).

    python research/pm_factors/gen_flags.py data/reports/gen_compare.csv
"""

from __future__ import annotations

import sys

import polars as pl


def main() -> None:
    d = pl.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data/reports/gen_compare.csv", schema_overrides={"month": pl.String})
    m = d.filter(pl.col("moved"))
    print(f"moved rows {m.height} of {d.height}; months {sorted(d['month'].unique().to_list())}")
    s = m.group_by("col", "ref_name").agg(pl.col("month").n_unique().alias("months"), pl.col("t50_shift").max(),
                                          pl.col("span_ratio").min().alias("span_min"), pl.col("span_ratio").max().alias("span_max"),
                                          pl.col("month").min().alias("first"), pl.col("month").max().alias("last"))
    with pl.Config(tbl_rows=-1, tbl_width_chars=200, float_precision=2):
        print(s.sort("months", descending=True))


if __name__ == "__main__":
    main()
