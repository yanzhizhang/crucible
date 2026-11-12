r"""A look at runS's raw numbers on one day, to read the rule off them: pm_orders' decisions (with the alg.cpp evaluation
line mp fa vpen vr poscst cpos asq hs folded in) and its census examples. Read-only; prints only.

* every column's quantiles and zero share;
* 30 decision rows: 10 that produced an order (nullptr 0), 10 with executp far from 0, 10 random;
* candidate identities, each as the share of rows where it holds to 1e-6 (relative): cpos == pos, poscst == pos * mp,
  executp == pos, executp == -pos, executp == cpos - pos, asq == sellable-like, lthr == -hs, lthr == -k * hs ...;
* correlations of executp with every other column.

    python research/pm_factors/runs_sample.py --orders /work/crucible_data/pm_orders --date 20261112
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    d = pl.read_parquet(a.orders / f"date={a.date}" / "decisions.parquet")
    num = [c for c, t in d.schema.items() if t.is_numeric()]
    print(f"{d.height:,} decisions, columns {d.columns}")
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, float_precision=6):
        q = pl.DataFrame([{"col": c, **{f"p{p}": float(np.nanpercentile(d[c].to_numpy().astype(float), p)) for p in (1, 10, 50, 90, 99)},
                           "zero": float((d[c] == 0).mean()), "int": float((d[c] == d[c].round(0)).mean())} for c in num])
        print(q)
        show = [c for c in ("sid", "src_epoch", "fa", "ev_mp", "ev_vpen", "ev_vr", "ev_poscst", "ev_cpos", "ev_asq", "ev_hs", "pos", "pp", "executp",
                            "asq", "lthr", "hs", "liid", "winsdQty", "asf", "nullptr") if c in d.columns]
        if "nullptr" in d.columns:
            print("\n-- decisions that produced an order (nullptr 0)")
            print(d.filter(pl.col("nullptr") == 0).select(show).head(10))
        print("\n-- |executp| largest")
        print(d.sort(pl.col("executp").abs(), descending=True).select(show).head(10))
        print("\n-- random")
        print(d.sample(10, seed=1).select(show))
    def holds(a_: pl.Expr, b_: pl.Expr) -> float:
        return float(d.select(((a_ - b_).abs() <= 1e-6 * (b_.abs() + 1e-9)).mean()).item())
    c = pl.col
    tests = {"ev_cpos == pos": (c("ev_cpos"), c("pos")), "ev_asq == asq": (c("ev_asq"), c("asq")), "ev_hs == hs": (c("ev_hs"), c("hs")),
             "ev_fa == fa": (c("ev_fa"), c("fa")), "ev_poscst == pos*mp": (c("ev_poscst"), c("pos") * c("ev_mp")),
             "ev_poscst == cpos*mp": (c("ev_poscst"), c("ev_cpos") * c("ev_mp")), "executp == pos": (c("executp"), c("pos")),
             "executp == -pos": (c("executp"), -c("pos")), "lthr == -hs": (c("lthr"), -c("hs")), "lthr == -2hs": (c("lthr"), -2 * c("hs")),
             "lthr == 0": (c("lthr"), pl.lit(0.0)), "executp == 0": (c("executp"), pl.lit(0.0))}
    print("\n-- identities (share of rows)")
    for k, (x, y) in tests.items():
        try:
            print(f"  {k:<24} {holds(x, y):.4f}")
        except Exception as e:  # noqa: BLE001 -- a column absent on this day
            print(f"  {k:<24} n/a ({str(e)[:40]})")
    if "executp" in d.columns:
        e = d["executp"].to_numpy().astype(float)
        print("\n-- corr(executp, x)")
        for col in num:
            if col == "executp":
                continue
            x = d[col].to_numpy().astype(float)
            m = np.isfinite(x) & np.isfinite(e)
            if m.sum() > 100 and x[m].std() > 0:
                print(f"  {col:<12} {np.corrcoef(x[m], e[m])[0, 1]:+.3f}")


if __name__ == "__main__":
    main()
