r"""Self-consistency audit of our whole filled matrix against PM's tree counts: which columns disagree with PM's trees?

Every internal node of PM's trees stores how many training rows reached it and how many went left. Send OUR rows
(the filled scoring matrix, NaN -> 0 as PM's missing type 'none' does) down every tree: at each node we get our row count
and our left share. A column whose values are wrong makes every node that splits on it, and every node below it, take
the wrong share -- across many different partners at once -- while a right column only carries the market-regime shift
that hits every column alike. Per column: count-weighted mean |our left share - PM's| at the nodes splitting on it
('own'), and at the nodes directly below them ('child'). The 4 PM-state columns calibrate the scale (they should sit in
the low end); settled stand-ins far above them are the ones to re-derive first. Per-day so stable columns can be told
from day effects. Read-only.

    python research/featgen/node_audit.py --days 20260430,20261112,20261113 --x-dir "data/reports/stage6_r99_{mmdd}" \
        --models /work/crucible_data/models_gen/B --out data/reports/node_audit
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pred_day as P  # noqa: E402
from col_probe import trees  # noqa: E402
from invert_cols import BASELINE  # noqa: E402
from model_atlas import SUBS, settled_formulas  # noqa: E402

PM_STATE = {"zzup", "zzvx.a", "zzxu.a", "zzxu.b"}


def audit(files: list[Path], X: np.ndarray, min_rows: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per column: weighted |share error| sums and weights at own-split nodes and at their child nodes."""
    nc = X.shape[1]
    own_e, own_w, ch_e, ch_w = (np.zeros(nc) for _ in range(4))
    Z = np.nan_to_num(X, nan=0.0)
    for f in files:
        for kv in trees(f):
            sf = [int(x) for x in kv["split_feature"]]
            th = [float(x) for x in kv["threshold"]]
            dt = [int(x) for x in kv["decision_type"]]
            lc, rc = [int(x) for x in kv["left_child"]], [int(x) for x in kv["right_child"]]
            ic = [float(x) for x in kv.get("internal_count", [])]
            lf = [float(x) for x in kv.get("leaf_count", [])]
            if not ic:
                continue

            def cnt(c: int) -> float:
                return ic[c] if c >= 0 else lf[~c]

            stack = [(0, np.arange(Z.shape[0]), -1)]
            while stack:
                n, rows, parent_feat = stack.pop()
                if dt[n] & 1 or ic[n] <= 0:
                    continue
                j = sf[n]
                go = Z[rows, j] <= th[n]
                if rows.size >= min_rows:
                    e = abs(go.mean() - cnt(lc[n]) / ic[n])
                    w = np.sqrt(ic[n])
                    own_e[j] += w * e
                    own_w[j] += w
                    if parent_feat >= 0:
                        ch_e[parent_feat] += w * e
                        ch_w[parent_feat] += w
                for c, sub in ((lc[n], rows[go]), (rc[n], rows[~go])):
                    if c >= 0 and sub.size:
                        stack.append((c, sub, j))
    return own_e, own_w, ch_e, ch_w


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--x-dir", required=True, help="dir with x.parquet (filled scoring matrix), {mmdd} / {date} allowed")
    ap.add_argument("--models", type=Path, default=Path("/work/crucible_data/models_gen/B"))
    ap.add_argument("--fit", default="livev1_fit1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/work/crucible_data/prod_models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--rows", type=int, default=150000)
    ap.add_argument("--min-rows", type=int, default=300)
    ap.add_argument("--out", type=Path, default=Path("data/reports/node_audit"))
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    formulas = settled_formulas(a.formulas)
    a.out.mkdir(parents=True, exist_ok=True)
    tabs = []
    for d in [x for x in a.days.split(",") if x]:
        xp = Path(a.x_dir.format(mmdd=d[4:], date=d)) / "x.parquet"
        x = pl.read_parquet(xp)
        if "date" in x.columns:
            x = x.filter(pl.col("date") == int(d))
        X = np.full((x.height, len(names)), np.nan)
        for i, n in enumerate(names):
            if n in x.columns:
                X[:, i] = x[n].cast(pl.Float64).to_numpy()
        if X.shape[0] > a.rows:
            X = X[np.sort(np.random.default_rng(0).choice(X.shape[0], a.rows, replace=False))]
        files = P.model_files(a.models, d, [a.fit], SUBS)
        oe, ow, ce, cw = audit(files, X, a.min_rows)
        with np.errstate(invalid="ignore", divide="ignore"):
            t = pl.DataFrame({"col": names, "own": oe / ow, "own_w": ow, "child": ce / cw, "child_w": cw,
                              "filled": np.isfinite(X).mean(axis=0), "date": [d] * len(names)})
        tabs.append(t)
        st = t.with_columns(pl.col("col").map_elements(lambda c: "pm-state" if c in PM_STATE else "settled" if c in formulas else "open", return_dtype=pl.String).alias("kind"))
        ref = st.filter(pl.col("kind") == "pm-state")["own"].drop_nans().drop_nulls()
        print(f"== {d}: {X.shape[0]:,} rows, {len(files)} model files ({files[0].parent.parent.parent.name if files else '-'})")
        print(f"   pm-state own error: {' '.join(f'{r['col']}={r['own']:.4f}' for r in st.filter(pl.col('kind') == 'pm-state').iter_rows(named=True))}")
        for kind in ("settled", "open"):
            sub = st.filter((pl.col("kind") == kind) & pl.col("own").is_not_nan() & (pl.col("own_w") > 0))
            print(f"   {kind}: {sub.height} cols, own error q10/50/90 {' '.join(f'{x:.4f}' for x in np.quantile(sub['own'].to_numpy(), [.1, .5, .9]))}"
                  + (f"; above the worst pm-state ({ref.max():.4f}): {sub.filter(pl.col('own') > ref.max()).height}" if ref.len() else ""))
        print("   settled, worst 20 by own error (filled share, child error):")
        for r in st.filter((pl.col("kind") == "settled") & pl.col("own").is_not_nan()).sort("own", descending=True).head(20).iter_rows(named=True):
            print(f"     {r['col']:<14} own {r['own']:.4f} child {r['child']:.4f} filled {r['filled']:.2f}  {formulas.get(r['col'], '')[:70]}")
    allt = pl.concat(tabs)
    allt.write_csv(a.out / "node_audit.csv")
    if len(tabs) > 1:
        agg = allt.group_by("col").agg(pl.col("own").mean().alias("own_mean"), pl.col("own").std().alias("own_sd"), pl.col("own").min().alias("own_min"))
        agg = agg.with_columns(pl.col("col").map_elements(lambda c: "pm-state" if c in PM_STATE else "settled" if c in formulas else "open", return_dtype=pl.String).alias("kind"))
        print("== across days: settled columns whose error is high on EVERY day (own_min), worst 25:")
        for r in agg.filter(pl.col("kind") == "settled").sort("own_min", descending=True).head(25).iter_rows(named=True):
            print(f"   {r['col']:<14} min {r['own_min']:.4f} mean {r['own_mean']:.4f} sd {r['own_sd']:.4f}  {formulas.get(r['col'], '')[:70]}")
        print("   pm-state: " + " ".join(f"{r['col']} min {r['own_min']:.4f}" for r in agg.filter(pl.col("kind") == "pm-state").iter_rows(named=True)))
    print(f"[done] csv {a.out / 'node_audit.csv'}")


if __name__ == "__main__":
    main()
