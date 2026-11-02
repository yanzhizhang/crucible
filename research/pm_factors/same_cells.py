r"""Which samplerS cells are carried unchanged from the previous day, and whether that follows time, stock, chance or our data.

Per (file, V) over consecutive export days: the share of cells with pm_d == pm_prev (same), its spread over bars of the
day and over stocks (min / max of the per-bar and per-stock shares), the persistence P(same_d | same_{d-1}) against the
base rate (1 = the same cells every day, base = independent draws), and, when the basics table has the day, the same
share in bars where our data has no trade vs at least one trade.

    python research/pm_factors/same_cells.py --dates 20260923,20260924,20260929,20260930,20261009
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from basic_match import BASICS
from rank_candidates import pm_frame


def load(exports: Path, name: str, d: str) -> tuple[pl.DataFrame, str] | None:
    f = exports / f"{name}_{d}.parquet"
    if not f.exists():
        return None
    pm, vals, kind = pm_frame(f, None, None, None, None, None)
    return pm.select("code", "minute", "V", pl.col(vals[0]).alias("x")), kind


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
    ap.add_argument("--basics", type=Path, default=BASICS)
    a = ap.parse_args()
    dates = sorted(x for x in a.dates.split(",") if x)
    names = sorted({f.stem.rsplit("_", 1)[0] for f in a.exports.glob("*.parquet")})
    out = []
    for name in names:
        frames = {d: r for d in dates if (r := load(a.exports, name, d))}
        days = sorted(frames)
        if len(days) < 2:
            continue
        kind = frames[days[0]][1]
        table = "m5" if kind == "time" else "d1"
        pairs = []
        for d0, d1 in itertools.pairwise(days):
            j = (frames[d1][0].join(frames[d0][0].rename({"x": "xp"}), on=["code", "minute", "V"])
                 .filter(pl.col("x").is_finite() & pl.col("xp").is_finite())
                 .with_columns((pl.col("x") == pl.col("xp")).alias("same"), pl.lit(d1).alias("day")))
            g = a.basics / f"date={d1}" / f"{table}.parquet"
            if g.exists():
                b = pl.read_parquet(g, columns=["symbol_id", "minute", "n"]).rename({"symbol_id": "code", "n": "trades"}).with_columns(
                    pl.col("code").cast(pl.Int64), pl.col("minute").cast(pl.Int64))
                j = j.join(b.drop("minute") if kind != "time" else b, on=["code"] if kind != "time" else ["code", "minute"], how="left")
            else:
                j = j.with_columns(pl.lit(None, pl.Float64).alias("trades"))
            pairs.append(j)
        t = pl.concat(pairs)
        for V in sorted(t["V"].unique().to_list()):
            s = t.filter(pl.col("V") == V)
            base = float(s["same"].mean())
            by_bar = s.group_by("minute").agg(pl.col("same").mean())["same"]
            by_stock = s.group_by("code").agg(pl.col("same").mean())["same"]
            # persistence: same on day d given same on the previous pair, over cells present in both pairs
            per = (s.select("code", "minute", "day", "same").sort("day").group_by("code", "minute")
                   .agg(pl.col("same").alias("seq")).with_columns(pl.col("seq").list.len().alias("k")).filter(pl.col("k") >= 2))
            cond = None
            if per.height:
                seqs = per["seq"].to_list()
                num = sum(1 for q in seqs for i in range(1, len(q)) if q[i - 1] and q[i])
                den = sum(1 for q in seqs for i in range(1, len(q)) if q[i - 1])
                cond = num / den if den else None
            tr = s.filter(pl.col("trades").is_not_null())
            no_tr = tr.filter(pl.col("trades") == 0)["same"].mean() if tr.height else None
            has_tr = tr.filter(pl.col("trades") > 0)["same"].mean() if tr.height else None
            out.append({"file": name, "V": V, "pairs": len(pairs), "same": base, "bar_min": float(by_bar.min()), "bar_max": float(by_bar.max()),
                        "stock_min": float(by_stock.min()), "stock_max": float(by_stock.max()), "p_same_given_same": cond,
                        "same_no_trade": no_tr, "same_with_trade": has_tr, "zero_trade_share": float((tr["trades"] == 0).mean()) if tr.height else None})
    with pl.Config(tbl_rows=300, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print("== cells carried unchanged (pm_d == pm_prev): base share, spread over bars and stocks, persistence, vs our trade count in the bar")
        print(pl.DataFrame(out))


if __name__ == "__main__":
    main()
