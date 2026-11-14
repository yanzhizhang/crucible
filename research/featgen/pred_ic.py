r"""Does the prediction predict? Cross-sectional rank IC of a score parquet against forward mid-price returns.

For each (sid, mark) of the scores: the mid (bid1 + ask1) / 2 of the latest snapshot received at the mark and at mark + h
(trading clock, receive time, as ic_score's marks), r = log(mid_h / mid_0). Per mark the Spearman correlation of the
score with r over the stocks, then the mean over marks, its t-stat, and the share of marks with IC > 0 -- for each
horizon. Several score sources side by side: our C++ chain (backfill pred_<D>.parquet: sid, mark, pred) and PM's own
prediction where one exists (a target parquet: sid, t, prediction -- its latest value at each mark).

    CRUCIBLE_SOURCE=dysl2 python research/featgen/pred_ic.py --date 20261111 --scores data/backfill_r80/days/20261111/pred/pred_20261111.parquet \
        --target /work/crucible_data/pm_target_r22/date=20261111.parquet --every 30
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pm_factors"))
import dcp_build as B  # noqa: E402


def mids(con: duckdb.DuckDBPyConnection, date: int, sids: list[int]) -> pl.DataFrame:
    con.register("want", pl.DataFrame({"sid": sids}))
    df = con.sql(f"""SELECT symbol_id AS sid, {B.tc(B.LSOD)}::DOUBLE AS t, (q.bid_px[1] + q.ask_px[1]) / 2.0 AS mid
                     FROM raw_quotation q JOIN want w ON w.sid = q.symbol_id
                     WHERE q.date = {date} AND q.bid_px[1] > 0 AND q.ask_px[1] > 0 AND {B.LSOD} >= 34200 ORDER BY sid, t""").pl()
    con.unregister("want")
    return df.with_columns(pl.col("sid").cast(pl.Int64))


def at(m: pl.DataFrame, keys: pl.DataFrame, shift: float) -> np.ndarray:
    k = keys.with_columns((pl.col("mark") + shift).alias("tq")).with_row_index("i").sort("tq")
    j = k.join_asof(m.sort("t"), left_on="tq", right_on="t", by="sid", strategy="backward").sort("i")
    return j["mid"].to_numpy().astype(float)


def ic_table(keys: pl.DataFrame, score: np.ndarray, rets: dict[int, np.ndarray]) -> dict[int, tuple[float, float, float, int]]:
    out = {}
    marks = keys["mark"].to_numpy()
    for h, r in rets.items():
        ics = []
        for mk in np.unique(marks):
            m = (marks == mk) & np.isfinite(score) & np.isfinite(r)
            if m.sum() >= 30:
                a, b = pl.Series(score[m]).rank().to_numpy(), pl.Series(r[m]).rank().to_numpy()
                if a.std() > 0 and b.std() > 0:
                    ics.append(float(np.corrcoef(a, b)[0, 1]))
        x = np.array(ics)
        out[h] = (float(x.mean()) if x.size else float("nan"), float(x.mean() / x.std() * np.sqrt(x.size)) if x.size > 2 and x.std() > 0 else float("nan"),
                  float(np.mean(x > 0)) if x.size else float("nan"), x.size)
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--scores", type=Path, required=True, help="sid, mark, pred (the C++ chain's pred_<D>.parquet)")
    ap.add_argument("--target", type=Path, help="PM's prediction of the day: sid, t, prediction")
    ap.add_argument("--catalog", type=Path, default=B.CATALOG)
    ap.add_argument("--horizons", default="30,60,300,900", help="seconds on the trading clock")
    ap.add_argument("--every", type=float, default=30.0, help="score only marks that are multiples of this (s), to keep it quick")
    a = ap.parse_args()
    s = pl.read_parquet(a.scores).with_columns(pl.col("sid").cast(pl.Int64), pl.col("mark").cast(pl.Float64))
    s = s.filter((pl.col("mark") % a.every).abs() < 1e-6).sort("sid", "mark")
    hs = [float(x) for x in a.horizons.split(",")]
    s = s.filter(pl.col("mark") <= 14220 - max(hs))
    con = duckdb.connect(str(a.catalog), read_only=True)
    m = mids(con, int(a.date), s["sid"].unique().to_list())
    con.close()
    keys = s.select("sid", "mark")
    m0 = at(m, keys, 0.0)
    rets = {int(h): np.log(at(m, keys, h) / m0) for h in hs}
    srcs = {"ours (C++ chain)": s["pred"].to_numpy().astype(float)}
    if a.target and a.target.exists():
        tg = pl.read_parquet(a.target, columns=["sid", "t", "prediction"]).with_columns(pl.col("sid").cast(pl.Int64), pl.col("t").cast(pl.Float64)).sort("t")
        k = keys.with_row_index("i").sort("mark")
        j = k.join_asof(tg, left_on="mark", right_on="t", by="sid", strategy="backward").sort("i")
        srcs["PM's prediction"] = j["prediction"].to_numpy().astype(float)
        p, o = srcs["PM's prediction"], srcs["ours (C++ chain)"]
        f = np.isfinite(p) & np.isfinite(o)
        print(f"{a.date}: PM's prediction found for {f.mean():.3f} of rows; rank corr ours vs PM pooled {pl.DataFrame({'a': o[f], 'b': p[f]}).select(pl.corr('a', 'b', method='spearman')).item():.4f}")
    print(f"{a.date}: {s.height:,} rows ({s['mark'].n_unique()} marks every {a.every:g} s, {s['sid'].n_unique()} stocks); mid found at the mark {np.isfinite(m0).mean():.3f}")
    for name, v in srcs.items():
        t = ic_table(keys, v, rets)
        print(f"  {name:18s} " + " | ".join(f"{h:>4d}s IC {ic:+.4f} t {tt:5.1f} >0 {pos:.2f} ({n})" for h, (ic, tt, pos, n) in t.items()))


if __name__ == "__main__":
    main()
