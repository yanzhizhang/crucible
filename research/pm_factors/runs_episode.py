r"""runS on 20261113, fourth pass: the morning / afternoon split and what starts a burst of orders.

R74: lthr = 0 through the whole morning and log(-lthr) ~ 2 log vpen + 6.4 in the afternoon (R^2 0.93; 0.995 with hs, vr);
decisions are ~90% afternoon; and orders come in bursts -- right after a stock's order ended the next decision places
20-40% of the time, after 300 s quiet 0.8%, with no earlier order 0.4%. Read-only on pm_orders' parquet; prints:

1  per hour: decisions, orders, lthr == 0 share, executp == 0 share, and the log file each decision came from is not
   kept, so the hour table is what shows a restart;
2  episodes: a stock's orders joined while the next is sent within --gap s of the previous one's end; their count,
   length, orders per episode, filled qty, and the decision that opened each one against the stock's decisions before
   it: was the threshold freshly crossed (the previous decision of the stock did not meet it), did |trade| jump, did
   executp change sign or size, how long since the previous decision;
3  the same rates on all decisions that meet the gate, by fresh crossing / not, so the start rule can be read off.

    python research/pm_factors/runs_episode.py --orders /work/crucible_data/pm_orders --date 20261113
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def q(x, qs=(10, 50, 90)) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return "/".join(f"{v:.4g}" for v in np.percentile(x, qs)) if x.size else "-"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--gap", type=float, default=10.0, help="seconds between one order's end and the next send that keep an episode going")
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    d = pl.read_parquet(root / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64))
    d = d.with_columns((c("executp") - c("ev_cpos")).alias("trade"), ((c("src_epoch") / 1e9 + 28800) % 86400 / 3600).floor().cast(pl.Int32).alias("hour"))
    d = d.with_columns((c("trade").abs() >= 100).alias("lot"),
                       pl.when(c("trade") < 0).then(c("fa") < c("lthr")).when(c("trade") > 0).then(c("fa") > -c("lthr")).otherwise(False).alias("thr"))
    d = d.with_columns((c("lot") & c("thr")).alias("gate"), (c("nullptr") == 0).alias("made")).sort("sid", "src_epoch")
    po = pl.read_parquet(root / "po.parquet").with_columns(c("sid").cast(pl.Int64), c("f3").alias("exoid"))
    po = po.with_columns(((c("po_ts") / 1e9 + 28800) % 86400 / 3600).floor().cast(pl.Int32).alias("hour"))

    print(f"==== {a.date}: {d.height:,} decisions, {po.height:,} orders")
    print("-- 1 per hour: decisions | orders | lthr == 0 | executp == 0 | gate met | made on gate")
    po_h = dict(po.group_by("hour").len().iter_rows())
    for h, n, z, e0, gm, mg in d.group_by("hour").agg(pl.len(), (c("lthr") == 0).mean(), (c("executp") == 0).mean(), c("gate").mean(),
                                                       c("made").filter(c("gate")).mean()).sort("hour").iter_rows():
        print(f"   {h:02d}h  {n:>9,} | {po_h.get(h, 0):>6,} | {z:.3f} | {e0:.3f} | {gm:.4f} | {mg if mg is not None else float('nan'):.4f}")
    first, lastd = d["src_epoch"].min(), d["src_epoch"].max()
    print(f"   first decision {(first / 1e9 + 28800) % 86400 / 3600:.3f} h, last {(lastd / 1e9 + 28800) % 86400 / 3600:.3f} h; orders first "
          f"{(po['po_ts'].min() / 1e9 + 28800) % 86400 / 3600:.3f} h, last {(po['po_ts'].max() / 1e9 + 28800) % 86400 / 3600:.3f} h")

    life = pl.read_parquet(root / "life.parquet")
    cx = life.filter(c("ev") == "OnECM").group_by("exoid").agg(c("ts").min().alias("t_cxl"))
    fl = life.filter(c("ev") == "OnEOM").group_by("exoid").agg(c("fill_qty").sum().alias("fq"), c("ts").max().alias("t_fill"))
    o = po.join(cx, on="exoid", how="left").join(fl, on="exoid", how="left").with_columns(c("fq").fill_null(0.0))
    o = o.with_columns(pl.coalesce(c("t_cxl"), c("t_fill"), c("po_ts") + 7_000_000_000).alias("t_end")).sort("sid", "po_ts")
    o = o.with_columns(((c("po_ts") - c("t_end").shift(1).over("sid")) / 1e9).alias("after_prev"))
    o = o.with_columns((c("after_prev").is_null() | (c("after_prev") > a.gap)).alias("opens"))
    o = o.with_columns(c("opens").cast(pl.Int32).cum_sum().over("sid").alias("ep"))
    ep = o.group_by("sid", "ep").agg(pl.len().alias("n"), c("po_ts").min().alias("t0"), c("t_end").max().alias("t1"), c("fq").sum().alias("filled"),
                                     c("qty").first().alias("q0"), c("side").first().alias("side"), c("side").n_unique().alias("sides"))
    print(f"-- 2 episodes (next send within {a.gap:g} s of the previous end): {ep.height:,} over {ep['sid'].n_unique()} stocks; orders per episode {q(ep['n'].to_numpy())} "
          f"(max {ep['n'].max()}); seconds long {q(((ep['t1'] - ep['t0']) / 1e9).to_numpy())}; filled qty {q(ep['filled'].to_numpy())}, any fill {np.mean(ep['filled'].to_numpy() > 0):.3f}; "
          f"one side only {np.mean(ep['sides'].to_numpy() == 1):.3f}; by side {dict(ep.group_by('side').len().iter_rows())}")
    starts = o.filter(c("opens")).select("sid", "po_ts", "side", "qty")
    dd = d.with_columns(c("gate").shift(1).over("sid").alias("prev_gate"), c("trade").shift(1).over("sid").alias("prev_trade"),
                        c("executp").shift(1).over("sid").alias("prev_ex"), ((c("src_epoch") - c("src_epoch").shift(1).over("sid")) / 1e9).alias("dt_prev"))
    dd = dd.with_columns((c("gate") & ~c("prev_gate").fill_null(False)).alias("fresh"))
    j = starts.sort("po_ts").join_asof(dd.sort("src_epoch").select("sid", "src_epoch", "gate", "fresh", "trade", "prev_trade", "executp", "prev_ex", "dt_prev", "fa", "lthr", "hs"),
                                       left_on="po_ts", right_on="src_epoch", by="sid", strategy="backward")
    g = lambda k: j[k].to_numpy().astype(float)  # noqa: E731
    print(f"   the decision that opened each episode ({j.height:,}): gate met {np.mean(j['gate'].fill_null(False).to_numpy()):.3f}, freshly crossed {np.mean(j['fresh'].fill_null(False).to_numpy()):.3f}; "
          f"|trade| {q(np.abs(g('trade')))} vs previous {q(np.abs(g('prev_trade')))}; executp changed {np.mean(g('executp') != g('prev_ex')):.3f}, "
          f"sign flipped {np.mean(np.sign(g('executp')) != np.sign(g('prev_ex'))):.3f}; seconds since the previous decision {q(g('dt_prev'))}; send - decision (s) {q((g('po_ts') - g('src_epoch')) / 1e9)}")
    print("-- 3 on decisions that meet the gate: made by fresh crossing")
    for fr, n, r in dd.filter(c("gate")).group_by("fresh").agg(pl.len(), c("made").mean()).sort("fresh").iter_rows():
        print(f"   fresh {fr}: {n:,} rows, made {r:.4f}")
    gt = dd.filter(c("gate"))
    tab = gt.with_columns(c("trade").abs().cut([100, 200, 500, 1000, 3000]).alias("b")).group_by("b").agg(pl.len(), c("made").mean()).sort("b")
    print("   made by |trade|: " + " | ".join(f"{b} {r:.4f} ({n:,})" for b, n, r in tab.iter_rows()))
    with np.errstate(divide="ignore", invalid="ignore"):
        m = gt.with_columns(pl.when(c("trade") < 0).then(c("lthr") - c("fa")).otherwise(c("fa") + c("lthr")).alias("over")).with_columns((c("over") / c("hs")).alias("over_hs"))
        tab = m.with_columns(c("over_hs").cut([0.25, 0.5, 1, 2, 4, 8]).alias("b")).group_by("b").agg(pl.len(), c("made").mean()).sort("b")
    print("   made by how far fa is past the threshold, in hs: " + " | ".join(f"{b} {r:.4f} ({n:,})" for b, n, r in tab.iter_rows()))


if __name__ == "__main__":
    main()
