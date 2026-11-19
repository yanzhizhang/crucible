r"""runS on a live day: when does a decision become an order, at what price, and where does the position go?

Three questions from the user (2026-11-19), all from the parsed 224 log (pm_orders/date=D):
1  the order gate. runS evaluates on every scored snapshot (769k decisions on 1118) but sends 3,063 orders. Per decision:
   order made (nullptr == 0) against the wanted trade = executp - cpos: its size raw and rounded down to lots, the side
   thresholds (sell fa < lthr, buy fa > -lthr), the sellable quantity asq, and the time since the stock's previous order
   (a live order blocks the next). The rule "lot-rounded trade >= 100 and past the threshold and no order in the last
   T s" is scored as a classifier for a grid of T.
2  the price: each order's price against the bid / ask of chrono's own ORD line (cross = the far touch, join = the near
   touch), per side; and the caps: does |executp| ever reach poscst, does position value ever reach nmvcst / tqub.
3  the position through the day: |pos| per 15-minute bucket (pos is the deviation the strategy holds around its base),
   summed over stocks in shares and in value, the share of stocks back at 0, and the same for cpos.
Read-only.

    python research/pm_factors/runs_gate.py --orders /work/crucible_data/pm_orders --date 20261118
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl


def q(v: np.ndarray, qs=(10, 50, 90)) -> str:
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.4g}" for x in np.percentile(v, qs)) if v.size else "-"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    tabs = {n: pl.read_parquet(root / f"{n}.parquet") for n in ("decisions", "po", "life", "trk") if (root / f"{n}.parquet").exists()}
    for n, t in tabs.items():
        print(f"-- {n}: {t.height:,} rows; columns {t.columns[:40]}")
    d = tabs["decisions"].with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64)).drop_nulls(["src_epoch", "executp"]).sort("src_epoch")
    num = [k for k in ("fa", "pos", "pp", "executp", "asq", "lthr", "hs", "winsdQty", "nullptr", "ev_mp", "ev_cpos", "ev_poscst", "ev_vpen", "ev_asq") if k in d.columns]
    d = d.with_columns(c(k).cast(pl.Float64) for k in num)
    d = d.with_columns(((c("src_epoch") / 1e9 + 28800) % 86400).alias("sod"))
    cp = "ev_cpos" if "ev_cpos" in d.columns else "pos"
    d = d.with_columns((c("executp") - c(cp)).alias("trade"), (c("nullptr") == 0).alias("order"))
    d = d.with_columns((c("trade").abs() // 100 * 100).alias("lot"), pl.when(c("trade") > 0).then(c("fa") > -c("lthr")).when(c("trade") < 0).then(c("fa") < c("lthr")).otherwise(False).alias("past"))
    n_ord = int(d["order"].sum())
    print(f"\n== 1 the gate: {d.height:,} decisions, {n_ord:,} with an order ({n_ord / d.height:.5f}); trade = executp - {cp}")
    print(f"   |trade| q10/50/90 {q(d['trade'].abs().to_numpy())}; lot-rounded >= 100 on {float((d['lot'] >= 100).mean()):.4f}; past the side threshold on {float(d['past'].mean()):.4f}")
    for lo, hi in ((0, 1), (1, 50), (50, 100), (100, 200), (200, 500), (500, 1000), (1000, 5000), (5000, 1e12)):
        s = d.filter((c("trade").abs() >= lo) & (c("trade").abs() < hi))
        if s.height:
            print(f"   |trade| in [{lo:g}, {hi:g}): {s.height:>8,} decisions, order on {float(s['order'].mean()):.5f}, past threshold {float(s['past'].mean()):.3f}")
    g = d.filter((c("lot") >= 100) & c("past"))
    print(f"   lot >= 100 and past the threshold: {g.height:,} decisions, order on {float(g['order'].mean()) if g.height else float('nan'):.5f}; orders outside that set: {int(d.filter(~((c('lot') >= 100) & c('past')))['order'].sum())}")
    o = d.filter(c("order")).select("sid", c("src_epoch").alias("prev_order"))
    g2 = g.join_asof(o, left_on="src_epoch", right_on="prev_order", by="sid", strategy="backward", allow_exact_matches=False).with_columns(((c("src_epoch") - c("prev_order")) / 1e9).alias("since"))
    print("   among those, by seconds since the stock's previous order (a live order should block):")
    for lo, hi in ((0, 0.7), (0.7, 3), (3, 5.3), (5.3, 10), (10, 30), (30, 120), (120, 1e9)):
        s = g2.filter((c("since") >= lo) & (c("since") < hi))
        if s.height:
            print(f"      [{lo:g}, {hi:g}) s: {s.height:>8,} decisions, order on {float(s['order'].mean()):.5f}")
    s = g2.filter(c("since").is_null())
    print(f"      no previous order today: {s.height:,} decisions, order on {float(s['order'].mean()) if s.height else float('nan'):.5f}")
    for T in (0.7, 3, 5.3, 6, 8, 10, 15):
        pred = ((d["lot"] >= 100) & d["past"]).to_numpy()
        dd = d.join_asof(o, left_on="src_epoch", right_on="prev_order", by="sid", strategy="backward", allow_exact_matches=False)
        since = ((dd["src_epoch"] - dd["prev_order"]) / 1e9).fill_null(1e9).to_numpy()
        pred = pred & (since >= T)
        true = d["order"].to_numpy()
        tp = float((pred & true).sum())
        print(f"   rule 'lot >= 100, past threshold, no order in the last {T:g} s': predicts {int(pred.sum()):,}; precision {tp / max(pred.sum(), 1):.4f}, recall {tp / max(true.sum(), 1):.4f}")
    if "asq" in d.columns:
        sl = g.filter(c("trade") < 0)
        print(f"   sells in the gated set: {sl.height:,}; asq >= lot on {float((sl['asq'] >= sl['lot']).mean()) if sl.height else float('nan'):.4f}; order rate with asq >= 100: "
              f"{float(sl.filter(c('asq') >= 100)['order'].mean() or 0):.5f}, with asq < 100: {float(sl.filter(c('asq') < 100)['order'].mean() or 0):.5f}")
    ords = d.filter(c("order"))
    print(f"   on the orders: trade sign + {float((ords['trade'] > 0).mean()):.3f}; |trade| q10/50/90 {q(ords['trade'].abs().to_numpy())}; lot q10/50/90 {q(ords['lot'].to_numpy())}")

    print("\n== 2 price and caps")
    if "trk" in tabs and "kind" in tabs["trk"].columns:
        t = tabs["trk"].filter(c("kind") == "ORD")
        keep = [k for k in t.columns if t[k].null_count() < t.height]
        print(f"   trk ORD: {t.height:,} rows; filled columns {keep}")
        if t.height:
            with pl.Config(tbl_cols=30, tbl_width_chars=220, tbl_rows=3):
                print(t.select(keep).head(3))
            pxk = next((k for k in ("px", "price") if k in keep), None)
            bk, ak = next((k for k in ("bid1", "bid", "b1") if k in keep), None), next((k for k in ("ask1", "ask", "a1") if k in keep), None)
            sk = next((k for k in ("side", "bs", "dir") if k in keep), None)
            if pxk and bk and ak and sk:
                t = t.with_columns(c(pxk).cast(pl.Float64).alias("_px"), c(bk).cast(pl.Float64).alias("_b"), c(ak).cast(pl.Float64).alias("_a"), c(sk).cast(pl.String).str.to_lowercase().alias("_s"))
                for side in t["_s"].unique().to_list():
                    s = t.filter(c("_s") == side)
                    buy = side.startswith("b") or side in ("1", "buy")
                    far, near = ("_a", "_b") if buy else ("_b", "_a")
                    e = lambda x, y: float(((s[x] - s[y]).abs() < 1e-9).mean())  # noqa: E731
                    rel = ((s["_px"] - s[near]) / (s[far] - s[near])).to_numpy()
                    print(f"   side {side}: {s.height:,} orders; px == far touch (cross) {e('_px', far):.3f}, == near touch (join) {e('_px', near):.3f}, strictly inside {float(((rel > 1e-9) & (rel < 1 - 1e-9)).mean()):.3f},"
                          f" beyond far {float((rel > 1 + 1e-9).mean()):.3f}, behind near {float((rel < -1e-9).mean()):.3f}; spread in ticks q10/50/90 {q(((s['_a'] - s['_b']) / 0.01).to_numpy())}")
            else:
                print(f"   cannot test cross / join: need px, bid1, ask1, side among the ORD columns (found px {pxk}, bid {bk}, ask {ak}, side {sk})")
    if "ev_poscst" in d.columns and "ev_mp" in d.columns:
        ex, pc, mp, cpv = (d[k].to_numpy() for k in ("executp", "ev_poscst", "ev_mp", cp))
        print(f"   caps: |executp| / poscst q50/90/99/max {q(np.abs(ex) / pc, (50, 90, 99, 100))}; == poscst on {float((np.abs(ex) == pc).mean()):.5f}; > poscst on {float((np.abs(ex) > pc).mean()):.5f}")
        print(f"         |{cp}| / poscst q50/90/99/max {q(np.abs(cpv) / pc, (50, 90, 99, 100))}; |{cp} x mp| max {np.nanmax(np.abs(cpv * mp)):.4g} (nmvcst = tqub = 1e7); |executp x mp| max {np.nanmax(np.abs(ex * mp)):.4g}")
        tot = d.group_by(c("src_epoch") // 60_000_000_000).agg((c(cp) * c("ev_mp")).abs().sum().alias("gross"), (c(cp) * c("ev_mp")).sum().alias("net"))
        print(f"         per-minute sums over the decisions of that minute (a rough book): gross value max {tot['gross'].max():.4g}, |net| max {tot['net'].abs().max():.4g}")

    print("\n== 3 position through the day (last value per stock in each 15-minute bucket)")
    last = d.with_columns((c("sod") // 900).cast(pl.Int64).alias("b")).group_by("b", "sid").agg(c("pos").last(), c(cp).last().alias("cpos"), c("ev_mp").last().alias("mp") if "ev_mp" in d.columns else pl.lit(1.0).alias("mp"))
    for (b,), s in sorted(last.group_by("b"), key=lambda kv: kv[0][0]):
        hh, mm = divmod(int(b) * 15, 60)
        p, cv, mp = s["pos"].to_numpy(), s["cpos"].to_numpy(), s["mp"].to_numpy()
        print(f"   {hh:02d}:{mm:02d}  stocks {s.height:>3}  sum |pos| {np.nansum(np.abs(p)):>10,.0f}  sum |pos x mp| {np.nansum(np.abs(p * mp)):>13,.0f}  net pos x mp {np.nansum(p * mp):>13,.0f}"
              f"  pos == 0 on {float((p == 0).mean()):.3f}  sum |cpos x mp| {np.nansum(np.abs(cv * mp)):>13,.0f}")
    print("[done]")


if __name__ == "__main__":
    main()
