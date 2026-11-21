r"""runS order replay: the decoded cOrd (side, price, quantity) against the orders PM really sent, and the day's order life.

For every decision line that made an order (and every one the order rule says should have): the snapshot it was made on is
looked up in the tick store (same stock, within --win seconds of the receive time, mid == the logged mp) to get the best
bid / ask and their sizes, then cOrd is replayed (alg.cpp @310-@491, normal mode):
    sa = fa asf; no order if the pending leg opposes sa, or |sa| < cr + hs + lthr;
    d = executp - cur limited to the participation room (the room itself limited to asq when cur > executp);
    buy (sa > 0, d > 0) at the ask / sell (sa < 0, d < 0) at the bid; quantity -> board cap (STAR 90,000 and none below 200,
    ChiNext 270,000, else 900,000) -> 5e6 yuan cap -> min(top-of-book size) -> down to 100; STAR: none below 200, max 100,000.
Scored against PM's SO lines (each SO line <-> the stock's decision nearest in time; book sizes tried as shares and as lots):
  A  the snapshot lookup: found share, receive-time lag, so the book fields can be trusted;
  B  order / no order with the book known (precision, recall; what the remaining false positives have in common);
  C  side, price == ask / bid, quantity == replay: exact shares, and which limit set the quantity; first mismatches;
  D  the day's order life, for the live-vs-sim view: SO lines by half hour, chrono rejects by reason and half hour,
     cancel requests (seconds after the order), cancels confirmed, fills (count, shares) against chrono's trk TRD.
Read-only.

    python research/pm_factors/runs_ord.py --orders /work/crucible_data/pm_orders --date 20261118 --catalog /work/crucible_data/catalog_s224.duckdb
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

CR, ASF = 0.0004, 0.8


def board(sid: int) -> str:
    s = f"{sid:06d}"
    return "star" if s.startswith(("688", "689")) else "chinext" if s.startswith("30") else "main"


def cord(r: dict, pend_key: str = "pp", unit: float = 1.0) -> tuple[str, float, float, str]:
    """(side, price, qty, what set the qty) of the decoded cOrd on one decision row with the book joined; side '' = no order."""
    sa = r["fa"] * ASF
    vf0 = r.get(pend_key) or 0.0
    if vf0 * sa < 0:
        return "", 0.0, 0.0, "pending leg opposes"
    if abs(sa) < CR + r["hs"] + r["lthr"]:
        return "", 0.0, 0.0, "below threshold"
    cur, ep, room = r["ev_cpos"], r["executp"], r["winsdQty"]
    if cur > ep:
        room = min(room, r["asq"])
    d = ep - cur
    d = min(d, room) if d > 0 else -min(-d, room)
    if not ((sa > 0 and d > 0) or (sa < 0 and d < 0)):
        return "", 0.0, 0.0, "step against alpha"
    buy = sa > 0
    px, top = (r["a1"], r["av1"]) if buy else (r["b1"], r["bv1"])
    top = top * unit if top is not None else None
    q, why = abs(d), ("room" if abs(d) < abs(ep - cur) else "step")
    b = board(r["sid"])
    if b == "star":
        if q < 200:
            return "", 0.0, 0.0, "star below 200"
        if q > 90000:
            q, why = 90000.0, "board cap"
    elif q > (270000 if b == "chinext" else 900000):
        q, why = (270000.0 if b == "chinext" else 900000.0), "board cap"
    if px and q * px > 5e6:
        q, why = 5e6 / px, "5e6 yuan"
    if top is not None and top == top and top < q:
        q, why = top, "top of book"
    q = np.floor(q / 100) * 100
    if q <= 0:
        return "", 0.0, 0.0, f"{why} below 100"
    if b == "star":
        if q <= 199:
            return "", 0.0, 0.0, f"{why}: star below 200"
        q = min(q, 100000.0)
    return ("buy" if buy else "sell"), float(px or np.nan), float(q), why


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--win", type=float, default=3.0)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--pair", default="backward", choices=("backward", "nearest", "forward"), help="which decision an SO line belongs to: the last one before its stamp, the nearest, the next")
    a = ap.parse_args()
    root = a.orders / f"date={a.date}"
    c = pl.col
    d = pl.read_parquet(root / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64)).drop_nulls(["src_epoch", "executp"])
    num = [k for k in ("fa", "pos", "pp", "executp", "asq", "lthr", "hs", "winsdQty", "nullptr", "ev_mp", "ev_cpos") if k in d.columns]
    d = d.with_columns(c(k).cast(pl.Float64) for k in num).with_columns((c("nullptr") == 0).alias("order")).sort("src_epoch").with_row_index("row")
    g = lambda k: d[k].to_numpy().astype(float)  # noqa: E731
    tr = g("executp") - g("ev_cpos")
    lot = np.floor(np.minimum(np.abs(tr), g("winsdQty")) / 100) * 100
    rule = (np.abs(g("fa")) * ASF >= CR + g("hs") + g("lthr")) & (lot >= 100) & (np.sign(tr) == np.sign(g("fa")))
    po = pl.read_parquet(root / "po.parquet").with_columns(c("sid").cast(pl.Int64), c("po_ts").cast(pl.Int64), c("px").cast(pl.Float64), c("qty").cast(pl.Float64)).sort("po_ts")
    pj = po.select("sid", "po_ts", c("px").alias("so_px"), c("qty").alias("so_qty"), *([c("side").alias("so_side")] if "side" in po.columns else []))
    pj = pj.join_asof(d.select("sid", "src_epoch", "row").sort("src_epoch"), left_on="po_ts", right_on="src_epoch", by="sid", strategy=a.pair, tolerance=3_000_000_000)
    hit = pj.drop_nulls("row")
    dup = hit.height - hit["row"].n_unique()
    lagp = ((hit["src_epoch"] - hit["po_ts"]).to_numpy() / 1e6) if hit.height else np.array([np.nan])
    print(f"== {a.date}: {d.height:,} decisions, {int(d['order'].sum()):,} flagged with an order, {int(rule.sum()):,} inside the log-only rule; SO lines {po.height:,}")
    print(f"   SO line -> the stock's decision by --pair {a.pair} (within 3 s of the SO stamp): found {hit.height:,}; two SO lines on one decision {dup}; decision receive time - SO stamp, ms, q1/10/50/90/99: {' / '.join(f'{v:.0f}' for v in np.nanpercentile(lagp, [1, 10, 50, 90, 99]))}")
    hit = hit.unique("row", keep="first")
    d = d.join(hit.select("row", "so_px", "so_qty", *(["so_side"] if "so_side" in hit.columns else []), "po_ts"), on="row", how="left").sort("row")
    has_so = d["so_qty"].is_not_null().to_numpy()
    flag = d["order"].fill_null(False).to_numpy()
    print(f"   decisions paired with an SO line: flagged as order {int((has_so & flag).sum()):,}, not flagged {int((has_so & ~flag).sum()):,}; flagged but no SO line {int((~has_so & flag).sum()):,}")
    sodh = ((d["src_epoch"].to_numpy() / 1e9 + 28800) % 86400) // 1800
    print("   by half hour (SO-paired / flagged): " + ", ".join(f"{int(b) // 2:02d}:{'30' if int(b) % 2 else '00'} {int((has_so & (sodh == b)).sum())}/{int((flag & (sodh == b)).sum())}" for b in sorted(set(sodh[has_so | flag].tolist()))))
    d = d.with_columns(pl.Series("order", has_so))
    x = d.filter(pl.Series(has_so | rule))
    print(f"   from here 'order' = paired with an SO line; replaying {x.height:,} decisions")

    import duckdb

    con = duckdb.connect(str(a.catalog), read_only=True)
    con.register("dec", x.select("row", "sid", "src_epoch", "ev_mp").to_arrow())
    w = int(a.win * 1e9)
    bk = con.sql(f"""SELECT d.row, q.spider_ts - d.src_epoch AS lag, q.time AS xt, q.bid_px[1]::DOUBLE AS b1, q.ask_px[1]::DOUBLE AS a1, q.bid_vol[1]::DOUBLE AS bv1, q.ask_vol[1]::DOUBLE AS av1, list_sum(q.bid_vol[1:5])::DOUBLE AS bv5, list_sum(q.ask_vol[1:5])::DOUBLE AS av5, list_sum(q.bid_vol)::DOUBLE AS bvx, list_sum(q.ask_vol)::DOUBLE AS avx
        FROM dec d JOIN raw_quotation q ON q.symbol_id = d.sid AND q.date = {int(a.date)} AND q.spider_ts BETWEEN d.src_epoch - {w} AND d.src_epoch + {w}
        WHERE abs((q.bid_px[1] + q.ask_px[1]) / 2 - d.ev_mp) < 1e-6 OR ((q.bid_px[1] <= 0 OR q.ask_px[1] <= 0) AND abs(q.price - d.ev_mp) < 1e-6)
        QUALIFY row_number() OVER (PARTITION BY d.row ORDER BY abs(q.spider_ts - d.src_epoch)) = 1""").pl()
    x = x.join(bk, on="row", how="left")
    found = x["b1"].is_not_null()
    lag = x.filter(found)["lag"].to_numpy() / 1e6
    print(f"-- A  snapshot found (mid == mp within {a.win:g} s) on {float(found.mean()):.4f}; tick-store receive time - PM's, ms, q1/10/50/90/99: {' / '.join(f'{v:.1f}' for v in np.percentile(lag, [1, 10, 50, 90, 99])) if lag.size else '-'}")

    for k in ("bv1", "av1", "bv5", "av5"):
        v = x.filter(found)[k].to_numpy()
        print(f"   {k} in the tick store: q1/10/50/90 {' / '.join(f'{t:.0f}' for t in np.nanpercentile(v, [1, 10, 50, 90]))}; a multiple of 100 on {float((v % 100 == 0).mean()):.3f}")
    so = {r["row"]: r for r in x.filter(c("order")).iter_rows(named=True)}

    for tag, kb, ka, unit in (("level 1, shares", "bv1", "av1", 1.0), ("level 1, lots", "bv1", "av1", 100.0), ("levels 1-5, lots", "bv5", "av5", 100.0), ("all levels, lots", "bvx", "avx", 100.0), ("no size cap", None, None, 1.0)):
        st, why_all, conf, fp_why, mism = Counter(), Counter(), Counter(), Counter(), []
        for r in x.filter(found).iter_rows(named=True):
            r = {**r, "bv1": r[kb] if kb else np.inf, "av1": r[ka] if ka else np.inf}
            side, px, q, why = cord(r, unit=unit)
            conf[("order" if r["order"] else "none", "order" if side else "none")] += 1
            if not r["order"]:
                fp_why[why if not side else "replay orders too"] += 1
                continue
            s = so.get(r["row"])
            if s is None or s.get("so_qty") is None:
                st["no SO pair"] += 1
                continue
            st["n"] += 1
            if not side:
                st["replay says no order"] += 1
                why_all[f"none: {why}"] += 1
                if len(mism) < a.show:
                    mism.append(f"   {r['sid']:06d} SO {s.get('so_side', '?')} {s['so_qty']:.0f} @ {s['so_px']} | replay none ({why}) | fa {r['fa']:.5g} hs {r['hs']:.5g} lthr {r['lthr']:.5g} cur {r['ev_cpos']:.0f} pp {r.get('pp')} ep {r['executp']:.1f} room {r['winsdQty']:.0f} asq {r['asq']:.0f}")
                continue
            sd_ok = ("so_side" not in s) or s["so_side"] is None or s["so_side"] == side
            px_ok, q_ok = abs(px - s["so_px"]) < 1e-6, abs(q - s["so_qty"]) < 1e-6
            st["side"] += sd_ok
            st["px"] += px_ok
            st["qty"] += q_ok
            st["all"] += sd_ok and px_ok and q_ok
            why_all[why] += 1
            if q_ok:
                st[f"qty ok when set by {why}"] += 1
            if not (sd_ok and px_ok and q_ok) and len(mism) < a.show:
                mism.append(f"   {r['sid']:06d} SO {s.get('so_side', '?')} {s['so_qty']:.0f} @ {s['so_px']} | replay {side} {q:.0f} @ {px} ({why}) | bid {r['b1']} x {r['bv1']:.0f} ask {r['a1']} x {r['av1']:.0f} cur {r['ev_cpos']:.0f} ep {r['executp']:.1f}"
                            f" room {r['winsdQty']:.0f} asq {r['asq']:.0f} pp {r.get('pp')}")
        tp, fp, fn = conf[("order", "order")], conf[("none", "order")], conf[("order", "none")]
        print(f"-- B  [book size: {tag}] order / no order with the book: replay orders {tp + fp:,}; precision {tp / max(tp + fp, 1):.4f}, recall {tp / max(tp + fn, 1):.4f}; rule-only rows that the replay drops, by reason: {dict(fp_why.most_common(8))}")
        n = max(st["n"], 1)
        print(f"-- C  {st['n']:,} orders paired with an SO line ({st['no SO pair']} unpaired): replay has an order on {1 - st['replay says no order'] / n:.4f}; side {st['side'] / n:.4f}, price {st['px'] / n:.4f}, quantity {st['qty'] / n:.4f}, all three {st['all'] / n:.4f}")
        print("   quantity set by: " + "; ".join(f"{k} {v:,} (exact {st[f'qty ok when set by {k}'] / max(v, 1):.3f})" for k, v in why_all.most_common(8)))
        print(f"   first {len(mism)} mismatches:")
        print("\n".join(mism))

    hh = lambda e: (((e / 1e9 + 28800) % 86400) // 1800).cast(pl.Int64)  # noqa: E731
    lab = lambda b: f"{int(b) // 2:02d}:{'30' if int(b) % 2 else '00'}"  # noqa: E731
    print("-- D  order life")
    print("   SO lines by half hour: " + ", ".join(f"{lab(b)} {k}" for b, k in sorted(po.group_by(hh(c("po_ts"))).len().iter_rows())))
    if (root / "trk.parquet").exists():
        trk = pl.read_parquet(root / "trk.parquet")
        tk = next((k for k in ("now", "ts", "ts_ns") if k in trk.columns), None)
        print(f"   chrono trk kinds: {dict(sorted(trk.group_by('kind').len().iter_rows(), key=lambda kv: -kv[1])[:12])}")
        rej = trk.filter(c("kind") == "REJ")
        if rej.height and "reason" in rej.columns:
            print("   REJ reasons: " + "; ".join(f"{k} x {str(rs)[:70]}" for rs, k in sorted(rej.group_by(c("reason").str.replace_all(r"\d+", "#")).len().iter_rows(), key=lambda kv: -kv[1])[:5]))
            if tk and rej[tk].null_count() < rej.height:
                print("   REJ by half hour: " + ", ".join(f"{lab(b)} {k}" for b, k in sorted(rej.drop_nulls(tk).group_by(hh(c(tk).cast(pl.Int64))).len().iter_rows())))
        t = trk.filter(c("kind") == "TRD")
        if t.height and "qty" in t.columns:
            print(f"   chrono TRD: {t.height:,} fills, {t['qty'].cast(pl.Float64).sum():,.0f} shares")
    if (root / "life.parquet").exists():
        life = pl.read_parquet(root / "life.parquet")
        print(f"   PM order-life lines: {dict(sorted(life.group_by('ev').len().iter_rows(), key=lambda kv: -kv[1]))}")
        if "fill_qty" in life.columns:
            f = life.filter(c("ev") == "OnEOM")
            print(f"   PM fills (OnEOM): {f.height:,} lines, {f['fill_qty'].sum():,.0f} shares")
        cx = life.filter(c("ev") == "cxl_req").drop_nulls("ts")
        if cx.height:
            pk = pl.read_parquet(root / "po.parquet")
            for key in ("f3", "f4"):
                j = cx.with_columns(c("exoid").cast(pl.Int64)).join(pk.select(c(key).cast(pl.Int64).alias("exoid"), c("po_ts").cast(pl.Int64)).unique("exoid"), on="exoid", how="inner")
                if j.height:
                    first = j.group_by("exoid").agg(((c("ts") - c("po_ts")).min() / 1e9).alias("s"))["s"].to_numpy()
                    print(f"   cancel requests keyed by SO {key}: {j.height:,} on {len(first):,} orders; first request seconds after the SO stamp q1/10/50/90/99: {' / '.join(f'{v:.2f}' for v in np.percentile(first, [1, 10, 50, 90, 99]))}")
    print("[done]")


if __name__ == "__main__":
    main()
