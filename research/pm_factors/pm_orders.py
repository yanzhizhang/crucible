r"""PM's order rules, read off 224's tee'd logs: when does a prediction become an order, of what size, at what time.

224's logs hold PM's strategy plugin (``libcatchRun.so`` inside chrono) next to chrono's own ``trk`` lines:

* untimestamped strategy lines -- ``qlive/catchRun.cpp`` (srcEpoch = PM's receive time of the snapshot, sid),
  ``sol/alg.cpp:169`` (fa = the prediction, emsOurVol / emsMktVol), ``sol/alg.cpp:251`` (a trade decision: pos, pp_,
  executp, asq, lthr, hs, liid, winsdQty, asf), ``sol/alg.cpp:252`` (``order is nullptr : N``) and any other path;
* chrono ``trk`` lines -- SIG (the signal, md_recv_ts = srcEpoch), ORD (the strategy's order: handle_ts = its SIG's),
  SEND / CXL / RPT / TRD (what happened to it).

Per day it writes ``<out>/date=D/{snaps,decisions,orders}.parquet`` and prints:

1. the census of strategy line shapes (source line, keys, count, one example) -- every code path the plugin logs;
2. snapshot -> decision: the share of snapshots that reach alg.cpp:251, by |fa| / hs and by sign;
3. the thresholds: lthr against hs (least squares), and fa against lthr on the decisions;
4. decision -> order: orders per decision, side against sign(executp / fa), qty against |executp|, asq, winsdQty;
5. timing: orders per half hour, signal-to-order latency, the gap between two orders of one stock (a cooldown shows as
   a floor), and orders per stock.

    python research/pm_factors/pm_orders.py --logs /work/crucible_data/chrono_logs --dates 20261104,20261105,20261106
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pm_signals import NUM, log_files, repair, trading_clock

SRC = re.compile(r"\[(?:/[^\]]*/)?([\w./]+\.(?:cpp|h|hpp):\d+)\]\s*(.*)$")  # [/work/sht-strategy/src/sol/alg.cpp:251] rest
PLUGIN = re.compile(r"\[(?:[^\]]*/)?(?:sol|qlive)/[^\]]*\]|\[(?:catchRun|alg|es|trd)\.(?:cpp|h|hpp):\d+\]")  # the strategy plugin's sources
TRK = re.compile(r"\btrk (\w+) (.*)$")
KV = re.compile(r"(\w+)=('([^']*)'|(\S*))")
GLUE = re.compile(r"(?=\[(?:INFO|WARN|WARNING|ERROR|DEBUG)\] \[/)|(?=\[20\d\d-\d\d-\d\d )")  # where one log line starts


def parse_day(files: list[str]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """(snapshots, decisions, chrono trk rows, signals, census) of one day's logs."""
    snaps, decs, trk, sigs, pos_, placed, life = [], [], [], [], [], [], []
    posm: list[dict] = []  # OnPosM (es.cpp:29x / 31x) and the plugin's position view (catchRun.cpp:180)
    census: dict[tuple[str, str], list] = {}
    for f in files:
        last: dict[str, float] = {}
        ev: dict[str, float] = {}  # the snapshot's alg.cpp evaluation line (mp ... hs), folded into its decision
        pending: dict | None = None  # the last decision, waiting for its alg.cpp:252 line
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in (part for raw in fh for part in GLUE.split(raw) if part.strip()):  # tee can glue two writers' lines
                m = TRK.search(line)
                if m:
                    kv = {k.group(1): k.group(3) if k.group(3) is not None else k.group(4) for k in KV.finditer(m.group(2))}
                    if m.group(1) == "SIG":
                        kv = repair(kv)
                        sigs.append({"handle_ts": kv.get("handle_ts"), "md_recv_ts": kv.get("md_recv_ts"), "symbol": kv.get("symbol"),
                                     "prediction": kv.get("prediction")})
                    else:
                        trk.append({"kind": m.group(1), **kv})
                    continue
                s = SRC.search(line)
                if not s:
                    continue
                if line.startswith("[20") and not PLUGIN.search(line):  # chrono's own lines; from 1112 the plugin logs through chrono too
                    continue
                where, rest = s.group(1), s.group(2)
                keys = tuple(k for k, _ in NUM.findall(rest))
                shape = (where, " ".join(keys) if keys else re.sub(r"\d+", "N", rest)[:60])
                c = census.setdefault(shape, [0, rest[:160]])
                c[0] += 1
                kv = {k.replace("this->", "").replace("book->", "book_").rstrip("_"): float(v) for k, v in NUM.findall(rest)}
                fname = where.rsplit(":", 1)[0].rsplit("/", 1)[-1]  # matched by file and keys: the line numbers move between plugin builds
                if fname == "catchRun.cpp" and "srcEpoch" in kv:
                    last, ev = kv, {}
                elif fname == "alg.cpp" and "emsMktVol" in kv and last:
                    snaps.append({"sid": int(last.get("sid", -1)), "src_epoch": round(last["srcEpoch"] * 1e9), "fa": kv.get("fa"),
                                  "ems_our_vol": kv.get("emsOurVol"), "ems_mkt_vol": kv.get("emsMktVol")})
                elif fname == "alg.cpp" and "vpen" in kv:  # from 1112: mp fa vpen vr poscst cpos asq hs, just before the decision
                    ev = {f"ev_{k}": v for k, v in kv.items()}
                elif fname == "alg.cpp" and "executp" in kv:
                    pending = {"src_epoch": round(last["srcEpoch"] * 1e9) if "srcEpoch" in last else None, **ev, **kv}
                    pending["sid"] = int(kv.get("sid", last.get("sid", -1)))
                    decs.append(pending)
                elif "OnPosM" in rest or where.endswith("catchRun.cpp:180"):
                    posm.append({"where": where, **{k.removeprefix("p.").replace("()", "").replace(".", "_"): float(v) for k, v in re.findall(r"([\w.()]+)\s*[:=]\s*(-?\d+(?:\.\d+)?)", rest)}})
                elif fname == "es.cpp" and "nowTs:" in rest:  # Po = the strategy places an order
                    f_ = rest.split("nowTs:", 1)[1].split(",")
                    try:
                        pos_.append({"po_ts": round(float(f_[1]) * 1e9), "f2": int(float(f_[2])), "f3": int(float(f_[3])), "f4": int(float(f_[4])),
                                     "sid": int(float(f_[5])), "px": float(f_[6]), "qty": float(f_[7].split()[0]), "stamp": f_[0][-2:]})
                    except (IndexError, ValueError):
                        pass
                elif fname == "es.cpp" and (ml := LIFE.search(rest)):  # order life: OnECM = cancel confirmed, OnEOM = fill (with qty, px)
                    tail = [x.strip() for x in ml.group(4).split(",") if x.strip()]
                    row = {"ev": ml.group(1), "ts": local_ns(ml.group(2)), "exoid": int(ml.group(3))}
                    if ml.group(1) == "OnEOM" and len(tail) >= 2:
                        row.update(fill_qty=float(tail[0]), fill_px=float(tail[1]))
                    life.append(row)
                elif fname == "es.cpp" and "cancelExOrd" in rest:  # a cancel request; no stamp of its own: the snapshot being handled
                    kv2 = dict(re.findall(r"(\w+): (-?[\d.]+)", rest))
                    life.append({"ev": "cxl_req", "ts": round(last["srcEpoch"] * 1e9) if "srcEpoch" in last else None, "exoid": int(float(kv2.get("exOid", -1))),
                                 "sid": int(float(kv2.get("sid", -1))), "cxl_px": float(kv2.get("prc", "nan")), "cxl_qty": float(kv2.get("exSoQ", "nan"))})
                elif fname == "catchRun.cpp" and "place_order" in rest:  # from 1113: the plugin's own order line carries the side
                    mp_ = re.search(r"symbol = (\d{6})\S*, volume = ([\d.]+), price = ([\d.]+), side = (\w+)", rest)
                    if mp_:
                        placed.append({"sid": int(mp_.group(1)), "qty": float(mp_.group(2)), "px": float(mp_.group(3)), "side": mp_.group(4).lower()})
                elif fname == "alg.cpp" and "nullptr" in rest and pending is not None:
                    mn = re.search(r"nullptr\s*:\s*(\d+|true|false)", rest)  # 0 / 1 before 1112, false / true from 1112
                    pending["nullptr"] = (1.0 if mn.group(1) == "true" else 0.0 if mn.group(1) == "false" else float(mn.group(1))) if mn else None
                    pending = None
    def frame(rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    global POSM, LIFE_DF
    POSM, LIFE_DF = frame(posm), frame(life)
    po = frame(pos_)
    if placed and po.height:  # each Po takes the side of the place_order line with the same stock, price and qty, in turn
        rk = [pl.col("sid"), pl.col("px").round(4), pl.col("qty")]
        a_ = po.with_columns(pl.col("px").round(4).alias("pxr"), pl.int_range(pl.len()).over(rk).alias("k"))
        b_ = frame(placed).with_columns(pl.col("px").round(4).alias("pxr"), pl.int_range(pl.len()).over(rk).alias("k"))
        po = a_.join(b_.select("sid", "pxr", "qty", "k", "side"), on=["sid", "pxr", "qty", "k"], how="left").drop("pxr", "k")
        print(f"place_order lines {len(placed):,}, Po lines {po.height:,}, Po with a side {po['side'].is_not_null().mean():.3f}: {dict(Counter(po['side'].to_list()))}")
    return frame(snaps), frame(decs), frame(trk), frame(sigs), census, po


POSM = pl.DataFrame()
LIFE_DF = pl.DataFrame()
LIFE = re.compile(r"(OnECM|OnEOM|OnFOM) (\d{8} \d\d:\d\d:\d\d\.\d+), ExOid (\d+)(.*)$")


def local_ns(stamp: str) -> int:
    """'20261113 09:30:07.120170' (Beijing time) -> epoch ns."""
    import datetime as dt

    t = dt.datetime.strptime(stamp, "%Y%m%d %H:%M:%S.%f").replace(tzinfo=dt.timezone(dt.timedelta(hours=8)))
    return round(t.timestamp() * 1e6) * 1000


def num(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Cast the given columns when present (trk values come in as strings): *_ts / now as Int64 (ns), the rest Float64."""
    return df.with_columns([pl.col(c).cast(pl.Int64 if c.endswith("_ts") or c == "now" else pl.Float64, strict=False)
                            for c in cols if c in df.columns])


def q(v: np.ndarray, qs=(10, 50, 90)) -> str:
    v = v[np.isfinite(v)]
    return "/".join(f"{np.percentile(v, x):.4g}" for x in qs) if v.size else "-"


def report(day: str, snaps: pl.DataFrame, decs: pl.DataFrame, trk: pl.DataFrame, sigs: pl.DataFrame, census: dict, out: Path) -> None:
    """The five checks of the module docstring."""
    print(f"\n==== {day}: {snaps.height:,} scored snapshots, {decs.height:,} decisions, "
          + ", ".join(f"{k} {v}" for k, v in Counter(trk["kind"].to_list() if trk.height else []).most_common()))
    print("-- 1 strategy line shapes (count, source, keys | example):")
    for (where, keys), (n, ex) in sorted(census.items(), key=lambda kv: -kv[1][0])[:25]:
        print(f"  {n:>9,}  {where:28s} {keys[:70]:70s} | {ex[:90]}")
    if snaps.is_empty():
        return
    # 2. snapshot -> decision
    key = lambda c: ((pl.col(c) + 50_000) // 100_000).alias("k")  # noqa: E731 -- PM prints epochs to 10 us
    d1 = decs.with_columns(key("src_epoch")) if decs.height else decs
    sn = snaps.with_columns(key("src_epoch"))
    if decs.height:
        sn = sn.join(d1.select("sid", "k", pl.lit(True).alias("dec"), "hs", "lthr"), on=["sid", "k"], how="left").with_columns(pl.col("dec").fill_null(False))
        hs_s = sn.group_by("sid").agg(pl.col("hs").drop_nulls().median().alias("hs_med"))
        sn = sn.join(hs_s, on="sid", how="left").with_columns((pl.col("fa").abs() / pl.col("hs_med")).alias("fa_hs"))
        print("-- 2 share of snapshots reaching a decision, by |fa| / hs (stock's median hs):")
        b = sn.filter(pl.col("fa_hs").is_finite()).with_columns(pl.col("fa_hs").cut([0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3]).alias("bin"))
        tab = b.group_by("bin", (pl.col("fa") > 0).alias("up")).agg(pl.len().alias("n"), pl.col("dec").mean().alias("p")).sort("bin", "up")
        print("  " + " | ".join(f"{r['bin']} {'+' if r['up'] else '-'} {r['p']:.3f} ({r['n']:,})" for r in tab.iter_rows(named=True)))
    if decs.is_empty():
        return
    # 3. thresholds
    dd = decs.filter(pl.col("hs").is_not_null() & pl.col("lthr").is_not_null())
    if dd.height > 10:
        x, y = dd["hs"].to_numpy(), dd["lthr"].to_numpy()
        A = np.column_stack([np.ones_like(x), x])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        print(f"-- 3 lthr = {coef[0]:+.6f} {coef[1]:+.4f} x hs (resid p10/p50/p90 {q(res)}, exact to 1e-5 {np.mean(np.abs(res) < 1e-5):.3f}); "
              f"distinct lthr {dd['lthr'].n_unique()}, hs {dd['hs'].n_unique()}")
        fa = dd["fa"].to_numpy()
        print(f"   on decisions: fa p10/p50/p90 {q(fa)}; fa < lthr {np.mean(fa < y):.3f}; fa > -lthr {np.mean(fa > -y):.3f}; "
              f"|fa| > hs {np.mean(np.abs(fa) > x):.3f}; executp != 0 {np.mean(dd['executp'].to_numpy() != 0):.3f}; "
              f"nullptr=1 {np.mean(dd['nullptr'].to_numpy() == 1) if 'nullptr' in dd.columns else float('nan'):.3f}")
        nz = dd.filter(pl.col("executp") != 0)
        if nz.height:
            print(f"   executp != 0: sign(executp) == sign(fa) {np.mean(np.sign(nz['executp'].to_numpy()) == np.sign(nz['fa'].to_numpy())):.3f}; "
                  f"|executp| p10/p50/p90 {q(np.abs(nz['executp'].to_numpy()))}; |executp| / winsdQty {q(np.abs(nz['executp'].to_numpy()) / nz['winsdQty'].to_numpy())}; "
                  f"pos p10/p50/p90 {q(nz['pos'].to_numpy())}; asq {q(nz['asq'].to_numpy())}")
    # 4. decision -> order
    if trk.height and "handle_ts" in trk.columns and sigs.height:
        ords = num(trk.filter(pl.col("kind") == "ORD"), ["qty", "px", "now", "handle_ts"])
        sg = num(sigs, ["handle_ts", "md_recv_ts", "prediction"]).with_columns(
            pl.col("symbol").str.slice(0, 6).cast(pl.Int64, strict=False).alias("sid"))
        if ords.height:
            o = ords.join(sg.select("handle_ts", "md_recv_ts", "sid", "prediction"), on="handle_ts", how="left")
            o = o.with_columns(((pl.col("md_recv_ts") + 50_000) // 100_000).alias("k"))
            o = o.join(d1.select("sid", "k", "fa", "pos", "executp", "asq", "lthr", "hs", "winsdQty"), on=["sid", "k"], how="left")
            o.write_parquet(out / "orders.parquet")
            side = o["side"].cast(pl.String).str.to_lowercase().str.slice(0, 1).to_numpy() if "side" in o.columns else np.array([])
            sgn = np.where(side == "b", 1, np.where(side == "s", -1, 0))
            print(f"-- 4 {o.height:,} orders; {o['fa'].is_not_null().mean():.3f} tied to a decision; "
                  f"side == sign(fa) {np.mean(sgn == np.sign(o['fa'].fill_null(0).to_numpy())):.3f}, == sign(executp) {np.mean(sgn == np.sign(o['executp'].fill_null(0).to_numpy())):.3f}; "
                  f"qty p10/p50/p90 {q(o['qty'].to_numpy())}; qty / |executp| {q(o['qty'].to_numpy() / np.abs(o['executp'].to_numpy()))}; "
                  f"|fa| / hs {q(np.abs(o['fa'].to_numpy()) / o['hs'].to_numpy())}")
            # 5. timing
            t = trading_clock(o["now"].to_numpy())
            lat = (o["now"] - o["md_recv_ts"]).to_numpy() / 1e6
            hh = np.floor(t / 1800).astype(int)
            print("-- 5 orders per half hour (09:30 = 0): " + " ".join(f"{k}:{v}" for k, v in sorted(Counter(hh.tolist()).items()))
                  + f"; PM's snapshot receive -> order (ms) p10/p50/p90 {q(lat)}; chrono's signal handling -> order (ms) {q((o['now'] - o['handle_ts']).to_numpy() / 1e6)}")
            gaps = o.sort("sid", "now").with_columns((pl.col("now").diff().over("sid") / 1e9).alias("gap"))["gap"].to_numpy()
            per = o.group_by("sid").len()["len"].to_numpy()
            print(f"   gap between two orders of a stock (s) p1/p10/p50/p90 {q(gaps, (1, 10, 50, 90))}, min {np.nanmin(gaps) if np.isfinite(gaps).any() else float('nan'):.3f}; "
                  f"orders per stock p10/p50/p90 {q(per.astype(float))}, stocks {per.size}")
            for k in ("SEND", "CXL", "RPT", "TRD"):
                n = trk.filter(pl.col("kind") == k).height
                if n:
                    print(f"   {k}: {n:,}")


def po_report(decs: pl.DataFrame, trk: pl.DataFrame, po: pl.DataFrame) -> None:
    """The strategy's own order lines (es.cpp:401 Po), each tied to the stock's latest decision before it; chrono's REJ."""
    if trk.height and "reason" in trk.columns:
        rj = trk.filter(pl.col("kind") == "REJ")
        if rj.height:
            print("-- 6 chrono REJ reasons: " + "; ".join(f"{r} {n:,}" for r, n in rj.group_by(pl.col("reason").str.slice(0, 60)).len().sort("len", descending=True).head(6).iter_rows()))
    if po.is_empty() or decs.is_empty():
        return
    d = decs.filter(pl.col("src_epoch").is_not_null()).with_columns(pl.col("src_epoch").cast(pl.Int64), pl.col("sid").cast(pl.Int64)).sort("src_epoch")
    p = po.with_columns(pl.col("sid").cast(pl.Int64)).sort("po_ts")
    j = p.join_asof(d.select("sid", pl.col("src_epoch").alias("dec_ts"), "fa", "executp", "pos", "asq", "lthr", "hs", "winsdQty",
                             *[c for c in ("pp", "ev_cpos") if c in d.columns]),
                    left_on="po_ts", right_on="dec_ts", by="sid", strategy="backward")
    ok = j.filter(pl.col("dec_ts").is_not_null())
    lag = ((ok["po_ts"] - ok["dec_ts"]) / 1e6).to_numpy()
    seq = po.sort("po_ts")
    print(f"-- 7 {po.height:,} Po lines; side tag (the 2 letters after the time): {dict(Counter(po['stamp'].to_list()))}; "
          f"f3 runs 1..{seq['f3'].max()} (exOid, +1 per order: {np.mean(np.diff(seq['f3'].to_numpy()) == 1):.3f}), "
          f"f4 = 100 x f3 + 1 {np.mean(po['f4'].to_numpy() == 100 * po['f3'].to_numpy() + 1):.3f}; tied to a decision {ok.height / po.height:.3f}, "
          f"decision -> Po (ms) p10/p50/p90 {q(lag)}")
    with np.errstate(divide="ignore", invalid="ignore"):
        by = "side" if "side" in ok.columns else "stamp"
        for tag in sorted(set(ok[by].drop_nulls().to_list())):
            o = ok.filter(pl.col(by) == tag)
            ex, fa = o["executp"].to_numpy(), o["fa"].to_numpy()
            print(f"   {tag}: n {o.height:,}; fa > 0 {np.mean(fa > 0):.3f}; executp > 0 {np.mean(ex > 0):.3f}, < 0 {np.mean(ex < 0):.3f}, = 0 {np.mean(ex == 0):.3f}; "
                  f"fa < lthr {np.mean(fa < o['lthr'].to_numpy()):.3f}, fa > -lthr {np.mean(fa > -o['lthr'].to_numpy()):.3f}; |fa| / hs p10/p50/p90 {q(np.abs(fa) / o['hs'].to_numpy())}; "
                  f"qty p10/p50/p90 {q(o['qty'].to_numpy())}, qty / |executp| {q(o['qty'].to_numpy() / np.abs(ex))}; pos {q(o['pos'].to_numpy())}; asq {q(o['asq'].to_numpy())}")
    if trk.height and "idx" in trk.columns:  # rejected orders: which side and do those stocks hold anything at boot (trk BHOLD / POS)
        ords = trk.filter(pl.col("kind") == "ORD").select("idx", "symbol", "side")
        rj = trk.filter(pl.col("kind") == "REJ").select("idx").join(ords, on="idx", how="left")
        hold = {}
        for k, col in (("BHOLD", "sellable_qty"), ("POS", "qty")):
            if col in trk.columns:
                for sym, v in trk.filter(pl.col("kind") == k).select("symbol", col).iter_rows():
                    if sym is not None and v not in (None, ""):
                        hold[str(sym)[:6]] = hold.get(str(sym)[:6], 0.0) + float(v)
        if rj.height:
            syms = rj["symbol"].cast(pl.String).str.slice(0, 6).to_list()
            print(f"   REJ: sides {dict(Counter(rj['side'].to_list()))}; on stocks holding nothing at boot {np.mean([hold.get(x, 0.0) <= 0 for x in syms]):.3f} "
                  f"({len(set(syms))} stocks, {sum(1 for x in set(syms) if hold.get(x, 0.0) <= 0)} of them with no holding); stocks with a boot holding: {sum(1 for v in hold.values() if v > 0)}")
    dn = d.with_columns(pl.col("executp").abs().alias("ax"))
    if "executp" in dn.columns:
        hit = dn.join_asof(p.select("sid", pl.col("po_ts").alias("po")), left_on="src_epoch", right_on="po", by="sid", strategy="forward",
                           tolerance=200_000_000).with_columns(pl.col("po").is_not_null().alias("placed"))
        tab = hit.with_columns(pl.col("ax").cut([0, 99, 199, 999, 9999]).alias("bin")).group_by("bin").agg(
            pl.len().alias("n"), pl.col("placed").mean().alias("p")).sort("bin")
        print("   decisions followed by a Po within 200 ms, by |executp|: " + " | ".join(f"{b} {pp:.3f} ({n:,})" for b, n, pp in tab.iter_rows()))
        fa, lt, ex = hit["fa"].to_numpy(), hit["lthr"].to_numpy(), hit["executp"].to_numpy()
        sel = hit.filter(pl.col("fa") < pl.col("lthr"))
        if sel.height and "nullptr" in sel.columns:  # what holds back a sell signal: a live order, nothing to sell, the position
            print("   fa < lthr, Po rate by: nullptr " + ", ".join(f"{k}: {v:.3f} ({n:,})" for k, n, v in sel.group_by("nullptr").agg(pl.len(), pl.col("placed").mean()).sort("nullptr").iter_rows())
                  + " | asq > 0 " + ", ".join(f"{k}: {v:.3f} ({n:,})" for k, n, v in sel.group_by(pl.col("asq") > 0).agg(pl.len(), pl.col("placed").mean()).sort("asq").iter_rows())
                  + " | pos > 0 " + ", ".join(f"{k}: {v:.3f} ({n:,})" for k, n, v in sel.group_by(pl.col("pos") > 0).agg(pl.len(), pl.col("placed").mean()).sort("pos").iter_rows()))
    if POSM.height:  # the strategy's own target / reserve / available, and the position view chrono gives it
        pm = POSM
        parts = []
        for col in ("target_qty", "reserve_qty", "current_qty", "qty_avl", "sellable_qty", "yesterday_qty"):
            if col in pm.columns:
                v = pm[col].drop_nulls().to_numpy()
                if v.size:
                    parts.append(f"{col} n {v.size:,} zero {np.mean(v == 0):.3f} p50 {np.median(v):.0f}")
        print("-- 8 positions: " + "; ".join(parts))
        for name, m in (("fa < lthr", fa < lt), ("lthr <= fa <= -lthr", (fa >= lt) & (fa <= -lt)), ("fa > -lthr", fa > -lt)):
            print(f"   {name:20s}: n {m.sum():,}, executp < 0 {np.mean(ex[m] < 0):.3f}, = 0 {np.mean(ex[m] == 0):.3f}, > 0 {np.mean(ex[m] > 0):.3f}, Po {hit['placed'].to_numpy()[m].mean():.3f}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--dates", required=True)
    ap.add_argument("--out", type=Path, default=Path("/work/crucible_data/pm_orders"))
    a = ap.parse_args()
    for d in [x for x in a.dates.split(",") if x]:
        files = log_files(a.logs, d)
        if not files:
            print(f"{d}: no logs")
            continue
        snaps, decs, trk, sigs, census, po = parse_day(files)
        out = a.out / f"date={d}"
        out.mkdir(parents=True, exist_ok=True)
        for name, df in (("snaps", snaps), ("decisions", decs), ("trk", trk), ("po", po), ("life", LIFE_DF)):
            if df.height:
                df.write_parquet(out / f"{name}.parquet")
        report(d, snaps, decs, trk, sigs, census, out)
        po_report(decs, trk, po)


if __name__ == "__main__":
    main()
