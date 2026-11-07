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
TRK = re.compile(r"\btrk (\w+) (.*)$")
KV = re.compile(r"(\w+)=('([^']*)'|(\S*))")
GLUE = re.compile(r"(?=\[(?:INFO|WARN|WARNING|ERROR|DEBUG)\] \[/)|(?=\[20\d\d-\d\d-\d\d )")  # where one log line starts


def parse_day(files: list[str]) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict]:
    """(snapshots, decisions, chrono trk rows, signals, census) of one day's logs."""
    snaps, decs, trk, sigs = [], [], [], []
    census: dict[tuple[str, str], list] = {}
    for f in files:
        last: dict[str, float] = {}
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
                if line.startswith("[20"):  # chrono's own timestamped lines other than trk
                    continue
                s = SRC.search(line)
                if not s:
                    continue
                where, rest = s.group(1), s.group(2)
                keys = tuple(k for k, _ in NUM.findall(rest))
                shape = (where, " ".join(keys) if keys else re.sub(r"\d+", "N", rest)[:60])
                c = census.setdefault(shape, [0, rest[:160]])
                c[0] += 1
                kv = {k.replace("this->", "").replace("book->", "book_").rstrip("_"): float(v) for k, v in NUM.findall(rest)}
                if where.endswith("catchRun.cpp:69") and "srcEpoch" in kv:
                    last = kv
                elif where.endswith("alg.cpp:169") and last:
                    snaps.append({"sid": int(last.get("sid", -1)), "src_epoch": round(last["srcEpoch"] * 1e9), "fa": kv.get("fa"),
                                  "ems_our_vol": kv.get("emsOurVol"), "ems_mkt_vol": kv.get("emsMktVol")})
                elif where.endswith("alg.cpp:251"):
                    pending = {"src_epoch": round(last["srcEpoch"] * 1e9) if "srcEpoch" in last else None, **kv}
                    pending["sid"] = int(kv.get("sid", last.get("sid", -1)))
                    decs.append(pending)
                elif where.endswith("alg.cpp:252") and pending is not None:
                    mn = re.search(r"nullptr\s*:\s*(\d+)", rest)  # tee can glue the next line onto this one
                    pending["nullptr"] = float(mn.group(1)) if mn else None
                    pending = None
    def frame(rows: list[dict]) -> pl.DataFrame:
        return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame()
    return frame(snaps), frame(decs), frame(trk), frame(sigs), census


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
        snaps, decs, trk, sigs, census = parse_day(files)
        out = a.out / f"date={d}"
        out.mkdir(parents=True, exist_ok=True)
        for name, df in (("snaps", snaps), ("decisions", decs), ("trk", trk)):
            if df.height:
                df.write_parquet(out / f"{name}.parquet")
        report(d, snaps, decs, trk, sigs, census, out)


if __name__ == "__main__":
    main()
