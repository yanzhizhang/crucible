r"""PM's live predictions, read from our executor's logs: chrono writes one ``trk SIG`` line per signal received, and PM
sends one signal per CSI 300 snapshot -- so the logs hold PM's model output at every snapshot, the end-to-end truth.

Per day: every ``trk SIG`` line of the chrono logs (``*_YYYYMMDD_*.log``, several files when chrono restarted, the same
reading as ``chrono/tools/pnl_report.py``) -> ``<out>/date=D.parquet`` (sid, md_ts = exchange time of the snapshot the
signal was built on, md_recv_ts = when PM received it, recv_ts, t = md_recv_ts on the trading clock -- PM gets a
snapshot ~40 ms after us, so the latest PM prediction at a mark matches our latest snapshot there --, prediction,
confidence, px, vol, name, src = sig / fa), plus a short report. Signals PM built on a late feed (md_recv_ts - md_ts over
``--max-delay``) are dropped: their snapshot is not the one our features see at that time.

224's logs also hold PM's own factor program output (the lines without a timestamp, ``qlive/catchRun.cpp`` +
``sol/alg.cpp``): per snapshot ``srcEpoch`` (PM's receive time), ``fa`` (the prediction, 5 decimals), ``emsMktVol`` /
``emsOurVol``, and at a trade decision ``pos / executp / asq / lthr / hs / winsdQty / asf``. chrono does not get every
signal, so a snapshot with an ``fa`` but no ``trk SIG`` joins as ``src = fa``; all of it is also written to
``<out>/state/date=D.parquet`` (``kind`` = snap / decision), and the report checks ``fa`` against the SIG prediction and
fits ``emsMktVol`` as an exponential moving sum of the snapshot volume.

* counts: signals, stocks, signals per stock, share of prediction == 0, prediction quantiles, signal names, and PM's
  feed delay md_recv_ts - md_ts (uniform = intervals, hence PM's clock-driven features, unchanged);
* against our snapshots: signals per snapshot (PM should be 1), share of md_ts equal to one of our snapshot exchange
  times, PM's receive time minus ours on those;
* what it predicts: rank IC (per 1-minute bucket) of the prediction against the forward mid return over 1 / 5 / 15 / 30
  minutes, raw and minus CSI 300 -- the peak names the label (tk5m = 5 minutes?).

``ic_score.py --target <out>`` then scores candidates against these predictions instead of the forward return.

    rsync -av zhangyanzhi@10.12.1.224:/tmp/zyq/sht-logs/ /work/crucible_data/chrono_logs/   # chrono + PM's program, tee'd
    python research/pm_factors/pm_signals.py --logs /work/crucible_data/chrono_logs --dates 20261105,20261106
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog
import dcp_ema as E  # tick loading
import feature_intraday as FI  # at_marks
import ic_score as IC  # index series
from pm_style_model import ic_buckets, summarize

TRK = re.compile(r"\btrk SIG (.*)$")
KV = re.compile(r"(\w+)=('([^']*)'|(\S*))")
INTS = ("md_ts", "md_recv_ts", "recv_ts", "send_ts", "handle_ts")
SHIFTED = ("send_ts", "md_ts", "md_recv_ts", "symbol", "px", "vol", "prediction", "confidence", "listener", "uuid", "name")


def repair(d: dict[str, str]) -> dict[str, str]:
    """chrono 3041e00 added ``send_ts={}`` to the SIG format without its argument, so every later value moved one key
    left (send_ts holds md_ts, md_ts holds md_recv_ts, md_recv_ts the symbol, ...): put them back."""
    if "send_ts" in d and not re.fullmatch(r"-?\d+", d.get("md_recv_ts", "")):
        vals = [d.get(k, "") for k in SHIFTED]
        d = {**d, **dict(zip(SHIFTED[1:], vals[:-1], strict=True))}
        d["send_ts"] = ""
    return d
FLOATS = ("prediction", "confidence", "px", "vol")
GLUE = re.compile(r"(?=\[(?:INFO|WARN|WARNING|ERROR|DEBUG)\] \[/)|(?=\[20\d\d-\d\d-\d\d )")  # where one log line starts
NUM = re.compile(r"([A-Za-z_][\w\->]*)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\b")


def log_files(log_dir: Path, day: str) -> list[str]:
    """chrono's own logs of the day and 224's tee'd ones (2026-11-06_09-25-52-757.log)."""
    dash = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    return sorted(set(glob.glob(os.path.join(str(log_dir), f"*_{day}_*.log")))
                  | set(glob.glob(os.path.join(str(log_dir), f"{dash}_*.log"))))


def read_factor(files: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """PM's factor program lines -> (one row per snapshot it scored, one row per trade decision).

    A ``catchRun.cpp`` line names the stock and its snapshot (srcEpoch); the ``alg.cpp`` lines after it carry
    book->epoch = that srcEpoch, so each is tied to the last catchRun line of the same file with the same epoch."""
    snaps, decisions, unmatched, loose = [], [], 0, 0
    for f in files:
        last: dict[str, float] = {}
        used = True  # the last catchRun line already has its alg line
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in (part for raw in fh for part in GLUE.split(raw) if part.strip()):  # tee can glue two writers' lines
                if "catchRun.cpp" in line and "srcEpoch" in line:
                    last, used = {k: float(v) for k, v in NUM.findall(line)}, False
                elif "alg.cpp" in line and "sidIdx" in line:
                    kv = {k: float(v) for k, v in NUM.findall(line)}
                    exact = bool(last) and abs(kv.get("book->epoch", -1.0) - last.get("srcEpoch", -2.0)) < 1e-4
                    # from 2026-11-09 book->epoch is no longer srcEpoch (a few ms to ~40 ms later): the next unused catchRun line
                    if exact or (last and "sid" in last and not used):
                        loose += not exact
                        used = True
                        snaps.append({"sid": int(last["sid"]), "src_epoch": round(last["srcEpoch"] * 1e9),
                                      "rec_epoch": round(last.get("recEpoch", float("nan")) * 1e9) if "recEpoch" in last else None,
                                      "book_epoch": round(kv["book->epoch"] * 1e9) if "book->epoch" in kv else None,
                                      "sid_idx": int(kv["sidIdx"]), "a_idx": int(kv.get("aIdx", 0)), "fa": kv.get("fa"),
                                      "ems_our_vol": kv.get("emsOurVol"), "ems_mkt_vol": kv.get("emsMktVol")})
                    else:
                        unmatched += 1
                elif "alg.cpp" in line and "executp" in line:
                    kv = {k.replace("this->", "").replace("_", ""): float(v) for k, v in NUM.findall(line.split("]", 2)[-1])}
                    decisions.append({"src_epoch": round(last["srcEpoch"] * 1e9) if "srcEpoch" in last else None, **kv})
    print(f"  factor lines: {len(snaps):,} matched ({loose:,} by order, not by epoch), {unmatched:,} alg lines without their catchRun line")
    if loose and snaps:  # an order match is right when each sidIdx keeps one sid and the epoch lag stays small
        sp = pl.DataFrame(snaps)
        top = sp.group_by("sid_idx", "sid").len().sort("len", descending=True).unique("sid_idx", keep="first")
        lag = ((sp["book_epoch"] - sp["src_epoch"]) / 1e6).drop_nulls().to_numpy()
        print(f"  order match check: {top['len'].sum() / sp.height:.4f} of rows on their sidIdx's main sid; "
              f"book->epoch - srcEpoch (ms) p1/p50/p99 " + "/".join(f"{np.percentile(lag, q):.1f}" for q in (1, 50, 99)))
    return pl.DataFrame(snaps), pl.DataFrame(decisions, infer_schema_length=None)


def ems_check(con: duckdb.DuckDBPyConnection, day: str, st: pl.DataFrame) -> None:
    """emsMktVol against exponential moving sums of our snapshot volume increments at PM's snapshot times."""
    st = st.filter(pl.col("ems_mkt_vol").is_not_null())
    if st.is_empty():
        return
    sids = sorted(st["sid"].unique().to_list())[:120]
    fast = E.load(con, int(day), sids, 54150.005)["fast"]
    hl = (30, 60, 120, 300, 600, 900, 1800, 3600)
    per = (0.9, 0.95, 0.98, 0.99, 0.995)
    got: dict[str, list] = {f"t{h}": [] for h in hl} | {f"a{al}": [] for al in per}
    ys = []

    def msum(dv: np.ndarray, dec: np.ndarray) -> np.ndarray:
        out, acc = np.empty(dv.size), 0.0
        for i in range(dv.size):
            acc = acc * dec[i] + dv[i]
            out[i] = acc
        return out

    for g in fast.partition_by("sid"):
        sid = int(g["sid"][0])
        p = st.filter(pl.col("sid") == sid)
        if p.height < 50 or g.height < 50:
            continue
        ar = E.arrays(g, "fast")
        t, dv = ar["t_local"], ar["dvol"]
        T = trading_clock(p["src_epoch"].to_numpy())
        ys.append(p["ems_mkt_vol"].to_numpy())
        dt = np.diff(t, prepend=t[0])
        for h in hl:
            got[f"t{h}"].append(FI.at_marks(t, msum(dv, np.exp(-dt * np.log(2) / h)), T))
        for al in per:
            got[f"a{al}"].append(FI.at_marks(t, msum(dv, np.full(t.size, al)), T))
    if not ys:
        return
    y = np.concatenate(ys)
    res = []
    for k, v in got.items():
        x = np.concatenate(v)
        m = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
        if m.sum() < 100:
            continue
        r = y[m] / x[m]
        res.append((k, np.median(r), np.mean(np.abs(r / np.median(r) - 1) < 1e-2), np.corrcoef(np.log(x[m]), np.log(y[m]))[0, 1]))
    res.sort(key=lambda z: -z[3])
    print("  emsMktVol vs moving sum of snapshot volume (t<half-life s> / a<decay per snapshot>), best 4 by log corr: "
          + "; ".join(f"{k} ratio {r:.3f} within 1% {w:.3f} logcorr {c:.4f}" for k, r, w, c in res[:4]))


def trading_clock(ns: np.ndarray) -> np.ndarray:
    """Epoch ns -> seconds since 09:30 local with the lunch break removed (the clock of ``dcp_build.tc``)."""
    s = (ns / 1e9 + 28800) % 86400
    return np.where(s <= 41400, s - 34200, np.where(s < 46800, 7200.0, s - 46800 + 7200))


def read_signals(log_dir: Path, day: str) -> pl.DataFrame:
    """Every ``trk SIG`` line of the day's chrono logs, one row each."""
    files = log_files(log_dir, day)
    rows = []
    for f in files:
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = TRK.search(line)
                if m:
                    rows.append(repair({k.group(1): k.group(3) if k.group(3) is not None else k.group(4) for k in KV.finditer(m.group(1))}))
    if not rows:
        return pl.DataFrame()
    df = pl.DataFrame(rows, infer_schema_length=None)
    df = df.with_columns([pl.col(c).cast(pl.Int64, strict=False) for c in INTS if c in df.columns]
                         + [pl.col(c).cast(pl.Float64, strict=False) for c in FLOATS if c in df.columns])
    sym = pl.col("symbol").str.strip_chars()
    df = df.with_columns(sym.str.slice(0, 6).cast(pl.Int64, strict=False).alias("sid"), pl.lit(int(day)).alias("date"))
    print(f"{day}: {len(files)} log file(s), {df.height:,} trk SIG lines")
    return df.filter(pl.col("sid").is_not_null() & pl.col("md_recv_ts").is_not_null())


def report(con: duckdb.DuckDBPyConnection, day: str, sig: pl.DataFrame) -> None:
    """The checks of the module docstring, a few lines per day."""
    p = sig["prediction"].to_numpy()
    per = sig.group_by("sid").len()["len"].to_numpy()
    print(f"  {sig.height:,} signals, {sig['sid'].n_unique()} stocks, per stock p10/p50/p90 "
          f"{np.percentile(per, 10):.0f}/{np.median(per):.0f}/{np.percentile(per, 90):.0f}; prediction == 0 {np.mean(p == 0):.3f}")
    nz = p[np.isfinite(p) & (p != 0)]
    if nz.size:
        print("  prediction (non-zero) p1/p10/p50/p90/p99 " + "/".join(f"{np.percentile(nz, q):.4g}" for q in (1, 10, 50, 90, 99))
              + f", distinct {np.unique(nz).size:,}")
    dl = (sig["md_recv_ts"] - sig["md_ts"]).to_numpy() / 1e9
    print("  PM feed delay md_recv_ts - md_ts (s) p1/p10/p50/p90/p99 " + "/".join(f"{np.percentile(dl, q):.2f}" for q in (1, 10, 50, 90, 99)))
    if "name" in sig.columns:
        print("  names: " + ", ".join(f"{n} {c}" for n, c in sig.group_by("name").len().sort("len", descending=True).head(5).iter_rows()))
    sids = sorted(sig["sid"].unique().to_list())
    up = E.load(con, int(day), sids, 54150.005)
    fast = up["fast"]
    if fast.is_empty():
        print("  no snapshots of ours for this day (decode it first): checks against the market skipped")
        return
    cont = fast.filter((pl.col("t_exch") > 0) & (pl.col("t_exch") < 14220))
    snaps = cont.group_by("sid").len().rename({"len": "snaps"})
    ratio = sig.group_by("sid").len().join(snaps, on="sid").with_columns((pl.col("len") / pl.col("snaps")).alias("r"))["r"].to_numpy()
    q = con.sql(f"SELECT symbol_id AS sid, time AS md_ts, spider_ts FROM raw_quotation WHERE date = {int(day)}").pl()
    j = sig.select("sid", "md_ts", "md_recv_ts").join(q, on=["sid", "md_ts"], how="left")
    hit = j["spider_ts"].is_not_null()
    lag = (j.filter(hit)["md_recv_ts"] - j.filter(hit)["spider_ts"]).to_numpy() / 1e6
    print(f"  signals per continuous-session snapshot p10/p50/p90 {np.percentile(ratio, 10):.2f}/{np.median(ratio):.2f}/{np.percentile(ratio, 90):.2f}; "
          f"md_ts = one of our snapshots {hit.mean():.3f}; PM receive - ours (ms) p10/p50/p90 "
          + ("/".join(f"{np.percentile(lag, x):.1f}" for x in (10, 50, 90)) if lag.size else "-"))
    ix = IC.index_series(con, int(day), 54150.005)
    t_all = trading_clock(sig["md_ts"].to_numpy())  # the snapshot's exchange time
    sid_all = sig["sid"].to_numpy()
    fwd = {h: np.full(sig.height, np.nan) for h in (60, 300, 900, 1800)}
    for g in fast.partition_by("sid"):
        s = int(g["sid"][0])
        a = E.arrays(g, "fast")
        ok = np.isfinite(a["mid"]) & (a["mid"] > 0)
        rows = np.flatnonzero(sid_all == s)
        T = t_all[rows]
        te = a["t_exch"][ok]  # forward returns on the exchange clock, from the snapshot PM used
        m0 = FI.at_marks(te, a["mid"][ok], T)
        for h in fwd:
            with np.errstate(divide="ignore", invalid="ignore"):
                fwd[h][rows] = np.where(T + h <= 14220, np.log(FI.at_marks(te, a["mid"][ok], T + h) / m0), np.nan)
    date = np.full(sig.height, int(day))
    for h, r in fwd.items():
        line = f"  IC vs forward {h // 60:>2} min: raw {summarize(ic_buckets(p, r, date, t_all))}"
        if "hs300" in ix:
            ti, li, _ = ix["hs300"]
            rex = r - (FI.at_marks(ti, li, t_all + h) - FI.at_marks(ti, li, t_all))
            line += f" | minus CSI 300 {summarize(ic_buckets(p, rex, date, t_all))}"
        print(line)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--logs", type=Path, required=True, help="directory of chrono logs (copied from 224)")
    ap.add_argument("--dates", required=True, help="comma list of YYYYMMDD")
    ap.add_argument("--out", type=Path, default=Path("/work/crucible_data/pm_signals"))
    ap.add_argument("--no-report", action="store_true", help="only write the parquet")
    ap.add_argument("--max-delay", type=float, default=2.0, help="drop signals PM built on a snapshot that reached it later than this (s)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    for d in [x for x in a.dates.split(",") if x]:
        t0 = time.perf_counter()
        sig = read_signals(a.logs, d)
        st, dec = read_factor(log_files(a.logs, d))
        if sig.is_empty() and st.is_empty():
            print(f"{d}: no trk SIG nor factor lines in {a.logs} (*_{d}_*.log or {d[:4]}-{d[4:6]}-{d[6:]}_*.log)")
            continue
        if not st.is_empty():
            (a.out / "state").mkdir(parents=True, exist_ok=True)
            pl.concat([st.with_columns(pl.lit("snap").alias("kind")), dec.with_columns(pl.lit("decision").alias("kind"))],
                      how="diagonal_relaxed").write_parquet(a.out / "state" / f"date={d}.parquet")
        if not sig.is_empty():
            late = ((sig["md_recv_ts"] - sig["md_ts"]) > a.max_delay * 1e9).to_numpy()
            print(f"  {sig.height:,} SIG lines, {int(late.sum()):,} dropped as built on a feed later than {a.max_delay:g} s")
            sig = sig.filter(pl.Series(~late))
            sig = sig.with_columns(pl.Series("t", trading_clock(sig["md_recv_ts"].to_numpy())), pl.lit("sig").alias("src"))
        if not st.is_empty():
            key = lambda c: ((pl.col(c) + 50_000) // 100_000).alias("k10us")  # noqa: E731 -- PM prints srcEpoch rounded to 10 us
            fa = st.with_columns(key("src_epoch"))
            if not sig.is_empty():
                both = sig.with_columns(key("md_recv_ts")).join(fa.select("sid", "k10us", "fa"), on=["sid", "k10us"], how="inner")
                diff = (both["prediction"] - both["fa"]).abs().to_numpy()
                cor = np.corrcoef(both["prediction"].to_numpy(), both["fa"].to_numpy())[0, 1] if both.height > 2 else float("nan")
                print(f"  factor program: {st.height:,} scored snapshots, {both.height:,} also in SIG "
                      f"(|prediction - fa| max {diff.max() if diff.size else float('nan'):.1e}, corr {cor:.6f}), {dec.height:,} trade decisions")
                fa = fa.join(sig.with_columns(key("md_recv_ts")).select("sid", "k10us"), on=["sid", "k10us"], how="anti")
            extra = fa.select("sid", pl.col("src_epoch").alias("md_recv_ts"), pl.col("fa").alias("prediction")).with_columns(
                pl.lit(int(d)).alias("date"), pl.Series("t", trading_clock(fa["src_epoch"].to_numpy())), pl.lit("fa").alias("src"))
            print(f"  {extra.height:,} snapshots scored by PM but missing from chrono's SIG: added from fa (5 decimals)")
            sig = extra if sig.is_empty() else pl.concat([sig, extra], how="diagonal_relaxed")
        keep = [c for c in ("date", "sid", "symbol", "md_ts", "md_recv_ts", "recv_ts", "t", "prediction", "confidence", "px", "vol", "name", "src")
                if c in sig.columns]
        sig.select(keep).sort("sid", "md_recv_ts").write_parquet(a.out / f"date={d}.parquet")
        if not a.no_report:
            on_sig = sig.filter(pl.col("src") == "sig") if "md_ts" in sig.columns else sig.clear()
            if on_sig.height:
                report(con, d, on_sig)
            if not st.is_empty():
                ems_check(con, d, st)
        print(f"  -> {a.out / f'date={d}.parquet'} ({time.perf_counter() - t0:.0f}s)")
    con.close()


if __name__ == "__main__":
    main()
