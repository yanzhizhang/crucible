r"""PM's live predictions, read from our executor's logs: chrono writes one ``trk SIG`` line per signal received, and PM
sends one signal per CSI 300 snapshot -- so the logs hold PM's model output at every snapshot, the end-to-end truth.

Per day: every ``trk SIG`` line of the chrono logs (``*_YYYYMMDD_*.log``, several files when chrono restarted, the same
reading as ``chrono/tools/pnl_report.py``) -> ``<out>/date=D.parquet`` (sid, md_ts = exchange time of the snapshot the
signal was built on, md_recv_ts = when PM received it, recv_ts, t = md_ts on the trading clock, prediction,
confidence, px, vol, name), plus a short report. Everything is aligned on md_ts, the snapshot's exchange time: a feed that
reaches PM late (224's test feed runs ~30 s behind) still names the snapshot each prediction was made on.

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


def trading_clock(ns: np.ndarray) -> np.ndarray:
    """Epoch ns -> seconds since 09:30 local with the lunch break removed (the clock of ``dcp_build.tc``)."""
    s = (ns / 1e9 + 28800) % 86400
    return np.where(s <= 41400, s - 34200, np.where(s < 46800, 7200.0, s - 46800 + 7200))


def read_signals(log_dir: Path, day: str) -> pl.DataFrame:
    """Every ``trk SIG`` line of the day's chrono logs, one row each."""
    dash = f"{day[:4]}-{day[4:6]}-{day[6:]}"  # 224's tee'd logs (chrono + PM's factor program): 2026-11-06_09-25-52-757.log
    files = sorted(set(glob.glob(os.path.join(str(log_dir), f"*_{day}_*.log")))
                   | set(glob.glob(os.path.join(str(log_dir), f"{dash}_*.log"))))
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
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    for d in [x for x in a.dates.split(",") if x]:
        t0 = time.perf_counter()
        sig = read_signals(a.logs, d)
        if sig.is_empty():
            print(f"{d}: no trk SIG lines in {a.logs} (*_{d}_*.log or {d[:4]}-{d[4:6]}-{d[6:]}_*.log)")
            continue
        sig = sig.with_columns(pl.Series("t", trading_clock(sig["md_ts"].to_numpy())))
        keep = [c for c in ("date", "sid", "symbol", "md_ts", "md_recv_ts", "recv_ts", "t", "prediction", "confidence", "px", "vol", "name")
                if c in sig.columns]
        sig.select(keep).sort("sid", "md_recv_ts").write_parquet(a.out / f"date={d}.parquet")
        if not a.no_report:
            report(con, d, sig)
        print(f"  -> {a.out / f'date={d}.parquet'} ({time.perf_counter() - t0:.0f}s)")
    con.close()


if __name__ == "__main__":
    main()
