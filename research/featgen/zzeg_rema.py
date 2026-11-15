r"""Rebuild PM's zzeg.c.rEma (an average of the 4-day channel's span, price units) and fit its clock to the .jz.

zzeg.c.rEma.sum / .wgt sits within cents of zzeg.c.span at the close (600519 1104: 1416.81 vs 1416.79), so it is a decayed
average of the span. Weight 1 per update, decayed by exp(-dt ln2 / half) on the trading clock (engine/state.Ema). Per
half-life x start (09:25 auction included or 09:30) x seed (none, or yesterday's .jz sum / wgt): the day-end wgt and value
against PM's .jz -- the wgt alone pins the clock, the value then the input. The best variant is written at the scoring
rows with the candidates rEma / span, span / rEma, px / rEma, rEma / px, (px - rEma) / rEma. Output <out>/zzrema_<D>.parquet.

    python research/featgen/zzeg_rema.py --date 20261113 --x data/reports/stage6_r99_1113/x.parquet --raw /work/crucible_data/dysl2_raw \
        --jz /work/crucible_data/seed_probe/HS300/20261113.jz --prev-jz /work/crucible_data/seed_probe/HS300/20261112.jz --out data/side_r111
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from ratio_extra import CST_NS, asof, mark_ns  # noqa: E402
from zzeg_channel import DATA, day_hl  # noqa: E402

LN2 = math.log(2.0)


def tclock(sod: np.ndarray) -> np.ndarray:
    """Seconds of day -> trading clock (09:30 = 0, lunch removed; before 09:30 negative)."""
    return np.where(sod <= 41400, sod - 34200, np.where(sod < 46800, 7200.0, sod - 46800 + 7200))


def ema(t: np.ndarray, x: np.ndarray, half: float, s0: float, w0: float, t0: float | None) -> tuple[np.ndarray, np.ndarray]:
    """Running (sum, wgt) after each update."""
    k = LN2 / half
    S, W = np.empty(t.size), np.empty(t.size)
    s, w, last = s0, w0, t0
    for i in range(t.size):
        d = 1.0 if last is None else math.exp(-(t[i] - last) * k)
        s, w, last = s * d + x[i], w * d + 1.0, t[i]
        S[i], W[i] = s, w
    return S, W


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--x", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--hist", default="dysl2_raw,feitu_raw")
    ap.add_argument("--jz", type=Path, default=None, help="PM's day-end state of --date, to fit the clock")
    ap.add_argument("--prev-jz", type=Path, default=None, help="PM's state of the previous day, for the seeded variants")
    ap.add_argument("--halves", default="60,120,180,200,225,240,250,270,300,360,600")
    ap.add_argument("--start", default="0930", help="with no --jz to fit against: 0925 (auction included) or 0930")
    ap.add_argument("--out", type=Path, default=Path("data/side_rema"))
    a = ap.parse_args()
    d = a.date
    key = pl.read_parquet(a.x, columns=["date", "sid", "mark"]).filter(pl.col("date") == int(d)).drop("date").with_columns(pl.col("sid").cast(pl.Int64))
    sids = key["sid"].unique().cast(pl.Int32).to_list()
    raws = [DATA / s for s in a.hist.split(",")]
    days = sorted({p.name[5:] for r in raws for k in ("quotation", "transaction") for p in (r / f"kind={k}").glob("date=*") if p.name[5:] < d})
    prev = []
    for dd in reversed(days):
        hl = day_hl(dd, sids, raws)
        if hl is not None and hl.height > 0.5 * len(sids):
            prev.append(hl)
        if len(prev) == 3:
            break
    ph = pl.concat(prev).group_by("symbol_id").agg(pl.col("h").max().alias("ph"), pl.col("l").min().alias("pl"))
    ph = {int(r[0]): (r[1], r[2]) for r in ph.iter_rows()}
    q = (pl.read_parquet(list(a.raw.glob(f"kind=quotation/date={d}/part-*.parquet")), columns=["symbol_id", "time", "spider_ts", "high", "low", "price"])
         .filter(pl.col("symbol_id").is_in(sids) & (pl.col("price") > 0)).sort("symbol_id", "spider_ts"))
    pm = {int(s): kv for s, kv in json.loads(gzip.open(a.jz).read()).items()} if a.jz and a.jz.exists() else {}
    pv = {int(s): kv for s, kv in json.loads(gzip.open(a.prev_jz).read()).items()} if a.prev_jz and a.prev_jz.exists() else {}
    per = {}
    for (sid,), g in q.group_by("symbol_id", maintain_order=True):
        if sid not in ph:
            continue
        H = np.maximum(g["high"].to_numpy(), ph[sid][0])
        lo = g["low"].to_numpy()
        L = np.minimum(np.where(lo > 0, lo, np.inf), ph[sid][1])
        sod = ((g["spider_ts"].to_numpy() + CST_NS) / 1e9) % 86400
        exch = ((g["time"].to_numpy() + CST_NS) / 1e9) % 86400
        per[sid] = (tclock(sod), (H + L) / 2, exch, g["time"].to_numpy(), g["price"].to_numpy())
    res = []
    for half in (float(h) for h in a.halves.split(",")):
        for start in ("0925", "0930"):
            for seed in ("none", "prev") if pv else ("none",):
                ok_w, ok_v, n = 0, 0, 0
                for sid, (t, span, exch, _, _) in per.items():
                    m = exch >= (33300 if start == "0925" else 34200)
                    if not m.any() or sid not in pm or "zzeg.c.rEma.wgt" not in pm[sid]:
                        continue
                    s0 = w0 = 0.0
                    t0 = None
                    if seed == "prev" and sid in pv and pv[sid].get("zzeg.c.rEma.wgt"):
                        s0, w0 = pv[sid]["zzeg.c.rEma.sum"], pv[sid]["zzeg.c.rEma.wgt"]  # yesterday's close, carried undecayed (the engine's c0)
                    stop = pm[sid].get("zzeg.c.rEma.t", np.inf)
                    mm = m & (t <= stop + 1e-6)
                    if not mm.any():
                        continue
                    S, W = ema(t[mm], span[mm], half, s0, w0, t0)
                    n += 1
                    ok_w += abs(W[-1] - pm[sid]["zzeg.c.rEma.wgt"]) <= 1e-3 * pm[sid]["zzeg.c.rEma.wgt"]
                    ok_v += abs(S[-1] / W[-1] - pm[sid]["zzeg.c.rEma.sum"] / pm[sid]["zzeg.c.rEma.wgt"]) <= 1e-4 * abs(pm[sid]["zzeg.c.rEma.sum"] / pm[sid]["zzeg.c.rEma.wgt"])
                if n:
                    res.append((ok_w / n + ok_v / n, half, start, seed, ok_w / n, ok_v / n, n))
    res.sort(reverse=True)
    for _, half, start, seed, fw, fv, n in res[:12]:
        print(f"  half {half:6.0f}s start {start} seed {seed:<4}: wgt equal {fw:.3f}  value equal {fv:.3f}  ({n} stocks)")
    if not res:
        print("  no PM state to fit against")
        best = (float(a.halves.split(",")[0]), a.start, "none")
    else:
        best = res[0][1:4]
    half, start, seed = best
    rows = []
    marks = np.sort(key["mark"].unique().to_numpy())
    for sid, (t, span, exch, texch, px) in per.items():
        m = exch >= (33300 if start == "0925" else 34200)
        if not m.any():
            continue
        S, W = ema(t[m], span[m], half, 0.0, 0.0, None)
        rows.append(pl.DataFrame({"sid": np.full(m.sum(), sid, np.int64), "time": texch[m], "rema": S / W, "span": span[m], "px": px[m]}))
    ev = pl.concat(rows)
    g = asof(ev, ["sid"], key.join(pl.DataFrame({"mark": marks, "t": mark_ns(d, marks)}), on="mark"))
    R, Sp, P = pl.col("rema"), pl.col("span"), pl.col("px")
    out = g.select("sid", "mark", (R / Sp).alias("rema/span"), (Sp / R).alias("span/rema"), (P / R).alias("px/rema"), (R / P).alias("rema/px"),
                   ((P - R) / R).alias("(px-rema)/rema"))
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"zzrema_{d}.parquet"
    out.write_parquet(dst)
    v = out["rema/span"]
    print(f"{d}: best half {half:.0f}s start {start} seed {seed}; rema/span quantiles " + " ".join(f"{v.quantile(x):.4f}" for x in (0.01, 0.1, 0.5, 0.9, 0.99)) + f" -> {dst}")


if __name__ == "__main__":
    main()
