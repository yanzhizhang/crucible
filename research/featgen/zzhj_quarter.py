r"""zzhj from quarter-hour volume: check zzhj.p against PM's .jz, then write candidates for the zzhj column.

zzhj.p.t sits at ~14190 (14:56:30, the last of the engine's HJ_SAMPLE points: 30 s before each quarter-hour end) and
zzhj.p.wgt ~14.9 (about one update per quarter, 16 a day), value ~1.6e6 (about a day's volume / 16). So zzhj.p looks like
an average of per-quarter volume sampled at those points. Per stock: cumulative volume (and amount) by local arrival at
each sample point, the per-sample increments v_k; the average of v_k (plain, and decayed with --halves) vs PM's
sum / wgt. Then at each scoring mark: the running quarter's volume so far, the last sampled quarter's volume, and their
ratios to the average of the samples so far. Output <out>/zzhj_<D>.parquet.

    python research/featgen/zzhj_quarter.py --date 20261113 --x data/reports/stage6_r99_1113/x.parquet --raw /work/crucible_data/dysl2_raw \
        --jz /work/crucible_data/seed_probe/HS300/20261113.jz --out data/side_r115
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
from ratio_extra import CST_NS  # noqa: E402

SAMPLE = np.array([min(k * 900.0 + 870.0, 14190.0) for k in range(16)])


def tclock(ns: np.ndarray) -> np.ndarray:
    """UTC epoch ns -> trading clock seconds (09:30 = 0, lunch removed)."""
    s = ((ns + CST_NS) / 1e9) % 86400
    return np.where(s <= 41400, s - 34200, np.where(s < 46800, 7200.0, s - 46800 + 7200))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--x", type=Path, required=True)
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--jz", type=Path, default=None)
    ap.add_argument("--halves", default="0,3600,14400,28800,57600,86400")
    ap.add_argument("--out", type=Path, default=Path("data/side_zzhj"))
    a = ap.parse_args()
    d = a.date
    key = pl.read_parquet(a.x, columns=["date", "sid", "mark"]).filter(pl.col("date") == int(d)).drop("date").with_columns(pl.col("sid").cast(pl.Int64))
    sids = key["sid"].unique().cast(pl.Int32).to_list()
    q = (pl.read_parquet(list(a.raw.glob(f"kind=quotation/date={d}/part-*.parquet")), columns=["symbol_id", "spider_ts", "price", "total_volume", "total_amount"])
         .filter(pl.col("symbol_id").is_in(sids) & (pl.col("price") > 0)).sort("symbol_id", "spider_ts"))
    pm = {int(s): kv for s, kv in json.loads(gzip.open(a.jz).read()).items()} if a.jz and a.jz.exists() else {}
    per = {}
    for (sid,), g in q.group_by("symbol_id", maintain_order=True):
        t = tclock(g["spider_ts"].to_numpy())
        cv, ca = g["total_volume"].to_numpy().astype(float), g["total_amount"].to_numpy()
        i = np.searchsorted(t, SAMPLE, side="right") - 1  # last snapshot at or before each sample point
        okk = i >= 0
        cvs = np.where(okk, cv[np.maximum(i, 0)], 0.0)
        cas = np.where(okk, ca[np.maximum(i, 0)], 0.0)
        per[int(sid)] = (t, cv, np.diff(np.concatenate([[0.0], cvs])), np.diff(np.concatenate([[0.0], cas])), cvs)
    if pm:
        for what, j in (("volume", 2), ("amount", 3)):
            for half in (float(h) for h in a.halves.split(",")):
                ok = n = 0
                okw = 0
                for sid, p in per.items():
                    kv = pm.get(sid, {})
                    if not kv.get("zzhj.p.wgt"):
                        continue
                    v = p[j]
                    w = np.ones(v.size) if half == 0 else np.exp(-math.log(2) / half * (SAMPLE[-1] - SAMPLE))
                    est, tgt = (v * w).sum() / w.sum(), kv["zzhj.p.sum"] / kv["zzhj.p.wgt"]
                    n += 1
                    ok += abs(est - tgt) <= 1e-3 * abs(tgt)
                    okw += abs(w.sum() - kv["zzhj.p.wgt"]) <= 1e-3 * kv["zzhj.p.wgt"]
                print(f"  zzhj.p vs mean of sampled quarter {what}, half {half:>7.0f}s: value equal {ok / max(n, 1):.3f}  wgt equal {okw / max(n, 1):.3f}  ({n} stocks)")
        ex = next(iter(per))
        kv = pm.get(ex, {})
        print(f"  e.g. {ex}: PM zzhj.p {kv.get('zzhj.p.sum', 0) / max(kv.get('zzhj.p.wgt', 1), 1e-9):.6g} wgt {kv.get('zzhj.p.wgt')}; sampled volumes {np.round(per[ex][2]).astype(int).tolist()}")
        bins = sorted(k for k in kv if k.startswith("zzhj.c."))
        print(f"  e.g. {ex}: zzhj.c bins {[(k, kv[k]) for k in bins][:16]}")
    rows = []
    marks = np.sort(key["mark"].unique().to_numpy())
    for sid, (t, cv, vq, aq, cvs) in per.items():
        i = np.searchsorted(t, marks, side="right") - 1
        cum = np.where(i >= 0, cv[np.maximum(i, 0)], np.nan)
        k = np.searchsorted(SAMPLE, marks, side="right")  # sample points already passed
        last_cum = np.where(k > 0, cvs[np.maximum(k - 1, 0)], 0.0)
        prev_cum = np.where(k > 1, cvs[np.maximum(k - 2, 0)], 0.0)
        mean_q = np.array([vq[:kk].mean() if kk > 0 else np.nan for kk in k])
        run = cum - last_cum
        lastq = last_cum - prev_cum
        frac = np.clip((marks - np.where(k > 0, SAMPLE[np.maximum(k - 1, 0)], 0.0)) / 900.0, 1e-3, None)
        rows.append(pl.DataFrame({"sid": np.full(marks.size, sid, np.int64), "mark": marks, "hj.run/mean-1": run / mean_q - 1, "hj.last/mean-1": lastq / mean_q - 1,
                                  "hj.run_full/mean-1": run / frac / mean_q - 1, "hj.run/mean": run / mean_q, "hj.last/mean": lastq / mean_q}))
    out = key.join(pl.concat(rows), on=["sid", "mark"], how="left")
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"zzhj_{d}.parquet"
    out.write_parquet(dst)
    v = out["hj.last/mean-1"]
    print(f"{d}: {out.height:,} rows; last/mean-1 quantiles " + " ".join(f"{v.quantile(x):.3f}" for x in (0.01, 0.1, 0.5, 0.9, 0.99)) + f" -> {dst}")


if __name__ == "__main__":
    main()
