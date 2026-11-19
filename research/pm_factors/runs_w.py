r"""The clock behind runS's close relaxation: lthr = -w (cr + msr / 2), and vpen = 0.001 sqrt(w') after 13:00.

R180: lthr / (-w (cr + msr / 2)) with w = (vpen / 0.001)^2 sits in 0.998 .. 1 but is never exact, so the two ramps do not
use the same clock or the same end. From the log alone (each decision's lthr, vpen, the snapshot's receive time
src_epoch) and the start-of-day msr:
  - the implied weights w_l = -lthr / (cr + msr / 2) and w_v = (vpen / 0.001)^2;
  - each regressed on the receive time: start (w = 0) and end (w = 1) as clock times, and the residual in seconds
    (a ramp driven by the EXCHANGE time of the snapshot shows a residual of the feed delay; by the local time, ~0);
  - the implied time t = start + w x length: is it a whole number of seconds / of 3 s (an exchange stamp), and its lag
    behind the receive time;
  - w_l / w_v through the afternoon.
Read-only.

    python research/pm_factors/runs_w.py --orders /work/crucible_data/pm_orders --date 20261118 --sts /work/crucible_data/sts_224
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
CR = 0.0004


def hms(s: float) -> str:
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--sts", type=Path, default=Path("/work/crucible_data/sts_224"))
    a = ap.parse_args()
    from ic_score import pm_daily
    c = pl.col
    d = pl.read_parquet(a.orders / f"date={a.date}" / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64)).drop_nulls(["src_epoch", "lthr", "ev_vpen"])
    d = d.with_columns(((c("src_epoch") / 1e9 + 28800) % 86400).alias("sod"), c("lthr").cast(pl.Float64), c("ev_vpen").cast(pl.Float64))
    d = d.join(pm_daily(a.sts, a.date, sorted(set(d["sid"].to_list())), "on").select("sid", "pm.msr"), on="sid", how="left").filter((c("sod") >= 46800) & (c("lthr") != 0) & c("pm.msr").is_not_null())
    sod, lt, vp, msr = (d[k].to_numpy().astype(float) for k in ("sod", "lthr", "ev_vpen", "pm.msr"))
    wl, wv = -lt / (CR + msr / 2), (vp / 0.001) ** 2
    wr = ((vp - 1e-7) / (0.001 - 1e-7)) ** 2  # decoded GVpen: vpen = (1 - r) cfg + r 0.001, r = sqrt(w), cfg = 1e-07
    print(f"== {a.date}: {d.height:,} decisions after 13:00 with lthr != 0; src_epoch has a fractional second on {float((np.mod(sod, 1) != 0).mean()):.3f} of them")
    for name, w in (("w_l = -lthr / (cr + msr / 2)", wl), ("w_v = (vpen / 0.001)^2", wv), ("w_r = ((vpen - 1e-7) / (0.001 - 1e-7))^2", wr)):
        A = np.column_stack([sod, np.ones(sod.size)])
        (b, a0), *_ = np.linalg.lstsq(A, w, rcond=None)
        res = (w - (a0 + b * sod)) / b
        t0, t1 = -a0 / b, (1 - a0) / b
        print(f"-- {name}: w in [{w.min():.6f}, {w.max():.6f}]; fit on the receive time: starts {hms(t0)}, ends {hms(t1)}, length {1 / b:.3f} s;"
              f" residual in seconds q1/10/50/90/99 {' / '.join(f'{x:.3f}' for x in np.percentile(res, [1, 10, 50, 90, 99]))}")
        for start, end, tag in ((46800.0, 53940.0, "13:00:00 -> 14:59:00"), (46800.0, 54000.0, "13:00:00 -> 15:00:00"), (46800.0, 53820.0, "13:00:00 -> 14:57:00")):
            t = start + w * (end - start)
            lag = sod - t
            fr = np.abs(t - np.round(t))
            print(f"   as {tag}: implied time is a whole second (1e-4) on {float((fr < 1e-4).mean()):.4f}, a multiple of 3 s on {float(((fr < 1e-4) & (np.mod(np.round(t), 3) == 0)).mean()):.4f};"
                  f" receive time - implied time q1/10/50/90/99 {' / '.join(f'{x:.3f}' for x in np.percentile(lag, [1, 10, 50, 90, 99]))} s")
    r = wl / wv
    print(f"-- w_l / w_v q1/10/50/90/99 {' / '.join(f'{x:.6f}' for x in np.percentile(r, [1, 10, 50, 90, 99]))}; by half hour (median): "
          + ", ".join(f"{hms(float(b) * 1800)[:5]} {np.median(r[(sod // 1800) == b]):.6f}" for b in sorted(set((sod // 1800).tolist()))))
    print(f"-- w_l / w_r q1/10/50/90/99 {' / '.join(f'{x:.9f}' for x in np.percentile(wl / wr, [1, 10, 50, 90, 99]))}; equal to 1e-9 on {float((np.abs(wl / wr - 1) < 1e-9).mean()):.4f}, to 1e-6 on {float((np.abs(wl / wr - 1) < 1e-6).mean()):.4f}")
    k = np.abs(lt) / (wv * (CR + msr / 2))
    print(f"   exact lthr == -w_v (cr + msr / 2) (1e-9) on {float((np.abs(k - 1) < 1e-9).mean()):.4f}; msr / 2 replaced by msr: ratio median {np.median(np.abs(lt) / (wv * (CR + msr))):.5f}")
    print("[done]")


if __name__ == "__main__":
    main()
