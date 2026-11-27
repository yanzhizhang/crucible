r"""r234: why PM's zzmh.r wgt is 8/7 of ours (r232: sum and wgt both x1.143, t equal, mean equal).

8/7 = 1 / (1 - 2^-3): a half-life-1 average updated every 3 s at the end. Replays zzmh.r (x = 4 ((hm - p) / hm)^2, hm = max
snapshot price from 09:30 exchange time, every snapshot an update, half 1) on the catalog feed and on PM's own .mp feed, both
clocks, and scores wgt against PM's .jz; then prints the last snapshots of a few stocks on both feeds. Read-only.

    python research/engine/mh_probe.py --date 20261119 --jz /work/crucible_data/seed_probe/HS300/20261119.jz \
        --catalog /work/crucible_data/catalog_dysl2.duckdb --pm-mp /work/crucible_data/pm_mp_cache/20261119
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))

from ema_rules import end_state  # noqa: E402


def replay(q, clk: str, mode: str) -> tuple[float, float, float]:
    px = q["price"].to_numpy().astype(float)
    s = q["s"].to_numpy().astype(float)
    t = q[clk].to_numpy().astype(float)
    keep = np.ones(len(px), bool)
    if mode == "dedup stamp":  # one update per new exchange stamp
        keep = np.concatenate([[True], np.diff(s) != 0])
    elif mode == "dedup cv":  # one update per volume change
        cv = q["cv"].to_numpy().astype(float)
        keep = np.concatenate([[True], np.diff(cv) != 0])
    hm = np.fmax.accumulate(np.where((s >= 34200) & (px > 0), px, np.nan))
    x = 4.0 * ((hm - px) / hm) ** 2
    m = keep & (px > 0) & np.isfinite(hm)
    return end_state(t[m], x[m], 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--pm-mp", type=Path, required=True)
    ap.add_argument("--show", type=int, default=4)
    a = ap.parse_args()
    import dcp_ema as E
    import duckdb
    import mp_load
    from stream import universe_from_state

    with gzip.open(a.jz) as f:
        pm = {int(s): kv for s, kv in json.load(f).items()}
    sids, stop = universe_from_state(a.jz)
    con = duckdb.connect(str(a.catalog), read_only=True)
    feeds = {"catalog": E.load(con, int(a.date), sids, stop)["fast"], "pm_mp": mp_load.load(a.pm_mp, sids, stop)["fast"]}
    by = {k: {int(g["sid"][0]): g for g in v.partition_by("sid")} for k, v in feeds.items()}
    ratio: dict[str, list[float]] = {}
    per: dict[int, float] = {}
    for sid in sids:
        ref = pm.get(sid, {})
        w, sm, tt = ref.get("zzmh.r.wgt"), ref.get("zzmh.r.sum"), ref.get("zzmh.r.t")
        if not isinstance(w, (int, float)) or not w:
            continue
        for feed, d in by.items():
            q = d.get(sid)
            if q is None:
                continue
            for clk in ("t_local", "t_exch"):
                for mode in ("every", "dedup stamp", "dedup cv"):
                    got = replay(q, clk, mode)
                    k = f"{feed:8s} {clk:8s} {mode:12s}"
                    r = ratio.setdefault(k, [])
                    r.append(w / got[1] if got[1] else math.nan)
                    ratio.setdefault(k + " t_eq", []).append(float(abs(got[2] - tt) < 1e-6) if isinstance(tt, (int, float)) else math.nan)
                    if feed == "catalog" and clk == "t_local" and mode == "every":
                        per[sid] = r[-1]
    print(f"zzmh.r wgt: PM / replay over {len(per)} stocks (q10 / q50 / q90); t_eq = share with the same last t")
    for k, v in ratio.items():
        v = np.array(v, float)
        if k.endswith("t_eq"):
            print(f"   {k:40s} {np.nanmean(v):.3f}")
        else:
            print(f"   {k:40s} {' / '.join(f'{x:.5f}' for x in np.nanpercentile(v, [10, 50, 90]))}")
    hi = sorted(per, key=lambda s: -per[s])[:a.show]
    lo = sorted(per, key=lambda s: per[s])[:2]
    for sid in hi + lo:
        ref = pm[sid]
        print(f"\n== {sid}: PM wgt {ref.get('zzmh.r.wgt'):.6f} t {ref.get('zzmh.r.t')}; PM / catalog replay {per[sid]:.5f}")
        for feed, d in by.items():
            q = d.get(sid)
            if q is None:
                continue
            tail = q.tail(8)
            print(f"   {feed}: n {len(q)}; last 8 (local sod | exch sod | t_local | t_exch | price | cv)")
            ls = (tail["ls"].to_numpy() / 1e9 + 28800) % 86400
            for i in range(len(tail)):
                print(f"      {ls[i]:10.3f} | {tail['s'][i]:9.3f} | {tail['t_local'][i]:9.3f} | {tail['t_exch'][i]:9.3f} | {tail['price'][i]:.3f} | {tail['cv'][i]:.0f}")
    print("[done]")


if __name__ == "__main__":
    main()
