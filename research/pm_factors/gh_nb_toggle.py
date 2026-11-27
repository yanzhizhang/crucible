r"""Which single trade makes our zzgh.nb equal PM's: for every stock not already exact, toggle each trade alone (drop it if
counted, add it if not, flip its side) and report the toggles that give PM's nb at 1e-9, with that trade's times and fields.

Window = local trading clock, t > 0, slots [0, n - 1) (the best rule so far); side = the feed's own flag (ZZGH reads the trade
record's own side byte). The summary counts how many stocks one toggle fixes and where those trades sit. Read-only.

    python research/pm_factors/gh_nb_toggle.py --date 20261120 --jz /work/crucible_data/seed_probe/HS300/20261120.jz --catalog /work/crucible_data/catalog_dysl2.duckdb
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "engine"))

SESSION = 14220.0


def hms(s: float) -> str:
    s = float(s)
    return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:06.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--term", default="zzgh")
    ap.add_argument("--slots", type=int, default=948)
    ap.add_argument("--show", type=int, default=25, help="stocks listed in detail")
    a = ap.parse_args()
    import dcp_ema as E
    import duckdb
    from stream import universe_from_state

    with gzip.open(a.jz) as f:
        pm = {int(s): kv for s, kv in json.load(f).items()}
    sids, stop = universe_from_state(a.jz)
    con = duckdb.connect(str(a.catalog), read_only=True)
    slow = {int(g["sid"][0]): g for g in E.load(con, int(a.date), sids, stop)["slow"].partition_by("sid")}
    step = SESSION / a.slots
    n_exact = n_fixed = n_none = 0
    kinds, where, shown = Counter(), Counter(), 0
    for sid in sids:
        g, p = slow.get(sid), pm.get(sid, {}).get(f"{a.term}.nb")
        if g is None or not isinstance(p, (int, float)):
            continue
        t, s, amt, side, px, vol = (g[c].to_numpy().astype(float) for c in ("t_local", "s", "amt", "side", "price", "vol"))
        keep = (t > 0) & (t < (a.slots - 1) * step)
        B, S = amt[keep & (side > 0)].sum(), amt[keep & (side < 0)].sum()
        f = lambda b, s_: (b - s_) / (b + s_)  # noqa: E731
        if abs(f(B, S) - p) <= 1e-9:
            n_exact += 1
            continue
        sgn_b, sgn_s = (side > 0).astype(float), (side < 0).astype(float)
        d_in = np.where(keep, -1.0, 1.0)  # dropping a counted trade / adding an uncounted one
        cand = {"toggle": f(B + d_in * amt * sgn_b, S + d_in * amt * sgn_s),
                "flip side": np.where(keep, f(B - amt * sgn_b + amt * sgn_s, S - amt * sgn_s + amt * sgn_b), np.nan)}
        hits = [(kn, i) for kn, v in cand.items() for i in np.flatnonzero(np.abs(v - p) <= 1e-9)]
        if not hits:
            n_none += 1
            continue
        n_fixed += 1
        for kn, i in hits[:3]:
            act = ("drop" if keep[i] else "add") if kn == "toggle" else "flip"
            zone = "t<=0" if t[i] <= 0 else ("last slot / after" if t[i] >= (a.slots - 1) * step else ("first 60 s" if t[i] < 60 else "middle"))
            kinds[act] += 1
            where[(act, zone)] += 1
            if shown < a.show:
                print(f"   {sid:06d} {act:<4} trade {i}/{len(t)}: local tc {t[i]:.3f} exch {hms(s[i])} side {side[i]:+.0f} px {px[i]:g} vol {vol[i]:g} ({zone})")
        shown += 1
    print(f"== {a.date} {a.term}.nb: exact already {n_exact}, fixed by one trade {n_fixed}, not fixed by one trade {n_none}")
    print("   by action:", dict(kinds))
    for (act, zone), c in where.most_common():
        print(f"   {c:>4}  {act:<4} {zone}")
    print("[done]")


if __name__ == "__main__":
    main()
