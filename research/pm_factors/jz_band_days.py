r"""Does PM's zzxu band carry from one day to the next? Consecutive day-end .jz files only.

In every variant tried the ACTIVE band (lower in an up trend, upper in a down trend) is never right, and the recomputed one
is. If the active band is a ratchet carried overnight, then for a stock in the same trend on D-1 and D it can only have
tightened: lower(D) >= lower(D-1) (up trend), upper(D) <= upper(D-1) (down trend), with equality when nothing tightened.
Per day pair and pooled:
  - the trend transitions (+1 -> +1, +1 -> -1, ...);
  - same-trend stocks: share with the active band unchanged (1e-9), tightened, loosened; the change in units of the range
    average; the inactive band's change for comparison;
  - where the active band is on D: (low - lower) / a, (lower - low(D-1)) / a ... quantiles; how often it equals yesterday's
    low / high / last / p or today's open-side values (exact matches would name the ratchet's input);
  - flipped stocks: the new active band against today's day-end recomputed value mid -+ 3 a.
Read-only.

    python research/pm_factors/jz_band_days.py --jz-dir /work/crucible_data/seed_probe/HS300 --days 20261110,20261111,20261112,20261113
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

KEYS = ("zzxu.upper", "zzxu.lower", "zzxu.dir", "zzxu.high", "zzxu.low", "zzxu.last", "zzpk.open")


def load(path: Path) -> dict[str, dict[str, float]]:
    with gzip.open(path) as f:
        d = json.load(f)
    out = {}
    for sid, kv in d.items():
        r = {k: (float(kv[k]) if kv.get(k) is not None else float("nan")) for k in KEYS}
        for k in ("zzxu.a", "zzxu.p"):
            w = kv.get(f"{k}.wgt") or 0.0
            r[k] = kv[f"{k}.sum"] / w if w and kv.get(f"{k}.sum") is not None else float("nan")
        out[sid] = r
    return out


def q(v: np.ndarray, qs=(1, 10, 50, 90, 99)) -> str:
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.5g}" for x in np.percentile(v, qs)) if v.size else "-"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jz-dir", type=Path, required=True)
    ap.add_argument("--days", required=True)
    a = ap.parse_args()
    ds = [x for x in a.days.split(",") if x and (a.jz_dir / f"{x}.jz").exists()]
    files = {d: load(a.jz_dir / f"{d}.jz") for d in ds}
    rows = []
    for d0, d1 in zip(ds, ds[1:], strict=False):
        for sid, r1 in files[d1].items():
            r0 = files[d0].get(sid)
            if r0:
                rows.append([r0[k] for k in (*KEYS, "zzxu.a", "zzxu.p")] + [r1[k] for k in (*KEYS, "zzxu.a", "zzxu.p")])
    A = np.array(rows)
    u0, l0, dr0, h0, lo0, la0, op0, a0, p0, u1, l1, dr1, h1, lo1, la1, op1, a1, p1 = A.T
    print(f"days {ds}: {A.shape[0]} stock-day pairs")
    print("-- trend transitions (D-1 -> D): " + "  ".join(f"{x:+.0f} -> {y:+.0f}: {int(((dr0 == x) & (dr1 == y)).sum())}" for x in (1, -1) for y in (1, -1)))
    eq = lambda x, y: np.abs(x - y) <= 1e-9 * np.maximum(np.abs(y), 1e-12)  # noqa: E731
    for name, m, act0, act1, ina0, ina1, sgn in (("up -> up (active = lower)", (dr0 == 1) & (dr1 == 1), l0, l1, u0, u1, 1.0),
                                                 ("down -> down (active = upper)", (dr0 == -1) & (dr1 == -1), u0, u1, l0, l1, -1.0)):
        n = int(m.sum())
        if n < 5:
            print(f"-- {name}: {n} pairs, skipped")
            continue
        ch = sgn * (act1[m] - act0[m])
        print(f"-- {name}: {n} pairs; active band unchanged {float(eq(act1[m], act0[m]).mean()):.3f}, tightened {float((ch > 1e-9 * np.abs(act0[m])).mean()):.3f},"
              f" loosened {float((ch < -1e-9 * np.abs(act0[m])).mean()):.3f}; change / a(D): {q(ch / a1[m])}")
        print(f"   inactive band unchanged {float(eq(ina1[m], ina0[m]).mean()):.3f}; its change / a(D): {q(sgn * (ina1[m] - ina0[m]) / a1[m])}")
        ext1, ext0 = (lo1, lo0) if sgn > 0 else (h1, h0)
        print(f"   active band's distance beyond the day's extreme, (low - lower) / a or (upper - high) / a: {q(sgn * (ext1[m] - act1[m]) / a1[m])}; against yesterday's extreme: {q(sgn * (ext0[m] - act1[m]) / a1[m])}")
        mid1, mid0 = (h1 + lo1) / 2, (h0 + lo0) / 2
        cands = {"yesterday's active band": act0, "yesterday's inactive band": ina0, "yesterday's low": lo0, "yesterday's high": h0, "yesterday's last": la0, "yesterday's p": p0,
                 "yesterday's mid": mid0, "today's low": lo1, "today's high": h1, "today's open": op1, "today's mid - 3 a": mid1 - 3 * a1, "today's mid + 3 a": mid1 + 3 * a1,
                 "yesterday's mid - 3 a(D-1)": mid0 - 3 * a0, "yesterday's mid + 3 a(D-1)": mid0 + 3 * a0, "yesterday's low - 2.5 a(D-1)": lo0 - 2.5 * a0,
                 "yesterday's high + 2.5 a(D-1)": h0 + 2.5 * a0, "today's open - 3 a(D-1)": op1 - 3 * a0, "today's open + 3 a(D-1)": op1 + 3 * a0}
        hits = sorted(((float(eq(act1[m], v[m]).mean()), k) for k, v in cands.items()), reverse=True)
        print("   active band on D equal to: " + ", ".join(f"{k} {s:.3f}" for s, k in hits[:6]))
    for name, m, act1, sgn in (("up -> down (new active = upper)", (dr0 == 1) & (dr1 == -1), u1, -1.0), ("down -> up (new active = lower)", (dr0 == -1) & (dr1 == 1), l1, 1.0)):
        n = int(m.sum())
        if n < 5:
            continue
        mid1 = (h1 + lo1) / 2
        ext1 = lo1 if sgn > 0 else h1
        print(f"-- {name}: {n} pairs; (mid -+ 3 a) relation, (active - mid) / a: {q((act1[m] - mid1[m]) / a1[m])}; beyond the extreme by / a: {q(sgn * (ext1[m] - act1[m]) / a1[m])}")
    print("[done]")


if __name__ == "__main__":
    main()
