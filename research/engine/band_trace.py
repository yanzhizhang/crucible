r"""Where in the day does PM's zzxu band value come from? A per-snapshot trace against PM's day-end upper / lower / dir.

For every stock the snapshot stream is replayed (snapshots only: the band's inputs are the snapshot price, the snapshot's
own high / low fields and snapshot-clock averages) keeping, per snapshot, the centres (mid of high / low, price, the 300 s
price average, high, low) and the widths (zzxu.a carried from yesterday "c0", zzxu.a from today only "a", zzvx.trEma "tr",
zzvx.trDEma "de"). PM's side per stock: dir > 0 -> lower is the ACTIVE band, upper the inactive one (and the mirror).

Printed:
1. inactive band = centre + mult x width at the LAST snapshot, per (centre, width, mult): share of stocks exact (the known rule
   should show mid / c0 / 3 at ~0.85);
2. active band, known formula (mid -+ 3 c0): the share of stocks where some snapshot's value equals PM's exactly; for those,
   when the hit is (half hour), whether PM's value is the running extreme of the formula from the hit on (a ratchet that started
   there), the extreme of the whole day, or EXCEEDED later (held, not ratcheted), and what the price did at the hit against the
   opposite band of the snapshot before (a flip);
3. active band, any formula: per (centre, width) the multiplier m = |centre - PM| / width at every snapshot; the count of
   stocks that have a snapshot with m within 1e-7 of a multiple of 0.25, per multiple (a real rule = one multiple on most stocks);
4. the same with the value frozen at the first / last snapshot of the afternoon or at fixed clock times is covered by 2 (the
   hit time histogram would pile up there).

    python research/engine/band_trace.py --date 20261120 --pm $SP/20261120.jz --prev $SP/20261119.jz --catalog /work/crucible_data/catalog_dysl2.duckdb
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

import duckdb
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))

from state import Ema

CENTRES = ("mid", "px", "p", "hi", "lo")
WIDTHS = ("c0", "a", "tr", "de")
MULTS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
TOL = 1e-9


def load_jz(path: Path) -> dict[int, dict[str, float]]:
    """A day-end state file: sid -> key -> value."""
    with gzip.open(path) as f:
        d = json.load(f)
    return {int(s): {k: v for k, v in kv.items() if isinstance(v, (int, float))} for s, kv in d.items()}


def trace(g: dict[str, np.ndarray], prev: dict[str, float] | None) -> dict[str, np.ndarray]:
    """Per snapshot with a price and both range fields: t, px, the centres and the widths, as the engine computes them."""
    p, a, c0, tr, de = Ema(300), Ema(300), Ema(300), Ema(30), Ema(30)
    if prev and prev.get("zzxu.a.wgt"):
        c0.s, c0.w = prev["zzxu.a.sum"], prev["zzxu.a.wgt"]
    rows = []
    for i in range(len(g["t"])):
        t, px = float(g["t"][i]), (float(g["price"][i]) if g["price"][i] > 0 else math.nan)
        hi, lo = (float(g["hi"][i]) if g["hi"][i] > 0 else math.nan), (float(g["lo"][i]) if g["lo"][i] > 0 else math.nan)
        hl = hi - lo
        p.update(t, px)
        for e in (a, c0, tr):
            e.update(t, hl)
        de.update(t, tr.value())
        if px == px and hl == hl:
            rows.append((t, float(g["s"][i]), px, (hi + lo) / 2, px, p.value(), hi, lo, c0.value(), a.value(), tr.value(), de.value()))
    m = np.array(rows, dtype=float).reshape(-1, 12)
    return {k: m[:, j] for j, k in enumerate(("t", "s", "price", "mid", "px", "p", "hi", "lo", "c0", "a", "tr", "de"))}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", type=int, required=True)
    ap.add_argument("--pm", type=Path, required=True)
    ap.add_argument("--prev", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="first N stocks only (0 = all)")
    ap.add_argument("--show", type=int, default=6, help="stocks to print in full")
    a = ap.parse_args()
    import dcp_ema as E
    from stream import universe_from_state

    pm, prev = load_jz(a.pm), load_jz(a.prev)
    sids, stop = universe_from_state(a.pm)
    sids = [s for s in sids if "zzxu.upper" in pm.get(s, {}) and "zzxu.lower" in pm[s]]
    if a.limit:
        sids = sids[:a.limit]
    con = duckdb.connect(str(a.catalog), read_only=True)
    fast = E.load(con, a.date, sids, stop)["fast"]
    parts = {int(g["sid"][0]): g for g in fast.partition_by("sid")}
    n = 0
    dirs = Counter()
    inact = Counter()
    hit = Counter()
    when = Counter()
    kind = Counter()
    flip = Counter()
    mult = {(c, w): Counter() for c in CENTRES for w in WIDTHS}
    shown = []
    for sid in sids:
        g = parts.get(sid)
        if g is None:
            continue
        tr = trace({"t": g["t_local"].to_numpy(), "s": g["s"].to_numpy(), "price": g["price"].to_numpy(), "hi": g["hi"].to_numpy(), "lo": g["lo"].to_numpy()}, prev.get(sid))
        if len(tr["t"]) < 10:
            continue
        n += 1
        d = pm[sid].get("zzxu.dir", math.nan)
        dirs[d] += 1
        up = d > 0  # dir up: lower is the active band, upper the inactive one
        act, ina = (pm[sid]["zzxu.lower"], pm[sid]["zzxu.upper"]) if up else (pm[sid]["zzxu.upper"], pm[sid]["zzxu.lower"])
        sa, si = (-1.0, 1.0) if up else (1.0, -1.0)  # sign of the width term: active, inactive
        tol = TOL * max(1.0, abs(act))
        for c in CENTRES:
            for w in WIDTHS:
                for m in MULTS:
                    if abs(tr[c][-1] + si * m * tr[w][-1] - ina) <= tol:
                        inact[(c, w, m)] += 1
                with np.errstate(divide="ignore", invalid="ignore"):
                    mm = sa * (act - tr[c]) / tr[w]
                q = np.round(mm * 4)
                ok = (np.abs(mm * 4 - q) <= 4e-7) & (q >= 1) & (q <= 40)
                for k in set(q[ok].astype(int).tolist()):
                    mult[(c, w)][k] += 1
        f = tr["mid"] + sa * 3.0 * tr["c0"]  # the known formula's active-side value at every snapshot
        idx = np.nonzero(np.abs(f - act) <= tol)[0]
        g_opp = tr["mid"] - sa * 3.0 * tr["c0"]
        if len(idx) == 0:
            hit["no snapshot gives PM's value"] += 1
            best = int(np.nanargmin(np.abs(f - act)))
            if len(shown) < a.show:
                shown.append(f"  {sid} dir {d:+.0f} active PM {act:.6f}: NO hit; nearest snapshot t {tr['t'][best]:.0f} value {f[best]:.6f} (off {f[best] - act:+.2e}); day extreme {(f.max() if up else f.min()):.6f}, last {f[-1]:.6f}")
            continue
        i0, i1 = int(idx[0]), int(idx[-1])
        hit["some snapshot gives PM's value"] += 1
        when[int(tr["t"][i0] // 1800)] += 1
        ext = (lambda x: x.max()) if up else (lambda x: x.min())
        beyond = (lambda x: x > act + tol) if up else (lambda x: x < act - tol)
        if abs(ext(f) - act) <= tol:
            kind["= extreme of the whole day"] += 1
        elif abs(ext(f[i0:]) - act) <= tol:
            kind["= extreme from the first hit on (ratchet started there)"] += 1
        elif i1 == len(f) - 1:
            kind["= the last snapshot's value (recomputed, not ratcheted)"] += 1
        else:
            kind[f"exceeded later (held from the hit; later snapshots beyond it: {'many' if beyond(f[i1 + 1:]).mean() > 0.2 else 'few'})"] += 1
        if i0 > 0:
            crossed = tr["px"][i0] > g_opp[i0 - 1] if up else tr["px"][i0] < g_opp[i0 - 1]
            flip["price beyond the previous snapshot's opposite band at the hit" if crossed else "no cross at the hit"] += 1
        else:
            flip["hit at the first snapshot"] += 1
        if len(shown) < a.show:
            shown.append(f"  {sid} dir {d:+.0f} active PM {act:.6f}: {len(idx)} hits, first t {tr['t'][i0]:.0f} (s {tr['s'][i0]:.0f}), last t {tr['t'][i1]:.0f} of {tr['t'][-1]:.0f}; "
                         f"day extreme {ext(f):.6f}, extreme from hit {ext(f[i0:]):.6f}, last {f[-1]:.6f}; later beyond {int(beyond(f[i1 + 1:]).sum())}/{len(f) - i1 - 1}")
    print(f"{a.date}: {n} stocks traced; PM dir counts {dict(dirs)}")
    print("1. inactive band at the last snapshot: (centre, width, mult) -> share of stocks exact (top 8)")
    for (c, w, m), k in inact.most_common(8):
        print(f"   {c:<4} {w:<3} {m:<4g} {k / n:.3f}")
    print(f"2. active band, mid -+ 3 c0: {dict(hit)}")
    print(f"   first hit by half hour of the trading clock: {dict(sorted(when.items()))}")
    for k, v in kind.most_common():
        print(f"   {v:>4}  {k}")
    for k, v in flip.most_common():
        print(f"   {v:>4}  {k}")
    print("3. active band, any formula: (centre, width) -> multiplier: stocks with an exact snapshot (top 4 multipliers each)")
    for (c, w), cnt in sorted(mult.items(), key=lambda kv: -max(kv[1].values(), default=0)):
        if cnt:
            print(f"   {c:<4} {w:<3} " + "  ".join(f"{k / 4:g}: {v}" for k, v in cnt.most_common(4)))
    print("examples:")
    print("\n".join(shown))


if __name__ == "__main__":
    main()
