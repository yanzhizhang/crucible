r"""Two structure tests on PM's day-end state (.jz), consecutive days.

1  zzvx: trEma* >= trDEma* and they differ on ~17% of stock-days (R158); rEma* is 1 with a tail above 1. Hypothesis:
   trD = the day's own range (high - low), tr = the TRUE range against the previous close
   (max(high, pc) - min(low, pc)), rEma = the average of tr / trD. With pc = the previous day's cdc.lp:
   share of stock-days where trEma / trDEma == TR / HL (to 1e-6), where both say "equal", and the same for rEma;
   the same against the open instead of the previous close.
2  zzxu: upper is always above the high, lower always below the low, dir is +-1. A band around a centre: for each
   centre X (p, last, high, low, (high + low) / 2, open) the quantiles of (upper - X) / a and (X - lower) / a
   (a = zzxu.a, also the zzvx range averages): a constant multiple = q10 == q90. dir against the signs of
   last - p, last - mid, and against which band is nearer. Then day-end values of the band ratios next to the two
   columns' training thresholds: zzxu.a 0.119 / 0.212 / 0.405, zzxu.b 0.848 / 0.981 / 1.261.
Read-only.

    python research/pm_factors/jz_bands.py --jz-dir /work/crucible_data/seed_probe/HS300 --days 20261110,20261111,20261112,20261113
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def load(path: Path) -> dict[str, dict[str, float]]:
    with gzip.open(path) as f:
        return json.load(f)


def val(kv: dict, k: str) -> float:
    """A key's value; an average as sum / wgt."""
    if f"{k}.sum" in kv:
        w = kv.get(f"{k}.wgt") or 0.0
        return kv[f"{k}.sum"] / w if w else float("nan")
    v = kv.get(k)
    return float("nan") if v is None else float(v)


def q(v: np.ndarray, qs=(1, 10, 50, 90, 99)) -> str:
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.6g}" for x in np.percentile(v, qs)) if v.size else "-"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jz-dir", type=Path, required=True)
    ap.add_argument("--days", required=True)
    a = ap.parse_args()
    ds = [x for x in a.days.split(",") if x and (a.jz_dir / f"{x}.jz").exists()]
    files = {d: load(a.jz_dir / f"{d}.jz") for d in ds}
    keys = ("zzvx.trEmaD", "zzvx.trDEmaD", "zzvx.rEmaD", "zzvx.high", "zzvx.low", "zzpk.open", "zzxu.last", "zzxu.p", "zzxu.a", "zzxu.upper", "zzxu.lower", "zzxu.dir", "cdc.lp")
    rows = []
    for i, d in enumerate(ds):
        prev = files[ds[i - 1]] if i else {}
        for sid, kv in files[d].items():
            r = [val(kv, k) for k in keys]
            pc = val(prev[sid], "cdc.lp") if sid in prev else float("nan")
            rows.append([*r, pc, float(i)])
    A = np.array(rows)
    tr, trd, rema, hi, lo, op, last, p, xa, up, lw, dr, _lp, pc, _ = A.T
    hl = hi - lo
    eq = lambda x, y, tol=1e-6: np.abs(x - y) <= tol * np.maximum(np.abs(y), 1e-12)  # noqa: E731
    print(f"days {ds}: {A.shape[0]} stock-days; with a previous close (from the previous file's cdc.lp): {int(np.isfinite(pc).sum())}")
    print("== 1 zzvx")
    with np.errstate(divide="ignore", invalid="ignore"):
        print(f"   trDEma / (high - low): {q(trd / hl)}; equal share {float(np.nanmean(eq(trd, hl))):.3f}")
        print(f"   trEma / trDEma: {q(tr / trd)}; share == 1: {float(np.nanmean(eq(tr, trd, 1e-9))):.3f}; rEma: {q(rema)}; share == 1: {float(np.nanmean(eq(rema, np.ones_like(rema), 1e-9))):.3f}")
        for name, ref in (("previous close", pc), ("open", op)):
            m = np.isfinite(ref) & np.isfinite(tr) & (hl > 0)
            trr = np.maximum(hi, ref) - np.minimum(lo, ref)
            ratio = trr / hl
            out = ratio > 1 + 1e-9
            print(f"   against the {name} (n {int(m.sum())}): ref outside the day's range on {float(out[m].mean()):.3f};"
                  f" trEma == TR: {float(eq(tr[m], trr[m]).mean()):.3f}; trEma / trDEma == TR / HL: {float(eq(tr[m] / trd[m], ratio[m]).mean()):.3f};"
                  f" both say equal-or-not alike: {float(((tr[m] / trd[m] > 1 + 1e-9) == out[m]).mean()):.3f}; rEma == TR / HL: {float(eq(rema[m], ratio[m]).mean()):.3f};"
                  f" rEma > 1 exactly where TR > HL: {float(((rema[m] > 1 + 1e-9) == out[m]).mean()):.3f}")
            dm = m & out
            if dm.sum() >= 20:
                print(f"      where the {name} is outside: (trEma / trDEma) / (TR / HL) {q(tr[dm] / trd[dm] / ratio[dm])}; (rEma - 1) / (TR / HL - 1) {q((rema[dm] - 1) / (ratio[dm] - 1))}")
        print("== 2 zzxu band")
        mid = (hi + lo) / 2
        for wn, w in (("zzxu.a", xa), ("zzvx.trDEma", trd), ("zzvx.trEma", tr), ("high - low", hl)):
            for cn, c in (("p", p), ("last", last), ("high", hi), ("low", lo), ("mid", mid), ("open", op)):
                u, l_ = (up - c) / w, (c - lw) / w
                su = np.nanpercentile(u, 90) / np.nanpercentile(u, 10) if np.nanpercentile(u, 10) > 0 else np.inf
                sl = np.nanpercentile(l_, 90) / np.nanpercentile(l_, 10) if np.nanpercentile(l_, 10) > 0 else np.inf
                if min(su, sl) < 3:
                    print(f"   width {wn:<12} centre {cn:<5}: (upper - X) / w {q(u)}  q90/q10 {su:.3f} | (X - lower) / w {q(l_)}  q90/q10 {sl:.3f}")
        print(f"   (upper - lower) / zzxu.a: {q((up - lw) / xa)}; (upper + lower) / 2 against p: {q((up + lw) / 2 / p)}, against last: {q((up + lw) / 2 / last)}, against mid: {q((up + lw) / 2 / mid)}")
        fin = np.isfinite(dr)
        print(f"   dir values: +1 {float((dr[fin] == 1).mean()):.3f}, -1 {float((dr[fin] == -1).mean()):.3f}, other {float(((dr[fin] != 1) & (dr[fin] != -1)).mean()):.3f};"
              f" dir == sign(last - p): {float((np.sign(last - p)[fin] == dr[fin]).mean()):.3f}; == sign(last - mid): {float((np.sign(last - mid)[fin] == dr[fin]).mean()):.3f};"
              f" == sign(last - open): {float((np.sign(last - op)[fin] == dr[fin]).mean()):.3f}; == sign(last - pc): {float((np.sign(last - pc)[fin & np.isfinite(pc)] == dr[fin & np.isfinite(pc)]).mean()):.3f};"
              f" == +1 where upper is the nearer band: {float(((up - last < last - lw) == (dr == 1))[fin].mean()):.3f}")
        act = np.where(dr > 0, lw, up)
        oth = np.where(dr > 0, up, lw)
        print("   day-end band ratios, q1 / q10 / q50 / q90 / q99 (thresholds: zzxu.a 0.119 / 0.212 / 0.405 at 1 / 50 / 99%; zzxu.b 0.848 / 0.981 / 1.261):")
        for nm, v in (("(upper - lower) / last", (up - lw) / last), ("(upper - lower) / p", (up - lw) / p), ("(upper - lower) / lower", (up - lw) / lw),
                      ("last / active band (lower if dir > 0 else upper)", last / act), ("last / other band", last / oth), ("active band / last", act / last),
                      ("(last - lower) / (upper - lower)", (last - lw) / (up - lw)), ("p / active band", p / act), ("last / upper", last / up), ("last / lower", last / lw),
                      ("|last - active| / zzxu.a", np.abs(last - act) / xa), ("(upper - last) / zzxu.a", (up - last) / xa), ("(last - lower) / zzxu.a", (last - lw) / xa)):
            print(f"      {nm:<50} {q(v)}")
    print("[done]")


if __name__ == "__main__":
    main()
