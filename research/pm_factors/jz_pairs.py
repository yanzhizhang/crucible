r"""What PM's own day-end state says about a family's keys: every key's level, and which pairs of keys are ORDERED.

zzvx.a's training thresholds sit just above 1 (1.0000005 .. 1.14): a ratio of two quantities where one is never below the
other and they are often equal -- e.g. a true range against the previous close over the day's range. PM's .jz holds the
family's keys for every stock; per day and pooled:
  - every key matching --key-re: share present, quantiles of the value and of value / last price (averages as sum / wgt;
    the .wgt and .t of an average are listed as keys of their own);
  - every ordered pair (K1, K2) of same-scale keys with K1 >= K2 on (nearly) every stock: the share where they are equal,
    and the quantiles of K1 / K2 where they differ -- these are the candidate ">= 1 with a mass at 1" ratios;
  - for the listed thresholds (--thr name=t01,t50,t99) the pairs / single ratios whose 1% / 50% / 99% quantiles are closest.
Read-only.

    python research/pm_factors/jz_pairs.py --jz-dir /work/crucible_data/seed_probe/HS300 --days 20261112,20261113 --key-re '^zz(vx|xu)\.'
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import re
from pathlib import Path

import numpy as np


def load(path: Path, kre: re.Pattern) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """One .jz -> key -> value per stock (averages as sum / wgt, plus their .wgt and .t), last price per stock."""
    with gzip.open(path) as f:
        d = json.load(f)
    sids = sorted(d)
    keys = sorted({k for s in sids for k in d[s] if kre.search(k) and not re.search(r"\.\d+$", k)})
    col = lambda k: np.array([np.nan if d[s].get(k) is None else float(d[s][k]) for s in sids])  # noqa: E731
    out: dict[str, np.ndarray] = {}
    for k in keys:
        if k.endswith(".sum") and f"{k[:-4]}.wgt" in keys:
            w = col(f"{k[:-4]}.wgt")
            with np.errstate(divide="ignore", invalid="ignore"):
                out[k[:-4]] = np.where(w != 0, col(k) / w, np.nan)
        else:
            out[k] = col(k)
    return out, col("cdc.lp")


def q(v: np.ndarray, qs=(1, 50, 99)) -> str:
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.6g}" for x in np.percentile(v, qs)) if v.size else "-"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jz-dir", type=Path, required=True)
    ap.add_argument("--days", required=True)
    ap.add_argument("--key-re", required=True)
    ap.add_argument("--thr", action="append", default=[], help="name=t01,t50,t99: a column's threshold quantiles to match")
    ap.add_argument("--top", type=int, default=8)
    a = ap.parse_args()
    kre = re.compile(a.key_re)
    ds = [x for x in a.days.split(",") if x and (a.jz_dir / f"{x}.jz").exists()]
    per = [load(a.jz_dir / f"{d}.jz", kre) for d in ds]
    keys = sorted(set.intersection(*[set(k) for k, _ in per])) if per else []
    K = {k: np.concatenate([kv[k] for kv, _ in per]) for k in keys}
    px = np.concatenate([p for _, p in per])
    print(f"days {ds}: {px.size} stock-days, {len(keys)} keys")
    print("-- keys: present share | value q1 / q50 / q99 | value / last price q1 / q50 / q99")
    for k in keys:
        v = K[k]
        print(f"   {k:<22} {float(np.isfinite(v).mean()):.3f} | {q(v)} | {q(v / px)}")
    val = [k for k in keys if not k.endswith((".wgt", ".t"))]
    cands: dict[str, np.ndarray] = {}
    print("-- ordered pairs K1 >= K2 on >= 99.5% of stock-days (same scale: ratio within [1, 5]): equal share | K1 / K2 where different q1 / q50 / q99 / max")
    with np.errstate(divide="ignore", invalid="ignore"):
        for k1, k2 in itertools.permutations(val, 2):
            r = K[k1] / K[k2]
            ok = np.isfinite(r) & (K[k2] > 0)
            if ok.sum() < 50:
                continue
            cands[f"{k1}/{k2}"] = r
            rr = r[ok]
            if float((rr >= 1 - 1e-12).mean()) >= 0.995 and np.percentile(rr, 99) <= 5:
                eq = np.abs(rr - 1) < 1e-12
                dif = rr[~eq]
                print(f"   {k1} >= {k2}: equal {float(eq.mean()):.3f} | {q(dif, (1, 50, 99, 100)) if dif.size else 'never different'}")
        for k in val:
            cands[f"{k}/px"] = K[k] / px
            cands[f"px/{k}"] = px / K[k]
            cands[f"(px-{k})/{k}"] = (px - K[k]) / K[k]
            cands[f"(px-{k})/px"] = (px - K[k]) / px
        for k1, k2, k3 in itertools.permutations(val, 3):
            if k2 < k3:
                continue
            cands[f"({k1}-{k2})/({k3}-{k2})"] = (K[k1] - K[k2]) / (K[k3] - K[k2])
            cands[f"({k1}-{k2})/{k3}"] = (K[k1] - K[k2]) / K[k3]
    for spec in a.thr:
        name, _, ts = spec.partition("=")
        t = np.array([float(x) for x in ts.split(",")])
        res = []
        for cn, v in cands.items():
            v = v[np.isfinite(v)]
            if v.size < 50:
                continue
            c = np.percentile(v, [1, 50, 99])
            sc = max(abs(t[2] - t[0]), 1e-12)
            res.append((float(np.abs(c - t).sum() / sc), cn, c))
        res.sort(key=lambda r: r[0])
        print(f"-- {name}: training thresholds q1 / q50 / q99 = {' / '.join(f'{x:.6g}' for x in t)}; closest day-end ratios (distance in threshold-range units):")
        for dist, cn, c in res[: a.top]:
            print(f"   {dist:.3f}  {cn:<46} {' / '.join(f'{x:.6g}' for x in c)}")
    print("[done]")


if __name__ == "__main__":
    main()
