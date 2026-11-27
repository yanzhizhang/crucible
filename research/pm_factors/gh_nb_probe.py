r"""zzgh / zzgh2 in PM's day-end .jz: what the two terms keep, and how nb relates.

Prints every key stem under zzgh / zzgh2 with its count and a sample value, whether zzgh.nb == zzgh2.nb per stock (if nb did
not depend on the slot grid they would be equal), the correlation of nb with the previous day's nb (a carried / blended
nb would show here), and nb against the imbalance of the stock's own slot codes. Reads only the .jz files.

    python research/pm_factors/gh_nb_probe.py --dir /work/crucible_data/seed_probe/HS300 --dates 20261119,20261120
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np


def load(p: Path) -> dict[int, dict]:
    with gzip.open(p) as f:
        return {int(s): kv for s, kv in json.load(f).items()}


def num(v) -> float:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=Path("/work/crucible_data/seed_probe/HS300"))
    ap.add_argument("--dates", required=True)
    a = ap.parse_args()
    for d in a.dates.split(","):
        cur = load(a.dir / f"{d}.jz")
        prev_d = sorted(p.stem for p in a.dir.glob("*.jz") if re.fullmatch(r"\d{8}", p.stem) and p.stem < d)
        prev = load(a.dir / f"{prev_d[-1]}.jz") if prev_d else {}
        print(f"== {d} ({len(cur)} stocks; previous {prev_d[-1] if prev_d else 'none'})")
        stems, sample = Counter(), {}
        for kv in cur.values():
            for k, v in kv.items():
                if k.startswith(("zzgh.", "zzgh2.")):
                    s = re.sub(r"\.\d+(\.|$)", r".N\1", k)
                    stems[s] += 1
                    sample.setdefault(s, v)
        for s, n in sorted(stems.items()):
            v = sample[s]
            print(f"   {s:<24} {n:>7}  e.g. {json.dumps(v)[:80]}")
        sids = sorted(cur)
        nb = np.array([num(cur[s].get("zzgh.nb")) for s in sids])
        nb2 = np.array([num(cur[s].get("zzgh2.nb")) for s in sids])
        ok = np.isfinite(nb) & np.isfinite(nb2)
        if ok.any():
            print(f"   zzgh.nb == zzgh2.nb: {np.mean(nb[ok] == nb2[ok]):.4f} of {ok.sum()}, max |diff| {np.abs(nb[ok] - nb2[ok]).max():.3g}")
        for tag, x in (("zzgh.nb", nb), ("zzgh2.nb", nb2)):
            pv = np.array([num(prev.get(s, {}).get(tag)) for s in sids])
            m = np.isfinite(x) & np.isfinite(pv)
            if m.sum() > 10:
                print(f"   {tag}: mean {np.nanmean(x):+.5f} sd {np.nanstd(x):.5f}; corr with previous day {np.corrcoef(x[m], pv[m])[0, 1]:+.4f};"
                      f" equal to previous {np.mean(x[m] == pv[m]):.4f}")
        for stem, x in (("zzgh", nb), ("zzgh2", nb2)):
            codes = []
            for s in sids:
                c = np.array([num(v) for k, v in cur[s].items() if re.fullmatch(rf"{stem}\.c\.\d+", k)])
                codes.append(c)
            if not any(len(c) for c in codes):
                continue
            n_slots = Counter(len(c) for c in codes).most_common(1)[0][0]
            hist = Counter(v for c in codes for v in c.tolist())
            mc = np.array([np.nanmean(np.where((c >= 0) & (c <= 4), c, np.nan)) if len(c) else np.nan for c in codes])
            m = np.isfinite(mc) & np.isfinite(x)
            print(f"   {stem}.c: {n_slots} slots per stock; code counts {dict(sorted(hist.items()))};"
                  f" corr(mean code, {stem}.nb) {np.corrcoef(mc[m], x[m])[0, 1]:+.4f}" if m.sum() > 10 else f"   {stem}.c: {n_slots} slots")
    print("[done]")


if __name__ == "__main__":
    main()
