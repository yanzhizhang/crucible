r"""Four read-only questions about PM's day-end state files (``<dcp>/YYYYMMDD.jz``, gzip JSON {sid: {key: value}}).

1 ``--stem``: every key of that family and its variants (zzro -> zzro.*, zzro2.* ...) with its value distribution;
2 ``--a`` / ``--b``: keys present on one day and not the other (0429 has 7532 keys a stock, the autumn days 7498), by stem;
3 weights: every ``*.wgt`` key -- typical value, cap, and day to day: the ratio wgt(D) / wgt(D-1) per stock over the last
  ``--days`` files (a key reset at the open has no memory of D-1; a carried one moves slowly);
4 ``--sod``: files of PM's start-of-day dir (2.23 sod/p1_intrading_dcp) against the dcp file of the same date and the one
  before: share of identical values per stem and the median ratio where they differ.

    python research/pm_factors/jz_probe.py --dcp /data/prod/h/HS300/p1_intrading_dcp --stem zzro --a 20260429 --b 20261106 \
        --sod /work/crucible_data/sod_dcp
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np


def load(p: Path) -> dict:
    raw = p.read_bytes()
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    return json.loads(raw)


def stem(k: str) -> str:
    return k.split(".")[0]


def q(v: list[float]) -> str:
    a = np.asarray([x for x in v if isinstance(x, (int, float)) and np.isfinite(x)], dtype=float)
    if not a.size:
        return "no finite values"
    p = np.quantile(a, [0.1, 0.5, 0.9])
    return f"n {a.size} p10/50/90 {p[0]:.6g} / {p[1]:.6g} / {p[2]:.6g} max {a.max():.6g} zero {np.mean(a == 0):.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dcp", type=Path, required=True)
    ap.add_argument("--stem", default="zzro")
    ap.add_argument("--a", default="20260429")
    ap.add_argument("--b", default="20261106")
    ap.add_argument("--days", type=int, default=6)
    ap.add_argument("--sod", type=Path)
    a = ap.parse_args()
    days = sorted(p.stem for p in a.dcp.glob("*.jz") if re.fullmatch(r"\d{8}", p.stem))

    print(f"== 1 keys of {a.stem} and its variants in {a.b}.jz")
    db = load(a.dcp / f"{a.b}.jz")
    keys = sorted({k for kv in db.values() for k in kv if re.fullmatch(rf"{a.stem}\d*", stem(k))})
    print(f"  {len(keys)} keys" + ("" if keys else f" -- no {a.stem}* family keeps day-end state"))
    for k in keys[:80]:
        print(f"  {k:<28} {q([kv.get(k) for kv in db.values()])}")

    print(f"\n== 2 key sets {a.a} vs {a.b}")
    da = load(a.dcp / f"{a.a}.jz")
    ka = {k for kv in da.values() for k in kv}
    kb = {k for kv in db.values() for k in kv}
    for name, only in ((f"only in {a.a}", ka - kb), (f"only in {a.b}", kb - ka)):
        by: dict[str, list[str]] = {}
        for k in sorted(only):
            by.setdefault(stem(k), []).append(k)
        print(f"  {name}: {len(only)} keys in {len(by)} stems")
        for s, ks in sorted(by.items(), key=lambda x: -len(x[1])):
            print(f"    {s:<8} {len(ks):3d}: {' '.join(ks[:12])}{' ...' if len(ks) > 12 else ''}")
    common = sorted(ka & kb)
    shift = []
    for k in common:
        x = np.array([da[s].get(k, np.nan) if s in da else np.nan for s in db], dtype=float)
        y = np.array([db[s].get(k, np.nan) for s in db], dtype=float)
        m = np.isfinite(x) & np.isfinite(y) & (x != 0)
        if m.sum() > 20:
            shift.append((k, float(np.median(y[m] / x[m]))))
    odd = [r for r in shift if not 0.5 < r[1] < 2]
    print(f"  common keys {len(common)}; median ratio {a.b}/{a.a} outside 0.5..2 (scale changed?) for {len(odd)}: "
          f"{', '.join(f'{k} {r:.3g}' for k, r in odd[:30])}")

    print(f"\n== 3 weights over the last {a.days} files: cap, and wgt(D) / wgt(D-1) per stock")
    tail = [load(a.dcp / f"{d}.jz") for d in days[-a.days:]]
    wk = sorted({k for kv in tail[-1].values() for k in kv if k.endswith(".wgt")})
    for k in wk:
        vals = [kv.get(k) for kv in tail[-1].values()]
        rat = []
        for prev, cur in zip(tail, tail[1:], strict=False):
            for s, kv in cur.items():
                x, y = prev.get(s, {}).get(k), kv.get(k)
                if isinstance(x, (int, float)) and isinstance(y, (int, float)) and x:
                    rat.append(y / x)
        r = np.asarray(rat)
        same = float(np.mean(np.abs(r - 1) < 1e-12)) if r.size else float("nan")
        print(f"  {k:<26} {q(vals)}; D/D-1 p10/50/90 "
              f"{np.quantile(r, 0.1):.4g} / {np.median(r):.4g} / {np.quantile(r, 0.9):.4g}, identical {same:.2f}" if r.size else f"  {k:<26} {q(vals)}")

    if a.sod and a.sod.exists():
        print(f"\n== 4 start-of-day files in {a.sod}")
        for f in sorted(p for p in a.sod.iterdir() if p.is_file()):
            m = re.search(r"(\d{8})", f.name)
            if not m:
                print(f"  {f.name}: no date in the name, skipped")
                continue
            d = m.group(1)
            try:
                s = load(f)
            except Exception as e:  # noqa: BLE001
                print(f"  {f.name}: not gzip / JSON ({str(e)[:60]})")
                continue
            before = [x for x in days if x < d]
            for ref in [x for x in (d, before[-1] if before else None) if x and (a.dcp / f"{x}.jz").exists()]:
                r_ = load(a.dcp / f"{ref}.jz")
                eq: dict[str, list[int]] = {}
                ratio: dict[str, list[float]] = {}
                for sid, kv in s.items():
                    for k, v in kv.items():
                        w = r_.get(sid, {}).get(k)
                        if not isinstance(v, (int, float)) or not isinstance(w, (int, float)):
                            continue
                        eq.setdefault(stem(k), []).append(int(v == w))
                        if v != w and w:
                            ratio.setdefault(stem(k), []).append(v / w)
                tot = sum(sum(x) for x in eq.values()) / max(sum(len(x) for x in eq.values()), 1)
                print(f"  {f.name} vs dcp {ref}: {len(s)} stocks, identical values {tot:.3f}; stems not identical "
                      f"(share equal, median ratio): " + ", ".join(
                          f"{st_} {np.mean(e):.2f} {np.median(ratio.get(st_, [np.nan])):.4g}" for st_, e in sorted(eq.items()) if np.mean(e) < 0.999)[:900])


if __name__ == "__main__":
    main()
