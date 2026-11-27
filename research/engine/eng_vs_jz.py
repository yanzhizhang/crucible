r"""The engine's day-end state (engine_<D>.json.gz) against PM's <D>.jz, for chosen key prefixes: per key, the share of
stocks within 1e-9 / 1e-6 / 1e-3 (relative), the median and q10 / q90 of PM / ours. Read-only.

    python research/engine/eng_vs_jz.py --engine data/engine/engine_20261119.json.gz --jz /work/crucible_data/seed_probe/HS300/20261119.jz --prefix zzhj.p,zztx.r,zzeb.r,zzui.r,zzui.k
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np


def load(p: Path) -> dict[int, dict]:
    with gzip.open(p) as f:
        d = json.load(f)
    if isinstance(d, dict) and "stocks" in d and isinstance(d["stocks"], dict):
        d = d["stocks"]
    return {int(k): v for k, v in d.items() if str(k).isdigit() and isinstance(v, dict)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    eng, pm = load(a.engine), load(a.jz)
    pre = tuple(a.prefix.split(","))
    keys = sorted({k for kv in pm.values() for k in kv if k.startswith(pre)} | {k for kv in eng.values() for k in kv if k.startswith(pre)})
    print(f"engine {len(eng)} stocks, PM {len(pm)} stocks, {len(keys)} keys")
    for k in keys:
        x = np.array([float(eng[s][k]) if isinstance(eng.get(s, {}).get(k), (int, float)) else np.nan for s in pm])
        y = np.array([float(pm[s][k]) if isinstance(pm[s].get(k), (int, float)) else np.nan for s in pm])
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            print(f"   {k:<16} ours {int(np.isfinite(x).sum())} / PM {int(np.isfinite(y).sum())} values: nothing to compare")
            continue
        rel = np.abs(x[ok] - y[ok]) / np.maximum(1e-12, np.abs(y[ok]))
        with np.errstate(divide="ignore", invalid="ignore"):
            r = y[ok] / x[ok]
        q10, q50, q90 = np.nanpercentile(r, [10, 50, 90])
        print(f"   {k:<16} n {ok.sum():>3}  within 1e-9 {np.mean(rel <= 1e-9):.3f}  1e-6 {np.mean(rel <= 1e-6):.3f}  1e-3 {np.mean(rel <= 1e-3):.3f}"
              f"  PM / ours {q50:.6g} ({q10:.5g} .. {q90:.5g})")
    print("[done]")


if __name__ == "__main__":
    main()
