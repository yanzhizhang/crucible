r"""Does PM store its state in float32 or float64? Per key family of a day-end .jz: the share of values that are exactly a
float32 (np.float32(v) == v), the share with <= 7 / <= 9 significant digits as written, and three raw examples. A float64
engine would show ~0 exact float32 values except on round numbers; float32 state shows ~1.000. Read-only.

    python research/engine/jz_precision.py /work/crucible_data/seed_probe/HS300/20261120.jz
"""

from __future__ import annotations

import gzip
import json
import re
import sys
from collections import defaultdict

import numpy as np


def main() -> None:
    """CLI entry point."""
    path = sys.argv[1]
    raw = gzip.open(path).read().decode()
    toks = defaultdict(list)
    for m in re.finditer(r'"([A-Za-z0-9_.]+)"\s*:\s*(-?[0-9][0-9.eE+-]*)', raw):
        k, v = m.group(1), m.group(2)
        fam = re.sub(r"\.\d+$", ".#", k)
        fam = fam.rsplit(".", 1)[-1] if fam.endswith((".sum", ".wgt", ".t")) and False else fam
        toks[fam].append(v)
    print(f"== {path}: {sum(len(v) for v in toks.values()):,} numbers in {len(toks)} keys")
    rows = []
    for k, vs in toks.items():
        x = np.array([float(v) for v in vs])
        nz = x[(x != 0) & np.isfinite(x)]
        nz = nz[np.abs(nz - np.round(nz)) > 0] if nz.size else nz  # integers are float32-exact anyway
        if nz.size < 10:
            continue
        f32 = float((nz.astype(np.float32).astype(np.float64) == nz).mean())
        digits = np.array([len(re.sub(r"[^0-9]", "", v.split("e")[0].split("E")[0]).lstrip("0")) for v in vs])
        rows.append((k, nz.size, f32, float((digits <= 7).mean()), float((digits <= 9).mean()), vs[:3]))
    rows.sort(key=lambda r: (-r[2], r[0]))
    print(f"{'key':<26}{'n (non-integer)':>16}{'float32-exact':>15}{'<=7 digits':>12}{'<=9 digits':>12}  examples")
    for k, n, f32, d7, d9, ex in rows:
        print(f"{k:<26}{n:>16,}{f32:>15.3f}{d7:>12.3f}{d9:>12.3f}  {', '.join(ex)}")
    allx = np.array([r[2] for r in rows])
    print(f"keys with float32-exact share >= 0.99: {int((allx >= 0.99).sum())} of {len(rows)}; <= 0.01: {int((allx <= 0.01).sum())}")


if __name__ == "__main__":
    main()
