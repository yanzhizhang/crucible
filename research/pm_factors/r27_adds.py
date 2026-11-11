r"""R27: R17 plus ONE of R24's new formulas for a column R17 left open, one json per column (the R24 gate's t >= 5 ones).

    python research/pm_factors/r27_adds.py --r17 data/reports/stage6_r17/ic_assign.json --r24 data/reports/stage6_r24/ic_assign.json --out data/reports/r27_add
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# R24's gate on 1109, new formulas for columns R17 had open, t >= 5 (r24.txt lines 270-358)
COLS = ["zzpe.e", "zzbo.c", "zzqo.d", "zzmo", "zzav", "zzie4.c", "zzro2.a", "zzie2.c", "zzrb", "zzag2.f", "zzfq.b", "zzie.c",
        "zzyj.c", "zzyj5.b", "zzvx.a", "zzaa.a", "zzxc.d", "zzxc.a", "zzko", "zzwf"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r17", type=Path, required=True)
    ap.add_argument("--r24", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r17, r24 = json.loads(a.r17.read_text()), json.loads(a.r24.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "base.json").write_text(json.dumps(r17, indent=1))
    n = 0
    for c in COLS:
        if c in r17 or c not in r24:
            print(f"  skip {c}: {'R17 has it' if c in r17 else 'not in R24'}")
            continue
        (a.out / f"add_{c}.json").write_text(json.dumps({**r17, c: r24[c]}, indent=1))
        print(f"  add_{c}: {r24[c]}")
        n += 1
    print(f"{n} variants -> {a.out}")


if __name__ == "__main__":
    main()
