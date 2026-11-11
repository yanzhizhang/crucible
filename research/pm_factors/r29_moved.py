r"""R29: the columns whose training thresholds moved between model generations (gen_flags: 202509-202608 vs 202612), dropped
from an assignment one at a time and all together -- for scoring on 0430 (generation A) against 1110 (generation C).

    python research/pm_factors/r29_moved.py --base data/reports/r27_pick_r25.json --out data/reports/r29_var
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MOVED = ["zzup", "zzdw", "zzcb", "zzal.a", "zzhi2.c", "zzrg", "zzal.e", "zzqf5.q.c4", "zzhk.d", "zzal.b", "zzal.f", "zzbs.b", "zzqo.a",
         "zzxc.d"]  # gen_flags columns 34 37 38 66 71 78 89 91 99 101 110 125 140 191


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    base = json.loads(a.base.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "base.json").write_text(json.dumps(base, indent=1))
    have = [c for c in MOVED if c in base]
    for c in have:
        (a.out / f"drop_{c}.json").write_text(json.dumps({k: v for k, v in base.items() if k != c}, indent=1))
        print(f"  drop_{c}: now {base[c]}")
    (a.out / "drop_moved.json").write_text(json.dumps({k: v for k, v in base.items() if k not in MOVED}, indent=1))
    print(f"{len(have) + 1} variants -> {a.out}; moved columns not assigned: {[c for c in MOVED if c not in base]}")


if __name__ == "__main__":
    main()
