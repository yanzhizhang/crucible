r"""Side by side: the formula of every column in several ic_assign json files, only the columns where they differ.

    python research/pm_factors/assign_diff.py data/reports/stage6_r17/ic_assign.json data/reports/stage6_r24/ic_assign.json data/reports/stage6_r24_gate/ic_assign_gated.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> None:
    paths = [Path(p) for p in sys.argv[1:]]
    if len(paths) < 2:
        sys.exit(__doc__)
    maps = [json.loads(p.read_text()) for p in paths]
    names = [p.parent.name for p in paths]
    print("files: " + "  ".join(f"{n} ({len(m)})" for n, m in zip(names, maps, strict=True)))
    cols = sorted(set().union(*maps))
    diff = [c for c in cols if len({m.get(c) for m in maps}) > 1]
    print(f"{len(diff)} of {len(cols)} columns differ; '-' = not assigned")
    for c in diff:
        print(f"{c:<14} " + " | ".join(str(m.get(c, "-"))[:60] for m in maps))
    for i, n in enumerate(names[1:], 1):
        lost = [c for c in maps[0] if c not in maps[i]]
        new = [c for c in maps[i] if c not in maps[0]]
        print(f"{n} vs {names[0]}: {len(new)} newly assigned, {len(lost)} dropped: {' '.join(lost)}")


if __name__ == "__main__":
    main()
