r"""R39: after R34 our prediction carries MORE short-term book imbalance / short reversal than PM's (residual "ours beyond PM":
imb1 +.42, microprice_minus_mid_rel +.41, rete60 / lpek20 / lpe60 / rete75 -.40). One variant per assigned column whose
formula uses such a base: that column dropped. Written as ic_assign json files next to the base.

    python research/pm_factors/r39_vars.py --base data/reports/r34g_baseline.json --out data/reports/r39_var
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HIT = re.compile(r"gen\.(imb1\b|imb2\b|microprice_minus_mid_rel|best_level_order_count_imbalance|rete(20|30|45|50|60|75|90|100)\b|"
                 r"lpek?(20|30|40|45|50|60|75|90)\b|rpe(20|25|30|40|50|60|75|90)\b|lpe(20|30|40|45|50|60|75|90)\b|depth_imbalance_levels1to[12]\b)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    base = json.loads(a.base.read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "base.json").write_text(json.dumps(base, indent=1))
    hits = [k for k, v in base.items() if v and HIT.search(v)]
    for k in hits:
        (a.out / f"drop_{k}.json").write_text(json.dumps({x: y for x, y in base.items() if x != k}, indent=1))
        print(f"  drop_{k}: {base[k]}")
    (a.out / "drop_all_hits.json").write_text(json.dumps({x: y for x, y in base.items() if x not in hits}, indent=1))
    print(f"{len(hits)} columns use short-term imbalance / reversal bases -> {a.out}")


if __name__ == "__main__":
    main()
