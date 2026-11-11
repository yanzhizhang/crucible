r"""R25: one-change variants of the R17 assignment, written as ic_assign json files for held-out scoring.

* drop_<col>: R17 without one column that agreed better with PM when missing (residual_breakdown "worth least");
* drop_all / drop_both: several at once;
* r17plus: R17 + R24's formulas for columns R17 left open (R24's replacements and drops ignored: the family lock without
  its vetoes);
* set_<col>_<i>: R17 with one column set to a state-key or long-horizon formula from the implied-value probe.

    python research/pm_factors/r25_variants.py --r17 data/reports/stage6_r17/ic_assign.json --r24 data/reports/stage6_r24/ic_assign.json --out data/reports/r25_var
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DROPS = ["zzzl2.f", "zzpe.c", "zzjb.b", "zzwa.a", "zzpe.b", "zzjx2.a", "zzpe.d"]
SETS = {
    "zzba": ["zzba.r / zzba.p", "cal:zzba.r / zzba.p"],
    "zzeg": ["zzpk.p / px", "cal:zzpk.p / px", "px / zzpk.p"],
    "zzro2.a": ["gen.lmie14400", "cal:gen.lmie14400", "gen.lee60_14400", "gen.lp_vwap", "gen.stoch"],
    "zzro2.c": ["gen.lee60_14400", "cal:gen.lee60_14400", "gen.lp_vwap", "gen.lmie14400"],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--r17", type=Path, required=True)
    ap.add_argument("--r24", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    r17 = json.loads(a.r17.read_text())
    r24 = json.loads(a.r24.read_text()) if a.r24.exists() else {}
    a.out.mkdir(parents=True, exist_ok=True)
    var: dict[str, dict] = {"base": dict(r17)}
    for c in DROPS:
        if c in r17:
            var[f"drop_{c}"] = {k: v for k, v in r17.items() if k != c}
    var["drop_both"] = {k: v for k, v in r17.items() if k not in ("zzzl2.f", "zzpe.c")}
    var["drop_all"] = {k: v for k, v in r17.items() if k not in DROPS}
    var["r17plus"] = {**r17, **{k: v for k, v in r24.items() if k not in r17}}
    for c, forms in SETS.items():
        for i, f in enumerate(forms):
            var[f"set_{c}_{i}"] = {**r17, c: f}
    for name, m in var.items():
        (a.out / f"{name}.json").write_text(json.dumps(m, indent=1))
    print(f"{len(var)} variants -> {a.out}: " + " ".join(var))
    print(f"  r17plus adds {len(var['r17plus']) - len(r17)} columns: " + ", ".join(f"{k}={v}" for k, v in r24.items() if k not in r17))
    for c, forms in SETS.items():
        print(f"  {c}: R17 has {r17.get(c, '-')!r}; tried {forms}")


if __name__ == "__main__":
    main()
