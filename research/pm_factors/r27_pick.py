r"""Pick the one-change variants that do not lose on any selection day, and combine them into one assignment.

Each variant is a full ic_assign json in ``--var-dir``; its change is its diff against ``--base`` (added, changed or removed
columns). Its IC on day D is read from ``<prefix><name>_<D>.txt`` (the ``IC: start`` line of an ic_score run). A variant is
picked when its IC is at least the base's on every ``--days`` day and above it on average by ``--min-gain``; picked changes
are applied together in order of their mean gain (a later change to a column already changed is skipped).

    python research/pm_factors/r27_pick.py --base data/reports/r25_var/base.json --var-dir data/reports/r25_var \
        --prefix data/reports/r25v_ --days 20261110,20260430 --out data/reports/r27_pick_drops.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

IC = re.compile(r"IC: start ([0-9.]+)")


def ic_of(p: Path) -> float | None:
    if not p.exists():
        return None
    m = IC.findall(p.read_text(errors="replace"))
    return float(m[-1]) if m else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--base-name", default="base", help="the base's own run name under --prefix")
    ap.add_argument("--var-dir", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--days", required=True)
    ap.add_argument("--min-gain", type=float, default=0.0005)
    ap.add_argument("--exclude", default="base,drop_all,drop_both", help="variants that are combinations themselves")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    days = [d for d in a.days.split(",") if d]
    base = json.loads(a.base.read_text())
    b_ic = {d: ic_of(Path(f"{a.prefix}{a.base_name}_{d}.txt")) for d in days}
    print(f"base ({len(base)} columns): " + "  ".join(f"{d} {b_ic[d]}" for d in days))
    if any(v is None for v in b_ic.values()):
        raise SystemExit("base IC missing on a selection day")
    excl = set(a.exclude.split(","))
    rows = []
    for f in sorted(a.var_dir.glob("*.json")):
        name = f.stem
        if name in excl:
            continue
        v = json.loads(f.read_text())
        ics = {d: ic_of(Path(f"{a.prefix}{name}_{d}.txt")) for d in days}
        if any(x is None for x in ics.values()):
            print(f"  {name:<22} missing a day: {ics}")
            continue
        gains = [ics[d] - b_ic[d] for d in days]
        diff = {k: v.get(k) for k in set(base) | set(v) if base.get(k) != v.get(k)}  # None = removed
        rows.append((name, gains, diff))
    rows.sort(key=lambda r: -sum(r[1]) / len(r[1]))
    print(f"{'variant':<22} " + " ".join(f"{d:>9}" for d in days) + "   mean  picked  change")
    out = dict(base)
    touched: set[str] = set()
    picked = []
    for name, gains, diff in rows:
        mean = sum(gains) / len(gains)
        ok = min(gains) >= 0 and mean >= a.min_gain and not (set(diff) & touched)
        if ok:
            for k, val in diff.items():
                if val is None:
                    out.pop(k, None)
                else:
                    out[k] = val
            touched |= set(diff)
            picked.append(name)
        desc = ", ".join(f"{k}={'-' if val is None else val}" for k, val in list(diff.items())[:3]) + (" ..." if len(diff) > 3 else "")
        print(f"{name:<22} " + " ".join(f"{g:+9.4f}" for g in gains) + f" {mean:+.4f}  {'yes' if ok else '   '}   {desc[:90]}")
    a.out.write_text(json.dumps(out, indent=1))
    print(f"picked {len(picked)}: {' '.join(picked)}\n-> {a.out} ({len(out)} columns)")


if __name__ == "__main__":
    main()
