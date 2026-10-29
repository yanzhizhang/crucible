r"""Stage 4: which of our candidate families a PM sampler file should be scored against -- by its column names, not its folder.

PM's folder names and column sets do not line up with the 20260430 sample our candidates were written from (the
``ZZGW`` folder now holds ``na nab nb qa qab qb``, ``ZZMS`` holds ``bl bm bs n q``, ...). So: read the export's ``V``
set and its slot count, pick the candidate family whose columns cover it best, and print the ``rank_candidates``
arguments (``--ours`` and, for the per-minute book columns that our ZZQI candidates carry under other names, ``--map``).

    python research/pm_factors/stage4_route.py --pm data/pm_export/ZZGW_20261009.parquet --date 20261009
"""

from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

CAND = Path("/work/crucible_data/store/pm_factors/candidates")
BOOK = {"qb": "c", "qa": "d", "qab": "cd"}  # PM per-minute resting book -> ZZQI's level-sum columns (lasttot variant)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pm", type=Path, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--cand", type=Path, default=CAND)
    a = ap.parse_args()
    pm = pl.read_parquet(a.pm)
    vcol = next((c for c in pm.columns if c.upper() == "V"), None)
    icol = next((c for c in pm.columns if c.upper() == "I"), None)
    if vcol is None:
        print("")
        return
    pv = set(pm[vcol].cast(pl.String).str.strip_chars().unique().to_list())
    slots = pm[icol].n_unique() if icol else 1
    best, score = None, 0
    for fam in sorted(a.cand.glob("family=*")):
        f = fam / f"date={a.date}" / "part.parquet"
        if not f.exists():
            continue
        ov = pl.scan_parquet(f).select(pl.col("V").str.split("@").list.first().unique()).collect()["V"].to_list()
        daily = pl.scan_parquet(f).select(pl.col("minute").max()).collect().item() == -1
        hit = len(pv & set(ov)) - (0.5 if daily != (slots == 1) else 0)  # a daily family for a daily file
        if hit > score:
            best, score = fam.name.split("=")[1], hit
    if pv <= set(BOOK) | {"na", "nb", "nab"} and slots > 1:
        f = a.cand / "family=ZZQI" / f"date={a.date}" / "part.parquet"
        if f.exists():
            print(f"--ours {f.parent.parent} --map " + ",".join(f"{o}={p}" for p, o in BOOK.items()))
            return
    print(f"--ours {a.cand / f'family={best}'}" if best else "")


if __name__ == "__main__":
    main()
