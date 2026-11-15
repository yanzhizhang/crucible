r"""PM's samplerS profile cells (hstats) at the scoring rows, as a candidate parquet for implied_scan --extra.

The cells ic_score joins its pool with (``profile_marks``: the cell at the mark's minute and the ones before / after),
for every family that has a samplerS.nc for the day (or the --fams list), at the (sid, mark) rows of an ic_score
--dump-x x.parquet. Output: <out>/hstats_<D>.parquet (sid, mark, <FAM>.<V>@<off> ...).

    python research/featgen/hstats_extra.py --date 20261112 --x data/reports/stage6_r70_1112/x.parquet --out data/side_r92
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pm_factors"))
sys.path.insert(0, str(HERE.parent))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--x", type=Path, required=True)
    ap.add_argument("--hstats", type=Path, default=Path("/work/crucible_data/hstats_view/HS300"))
    ap.add_argument("--cache", type=Path, default=Path("data/pm_export_all"))
    ap.add_argument("--fams", default="all", help="comma list, or 'all' = every family dir with a samplerS.nc for the day")
    ap.add_argument("--out", type=Path, default=Path("data/side_hstats"))
    ap.add_argument("--prev", action="store_true", help="each family's latest samplerS.nc BEFORE the day (what PM had that morning; no look-ahead)")
    a = ap.parse_args()
    import ic_score as IC

    d = a.date
    x = pl.read_parquet(a.x, columns=["date", "sid", "mark"]).filter(pl.col("date") == int(d)).drop("date").with_columns(pl.col("sid").cast(pl.Int64))
    marks = np.sort(x["mark"].unique().to_numpy())
    def src(fam: str) -> str | None:  # the day whose file stands in for d
        days = sorted(p.name for p in (a.hstats / fam).glob("20*") if (p / "samplerS.nc").exists() and (p.name < d if a.prev else p.name == d))
        return days[-1] if days else None

    fams = sorted(p.name for p in a.hstats.iterdir()) if a.fams == "all" else [f for f in a.fams.split(",") if f]
    used = {f: s for f in fams if (s := src(f))}
    fams = sorted(used)
    prof: dict[int, dict[str, np.ndarray]] = {}
    for f, s in used.items():  # one family at a time, each from its own source day; cells are keyed by minute of day
        for sid, kv in IC.profile_marks(a.hstats, a.cache, [f], s, marks).items():
            prof.setdefault(sid, {}).update(kv)
    keys = sorted({k for kv in prof.values() for k in kv})
    rows = {"sid": np.repeat(np.array(sorted(prof), np.int64), marks.size), "mark": np.tile(marks, len(prof))}
    for k in keys:
        rows[k] = np.concatenate([np.asarray(prof[s].get(k, np.full(marks.size, np.nan)), float) for s in sorted(prof)])
    df = x.join(pl.DataFrame(rows), on=["sid", "mark"], how="left")
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"hstats{'_prev' if a.prev else ''}_{d}.parquet"
    df.write_parquet(dst)
    print(f"{d}: families {fams} from {sorted(set(used.values()))};{len(keys)} cells x {df.height:,} rows ({len(prof)} stocks with cells) -> {dst}")


if __name__ == "__main__":
    main()
