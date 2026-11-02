r"""How the share of unchanged samplerS cells decays with the gap in trading days -- tells the update rule apart.

Expected share of cells with pm_d == pm_{d-k} (iid daily inputs; simulated):

    model            k=1    k=2    k=3    k=4
    median of 3      0.333  0.133  0      0
    max / min of 2   0.333  0      0      0
    median of 5      0.400  0.229  0.129  0.057
    hold w.p. 1/3    0.333  0.111  0.037  0.012

Reads every PM day of the families from the hstats tree (cached as parquet under --cache) and prints the share per gap.

    python research/pm_factors/same_gap.py --hstats /data/prod/hstats/HS300 --families ZZAL,ZZMS,ZZQO,ZZTS,ZZXC,ZZDS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from compare_pm import load_pm  # noqa: E402
from rank_candidates import pm_frame  # noqa: E402


def day_frame(hstats: Path, fam: str, d: str, cache: Path) -> pl.DataFrame | None:
    """(code, minute, V, x) of one family and day, exported once."""
    dst = cache / f"{fam}_{d}.parquet"
    if not dst.exists():
        src = hstats / fam / d / "samplerS.nc"
        if not src.exists():
            return None
        cache.mkdir(parents=True, exist_ok=True)
        load_pm(src).write_parquet(dst)
    pm, vals, _ = pm_frame(dst, None, None, None, None, None)
    return pm.select("code", "minute", "V", pl.col(vals[0]).alias("x"))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--hstats", type=Path, default=Path("/data/prod/hstats/HS300"))
    ap.add_argument("--families", default="ZZAL,ZZMS,ZZQO,ZZTS,ZZXC,ZZDS")
    ap.add_argument("--start", default="20260901")
    ap.add_argument("--end", default="20261030")
    ap.add_argument("--cache", type=Path, default=Path("data/pm_export_all"))
    ap.add_argument("--max-gap", type=int, default=4)
    a = ap.parse_args()
    rows = []
    for fam in a.families.split(","):
        days = sorted(p.name for p in (a.hstats / fam).glob("2026*") if a.start <= p.name <= a.end and (p / "samplerS.nc").exists())
        frames = {d: f for d in days if (f := day_frame(a.hstats, fam, d, a.cache)) is not None}
        days = sorted(frames)  # PM days = trading days
        for V in sorted(frames[days[0]]["V"].unique().to_list()):
            row = {"file": fam, "V": V, "days": len(days)}
            for k in range(1, a.max_gap + 1):
                shares = []
                for i in range(k, len(days)):
                    j = (frames[days[i]].filter(pl.col("V") == V).join(frames[days[i - k]].filter(pl.col("V") == V).rename({"x": "xp"}), on=["code", "minute", "V"])
                         .filter(pl.col("x").is_finite() & pl.col("xp").is_finite()))
                    if j.height:
                        shares.append(float((j["x"] == j["xp"]).mean()))
                row[f"gap{k}"] = sum(shares) / len(shares) if shares else None
            rows.append(row)
        print(f"{fam}: {len(days)} days", flush=True)
    with pl.Config(tbl_rows=200, tbl_cols=-1, tbl_width_chars=160, float_precision=3):
        print("== share of cells with pm_d == pm_{d-k} by trading-day gap k "
              "(median3 .333/.133/0/0, max2 .333/0/0/0, median5 .400/.229/.129/.057, hold1/3 .333/.111/.037/.012)")
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    main()
