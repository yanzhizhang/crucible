r"""The formula layer's side tables of one day without running ic_score: side_<D>.npz (CED sids, Barra industry,
exposures) and cal_<D>.json (the model generation's root-split training-CDF points).

The same calls ic_score makes for ``--dump-x`` (defaults of the baseline runs: CED daily candidates joined with PM's
stsfile daily inputs; ``Pool.add_block`` for industry and exposures; ``generation_cdf`` of the model files picked for
the day), so the output must equal ic_score's; ``--check DIR`` compares with an earlier ``--dump-x`` directory.

    python research/featgen/side_day.py --date 20261112 --state $SEED/20261112.jz --models $DATA/models_gen/B --out data/side \
        --check data/reports/stage6_r70_1112
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pm_factors"))
sys.path.insert(0, str(HERE.parent))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--state", type=Path, required=True, help="PM's state file of the day (.jz): the stocks (an earlier day's file when the day has none)")
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--pm-stsfile", type=Path, default=Path("/work/crucible_data/pm_stsfile"))
    ap.add_argument("--pm-sts-day", choices=("before", "on"), default="before")
    ap.add_argument("--pm-sts-map", type=Path, help="json {day: stsfile day} (sts_days.py), as ic_score --pm-sts-map")
    ap.add_argument("--out", type=Path, default=Path("data/side"))
    ap.add_argument("--check", type=Path, help="an ic_score --dump-x output dir holding side_<D>.npz / cal_<D>.json: compare")
    a = ap.parse_args()
    import ced_match as CM
    import dcp_build as B
    import ic_score as IC
    import polars as pl

    t0 = time.perf_counter()
    d = a.date
    if a.pm_sts_map:
        IC.STS_MAP.update(json.loads(a.pm_sts_map.read_text()))
    if not a.state.exists():
        a.state = sorted(p for p in a.state.parent.glob("2*.jz") if p.stem < d)[-1]
    sids = sorted(B.load_state(a.state)["sid"].unique().to_list())
    dc = CM.daily_candidates(d, sids)
    pmd = IC.pm_daily(a.pm_stsfile, d, sids, a.pm_sts_day, IC.STS_MAP.get(d)) if a.pm_stsfile.exists() else None
    if pmd is not None:
        dc = pmd if dc is None else dc.join(pmd.with_columns(pl.col("sid").cast(dc.schema["sid"])), on="sid", how="left")
    pool = IC.Pool(False, True)
    pool.add_block([], dc)
    lines = [ln.strip() for ln in a.feature_csv.read_text(errors="replace").splitlines() if ln.strip()]
    names = [ln.split(",")[0].strip() for ln in lines]
    names = names[1:] if names and names[0].lower() == "name" else names
    mf = IC.model_files(a.models, d, a.fits.split(","))
    pts = IC.generation_cdf(mf, names) if mf else {}
    a.out.mkdir(parents=True, exist_ok=True)
    side, cal = a.out / f"side_{d}.npz", a.out / f"cal_{d}.json"
    np.savez_compressed(side, dsid=pool.daily[0][0], ind=pool.ind[0], expo=pool.expo[0])
    cal.write_text(json.dumps({f: [[float(x) for x in t], [float(x) for x in q]] for f, (t, q) in pts.items()}))
    print(f"{d}: {len(sids)} stocks from {a.state.name}; CED rows {pool.daily[0][0].size}, {pool.expo[0].shape[1]} exposures, "
          f"{int((pool.ind[0] >= 0).sum())} with an industry; calibration points for {len(pts)} features from {[str(f.relative_to(a.models)) for f in mf]}; "
          f"{time.perf_counter() - t0:.1f}s -> {side}, {cal}")
    if a.check:
        rs, rc = a.check / f"side_{d}.npz", a.check / f"cal_{d}.json"
        if rs.exists():
            z0, z1 = np.load(rs), np.load(side)
            for k in ("dsid", "ind", "expo"):
                x, y = z0[k], z1[k]
                same = x.shape == y.shape and bool(np.array_equal(x, y, equal_nan=True)) if x.dtype.kind == "f" else x.shape == y.shape and bool(np.array_equal(x, y))
                print(f"  {k}: shape {x.shape} vs {y.shape}, identical {same}")
        if rc.exists():
            c0, c1 = json.loads(rc.read_text()), json.loads(cal.read_text())
            print(f"  cal: {len(c0)} vs {len(c1)} features, identical {c0 == c1}")


if __name__ == "__main__":
    main()
