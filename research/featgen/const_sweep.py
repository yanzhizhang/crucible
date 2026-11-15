r"""Which open columns act like market-level inputs: fill each with ONE value for every row and see how much that buys.

A cross-sectionally constant input (an index, a futures basis, the time of day) still reorders stocks in a tree model:
the trees branch on it first. If such a column is missing (NaN), every tree that uses it takes the missing branch for
every stock. Per open column, every split interval of PM's trees (a point inside each, below the lowest, above the
highest) is tried as a constant for all rows of a day, scored in one lgbpred --batch run; the best constant's gain over
NaN and its value are printed per day. A column where one value lifts the trusted day a lot behaves like a market-level
input (or its NaN is far from where PM's rows sit); the per-day best value is then the series to match.

    python research/featgen/const_sweep.py --days 20260430,20261112,20261113 --x-dir "data/reports/stage6_r99_{mmdd}" \
        --base "data/featgen_r99/base_{date}.parquet" --models $DATA/models_gen/B
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pm_factors"))
import pred_day as P  # noqa: E402
from implied_scan import Day  # noqa: E402
from invert_cols import BASELINE, points, thresholds  # noqa: E402
from missing_damage import sim  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--x-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--cap", type=int, default=48, help="most constants per column")
    ap.add_argument("--work", type=Path, default=Path("data/const_sweep_work"))
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    settled = {ln.split()[0] for ln in a.formulas.read_text().splitlines() if ln.strip() and not ln.startswith("#")}
    targets = [n for n in names if n not in settled]
    ix = {n: i for i, n in enumerate(names)}
    ds = [x.strip() for x in a.days.split(",") if x.strip()]
    days = {d: Day(d, Path(a.x_dir.format(mmdd=d[4:], date=d)), Path(a.base.format(date=d, mmdd=d[4:])), names) for d in ds}
    files = {d: P.model_files(a.models, d, a.fits.split(","), a.subs.split(",")) for d in ds}
    a.work.mkdir(parents=True, exist_ok=True)
    exe = P.ensure_binary()

    def run(d: str, X: np.ndarray, col: int | None, vals: list[float]) -> np.ndarray:
        P.write_x(a.work / "x.bin", X)
        args = [str(exe), str(a.work / "x.bin"), str(a.work / "b.bin" if col is not None else a.work / "p.bin"), *map(str, files[d]), "--threads", str(a.threads)]
        if col is not None:
            with (a.work / "c.bin").open("wb") as f:
                f.write(b"CRULC001")
                f.write(np.array([len(vals), X.shape[0]], "<u8").tobytes())
                for v in vals:
                    f.write(np.array([col], "<u8").tobytes())
                    f.write(np.full(X.shape[0], v, "<f8").tobytes())
            args += ["--batch", str(a.work / "c.bin")]
        subprocess.run(args, check=True, stdout=subprocess.DEVNULL)
        if col is None:
            return P.read_scores(a.work / "p.bin").copy()[None, :]
        raw = (a.work / "b.bin").read_bytes()
        n, r = (int(x) for x in np.frombuffer(raw, "<u8", 2, 8))
        return np.frombuffer(raw, "<f8", n * r, 24).reshape(n, r)

    s0 = {d: sim(run(d, D.X, None, [])[0], D.pm, D.marks) for d, D in days.items()}
    print("baseline " + ", ".join(f"{d} {v:.4f}" for d, v in s0.items()), flush=True)
    th = thresholds(files[ds[0]], len(names))
    rows = []
    for t in targets:
        j = ix[t]
        if not th[j].size:
            continue
        pts = points(th[j], a.cap).tolist()
        best = {}
        for d, D in days.items():
            g = np.array([sim(sc, D.pm, D.marks) for sc in run(d, D.X, j, pts)]) - s0[d]
            k = int(np.argmax(g))
            best[d] = (float(g[k]), pts[k])
        rows.append((t, best))
        print(f"  {t:<10} " + "  ".join(f"{d[4:]} {best[d][0]:+.4f} @ {best[d][1]:.5g}" for d in ds), flush=True)
    tr = ds[0]
    print(f"\nby the best constant's gain on {tr}:")
    for t, best in sorted(rows, key=lambda r: -r[1][tr][0])[:25]:
        print(f"  {t:<10} " + "  ".join(f"{d[4:]} {best[d][0]:+.4f} @ {best[d][1]:.5g}" for d in ds))
    for f in ("x.bin", "p.bin", "c.bin", "b.bin"):
        (a.work / f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
