r"""Score a short list of single-column fills on one day, every fill alone in one lgbpred --batch run, with its residual profile.

For the full 3 s grid (data/full_0430: 1.4M rows) a generic scan of every base is too slow; this takes the leads only:
--fills "col:transform:candidate,..." (transforms as implied_scan's raw ones). Per fill: the similarity gain over NaN, and
the gain inside drawdown bins (sqrt(zzmh.r) / 2 from the day's base) and in the first 30 minutes, where R126 found the
residual concentrated (corr 0.41 in 09:30-10:00; PM above ours by +0.34 z near the day's high). Read-only.

    python research/featgen/single_fill.py --day 20260430 --x-dir data/full_0430 --base data/full_0430/base_{date}.parquet \
        --extra data/full_0430/extra_{date}.parquet --fills "zzmh:id:x.mh.2dd,zzjx3.b:-dm:ret90" --models $DATA/models_gen/B
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pm_factors"))
import pred_day as P  # noqa: E402
from implied_scan import RAW, Day, raw_xs  # noqa: E402
from missing_damage import sim  # noqa: E402


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--day", required=True)
    ap.add_argument("--x-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--fills", required=True)
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--work", type=Path, default=Path("data/single_fill_work"))
    ap.add_argument("--threads", type=int, default=64)
    a = ap.parse_args()
    d = a.day
    names = P.feature_names(a.feature_csv)
    ix = {n: i for i, n in enumerate(names)}
    D = Day(d, Path(a.x_dir.format(mmdd=d[4:], date=d)), Path(a.base.format(date=d, mmdd=d[4:])), names,
            [Path(e.format(date=d, mmdd=d[4:])) for e in a.extra.split(",") if e])
    files = P.model_files(a.models, d, a.fits.split(","), a.subs.split(","))
    a.work.mkdir(parents=True, exist_ok=True)
    exe = P.ensure_binary()
    fills = []
    for spec in [x for x in a.fills.split(",") if x]:
        col, tr, cand = spec.split(":", 2)
        if col not in ix or tr not in RAW or cand not in D.bases:
            print(f"  skip {spec}: {'no column' if col not in ix else 'no transform' if tr not in RAW else 'no candidate'}")
            continue
        v = D.bases[cand]
        with np.errstate(divide="ignore", invalid="ignore"):
            v = v if tr == "id" else -v if tr == "-id" else 1.0 / v if tr == "1/b" else raw_xs(v, D.marks, tr)
        fills.append((spec, ix[col], v))
    P.write_x(a.work / "x.bin", D.X)
    with (a.work / "c.bin").open("wb") as f:
        f.write(b"CRULC001")
        f.write(np.array([len(fills), D.X.shape[0]], "<u8").tobytes())
        for _, j, v in fills:
            f.write(np.array([j], "<u8").tobytes())
            f.write(np.ascontiguousarray(v, "<f8").tobytes())
    subprocess.run([str(exe), str(a.work / "x.bin"), str(a.work / "p.bin"), *map(str, files), "--threads", str(a.threads)], check=True, stdout=subprocess.DEVNULL)
    s0 = P.read_scores(a.work / "p.bin").copy()
    subprocess.run([str(exe), str(a.work / "x.bin"), str(a.work / "b.bin"), *map(str, files), "--batch", str(a.work / "c.bin"), "--threads", str(a.threads)],
                   check=True, stdout=subprocess.DEVNULL)
    raw = (a.work / "b.bin").read_bytes()
    n, r = (int(x) for x in np.frombuffer(raw, "<u8", 2, 8))
    S = np.frombuffer(raw, "<f8", n * r, 24).reshape(n, r)
    dd = D.bases.get("x.mh.dd")
    sel = {"all": np.ones(D.marks.size, bool), "first30m": D.marks <= 1800}
    if dd is not None:
        sel.update({"dd<0.2%": dd < 0.002, "dd 0.2-1%": (dd >= 0.002) & (dd < 0.01), "dd>=1%": dd >= 0.01})

    def sims(sc: np.ndarray) -> dict[str, float]:
        return {k: sim(np.where(m, sc, np.nan), np.where(m, D.pm, np.nan), D.marks) for k, m in sel.items()}

    b0 = sims(s0)
    print(f"{d}: {D.X.shape[0]:,} rows; baseline " + "  ".join(f"{k} {v:.4f}" for k, v in b0.items()))
    rows = []
    for (spec, _, _), sc in zip(fills, S, strict=True):
        g = {k: v - b0[k] for k, v in sims(sc).items()}
        rows.append((g["all"], spec, g))
    for _, spec, g in sorted(rows, reverse=True):
        print(f"  {spec:<55} " + "  ".join(f"{k} {v:+.4f}" for k, v in g.items()))
    for f in ("x.bin", "p.bin", "c.bin", "b.bin"):
        (a.work / f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
