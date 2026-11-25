r"""Fill many open columns at once with their best units-fingerprint candidates and score the days together.

Single-column scans cannot tell candidates apart (one fill moves a day by ~0.001 while 64 other columns stay NaN). The
units fingerprint is target-free; this asks whether the fills it likes help TOGETHER. From fingerprint.py reports
(lines "   <score>  <tr> <candidate>  below ... "), each column's best candidate with score <= --max-score is taken; all
are filled at once and every day re-scored; then a leave-one-out pass shows each fill's share of the joint change, and a
cumulative curve adds them in fingerprint order. Read-only.

    python research/featgen/joint_fill.py --days 20260430,20261112,20261113 --reports data/reports/r108.txt,data/reports/r111b.txt \
        --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" --extra "..." --models $DATA/models_gen/B
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pm_factors"))
import pred_day as P  # noqa: E402
from implied_scan import RAW, Day, raw_xs  # noqa: E402
from missing_damage import sim  # noqa: E402

HEAD = re.compile(r"^-- (\S+):")
ROW = re.compile(r"^\s+([0-9.]+)\s+(\S+)\s+(\S+)\s+below\s+([0-9.]+)\s+above\s+([0-9.]+)\s+intervals\s+([0-9.]+)\s+root gap\s+([0-9.]+)")


def best_fills(reports: list[Path], max_score: float) -> dict[str, tuple[float, str, str]]:
    """Column -> (score, transform, candidate) of its lowest-score fingerprint row over the reports."""
    out: dict[str, tuple[float, str, str]] = {}
    for rp in reports:
        col = None
        for ln in rp.read_text(errors="replace").splitlines():
            m = HEAD.match(ln)
            if m:
                col = m.group(1)
                continue
            r = ROW.match(ln)
            if r and col:
                sc, tr, cand = float(r.group(1)), r.group(2), r.group(3)
                if sc <= max_score and (col not in out or sc < out[col][0]):
                    out[col] = (sc, tr, cand)
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--reports", required=True, help="comma list of fingerprint.py outputs")
    ap.add_argument("--max-score", type=float, default=0.02)
    ap.add_argument("--x-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--models", type=Path, default=Path("/work/crucible_data/prod_models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/work/crucible_data/prod_models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--work", type=Path, default=Path("data/joint_fill_work"))
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    names = P.feature_names(a.feature_csv)
    ix = {n: i for i, n in enumerate(names)}
    ds = [x.strip() for x in a.days.split(",") if x.strip()]
    ext = [x for x in a.extra.split(",") if x]
    days = {d: Day(d, Path(a.x_dir.format(mmdd=d[4:], date=d)), Path(a.base.format(date=d, mmdd=d[4:])), names,
                   [Path(e.format(date=d, mmdd=d[4:])) for e in ext]) for d in ds}
    files = {d: P.model_files(a.models, d, a.fits.split(","), a.subs.split(",")) for d in ds}
    a.work.mkdir(parents=True, exist_ok=True)
    exe = P.ensure_binary()

    def score(d: str, M: np.ndarray) -> float:
        P.write_x(a.work / "x.bin", M)
        subprocess.run([str(exe), str(a.work / "x.bin"), str(a.work / "p.bin"), *map(str, files[d]), "--threads", str(a.threads)], check=True, stdout=subprocess.DEVNULL)
        return sim(P.read_scores(a.work / "p.bin").copy(), days[d].pm, days[d].marks)

    fills = best_fills([Path(r) for r in a.reports.split(",")], a.max_score)
    vals: dict[str, dict[str, np.ndarray]] = {}
    for col, (sc, tr, cand) in sorted(fills.items(), key=lambda kv: kv[1][0]):
        b = cand  # fingerprint prints the base key as Day holds it ("x." = from --extra)
        if col not in ix or tr not in RAW or any(b not in D.bases for D in days.values()):
            print(f"  skip {col}: {tr} {cand} (not a model column / transform / candidate on every day)")
            continue
        per = {}
        for d, D in days.items():
            v = D.bases[b]
            with np.errstate(divide="ignore", invalid="ignore"):
                per[d] = v if tr == "id" else -v if tr == "-id" else 1.0 / v if tr == "1/b" else raw_xs(v, D.marks, tr)
        vals[col] = per
        print(f"  fill {col:<10} <- {tr:>6} {cand:<48} fingerprint {sc:.3f}")
    if not vals:
        print("nothing to fill")
        return
    s0 = {d: score(d, D.X) for d, D in days.items()}

    def filled(d: str, cols: list[str]) -> np.ndarray:
        M = days[d].X.copy()
        for c in cols:
            M[:, ix[c]] = vals[c][d]
        return M

    cols = list(vals)
    sj = {d: score(d, filled(d, cols)) for d in ds}
    print("baseline " + "  ".join(f"{d} {s0[d]:.4f}" for d in ds))
    print(f"all {len(cols)} fills  " + "  ".join(f"{d} {sj[d]:.4f} ({sj[d] - s0[d]:+.4f})" for d in ds), flush=True)
    print("leave one out (joint minus without it; + = the fill helps inside the set):")
    for c in cols:
        lo = {d: score(d, filled(d, [x for x in cols if x != c])) for d in ds}
        print(f"  {c:<10} " + "  ".join(f"{d[4:]} {sj[d] - lo[d]:+.4f}" for d in ds), flush=True)
    print("cumulative, fingerprint order:")
    for k in range(1, len(cols) + 1):
        sk = {d: score(d, filled(d, cols[:k])) for d in ds}
        print(f"  +{cols[k - 1]:<10} " + "  ".join(f"{d[4:]} {sk[d] - s0[d]:+.4f}" for d in ds), flush=True)
    for f in ("x.bin", "p.bin"):
        (a.work / f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
