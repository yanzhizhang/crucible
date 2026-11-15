r"""Greedy joint fill: add the units-fingerprint fill that raises the WORST day most, keep it, repeat.

R114 showed fills interact (zzeg's leave-one-out +0.07..+0.12 inside a set, ~0 alone), so columns must be chosen in
context. Options = the top --per-col fingerprint rows of every open column (fingerprint.py reports). Each step scores
every remaining option on top of the current fills in ONE lgbpred --batch run per day (each batch vector may target its
own column), adds the option whose smallest gain over the days is largest, and stops when that falls below --min-gain.
A column takes at most one fill. Read-only.

    python research/featgen/greedy_fill.py --days 20260430,20261112,20261113 --reports data/reports/r108.txt,... \
        --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" --extra "..." --models $DATA/models_gen/B
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
from implied_scan import RAW, Day, raw_xs  # noqa: E402
from joint_fill import HEAD, ROW  # noqa: E402
from missing_damage import sim  # noqa: E402


def options(reports: list[Path], per_col: int, max_score: float) -> list[tuple[str, str, str, float]]:
    """(column, transform, candidate, fingerprint score): each column's best --per-col distinct rows over the reports."""
    rows: dict[str, list[tuple[float, str, str]]] = {}
    for rp in reports:
        col = None
        for ln in rp.read_text(errors="replace").splitlines():
            m = HEAD.match(ln)
            if m:
                col = m.group(1)
                continue
            r = ROW.match(ln)
            if r and col and float(r.group(1)) <= max_score:
                rows.setdefault(col, []).append((float(r.group(1)), r.group(2), r.group(3)))
    out = []
    for col, rs in rows.items():
        seen = set()
        for sc, tr, cand in sorted(rs):
            if (tr, cand) in seen:
                continue
            seen.add((tr, cand))
            out.append((col, tr, cand, sc))
            if len(seen) == per_col:
                break
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--reports", required=True)
    ap.add_argument("--per-col", type=int, default=4)
    ap.add_argument("--max-score", type=float, default=0.10)
    ap.add_argument("--min-gain", type=float, default=0.0005)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--x-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--extra", default="")
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--work", type=Path, default=Path("data/greedy_fill_work"))
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
    opts = []
    for col, tr, cand, sc in options([Path(r) for r in a.reports.split(",")], a.per_col, a.max_score):
        if col not in ix or tr not in RAW or any(cand not in D.bases for D in days.values()):
            continue
        vals = {}
        for d, D in days.items():
            v = D.bases[cand]
            with np.errstate(divide="ignore", invalid="ignore"):
                vals[d] = v if tr == "id" else -v if tr == "-id" else 1.0 / v if tr == "1/b" else raw_xs(v, D.marks, tr)
        opts.append((col, tr, cand, sc, vals))
    print(f"{len(opts)} options over {len({o[0] for o in opts})} columns, days {ds}", flush=True)

    def score(d: str, M: np.ndarray) -> float:
        P.write_x(a.work / "x.bin", M)
        subprocess.run([str(exe), str(a.work / "x.bin"), str(a.work / "p.bin"), *map(str, files[d]), "--threads", str(a.threads)], check=True, stdout=subprocess.DEVNULL)
        return sim(P.read_scores(a.work / "p.bin").copy(), days[d].pm, days[d].marks)

    def batch(d: str, M: np.ndarray, todo: list[int]) -> list[float]:
        """Similarity of M with each option in todo applied on its own column, one lgbpred run."""
        P.write_x(a.work / "x.bin", M)
        with (a.work / "c.bin").open("wb") as f:
            f.write(b"CRULC001")
            f.write(np.array([len(todo), M.shape[0]], "<u8").tobytes())
            for k in todo:
                f.write(np.array([ix[opts[k][0]]], "<u8").tobytes())
                f.write(np.ascontiguousarray(opts[k][4][d], "<f8").tobytes())
        subprocess.run([str(exe), str(a.work / "x.bin"), str(a.work / "b.bin"), *map(str, files[d]), "--batch", str(a.work / "c.bin"), "--threads", str(a.threads)],
                       check=True, stdout=subprocess.DEVNULL)
        raw = (a.work / "b.bin").read_bytes()
        n, r = (int(x) for x in np.frombuffer(raw, "<u8", 2, 8))
        S = np.frombuffer(raw, "<f8", n * r, 24).reshape(n, r)
        return [sim(S[i], days[d].pm, days[d].marks) for i in range(n)]

    M = {d: D.X.copy() for d, D in days.items()}
    s0 = {d: score(d, M[d]) for d in ds}
    cur = dict(s0)
    print("baseline " + "  ".join(f"{d} {s0[d]:.4f}" for d in ds), flush=True)
    used: set[str] = set()
    for step in range(a.steps):
        todo = [k for k, o in enumerate(opts) if o[0] not in used]
        if not todo:
            break
        res = {d: batch(d, M[d], todo) for d in ds}
        gains = [(min(res[d][i] - cur[d] for d in ds), k, i) for i, k in enumerate(todo)]
        g, k, i = max(gains)
        top = sorted(gains, reverse=True)[:3]
        print(f"step {step + 1}: best " + " | ".join(f"{opts[kk][0]} <- {opts[kk][1]} {opts[kk][2]} min {gg:+.4f}" for gg, kk, _ in top), flush=True)
        if g < a.min_gain:
            print("  stop: no option raises every day by --min-gain")
            break
        col, tr, cand, sc, vals = opts[k]
        for d in ds:
            M[d][:, ix[col]] = vals[d]
            cur[d] = res[d][i]
        used.add(col)
        print(f"  KEEP {col:<10} <- {tr:>6} {cand:<48} (fingerprint {sc:.3f})  now " + "  ".join(f"{d[4:]} {cur[d]:.4f} ({cur[d] - s0[d]:+.4f})" for d in ds), flush=True)
    print("final " + "  ".join(f"{d} {s0[d]:.4f} -> {cur[d]:.4f} ({cur[d] - s0[d]:+.4f})" for d in ds))
    for f in ("x.bin", "p.bin", "c.bin", "b.bin"):
        (a.work / f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
