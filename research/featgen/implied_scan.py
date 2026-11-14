r"""A fast implied scan for open model columns: put each C++ base quantity (calibrated onto PM's training CDF) into the
open column, score with the C++ predictor, keep what raises the similarity to PM's prediction on every day given.

R83 showed which open columns move today's scores most (zzbt.b, zzaa.b, zzxo, zzpp.b, zzop.i ...); a column like that is
worth a wide search. For each target column with training-CDF points, for each base quantity b of run_day.py's
base_<D>.parquet at the scoring marks, and each of the transforms b, -b, |b|, rank(b) per mark, -rank(b): the column is
set to ``calibrate(transform)`` (ic_score.calibrate: the pooled mid-rank ECDF through PM's points, i.e. a ``cal:``
formula) and the day re-scored; the similarity change against the NaN baseline is recorded per day. Candidates are
ranked by the smaller of the days' gains (both must help).

    python research/featgen/implied_scan.py --days 20261112,20260430 --target zzbt.b,zzaa.b,zzxo,zzpp.b,zzop.i \
        --x-dir data/reports/stage6_r70_{mmdd} --base data/featgen_r67/base_{date}.parquet --models $DATA/models_gen/B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pm_factors"))
import pred_day as P  # noqa: E402
from missing_damage import sim  # noqa: E402


class Day:
    """One day: the scored matrix, PM's prediction, the bases at the same rows, the calibration points."""

    def __init__(self, d: str, x_dir: Path, base: Path, names: list[str]) -> None:
        x = pl.read_parquet(x_dir / "x.parquet").filter(pl.col("date") == int(d)).drop("date").with_columns(pl.col("sid").cast(pl.Int64)).sort("sid", "mark")
        pr = pl.read_parquet(x_dir / "pred.parquet").filter(pl.col("date") == int(d)).with_columns(pl.col("sid").cast(pl.Int64))
        key = x.select("sid", "mark")
        self.pm = key.join(pr.select("sid", "mark", "pm"), on=["sid", "mark"], how="left")["pm"].to_numpy().astype(float)
        self.marks = key["mark"].to_numpy()
        b = pl.read_parquet(base).with_columns(pl.col("sid").cast(pl.Int64))
        bj = key.join(b, on=["sid", "mark"], how="left")
        self.bases = {c: bj[c].to_numpy().astype(float) for c in b.columns if c not in ("sid", "mark")}
        self.found = float(bj[next(iter(self.bases))].is_not_null().mean()) if self.bases else 0.0
        ix = {n: i for i, n in enumerate(names)}
        self.X = np.full((x.height, len(names)), np.nan)
        for c in x.columns:
            if c in ix:
                self.X[:, ix[c]] = x[c].to_numpy().astype(float)
        self.cal = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.loads((x_dir / f"cal_{d}.json").read_text()).items()}


def per_mark_rank(v: np.ndarray, marks: np.ndarray, sign: float) -> np.ndarray:
    out = np.full(v.size, np.nan)
    for mk in np.unique(marks):
        m = (marks == mk) & np.isfinite(v)
        if m.sum() > 1:
            out[m] = pl.Series(sign * v[m]).rank("average").to_numpy() / m.sum()
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", required=True)
    ap.add_argument("--target", required=True, help="comma list of open model columns")
    ap.add_argument("--x-dir", required=True, help="dir of ic_score --dump-x/--dump-pred output, {mmdd} / {date} filled per day")
    ap.add_argument("--base", required=True, help="run_day.py's base parquet at the same marks, {date} filled per day")
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--work", type=Path, default=Path("data/implied_scan_work"))
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    import ic_score as IC

    names = P.feature_names(a.feature_csv)
    ix = {n: i for i, n in enumerate(names)}
    ds = [x.strip() for x in a.days.split(",") if x.strip()]
    days = {d: Day(d, Path(a.x_dir.format(mmdd=d[4:], date=d)), Path(a.base.format(date=d, mmdd=d[4:])), names) for d in ds}
    files = {d: P.model_files(a.models, d, a.fits.split(","), a.subs.split(",")) for d in ds}
    a.work.mkdir(parents=True, exist_ok=True)
    exe = P.ensure_binary()

    def score(d: str, M: np.ndarray) -> np.ndarray:
        P.write_x(a.work / "x.bin", M)
        subprocess.run([str(exe), str(a.work / "x.bin"), str(a.work / "p.bin"), *map(str, files[d]), "--threads", str(a.threads)], check=True,
                       stdout=subprocess.DEVNULL)
        return P.read_scores(a.work / "p.bin").copy()

    s0 = {d: sim(score(d, D.X), D.pm, D.marks) for d, D in days.items()}
    print("baseline similarity " + ", ".join(f"{d} {v:.4f} (bases found at {days[d].found:.3f} of rows)" for d, v in s0.items()), flush=True)
    bases = sorted(set.intersection(*[set(D.bases) for D in days.values()]))
    for tgt in [t.strip() for t in a.target.split(",") if t.strip()]:
        if tgt not in ix or any(tgt not in D.cal for D in days.values()):
            print(f"-- {tgt}: not a model column or no training-CDF points on every day, skipped")
            continue
        t0 = time.perf_counter()
        rows = []
        for b in bases:
            for tr in ("b", "-b", "|b|", "rank", "-rank"):
                gain = {}
                for d, D in days.items():
                    v = D.bases[b]
                    v = (-v if tr == "-b" else np.abs(v) if tr == "|b|" else per_mark_rank(v, D.marks, 1.0) if tr == "rank"
                         else per_mark_rank(v, D.marks, -1.0) if tr == "-rank" else v)
                    if np.isfinite(v).sum() < 0.5 * v.size:
                        break
                    M = D.X.copy()
                    M[:, ix[tgt]] = IC.calibrate(v, D.cal[tgt])
                    gain[d] = sim(score(d, M), D.pm, D.marks) - s0[d]
                if len(gain) == len(days):
                    label = {"b": f"cal:gen.{b}", "-b": f"cal:-gen.{b}", "|b|": f"cal:|gen.{b}|", "rank": f"cal:rank:gen.{b}", "-rank": f"cal:rank:-gen.{b}"}[tr]
                    rows.append((min(gain.values()), label, gain))
        rows.sort(key=lambda r: -r[0])
        print(f"-- {tgt}: {len(rows)} candidates in {time.perf_counter() - t0:.0f}s; best by the smaller day gain:")
        for mn, name, gain in rows[: a.top]:
            print(f"   {name:<52} min {mn:+.4f}  " + "  ".join(f"{d} {g:+.4f}" for d, g in gain.items()))
        print(f"   share of candidates helping on every day: {np.mean([r[0] > 0 for r in rows]) if rows else float('nan'):.3f}", flush=True)
    (a.work / "x.bin").unlink(missing_ok=True)
    (a.work / "p.bin").unlink(missing_ok=True)


if __name__ == "__main__":
    main()
