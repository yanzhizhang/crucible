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

    def __init__(self, d: str, x_dir: Path, base: Path, names: list[str], extra: list[Path] | None = None) -> None:
        x = pl.read_parquet(x_dir / "x.parquet").filter(pl.col("date") == int(d)).drop("date").with_columns(pl.col("sid").cast(pl.Int64)).sort("sid", "mark")
        pr = pl.read_parquet(x_dir / "pred.parquet").filter(pl.col("date") == int(d)).with_columns(pl.col("sid").cast(pl.Int64))
        key = x.select("sid", "mark")
        self.key = key
        self.pm = key.join(pr.select("sid", "mark", "pm"), on=["sid", "mark"], how="left")["pm"].to_numpy().astype(float)
        self.marks = key["mark"].to_numpy()
        b = pl.read_parquet(base).with_columns(pl.col("sid").cast(pl.Int64))
        bj = key.join(b, on=["sid", "mark"], how="left")
        self.bases = {c: bj[c].to_numpy().astype(float) for c in b.columns if c not in ("sid", "mark")}
        self.found = float(bj[next(iter(self.bases))].is_not_null().mean()) if self.bases else 0.0
        for f in extra or []:  # more candidate sources: per (sid, mark) like engine marks, or per sid (a daily table)
            e = pl.read_parquet(f).with_columns(pl.col("sid").cast(pl.Int64))
            on = ["sid", "mark"] if "mark" in e.columns else ["sid"]
            if "mark" in e.columns and e.schema["mark"] != key.schema["mark"]:  # engine marks are float seconds, the matrix's are ints
                e = e.with_columns(pl.col("mark").round().cast(key.schema["mark"]))
            ej = key.join(e, on=on, how="left")
            sums = {c[:-4] for c in e.columns if c.endswith(".sum") and f"{c[:-4]}.wgt" in e.columns}
            for c in e.columns:
                if c in on or c.endswith(".wgt") or e.schema[c] not in (pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8, pl.UInt32, pl.UInt64, pl.Boolean):
                    continue
                if c.endswith(".sum") and c[:-4] in sums:  # an EMA: sum / wgt, as feature_match reads it
                    w = ej[f"{c[:-4]}.wgt"].to_numpy().astype(float)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        self.bases[f"x.{c[:-4]}"] = np.where(w != 0, ej[c].to_numpy().astype(float) / w, np.nan)
                else:
                    self.bases[f"x.{c}"] = ej[c].to_numpy().astype(float)
        ix = {n: i for i, n in enumerate(names)}
        self.X = np.full((x.height, len(names)), np.nan)
        for c in x.columns:
            if c in ix:
                self.X[:, ix[c]] = x[c].to_numpy().astype(float)
        self.cal = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.loads((x_dir / f"cal_{d}.json").read_text()).items()}


RAW_XS = {"dm", "-dm", "srank", "-srank", "abs4", "log", "log|b|", "-2b", "2b", "4b2", "sqrt"}
RAW = {"id", "1/b", "-id"} | RAW_XS


def raw_xs(v: np.ndarray, marks: np.ndarray, tr: str) -> np.ndarray:
    """dm / -dm: minus the mark's cross-sectional mean; srank / -srank: rank in [-1, 1] per mark; abs4: |v| * 1e4."""
    if tr in ("-2b", "2b", "4b2", "sqrt"):  # scalings seen in PM's states: zzmh.r = 4 dd^2 -> zzmh ~ 2 dd = sqrt(zzmh.r)
        with np.errstate(invalid="ignore"):
            return -2 * v if tr == "-2b" else 2 * v if tr == "2b" else 4 * v * v if tr == "4b2" else np.sqrt(v)
    if tr in ("log", "log|b|"):  # natural log of the value (non-positive -> NaN) or of its magnitude
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(np.abs(v)) if tr == "log|b|" else np.where(v > 0, np.log(np.where(v > 0, v, 1.0)), np.nan)
    if tr == "abs4":
        return np.abs(v) * 1e4
    out = np.full(v.size, np.nan)
    if tr in ("srank", "-srank"):
        for mk in np.unique(marks):
            m = (marks == mk) & np.isfinite(v)
            if m.sum() > 1:
                r = pl.Series((1.0 if tr == "srank" else -1.0) * v[m]).rank("average").to_numpy()
                out[m] = 2 * (r - 0.5) / m.sum() - 1
        return out
    sg = 1.0 if tr == "dm" else -1.0
    for mk in np.unique(marks):
        m = (marks == mk) & np.isfinite(v)
        if m.any():
            out[m] = sg * (v[m] - v[m].mean())
    return out


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
    ap.add_argument("--models", type=Path, default=Path("/work/crucible_data/prod_models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/work/crucible_data/prod_models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--extra", default="", help="comma list of more candidate parquets ({date} / {mmdd} filled): (sid, mark, ...) or (sid, ...)")
    ap.add_argument("--only-extra", action="store_true", help="candidates from --extra only, not the tick bases")
    ap.add_argument("--transforms", default="b,-b,|b|,rank,-rank", help="subset of b, -b, |b|, rank, -rank")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--work", type=Path, default=Path("data/implied_scan_work"))
    ap.add_argument("--fill-from", default="", help="invert_cols implied_{date}.parquet: every other column it holds is set to its implied value first")
    ap.add_argument("--target-nan", action="store_true", help="with --fill-from: the targets themselves start as NaN (baseline = context without them)")
    ap.add_argument("--rank-day", default="", help="also list the best candidates on this one day (the trusted one); every gain goes to <work>/gains_<col>.csv")
    ap.add_argument("--cal-implied", default="", help="for a target without training-CDF points: invert_cols implied_{date}.parquet, pooled over --days, as the CDF")
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    import ic_score as IC

    names = P.feature_names(a.feature_csv)
    ix = {n: i for i, n in enumerate(names)}
    ds = [x.strip() for x in a.days.split(",") if x.strip()]
    ext = [x for x in a.extra.split(",") if x]
    days = {d: Day(d, Path(a.x_dir.format(mmdd=d[4:], date=d)), Path(a.base.format(date=d, mmdd=d[4:])), names,
                   [Path(e.format(date=d, mmdd=d[4:])) for e in ext]) for d in ds}
    files = {d: P.model_files(a.models, d, a.fits.split(","), a.subs.split(",")) for d in ds}
    a.work.mkdir(parents=True, exist_ok=True)
    exe = P.ensure_binary()

    def score(d: str, M: np.ndarray) -> np.ndarray:
        P.write_x(a.work / "x.bin", M)
        subprocess.run([str(exe), str(a.work / "x.bin"), str(a.work / "p.bin"), *map(str, files[d]), "--threads", str(a.threads)], check=True,
                       stdout=subprocess.DEVNULL)
        return P.read_scores(a.work / "p.bin").copy()

    def batch(d: str, X: np.ndarray, col: int, cols: list[np.ndarray]) -> np.ndarray:
        P.write_x(a.work / "x.bin", X)
        with (a.work / "c.bin").open("wb") as f:
            f.write(b"CRULC001")
            f.write(np.array([len(cols), X.shape[0]], "<u8").tobytes())
            for v in cols:
                f.write(np.array([col], "<u8").tobytes())
                f.write(np.ascontiguousarray(v, "<f8").tobytes())
        subprocess.run([str(exe), str(a.work / "x.bin"), str(a.work / "b.bin"), *map(str, files[d]), "--batch", str(a.work / "c.bin"),
                        "--threads", str(a.threads)], check=True, stdout=subprocess.DEVNULL)
        raw = (a.work / "b.bin").read_bytes()
        assert raw[:8] == b"CRULB001", "not an lgbpred batch output"
        n, r = (int(x) for x in np.frombuffer(raw, "<u8", 2, 8))
        return np.frombuffer(raw, "<f8", n * r, 24).reshape(n, r)

    if a.fill_from:  # the other open columns at their invert_cols implied values: judge a candidate in PM-like context
        tg = {t.strip() for t in a.target.split(",")}
        for d, D in days.items():
            imp = D.key.join(pl.read_parquet(a.fill_from.format(date=d, mmdd=d[4:])), on=["sid", "mark"], how="left")
            cols = [c for c in imp.columns if c in ix and c not in ("sid", "mark") and c not in tg]
            for c in cols:
                D.X[:, ix[c]] = imp[c].to_numpy().astype(float)
            print(f"  {d}: {len(cols)} columns filled from {a.fill_from.format(date=d, mmdd=d[4:])}; targets left as they were", flush=True)
            if a.target_nan:
                for t in tg:
                    if t in ix:
                        D.X[:, ix[t]] = np.nan
    s0 = {d: sim(score(d, D.X), D.pm, D.marks) for d, D in days.items()}
    print("baseline similarity " + ", ".join(f"{d} {v:.4f} (bases found at {days[d].found:.3f} of rows)" for d, v in s0.items()), flush=True)
    bases = sorted(set.intersection(*[set(D.bases) for D in days.values()]))
    if a.only_extra:
        bases = [b for b in bases if b.startswith("x.")]
    print(f"{len(bases)} candidate bases ({sum(b.startswith('x.') for b in bases)} from --extra)", flush=True)
    want = []
    for t in (t.strip() for t in a.target.split(",") if t.strip()):  # "zzro*" = every model column of the family (zzro, zzro2, ...)
        want += [n for n in names if n.split(".")[0].rstrip("0123456789") == t[:-1]] if t.endswith("*") else [t]
    for tgt in dict.fromkeys(want):
        if tgt in ix and a.cal_implied and any(tgt not in D.cal for D in days.values()):  # stand-in CDF: the implied values' quantiles
            v = np.concatenate([pl.read_parquet(a.cal_implied.format(date=d, mmdd=d[4:]), columns=[tgt])[tgt].to_numpy() for d in ds])
            v = np.sort(v[np.isfinite(v)])
            if v.size >= 100:
                q = np.linspace(0, 1, 201)
                for D in days.values():
                    D.cal.setdefault(tgt, (np.quantile(v, q), q))
                print(f"-- {tgt}: no training-CDF points on every day, using {v.size:,} implied values as the CDF")
        raw_only = set(a.transforms.split(",")) <= RAW
        if tgt not in ix or (not raw_only and any(tgt not in D.cal for D in days.values())):
            print(f"-- {tgt}: not a model column or no training-CDF points on every day, skipped")
            continue
        t0 = time.perf_counter()
        cands = [(b, tr) for b in bases for tr in a.transforms.split(",")]
        gains: dict[tuple[str, str], dict[str, float]] = {k: {} for k in cands}
        for d, D in days.items():  # every candidate of this column on this day in one lgbpred run (--batch)
            keep, cols = [], []
            for b, tr in cands:
                v = D.bases[b]
                v = (-v if tr == "-b" else np.abs(v) if tr == "|b|" else per_mark_rank(v, D.marks, 1.0) if tr == "rank"
                     else per_mark_rank(v, D.marks, -1.0) if tr == "-rank" else 1.0 / v if tr == "1/b" else -v if tr == "-id" else v)
                if tr in RAW_XS:  # ic_score's formula prefixes, PM's units: dm: (minus the mark's mean), srank: (rank in [-1, 1]), |b|*1e4
                    v = raw_xs(D.bases[b], D.marks, tr)
                if np.isfinite(v).sum() >= 0.5 * v.size:
                    keep.append((b, tr))
                    cols.append(v if tr in RAW else IC.calibrate(v, D.cal[tgt]))  # raw transforms: the value as is, PM's units
            if not keep:
                continue
            for k, sc in zip(keep, batch(d, D.X, ix[tgt], cols), strict=True):
                gains[k][d] = sim(sc, D.pm, D.marks) - s0[d]
        rows = []
        for (b, tr), gain in gains.items():
            if len(gain) == len(days):
                nm = b[2:] if b.startswith("x.") else f"gen.{b}"
                label = {"b": f"cal:{nm}", "-b": f"cal:-{nm}", "|b|": f"cal:|{nm}|", "rank": f"cal:rank:{nm}", "-rank": f"cal:rank:-{nm}", "id": nm, "1/b": f"1/{nm}", "-id": f"-{nm}", "dm": f"dm:{nm}", "-dm": f"dm:-{nm}", "srank": f"srank:{nm}", "-srank": f"srank:-{nm}", "abs4": f"|{nm}|*1e4", "log": f"log:{nm}", "log|b|": f"log|{nm}|", "-2b": f"-2*{nm}", "2b": f"2*{nm}", "4b2": f"4*{nm}^2", "sqrt": f"sqrt:{nm}"}[tr]
                rows.append((min(gain.values()), label, gain))
        rows.sort(key=lambda r: -r[0])
        print(f"-- {tgt}: {len(rows)} candidates in {time.perf_counter() - t0:.0f}s; best by the smaller day gain:")
        for mn, name, gain in rows[: a.top]:
            print(f"   {name:<52} min {mn:+.4f}  " + "  ".join(f"{d} {g:+.4f}" for d, g in gain.items()))
        print(f"   share of candidates helping on every day: {np.mean([r[0] > 0 for r in rows]) if rows else float('nan'):.3f}", flush=True)
        if a.rank_day and rows and a.rank_day in rows[0][2]:  # the trusted day first (0430: PM's production alpha)
            print(f"   best on {a.rank_day}:")
            for mn, name, gain in sorted(rows, key=lambda r: -r[2][a.rank_day])[: a.top]:
                print(f"   {name:<52} {a.rank_day} {gain[a.rank_day]:+.4f}  min {mn:+.4f}  " + "  ".join(f"{d[4:]} {g:+.4f}" for d, g in gain.items() if d != a.rank_day))
        if rows:
            pl.DataFrame([{"cand": n, "min": m, **g} for m, n, g in rows]).write_csv(a.work / f"gains_{tgt}.csv")
    (a.work / "x.bin").unlink(missing_ok=True)
    for f in ("p.bin", "c.bin", "b.bin"):
        (a.work / f).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
