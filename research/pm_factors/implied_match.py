r"""Stage 4, A2: which quantity PM feeds a samplerS column -- found by inverting its cross-day update, not by guessing blends.

PM's profile columns update once a day as ``pm_d = a pm_{d-1} + b x_d``. Instead of trying blends of our days against
``pm_d`` (the old ``basic_match``), take PM's own previous file as the state and solve for the input it was fed:
``y_d(a) = pm_d - a pm_{d-1} = b x_d``. For a small grid of ``a`` (0 = no memory, so plain same-day matching is a special
case) the implied input is scored against EVERY basic column at once (one join, then numpy over an N x K matrix):
``exact_scaled`` (share of cells within 1e-3 after dividing by the median ratio), ``spearman``, the median ``ratio``.
The right ``a`` is where the right basic becomes exact.

Columns no single basic explains get a structure search: a library of transforms of the 20 best-correlated basics
(log, inverse, square root, pairwise ratios and products of the top 10) scored the same way, then 2-term least squares
over the best 15 single terms (OMP style) -- the best formulas per column are printed with their coefficients.

    python research/pm_factors/implied_match.py --dates 20260923,20260924 --out data/reports/stage4/implied_match.csv
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from basic_match import BASICS
from rank_candidates import pm_frame

A_GRID = (0.0, 0.5, 0.8, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995)
MAX_CELLS = 60000  # cells scored per (file, V, day); a random subset is plenty for ranks and medians
TOL = 1e-3


def rank(x: np.ndarray) -> np.ndarray:
    """Average ranks of a 1-D array (NaN kept as NaN)."""
    out = np.full(x.shape, np.nan)
    ok = np.isfinite(x)
    v = x[ok]
    order = np.argsort(v, kind="stable")
    r = np.empty(v.size)
    r[order] = np.arange(v.size, dtype=float)
    # ties -> mean rank
    sv = v[order]
    starts = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1]])
    ends = np.r_[starts[1:], sv.size]
    mean_rank = np.repeat((starts + ends - 1) / 2.0, ends - starts)
    rr = np.empty(v.size)
    rr[order] = mean_rank
    out[ok] = rr
    return out


def rank_cols(B: np.ndarray) -> np.ndarray:
    """Ranks of every column of B (each over its own finite cells), computed once per join."""
    return np.column_stack([rank(B[:, k]) for k in range(B.shape[1])]) if B.shape[1] else B


def col_scores(y: np.ndarray, B: np.ndarray, RB: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per column of B: n, median ratio y / B, exact_scaled share, spearman with y (column ranks RB reused across a)."""
    n_cols = B.shape[1]
    ratio = np.full(n_cols, np.nan)
    exact = np.zeros(n_cols)
    rho = np.full(n_cols, np.nan)
    n = np.zeros(n_cols, dtype=int)
    ry = rank(y)
    if RB is None:
        RB = rank_cols(B)
    for k in range(n_cols):
        b = B[:, k]
        ok = np.isfinite(y) & np.isfinite(b)
        n[k] = int(ok.sum())
        if n[k] < 100:
            continue
        nz = ok & (b != 0)
        if nz.sum() >= 100:
            r = np.median(y[nz] / b[nz])
            ratio[k] = r
            exact[k] = float(np.mean(np.abs(y[ok] - r * b[ok]) <= TOL * np.abs(y[ok]) + 1e-12))
        a_, b_ = ry[ok], RB[ok, k]
        sa, sb = a_.std(), b_.std()
        if sa > 0 and sb > 0:
            rho[k] = float(np.mean((a_ - a_.mean()) * (b_ - b_.mean())) / (sa * sb))
    return n, ratio, exact, rho


def load_basics(basics_dir: Path, day: str, table: str) -> pl.DataFrame | None:
    """One basics table with Int64 keys, or None when the day is missing."""
    g = basics_dir / f"date={day}" / f"{table}.parquet"
    if not g.exists():
        return None
    return pl.read_parquet(g).rename({"symbol_id": "code"}).with_columns(pl.col("code").cast(pl.Int64), pl.col("minute").cast(pl.Int64))


def pm_cells(path: Path, V: str, val: str, name: str) -> pl.DataFrame:
    """PM file -> (code, minute, <name>) for one V and value column."""
    pm, _, _ = pm_frame(path, None, None, None, None, None)
    return pm.filter(pl.col("V") == V).select("code", "minute", pl.col(val).alias(name))


def lin_fit(X: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float, float] | None:
    """Least squares x ~ X (finite rows only): coefficients, ok3 share, correlation of fit and x."""
    ok = np.isfinite(x) & np.all(np.isfinite(X), axis=1)
    if ok.sum() < 100:
        return None
    try:
        coef, *_ = np.linalg.lstsq(X[ok], x[ok], rcond=None)
    except np.linalg.LinAlgError:
        return None
    fit = X[ok] @ coef
    ok3 = float(np.mean(np.abs(x[ok] - fit) <= TOL * np.abs(x[ok]) + 1e-12))
    rho = float(np.corrcoef(fit, x[ok])[0, 1]) if fit.std() > 0 else float("nan")
    return coef, ok3, rho


def structure_search(x: np.ndarray, xp: np.ndarray, B: np.ndarray, names: list[str], RB: np.ndarray) -> list[dict]:
    """Fit pm_d = a pm_prev + c1 L_i (+ c2 L_j) over a library of transforms of the basics best correlated with the memory-free residual."""
    has_prev = np.isfinite(xp).mean() > 0.5
    if has_prev:
        f0 = lin_fit(xp[:, None], x)
        y0 = x - (f0[0][0] * xp if f0 else 0.0)  # pm_d with its own memory partialled out: what the input must explain
    else:
        y0 = x
    _, _, _, rho = col_scores(y0, B, RB)
    top = [k for k in np.argsort(-np.nan_to_num(np.abs(rho)))[:20] if np.isfinite(rho[k])]
    lib: list[tuple[str, np.ndarray]] = []
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in top:
            b = B[:, k]
            lib += [(names[k], b), (f"log({names[k]})", np.log(np.where(b > 0, b, np.nan))), (f"1/{names[k]}", 1 / np.where(b != 0, b, np.nan)),
                    (f"sqrt({names[k]})", np.sqrt(np.where(b >= 0, b, np.nan)))]
        for i, j in itertools.combinations(top[:10], 2):
            bi, bj = B[:, i], B[:, j]
            lib += [(f"{names[i]}/{names[j]}", bi / np.where(bj != 0, bj, np.nan)), (f"{names[j]}/{names[i]}", bj / np.where(bi != 0, bi, np.nan)),
                    (f"{names[i]}*{names[j]}", bi * bj)]
    if not lib:
        return []
    L = np.column_stack([v for _, v in lib])
    lnames = [nm for nm, _ in lib]
    _, _, _, rho1 = col_scores(y0, L)
    mem = [xp] if has_prev else []

    def row(cols: list[int]) -> dict | None:
        f = lin_fit(np.column_stack([*mem, *[L[:, c] for c in cols]]), x)
        if f is None:
            return None
        coef, ok3, r = f
        a_ = float(coef[0]) if has_prev else 0.0
        cs = coef[1:] if has_prev else coef
        return {"formula": " + ".join(f"{c:.6g} * {lnames[k]}" for c, k in zip(cs, cols, strict=True)), "a": a_, "terms": len(cols), "ok3": ok3, "rho": r}

    best = [k for k in np.argsort(-np.nan_to_num(np.abs(rho1)))[:15]]
    singles = [(k, r) for k in best if (r := row([k]))]
    out = [r for _, r in singles]
    out += [r for i, j in itertools.combinations(best, 2) if (r := row([i, j]))]
    if singles:  # OMP step: the second term is whatever best explains the residual of the best single-term fit
        k1 = max(singles, key=lambda kr: (kr[1]["ok3"], abs(kr[1]["rho"]) if np.isfinite(kr[1]["rho"]) else 0))[0]
        f1 = lin_fit(np.column_stack([*mem, L[:, k1]]), x)
        if f1 is not None:
            res = x - np.column_stack([*mem, L[:, k1]]) @ f1[0]
            _, _, _, rho2 = col_scores(res, L)
            out += [r for k in np.argsort(-np.nan_to_num(np.abs(rho2)))[:15] if k != k1 and (r := row([k1, int(k)]))]
    out.sort(key=lambda r: (-r["ok3"], -(abs(r["rho"]) if np.isfinite(r["rho"]) else 0)))
    return out[:4]


def match_file(f: Path, d: str, dp: str | None, basics_dir: Path, search: bool, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """Score one PM file of day d: every (V, value column, ours day, shift, a) against every basic; structure search on the weak ones."""
    rng = np.random.default_rng(seed)
    pm, vals, kind = pm_frame(f, None, None, None, None, None)
    name = f.stem.rsplit("_", 1)[0]
    if kind == "time":
        step = int(pm.select(pl.col("minute").unique().sort().diff().drop_nulls().min()).item() or 1)
        table, keys, shifts = ("m5" if step >= 5 else "m1"), ["code", "minute"], (0, step, -step)
    elif kind == "daily":
        table, keys, shifts, step = "d1", ["code"], (0,), 0
    else:
        print(f"{f.name}: I kind {kind} not handled")
        return [], []
    prev_path = f.with_name(f"{name}_{dp}.parquet") if dp else None
    pmp = pl.read_parquet(prev_path) if prev_path and prev_path.exists() else None
    rows, forms = [], []
    for V in sorted(pm["V"].unique().to_list()):
        for val in vals:
            cur = pm.filter(pl.col("V") == V).select(*keys, pl.col(val).alias("x"))
            if pmp is not None:
                cur = cur.join(pm_cells(prev_path, V, val, "xp").select(*keys, "xp"), on=keys, how="left")
            else:
                cur = cur.with_columns(pl.lit(None, pl.Float64).alias("xp"))
            for oday in ([d, dp] if dp else [d]):
                t = load_basics(basics_dir, oday, table)
                if t is None:
                    continue
                names = [c for c in t.columns if c not in ("code", "minute")]
                for sh in shifts:
                    p = cur.with_columns(pl.col("minute") + sh) if kind == "time" else cur
                    j = p.join(t.drop("minute") if kind == "daily" else t, on=keys)
                    j = j.filter(pl.col("x").is_finite())
                    if j.height < 100:
                        continue
                    if j.height > MAX_CELLS:
                        j = j[np.sort(rng.choice(j.height, MAX_CELLS, replace=False))]
                    x, xp = j["x"].to_numpy(), j["xp"].to_numpy().astype(float)
                    B = j.select(names).to_numpy().astype(float)
                    RB = rank_cols(B)
                    has_prev = np.isfinite(xp).mean() > 0.5
                    best_exact = 0.0
                    for a_ in (A_GRID if has_prev else (0.0,)):
                        y = x - a_ * np.nan_to_num(xp) if a_ else x
                        if a_:
                            y = np.where(np.isfinite(xp), y, np.nan)
                        n, ratio, exact, rho = col_scores(y, B, RB)
                        for k, nm in enumerate(names):
                            if n[k] >= 100:
                                rows.append({"file": name, "V": V, "value_col": val, "day": d, "ours_day": "d" if oday == d else "d-1", "shift": sh,
                                             "a": a_, "basic": nm, "n": int(n[k]), "ratio": float(ratio[k]), "exact_scaled": float(exact[k]), "spearman": float(rho[k])})
                        best_exact = max(best_exact, float(exact.max()))
                    if search and oday == d and sh == 0 and best_exact < 0.5:
                        for r in structure_search(x, xp, B, names, RB):
                            forms.append({"file": name, "V": V, "value_col": val, "day": d, **r})
    print(f"{d} {name}: {kind}, {len(vals)} value columns, {pm['V'].n_unique()} V, prev {dp}", flush=True)
    return rows, forms


def _job(args: tuple) -> tuple[list[dict], list[dict]]:
    return match_file(*args)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
    ap.add_argument("--basics", type=Path, default=BASICS)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1, help="(day, file) pairs scored in parallel")
    ap.add_argument("--no-search", action="store_true", help="skip the structure search")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4/implied_match.csv"))
    a = ap.parse_args()
    dates = [x for x in a.dates.split(",") if x]
    pm_days = sorted({f.stem.rsplit("_", 1)[1] for f in a.exports.glob("*.parquet")})
    jobs = []
    for d in dates:
        prevs = [x for x in pm_days if x < d]
        for f in sorted(a.exports.glob(f"*_{d}.parquet")):
            jobs.append((f, d, prevs[-1] if prevs else None, a.basics, not a.no_search))
    rows, forms = [], []
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            for r, fm in pool.imap_unordered(_job, jobs):
                rows += r
                forms += fm
    else:
        for job in jobs:
            r, fm = match_file(*job)
            rows += r
            forms += fm
    if not rows:
        print("nothing scored")
        return
    t = pl.DataFrame(rows)
    agg = (t.group_by("file", "value_col", "V", "basic", "ours_day", "shift", "a").agg(
                pl.len().alias("days"), pl.col("n").sum(), pl.col("exact_scaled").mean(), pl.col("ratio").median(), pl.col("spearman").mean())
           .sort("file", "V", "exact_scaled", "spearman", descending=[False, False, True, True], nulls_last=True))
    agg.write_csv(a.out)
    best = agg.group_by("file", "value_col", "V", maintain_order=True).head(a.top)
    with pl.Config(tbl_rows=500, tbl_cols=-1, tbl_width_chars=220, float_precision=3):
        print(f"\n== implied input y = pm_d - a pm_prev vs every basic, best {a.top} per column over {len(dates)} day(s) "
              f"(exact_scaled = share of cells within 1e-3 after scaling by ratio; a = 0 is plain same-day matching)")
        print(best.select("file", "V", "basic", "a", "ours_day", "shift", "days", "n", "exact_scaled", "ratio", "spearman"))
    if forms:
        ft = pl.DataFrame(forms)
        fagg = (ft.group_by("file", "value_col", "V", "formula", "terms").agg(pl.len().alias("days"), pl.col("a").median(), pl.col("ok3").mean(), pl.col("rho").mean())
                .sort("file", "V", "ok3", descending=[False, False, True], nulls_last=True))
        fagg.write_csv(a.out.with_name("implied_formulas.csv"))
        with pl.Config(tbl_rows=300, tbl_cols=-1, tbl_width_chars=220, float_precision=3, fmt_str_lengths=80):
            print("\n== structure search on the columns no single basic explains (exact_scaled < 0.5): best 3 formulas per column")
            print(fagg.group_by("file", "value_col", "V", maintain_order=True).head(3).select("file", "V", "formula", "a", "days", "ok3", "rho"))
    print(f"table: {a.out}")


if __name__ == "__main__":
    main()
