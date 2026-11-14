r"""One day's model columns from the C++ formula layer (featvec), assembled from the C++ base quantities and the side tables.

Inputs, all on the same time grid:
  --base       run_day.py's base_<D>.parquet                       (sid, mark, the tick base quantities)
  --full-book  full_book.py's date=<D>.parquet                     (sid, mark, best_level_order_count_imbalance ...)
  --engine     run_cpp.py's marks_<D>.parquet                      (sid, mark, the state engine's keys)
  --side       ic_score --dump-x's side_<D>.npz, --cal cal_<D>.json (CED sids, Barra industry, exposures; training-CDF points)
Output: <out>/vec_<D>.parquet (sid, mark, one column per formula of --formulas).
With --x (ic_score --dump-x's x.parquet) every column is compared with the matrix ic_score scored.

    python research/featgen/vec_day.py --date 20260430 --base data/featgen_x0/base_20260430.parquet \
        --full-book $DATA/store/pm_factors/full_book_m7/date=20260430.parquet --engine data/engine_m7/marks_20260430.parquet \
        --side $R/stage6_r62_0430/side_20260430.npz --cal $R/stage6_r62_0430/cal_20260430.json --x $R/stage6_r62_0430/x.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import featio as IO  # noqa: E402

BASELINE = Path(__file__).resolve().parents[1] / "engine" / "baseline_r41c.txt"
BOOK = ["best_level_order_count_imbalance"]


def state_column(mk: pl.DataFrame, name: str) -> pl.Expr | None:
    """A state name as ``feature_match.atoms`` reads it: an EMA is <name>.sum / <name>.wgt, anything else the key itself."""
    if f"{name}.sum" in mk.columns:
        w = pl.col(f"{name}.wgt") if f"{name}.wgt" in mk.columns else pl.lit(1.0)
        return pl.when(w != 0).then(pl.col(f"{name}.sum") / w).otherwise(None).alias(name)
    return pl.col(name) if name in mk.columns else None


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--full-book", type=Path)
    ap.add_argument("--engine", type=Path)
    ap.add_argument("--side", type=Path)
    ap.add_argument("--cal", type=Path)
    ap.add_argument("--formulas", type=Path, default=BASELINE)
    ap.add_argument("--x", type=Path, help="ic_score --dump-x's x.parquet: compare every column with it")
    ap.add_argument("--out", type=Path, default=Path("data/featvec"))
    ap.add_argument("--threads", type=int, default=32)
    a = ap.parse_args()
    t0 = time.perf_counter()
    forms = IO.read_formulas(a.formulas)
    base = pl.read_parquet(a.base).sort("sid", "mark")
    sids, T = base["sid"].unique().sort().to_numpy(), base["mark"].unique().sort().to_numpy()
    ns, nt = sids.size, T.size
    assert base.height == ns * nt, "base parquet is not a full sid x mark grid"
    key = base.select("sid", "mark")
    cols: dict[str, np.ndarray] = {("px" if c == "px" else f"gen.{c}"): base[c].to_numpy() for c in base.columns if c not in ("sid", "mark")}
    note = []
    if a.full_book and a.full_book.exists():
        fb = key.join(pl.read_parquet(a.full_book, columns=["sid", "mark", *BOOK]).with_columns(pl.col("sid").cast(pl.Int64)), on=["sid", "mark"], how="left").sort("sid", "mark")
        for c in BOOK:
            cols[f"gen.{c}"] = fb[c].to_numpy().astype(float)
        note.append(f"book rows matched {fb[BOOK[0]].is_not_null().mean():.3f}")
    need = sorted({k for _, e in forms for k in re.findall(r"[A-Za-z_][A-Za-z0-9_.]*", e.removeprefix("cal:")) if k.startswith("zz")})
    if a.engine and a.engine.exists():
        mk = pl.read_parquet(a.engine)
        ex = [x for x in (state_column(mk, k) for k in need) if x is not None]
        st = key.join(mk.select(pl.col("sid").cast(pl.Int64), "mark", *ex), on=["sid", "mark"], how="left").sort("sid", "mark")
        for k in need:
            if k in st.columns:
                cols[k] = st[k].to_numpy().astype(float)
        note.append(f"state names found {sum(k in st.columns for k in need)} of {len(need)}")
    names = list(cols)
    missing = sorted({k for _, e in forms for k in re.findall(r"gen\.[A-Za-z0-9_]+|zz[a-z0-9]+\.[A-Za-z.]+|\bpx\b", e)} - set(names))
    for k in missing:  # a name nobody supplied: the columns built on it come out NaN
        cols[k] = np.full(ns * nt, np.nan)
    names = list(cols)
    val = np.stack([np.asarray(v, float).reshape(ns, nt) for v in cols.values()], axis=1)  # [ns, nv, nt]
    industry, expo = np.full(ns, -1), np.zeros((ns, 0))
    if a.side and a.side.exists():
        z = np.load(a.side)
        i = np.searchsorted(z["dsid"], sids)
        ok = (i < z["dsid"].size) & (z["dsid"][np.clip(i, 0, max(z["dsid"].size - 1, 0))] == sids)
        ic = np.clip(i, 0, max(z["dsid"].size - 1, 0))
        industry = np.where(ok, z["ind"][ic], -1) if z["ind"].size else industry
        expo = np.where(ok[:, None], z["expo"][ic], np.nan) if z["expo"].shape[1] else expo
        note.append(f"CED rows {int(ok.sum())} of {ns}, {expo.shape[1]} exposures, {int((industry >= 0).sum())} with an industry")
    cal = {}
    if a.cal and a.cal.exists():
        cal = {k: (np.array(v[0]), np.array(v[1])) for k, v in json.loads(a.cal.read_text()).items()}
    work = a.out / "work" / a.date
    work.mkdir(parents=True, exist_ok=True)
    IO.write_vec_inputs(work, sids, industry, expo, names, val, cal)
    t1 = time.perf_counter()
    IO.run_vec(work, a.formulas, a.threads)
    t2 = time.perf_counter()
    got = IO.read_vec_output(work)  # [n_col, nt, ns]
    out = pl.DataFrame({"sid": np.tile(sids, nt), "mark": np.repeat(T, ns), **{c: got[i].ravel() for i, (c, _) in enumerate(forms)}})
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"vec_{a.date}.parquet"
    out.write_parquet(dst)
    for f in ("vec_in.bin", "vec_out.bin"):
        (work / f).unlink()
    print(f"{a.date}: {ns} stocks x {nt} times, {len(names)} names ({'; '.join(note)}); names nobody supplied: {missing or 'none'}; "
          f"calibration points for {sum(c in cal for c, e in forms if e.startswith('cal:'))} of {sum(e.startswith('cal:') for _, e in forms)} cal columns")
    print(f"{a.date}: assemble {t1 - t0:.1f}s, featvec {t2 - t1:.1f}s; {out.height:,} rows x {len(forms)} columns -> {dst}")
    if a.x and a.x.exists():
        x = pl.read_parquet(a.x).filter(pl.col("date") == int(a.date)).with_columns(pl.col("sid").cast(pl.Int64))
        j = x.join(out, on=["sid", "mark"], how="inner", suffix="__cpp")
        print(f"{a.date}: {x.height:,} rows in x.parquet, {j.height:,} matched")
        bad, rows = 0, []
        for c, e in forms:
            if c not in x.columns:
                rows.append(f"  --  {c:<12} {e:<58} not in x.parquet (dropped by ic_score?)")
                continue
            r, g = j[c].to_numpy().astype(float), j[c + "__cpp"].to_numpy().astype(float)
            fr, fg = np.isfinite(r), np.isfinite(g)
            both = fr & fg
            d = np.abs(g[both] - r[both])
            tol = 1e-7 * np.maximum(np.abs(r[both]), 1e-6) + 1e-10
            nd, nn = int((d > tol).sum()), int((fr != fg).sum())
            corr = float(np.corrcoef(r[both], g[both])[0, 1]) if both.sum() > 2 and np.std(r[both]) > 0 and np.std(g[both]) > 0 else float("nan")
            off = nd > 0.002 * max(int(fr.sum()), 1) or nn > 0.002 * r.size
            bad += off
            if off or nd or nn:
                rows.append(f"  {'OFF' if off else 'ok '} {c:<12} {e:<58} finite {int(fr.sum()):>6} NaN mismatches {nn:>6} cells differing {nd:>6} "
                            f"max abs diff {float(d.max()) if d.size else 0:.3g} corr {corr:.6f}")
        print("\n".join(rows))
        print(f"{a.date}: {len(forms) - bad} of {len(forms)} columns agree with ic_score's matrix, {bad} off")


if __name__ == "__main__":
    main()
