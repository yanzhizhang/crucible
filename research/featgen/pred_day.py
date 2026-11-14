r"""PM's LightGBM model on a feature parquet, through the C++ predictor (cpp/lgbpred.cpp): one score per (sid, mark).

The feature parquet has ``sid``, ``mark`` and one column per model feature it fills (``vec_day.py``'s output, or
``ic_score --dump-x``'s x.parquet); the model's other columns are NaN, as in ``ic_score``. The model files are picked as
``ic_score.load_models`` picks them: the latest month folder not after the day, ``<month>/lgb/<fit>/model_<sub>.txt``.

    python research/featgen/pred_day.py --date 20260430 --x $R/stage6_r70_0430/x.parquet --models $DATA/models_gen/B \
        --check $R/stage6_r70_0430/pred.parquet                      # the same matrix ic_score scored: the scores must be equal
    python research/featgen/pred_day.py --date 20261112 --x data/featvec_3s/vec_20261112.parquet --models $DATA/models_gen/B
    python research/featgen/pred_day.py --selftest [--stage make|check --work W]   # made-up trees against ic_score.TreeModel

With ``--pq <lgbpred_pq>`` (the CMake build inside the conda env, Arrow + Parquet) the C++ program reads the feature parquet
and writes the score parquet itself: no 2 GB matrix file in between. ``--selftest --pq ...`` checks that way against the
binary way on a made-up parquet.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pm_factors"))

CPP = HERE / "cpp"


def ensure_binary(build: Path | None = None) -> Path:
    build = build or CPP / "build"
    exe, src = build / "lgbpred", CPP / "lgbpred.cpp"
    if exe.exists() and exe.stat().st_mtime >= src.stat().st_mtime:
        return exe
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run([shutil.which("g++") or "g++", "-std=c++20", "-O2", "-DNDEBUG", "-ffp-contract=off", "-pthread", str(src), "-o", str(exe)], check=True)
    return exe


def write_x(dst: Path, X: np.ndarray) -> None:
    with dst.open("wb") as f:
        f.write(b"CRULX001")
        f.write(np.array(X.shape, "<u8").tobytes())
        f.write(np.ascontiguousarray(X, "<f8").tobytes())


def read_scores(src: Path) -> np.ndarray:
    b = src.read_bytes()
    assert b[:8] == b"CRULP001", "not an lgbpred output file"
    return np.frombuffer(b, "<f8", int(np.frombuffer(b, "<u8", 1, 8)[0]), 16)


def feature_names(csv: Path) -> list[str]:
    """The model's columns in order (ic_score's reading of lgb_fn_v1_209_cte.csv)."""
    lines = [ln.strip() for ln in csv.read_text(errors="replace").splitlines() if ln.strip()]
    names = [ln.split(",")[0].strip() for ln in lines]
    return names[1:] if names and names[0].lower() == "name" else names


def model_files(root: Path, date: str, fits: list[str], subs: list[str]) -> list[Path]:
    months = sorted(p.name for p in root.iterdir() if re.fullmatch(r"\d{6}", p.name) and p.name <= date[:6])
    if not months:
        raise SystemExit(f"no model month <= {date[:6]} under {root}")
    return [f for f in (root / months[-1] / "lgb" / ft / f"model_{s}.txt" for ft in fits for s in subs) if f.exists()]


def synth_model(dst: Path, rng: np.random.Generator, n_trees: int, n_cols: int) -> None:
    """A made-up model in LightGBM's text layout: every missing type, default directions, one-leaf trees."""
    out = ["tree", "version=v3", f"max_feature_idx={n_cols - 1}", ""]
    for t in range(n_trees):
        nl = int(rng.integers(1, 40))
        left, right, where = [], [], {0: None}  # leaf -> (node, side) it hangs on
        for k in range(nl - 1):  # LightGBM's numbering: splitting leaf l makes node k, its left child stays leaf l
            leaf = int(rng.choice(list(where)))
            at = where[leaf]
            if at is not None:
                (left if at[1] == 0 else right)[at[0]] = k
            left.append(~leaf)
            right.append(~(k + 1))
            where[leaf], where[k + 1] = (k, 0), (k, 1)
        out += [f"Tree={t}", f"num_leaves={nl}", "num_cat=0"]
        if nl > 1:
            dt = [(int(rng.integers(0, 3)) << 2) | (int(rng.integers(0, 2)) << 1) for _ in range(nl - 1)]
            thr = rng.normal(0, 1, nl - 1)
            thr[rng.random(nl - 1) < 0.1] = 0.0
            out += ["split_feature=" + " ".join(str(int(x)) for x in rng.integers(0, n_cols, nl - 1)),
                    "threshold=" + " ".join(repr(float(x)) for x in thr), "decision_type=" + " ".join(map(str, dt)),
                    "left_child=" + " ".join(map(str, left)), "right_child=" + " ".join(map(str, right))]
        out += ["leaf_value=" + " ".join(repr(float(x)) for x in rng.normal(0, 1e-3, nl)), "shrinkage=0.03", "", ""]
    dst.write_text("\n".join(out) + "\nend of trees\n\nfeature_importances:\n", newline="\n")


def selftest(a: argparse.Namespace) -> None:
    a.work.mkdir(parents=True, exist_ok=True)
    files = [a.work / "model_a.txt", a.work / "model_b.txt"]
    if a.stage in ("all", "make"):
        import ic_score as IC

        rng = np.random.default_rng(21)
        for f in files:
            synth_model(f, rng, 300, 60)
        X = rng.normal(0, 1, (20000, 60))
        X[rng.random(X.shape) < 0.15] = np.nan
        X[rng.random(X.shape) < 0.05] = 0.0
        write_x(a.work / "x.bin", X)
        np.save(a.work / "pred_ref.npy", np.mean([IC.TreeModel(f).predict(X) for f in files], axis=0))
        print(f"2 made-up models x 300 trees, {X.shape[0]:,} rows -> {a.work}")
    if a.stage == "all":
        subprocess.run([str(ensure_binary()), str(a.work / "x.bin"), str(a.work / "pred.bin"), *map(str, files)], check=True)
    if a.pq and a.stage in ("all", "check"):  # the parquet way on the same matrix: the scores must be the same doubles
        X = np.frombuffer((a.work / "x.bin").read_bytes(), "<f8", offset=24).reshape(-1, 60)
        keep = [c for c in range(60) if c % 7]  # some model columns left out: they must count as NaN
        (a.work / "names.csv").write_text("name\n" + "".join(f"f{c},x\n" for c in range(60)), newline="\n")
        pl.DataFrame({"sid": np.arange(X.shape[0], dtype=np.int64) % 300, "mark": np.arange(X.shape[0]) // 300 * 3.0,
                      **{f"f{c}": X[:, c] for c in keep}}).write_parquet(a.work / "x.parquet")
        Xk = X.copy()
        Xk[:, [c for c in range(60) if c % 7 == 0]] = np.nan
        write_x(a.work / "xk.bin", Xk)
        subprocess.run([str(ensure_binary()), str(a.work / "xk.bin"), str(a.work / "predk.bin"), *map(str, files)], check=True)
        subprocess.run([str(a.pq), "--parquet", str(a.work / "x.parquet"), str(a.work / "pred.parquet"), "--features", str(a.work / "names.csv"),
                        *map(str, files)], check=True)
        p, b = pl.read_parquet(a.work / "pred.parquet"), read_scores(a.work / "predk.bin")
        n = int((p["pred"].to_numpy() != b).sum()) + int((p["sid"].to_numpy() != np.arange(X.shape[0]) % 300).sum())
        print(f"parquet way: {p.height:,} rows, {n} differ from the binary way")
        if n:
            sys.exit(1)
    if a.stage in ("all", "check"):
        got, ref = read_scores(a.work / "pred.bin"), np.load(a.work / "pred_ref.npy")
        n = int((got != ref).sum())
        print(f"{ref.size:,} rows: {n} scores differ from ic_score.TreeModel (largest difference {float(np.abs(got - ref).max()):.3g})")
        sys.exit(1 if n else 0)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", choices=["all", "make", "check"], default="all")
    ap.add_argument("--work", type=Path, default=Path("data/lgbpred_work"))
    ap.add_argument("--date")
    ap.add_argument("--x", type=Path, help="feature parquet: sid, mark, one column per model feature filled")
    ap.add_argument("--models", type=Path, default=Path("/data/prod/models/tk5m"))
    ap.add_argument("--fits", default="livev1_fit1")
    ap.add_argument("--subs", default="s1")
    ap.add_argument("--feature-csv", type=Path, default=Path("/data/prod/models/lgb_fn_v1_209_cte.csv"))
    ap.add_argument("--out", type=Path, default=Path("data/featpred"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--check", type=Path, help="ic_score --dump-pred's pred.parquet: compare with its 'ours'")
    ap.add_argument("--pq", type=Path, help="the lgbpred_pq binary (CMake build with Arrow): parquet in and out, no matrix file")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a)
    t0 = time.perf_counter()
    files = model_files(a.models, a.date, a.fits.split(","), a.subs.split(","))
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"pred_{a.date}.parquet"
    if a.pq and "date" not in pl.read_parquet_schema(a.x):
        subprocess.run([str(a.pq), "--parquet", str(a.x), str(dst), "--features", str(a.feature_csv), *map(str, files), "--threads", str(a.threads)], check=True)
        out = pl.read_parquet(dst).with_columns(pl.col("sid").cast(pl.Int64))
        print(f"{a.date}: {out.height:,} rows, models {[str(f.relative_to(a.models)) for f in files]}; lgbpred_pq {time.perf_counter() - t0:.1f}s -> {dst}")
    else:
        names = feature_names(a.feature_csv)
        df = pl.read_parquet(a.x)
        if "date" in df.columns:
            df = df.filter(pl.col("date") == int(a.date)).drop("date")
        have = [c for c in df.columns if c in set(names)]
        X = np.full((df.height, len(names)), np.nan)
        ix = {n: i for i, n in enumerate(names)}
        for c in have:
            X[:, ix[c]] = df[c].to_numpy().astype(float)
        a.work.mkdir(parents=True, exist_ok=True)
        xb, pb = a.work / f"x_{a.date}.bin", a.work / f"pred_{a.date}.bin"
        write_x(xb, X)
        t1 = time.perf_counter()
        subprocess.run([str(ensure_binary()), str(xb), str(pb), *map(str, files), "--threads", str(a.threads)], check=True)
        t2 = time.perf_counter()
        out = df.select(pl.col("sid").cast(pl.Int64), "mark").with_columns(pl.Series("pred", read_scores(pb)))
        out.write_parquet(dst)
        xb.unlink()
        pb.unlink()
        print(f"{a.date}: {df.height:,} rows, {len(have)} of {len(names)} model columns filled, models {[str(f.relative_to(a.models)) for f in files]}; "
              f"assemble {t1 - t0:.1f}s, lgbpred {t2 - t1:.1f}s -> {dst}")
    if a.check and a.check.exists():
        r = pl.read_parquet(a.check).filter(pl.col("date") == int(a.date)).with_columns(pl.col("sid").cast(pl.Int64))
        j = out.join(r, on=["sid", "mark"], how="inner")
        g, o = j["pred"].to_numpy(), j["ours"].to_numpy()
        f = np.isfinite(g) & np.isfinite(o)
        d = np.abs(g[f] - o[f])
        corr = float(np.corrcoef(g[f], o[f])[0, 1]) if f.sum() > 2 else float("nan")
        print(f"{a.date}: against {a.check}: {j.height:,} rows matched; scores bit-identical {int((g == o).sum()):,}, differing {int((g != o).sum()):,}; "
              f"largest difference {float(d.max()) if d.size else 0:.3g} (scores span {float(np.ptp(o[f])) if f.any() else 0:.3g}), correlation {corr:.9f}")
        pm = j["pm"].to_numpy()
        ok = np.isfinite(pm) & np.isfinite(g)
        ic = [float(pl.DataFrame({"a": g[ok][m], "b": pm[ok][m]}).select(pl.corr("a", "b", method="spearman")).item())
              for mk in np.unique(j["mark"].to_numpy()[ok]) if (m := j["mark"].to_numpy()[ok] == mk).sum() >= 30]
        print(f"{a.date}: similarity of the C++ scores to PM's prediction (rank IC per mark, mean over {len(ic)} marks): {np.nanmean(ic):.4f}")


if __name__ == "__main__":
    main()
