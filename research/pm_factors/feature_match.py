r"""Stage 5c: guess each model feature from its family's day-end state by matching distributions.

PM shipped no values for most model features, but two things are known about each: the model
files' fingerprint (``pm_models_probe.py`` -> ``fingerprint.parquet``: training range and the split
thresholds' p10 / median / p90) and, for families that keep state, the decoded day-end state
(``h/HS300/p1_intrading_dcp/<D>.jz``: EMA values ``sum / wgt``, scalars, the 10 s price lists).

For every feature whose stem (the name before the first dot: ``zzup``, ``zzxu`` of ``zzxu.a``,
``zzab2``) has state keys, this builds candidate expressions from that stem's state -- each
quantity, every pairwise ratio / log ratio / difference, each relative to the last price -- and
ranks them by how well their cross-stock distribution (pooled over days) matches the feature's
threshold quantiles, plus the share of stocks inside the training range. The thresholds trace the
feature's distribution only roughly and the state is one moment (the close), so this ranks
hypotheses; it does not prove them.

    python research/pm_factors/feature_match.py --fingerprint data/reports/stage5/fingerprint.parquet
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # state loader, default state directory


def atoms(pm: pl.DataFrame, stem: str) -> pl.DataFrame:
    """Per stock, the stem's state quantities as (sid, name, v): EMA values, scalars, price-list statistics."""
    x = pm.filter(pl.col("key").str.starts_with(stem + "."))
    out = []
    tri = x.filter(pl.col("key").str.contains(r"\.(sum|wgt)$")).with_columns(
        pl.col("key").str.extract(r"^(.*)\.(sum|wgt)$", 1).alias("p"), pl.col("key").str.extract(r"\.(sum|wgt)$", 1).alias("f"))
    if tri.height:
        w = tri.pivot(on="f", index=["sid", "p"], values="pm")
        if {"sum", "wgt"} <= set(w.columns):
            out.append(w.filter(pl.col("wgt") != 0).select("sid", pl.col("p").alias("name"), (pl.col("sum") / pl.col("wgt")).alias("v")))
    lists = x.filter(pl.col("key").str.contains(r"\.r\.\d+$"))
    scal = x.filter(~pl.col("key").str.contains(r"\.(sum|wgt|t)$") & ~pl.col("key").str.contains(r"\.r\.\d+$"))
    out.append(scal.select("sid", pl.col("key").alias("name"), pl.col("pm").alias("v")))
    if lists.height:
        lp = lists.with_columns(pl.col("key").str.extract(r"^(.*\.r)\.\d+$", 1).alias("lst"),
                                pl.col("key").str.extract(r"\.(\d+)$", 1).cast(pl.Int32).alias("k")).sort("sid", "lst", "k")
        r = lp.with_columns(pl.col("pm").log().diff().over("sid", "lst").alias("lr"))
        st = r.group_by("sid", "lst").agg(
            pl.col("pm").last().alias("last"), pl.col("pm").mean().alias("mean"),
            (pl.col("lr") ** 2).sum().log().alias("lrv"),  # log realized variance of the list's steps
            pl.col("lr").std().log().alias("lsd"), (pl.col("pm").last() / pl.col("pm").first()).log().alias("ret"))
        for c in ("last", "mean", "lrv", "lsd", "ret"):
            out.append(st.select("sid", (pl.col("lst") + "." + c).alias("name"), pl.col(c).alias("v")))
    px = pm.filter(pl.col("key") == "zzxu.last").select("sid", pl.lit("px").alias("name"), pl.col("pm").alias("v"))
    return pl.concat([*out, px]).filter(pl.col("v").is_finite())


def candidates(a: pl.DataFrame) -> dict[str, np.ndarray]:
    """Expressions over one day's atoms (wide by stock): each atom, pairwise ratio / log ratio / difference, relative to price."""
    w = a.pivot(on="name", index="sid", values="v", aggregate_function="first").sort("sid")
    names = [c for c in w.columns if c != "sid"]
    col = {n: w[n].to_numpy().astype(float) for n in names}
    out = {n: v for n, v in col.items() if n != "px"}
    with np.errstate(divide="ignore", invalid="ignore"):
        for x, y in itertools.permutations(names, 2):
            out[f"{x} / {y}"] = col[x] / col[y]
            out[f"log({x} / {y})"] = np.log(col[x] / col[y])
            if x < y:
                out[f"{x} - {y}"] = col[x] - col[y]
    return out


def pool(pooled: dict[str, list[np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate each candidate's samples, finite values only, dropping candidates with under 50."""
    cand = {k: np.concatenate(v) for k, v in pooled.items()}
    cand = {k: v[np.isfinite(v)] for k, v in cand.items()}
    return {k: v for k, v in cand.items() if v.size >= 50}


def rank(features: pl.DataFrame, cand: dict[str, np.ndarray], top: int, levels: dict[str, float] | None = None) -> list[dict]:
    """For each fingerprint row, the ``top`` best candidates.

    Without ``levels``: p10 / median / p90 closest to the thresholds' (in threshold-span units). With
    ``levels`` (``fingerprint_calib.json``): where the thresholds fall in the candidate's own distribution
    against where they fall in true feature values -- immune to the trees' tail bias.
    """
    rows = []
    for f in features.iter_rows(named=True):
        if f["t_med"] is None:
            continue
        tq = np.array([f["t_p10"], f["t_med"], f["t_p90"]])
        span = max(tq[2] - tq[0], 1e-12)
        scored = []
        for k, v in cand.items():
            cq = np.quantile(v, [0.1, 0.5, 0.9])
            inr = float(np.mean((v >= f["lo"]) & (v <= f["hi"]))) if f["lo"] is not None else np.nan
            if levels:
                fq = np.array([np.mean(v <= x) for x in tq])
                sc = float(np.abs(fq - np.array([levels["p10"], levels["med"], levels["p90"]])).sum())
            else:
                sc = float(np.abs(cq - tq).sum() / span)
            scored.append((sc, k, cq, inr))
        scored.sort(key=lambda t: t[0])
        for r, (sc, k, cq, inr) in enumerate(scored[:top], 1):
            rows.append({"feature": f["feature"], "gain_share": f["gain_share"], "kind": f["kind"], "rank": r, "candidate": k,
                         "score": sc, "in_range": inr, "t_p10": tq[0], "t_med": tq[1], "t_p90": tq[2],
                         "c_p10": cq[0], "c_med": cq[1], "c_p90": cq[2]})
    return rows


def load_levels(path: Path | None) -> dict[str, float] | None:
    """Calibration levels from ``fingerprint_calib.json``, or None (with a warning when the file is missing)."""
    if path is None:
        return None
    if not path.exists():
        print(f"WARNING: {path} not found -- scoring on raw threshold quantiles")
        return None
    return json.loads(path.read_text())["levels"]


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--since", default="20260901", help="state files from this day on")
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--calib", type=Path, help="fingerprint_calib.json: score by calibrated threshold levels")
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage5"))
    args = ap.parse_args()
    fp = pl.read_parquet(args.fingerprint).with_columns(pl.col("feature").str.extract(r"^([^.]+)", 1).alias("stem"))
    files = sorted(f for f in args.dcp_dir.glob("*.jz") if f.stem >= args.since)
    states = [B.load_state(f) for f in files]
    print(f"{len(files)} state files from {args.since}")
    stems = sorted(s for s in fp["stem"].unique().to_list() if any(st["key"].str.starts_with(s + ".").any() for st in states[:1]))
    print(f"stems with state: {stems}")
    rows = []
    for stem in stems:
        pooled: dict[str, list[np.ndarray]] = {}
        for st in states:
            for k, v in candidates(atoms(st, stem)).items():
                pooled.setdefault(k, []).append(v)
        cand = pool(pooled)
        rows += rank(fp.filter(pl.col("stem") == stem), cand, args.top, load_levels(args.calib))
        print(f"done {stem}: {len(cand)} candidates")
    if not rows:
        return
    out = pl.DataFrame(rows).sort(["gain_share", "rank"], descending=[True, False])
    args.out.mkdir(parents=True, exist_ok=True)
    out.write_csv(args.out / "feature_match.csv")
    with pl.Config(tbl_rows=200, tbl_cols=14, tbl_width_chars=250, fmt_str_lengths=40, float_precision=5):
        print(f"\n== best {args.top} candidates per feature (score: |quantile gap| summed / threshold p10-p90 span, 0 = same;"
              " in_range: share inside the training range)")
        print(out.select("feature", "gain_share", "kind", "rank", "candidate", "score", "in_range",
                         "t_p10", "c_p10", "t_med", "c_med", "t_p90", "c_p90"))
    print(f"table: {args.out / 'feature_match.csv'}")


if __name__ == "__main__":
    main()
