r"""What is in PM's day-end state (``<dcp>/YYYYMMDD.jz``): every key by family and kind, which ones carry across days,
and how each family maps onto the 209 model columns. Read-only.

Kinds of key (by name): ``bin`` = ``<p>.c.<N>`` histogram / profile cells; ``ema`` = ``.sum`` / ``.wgt`` of an
exponential average; ``clock`` = ``.t`` (trading-clock seconds of the last update); ``list`` = ``.r.<k>`` / ``.<k>``
grids (zzxo.r.0..29 ...); ``scalar`` = the rest (zzpk.high, zzgh.nb ...). ``dcp_build.load_state`` drops the bins, which is
why other tools count ~1573 keys a stock where the file has ~7500.

Carry test over consecutive files D-1 -> D, per key: share of stocks with an identical value; Spearman corr across
stocks; for a family's bins the corr of each stock's normalised histogram with yesterday's (shape memory) and the
median ratio of its total mass today / yesterday (~1 and stable = steady state of a decaying count, or a daily reset to
a similar level; growing = accumulates). A key reset every morning still correlates with yesterday across stocks (big
stocks stay big), so corr alone is not carry: compare it with the family's other keys and with identical shares.

    python research/pm_factors/jz_keys.py --dcp /work/crucible_data/seed_probe/HS300 --from 20261103 --to 20261111 \
        --assign data/reports/r27_both.json --out data/reports/jz_keys
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import numpy as np
import polars as pl

BIN = re.compile(r"^(.*)\.c\.(\d+)$")
LIST = re.compile(r"^(.*?)\.(?:r\.)?(\d+)$")


def kind(k: str) -> str:
    if BIN.match(k):
        return "bin"
    if k.endswith((".sum", ".wgt")):
        return "ema"
    if k.endswith(".t"):
        return "clock"
    if LIST.match(k):
        return "list"
    return "scalar"


def load(p: Path) -> dict:
    with gzip.open(p) as f:
        return json.loads(f.read())


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 20:
        return float("nan")
    rx, ry = np.argsort(np.argsort(x[m])), np.argsort(np.argsort(y[m]))
    return float(np.corrcoef(rx, ry)[0, 1]) if rx.std() > 0 and ry.std() > 0 else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dcp", type=Path, required=True)
    ap.add_argument("--from", dest="start", default="20261103")
    ap.add_argument("--to", dest="end", default="20261111")
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    ap.add_argument("--assign", type=Path, default=Path("data/reports/r27_both.json"))
    ap.add_argument("--show", default="zzui,zzhj", help="families whose every key is printed")
    ap.add_argument("--out", type=Path, default=Path("data/reports/jz_keys"))
    a = ap.parse_args()
    days = sorted(p.stem for p in a.dcp.glob("*.jz") if re.fullmatch(r"\d{8}", p.stem) and a.start <= p.stem <= a.end)
    print(f"{len(days)} state files: {days}")
    st = [load(a.dcp / f"{d}.jz") for d in days]
    sids = sorted(set.intersection(*[set(x) for x in st]))
    keys = sorted({k for x in st for kv in x.values() for k in kv})
    per_stock = [len(kv) for kv in st[-1].values()]
    print(f"{len(sids)} stocks in every file; {len(keys)} distinct keys; keys a stock on {days[-1]}: median {int(np.median(per_stock))}, "
          f"max {max(per_stock)}")

    def col(x: dict, k: str) -> np.ndarray:
        return np.array([x[s].get(k, np.nan) if isinstance(x[s].get(k), (int, float)) else np.nan for s in sids], float)

    rows = []
    for k in keys:
        v = [col(x, k) for x in st]
        same = [float(np.mean((p == c) & np.isfinite(c))) for p, c in zip(v, v[1:], strict=False)]
        rho = [spearman(p, c) for p, c in zip(v, v[1:], strict=False)]
        last = v[-1][np.isfinite(v[-1])]
        rows.append({"key": k, "stem": k.split(".")[0], "kind": kind(k), "stocks": float(np.isfinite(v[-1]).mean()),
                     "p50": float(np.median(last)) if last.size else None, "zero": float(np.mean(last == 0)) if last.size else None,
                     "integer": bool(last.size and np.all(last == np.round(last))),
                     "same_prev": float(np.mean(same)) if same else None, "rho_prev": float(np.nanmean(rho)) if rho and not np.all(np.isnan(rho)) else None})
    kt = pl.DataFrame(rows, infer_schema_length=None)
    a.out.mkdir(parents=True, exist_ok=True)
    kt.write_csv(a.out / "keys.csv")

    # histograms: per family, shape memory and total-mass ratio
    bins: dict[str, list[str]] = {}
    for k in keys:
        m = BIN.match(k)
        if m:
            bins.setdefault(m.group(1), []).append(k)
    hrows = []
    for fam, ks in sorted(bins.items()):
        ks = sorted(ks, key=lambda x: int(BIN.match(x).group(2)))
        H = [np.stack([col(x, k) for k in ks], axis=1) for x in st]  # day -> stocks x bins
        mass = [np.nansum(h, axis=1) for h in H]
        shape, ratio = [], []
        for p, c, mp, mc in zip(H, H[1:], mass, mass[1:], strict=False):
            ok = (mp > 0) & (mc > 0)
            for i in np.flatnonzero(ok):
                a1, a2 = p[i] / mp[i], c[i] / mc[i]
                if np.nanstd(a1) > 0 and np.nanstd(a2) > 0:
                    shape.append(np.corrcoef(np.nan_to_num(a1), np.nan_to_num(a2))[0, 1])
            ratio.append(float(np.median(mc[ok] / mp[ok])) if ok.any() else np.nan)
        hrows.append({"family": fam, "bins": len(ks), "mass_p50_last": float(np.nanmedian(mass[-1])),
                      "mass_ratio_by_day": " ".join(f"{r:.3f}" for r in ratio), "shape_corr_prev": float(np.nanmedian(shape)) if shape else None,
                      "integer_mass": bool(np.all(np.nan_to_num(mass[-1]) == np.round(np.nan_to_num(mass[-1]))))})
    ht = pl.DataFrame(hrows, infer_schema_length=None)
    ht.write_csv(a.out / "histograms.csv")

    # family -> model columns
    fp = pl.read_parquet(a.fingerprint).sort("gain_share", descending=True).unique("feature", maintain_order=True) if a.fingerprint.exists() else None
    assign = json.loads(a.assign.read_text()) if a.assign.exists() else {}
    fam_cols: dict[str, list[tuple[str, float]]] = {}
    if fp is not None:
        for f, g in zip(fp["feature"].to_list(), fp["gain_share"].to_list(), strict=True):
            fam_cols.setdefault(f.split(".")[0], []).append((f, g))
    srows = []
    for (stem,), g in kt.group_by("stem", maintain_order=True):
        kc = {k_: int((g["kind"] == k_).sum()) for k_ in ("scalar", "ema", "clock", "list", "bin")}
        nb = g.filter(pl.col("kind") != "bin")
        cols = fam_cols.get(stem, [])
        srows.append({"stem": stem, "keys": g.height, **kc,
                      "same_prev_max": nb["same_prev"].max() if nb.height else None, "rho_prev_median": nb["rho_prev"].median() if nb.height else None,
                      "model_cols": len(cols), "gain": sum(x[1] for x in cols),
                      "assigned": sum(1 for x in cols if x[0] in assign), "cols": " ".join(c for c, _ in cols[:8])})
    stt = pl.DataFrame(srows, infer_schema_length=None).sort("gain", descending=True)
    stt.write_csv(a.out / "stems.csv")
    no_state = sorted({s for s in fam_cols if s not in set(kt["stem"].to_list())}, key=lambda s: -sum(x[1] for x in fam_cols[s]))

    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=230, fmt_str_lengths=70, float_precision=3):
        print("\n== 1 keys by family and kind (scalar / ema sum+wgt / clock .t / list grid / histogram bins), model columns and gain")
        print(stt)
        print(f"\nmodel families with NO state keys ({len(no_state)}): {' '.join(no_state)}")
        print("\n== 2 carry: non-bin keys identical to the previous day for at least 5 % of stocks (a value that survives the night)")
        print(kt.filter((pl.col("kind") != "bin") & (pl.col("same_prev") >= 0.05)).sort("same_prev", descending=True).head(60))
        print("\n== 3 histograms / profiles per family: total mass today / yesterday per day pair, shape corr with yesterday")
        print(ht)
        for fam in [x for x in a.show.split(",") if x]:
            print(f"\n== 4 every key of {fam}")
            print(kt.filter(pl.col("stem") == fam).sort("key"))
    print(f"\n-> {a.out}/keys.csv, histograms.csv, stems.csv")


if __name__ == "__main__":
    main()
