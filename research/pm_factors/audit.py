r"""How far can the current assignment be trusted? A scan keeps a formula for the IC it adds, which a correlated proxy also does.
This grades every assigned column by a second, target-free ruler -- PM's training CDF (root split points of the model itself):

  units  the formula as built sits on the CDF (raw gap <= --gap): right quantity, right units
  shape  only after an affine map (shape gap <= --gap): right shape, wrong units (cal: hides this)
  proxy  the column carries IC (t >= 2) but its shape is off: likely a stand-in for the true input
  idle   no significant IC without it

then the model gain behind each grade and behind the open columns, the open columns by gain with their fingerprint, and how
well our prediction finds PM's extremes (top / bottom decile overlap per mark) -- orders come from the tails, not the middle.

    python research/pm_factors/audit.py --run data/reports/stage6_r41_audit --out data/reports/r41_audit.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl


def tails(pred: Path, q: float) -> None:
    d = pl.read_parquet(pred).filter(pl.col("ours").is_finite() & pl.col("pm").is_finite())
    rows = []
    for _, g in d.group_by("date", "mark"):
        o, p = g["ours"].to_numpy(), g["pm"].to_numpy()
        n = o.size
        k = int(n * q)
        if k < 5:
            continue
        io, ip = np.argsort(o), np.argsort(p)
        top = len(set(io[-k:]) & set(ip[-k:])) / k
        bot = len(set(io[:k]) & set(ip[:k])) / k
        ro = np.empty(n)
        ro[io] = np.arange(n) / (n - 1)
        rows.append((top, bot, float(ro[ip[-k:]].mean()), float(ro[ip[:k]].mean()), float(np.mean(np.sign(o - np.median(o)) == np.sign(p - np.median(p))))))
    a = np.array(rows)
    print(f"\n== tails over {a.shape[0]} marks, decile = {q:.0%} (random: overlap {q:.2f}, our rank 0.50)")
    print(f"  PM's top decile:    {a[:, 0].mean():.3f} of it is in our top decile; our mean rank of its stocks {a[:, 2].mean():.3f}")
    print(f"  PM's bottom decile: {a[:, 1].mean():.3f} of it is in our bottom decile; our mean rank of its stocks {a[:, 3].mean():.3f}")
    print(f"  same side of the median: {a[:, 4].mean():.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="an ic_score --out dir: ic_evidence.parquet, ic_assign.json, pred.parquet")
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    ap.add_argument("--gap", type=float, default=0.05)
    ap.add_argument("--q", type=float, default=0.1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ev = pl.read_parquet(a.run / "ic_evidence.parquet")
    fp = pl.read_parquet(a.fingerprint).unique("feature", keep="first")
    gs = dict(zip(fp["feature"].to_list(), fp["gain_share"].to_list(), strict=True))
    tot = sum(v for v in gs.values() if v is not None)

    def grade(r: dict) -> str:
        t = r["t_loss"] if r["t_loss"] is not None and np.isfinite(r["t_loss"]) else 0.0
        fin = lambda x: x is not None and np.isfinite(x)  # noqa: E731
        if t < 2:
            return "idle"
        if fin(r["raw_gap"]) and r["raw_gap"] <= a.gap:
            return "units"
        if fin(r["shape_gap"]) and r["shape_gap"] <= a.gap:
            return "shape"
        return "proxy" if fin(r["shape_gap"]) or fin(r["raw_gap"]) else "nocdf"

    rows = [{**r, "tier": grade(r), "gain": (gs.get(r["feature"]) or 0.0) / tot} for r in ev.iter_rows(named=True)]
    t = pl.DataFrame(rows).select("feature", "formula", "source", "tier", "gain", "raw_gap", "shape_gap", "ic_loss", "t_loss")
    assigned = set(t["feature"].to_list())
    opn = fp.filter(~pl.col("feature").is_in(list(assigned))).with_columns((pl.col("gain_share") / tot).alias("gain")).sort("gain", descending=True)
    with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=230, fmt_str_lengths=60, float_precision=4):
        print(f"== {t.height} assigned columns by tier (gap <= {a.gap}; gain = share of the model's total split gain)")
        print(t.group_by("tier").agg(pl.len().alias("columns"), pl.col("gain").sum(), pl.col("ic_loss").sum().alias("sum_ic_loss")).sort("gain", descending=True))
        print(f"open: {opn.height} columns, gain {opn['gain'].sum():.4f}; assigned gain {t['gain'].sum():.4f}")
        print("  by source: " + ", ".join(f"{s} {n} cols gain {g:.3f}" for s, n, g in t.group_by("source").agg(pl.len(), pl.col("gain").sum()).sort("gain", descending=True).iter_rows()))
        for tier in ("proxy", "shape", "nocdf", "idle", "units"):
            x = t.filter(pl.col("tier") == tier).sort("gain", descending=True)
            if x.height:
                print(f"\n== tier {tier}: {x.height} columns")
                print(x.drop("tier"))
        keep = [c for c in ("feature", "gain", "kind", "lo", "hi", "t_p10", "t_med", "t_p90", "q_p10", "q_med", "q_p90", "nan_share", "n_splits") if c in opn.columns]
        print(f"\n== open columns by gain (fingerprint columns available: {[c for c in fp.columns if c not in keep][:30]})")
        print(opn.select(keep).head(80))
    if (a.run / "pred.parquet").exists():
        tails(a.run / "pred.parquet", a.q)
    out = {tier: t.filter(pl.col("tier") == tier).sort("gain", descending=True)["feature"].to_list() for tier in ("proxy", "shape", "nocdf", "idle", "units")}
    out["open"] = opn["feature"].to_list()
    a.out.write_text(json.dumps(out, indent=1))
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
