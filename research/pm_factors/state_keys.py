r"""PM's state-file keys for the model families we have not matched yet: what each family keeps, as a start for exact matching.

The families we solved were solved state first: each ``.jz`` key (an EMA's sum / weight, a clock, a running count) matched
exactly from the ticks, then the column built from those keys. This lists, for every stem with an unassigned model column,
its keys in a day's state file with the value distribution that hints at the quantity: share of stocks that have it,
10 / 50 / 90 % points, share of zeros, whether every value is an integer (a count / a clock) and the number of distinct
values. Columns of the stem and whether each is assigned are printed above its keys.

With ``--hstats`` it also sorts every unassigned column into one group by where its family's inputs can live: ``state``
(keys in the .jz, maybe hstats too), ``hstats`` (an hstats family, no state keys) or ``neither`` (no state, no hstats:
stateless from ticks, or a static input outside both -- stsfile, CED, neutralisation); per group the columns and their
gain, written to ``<out stem>_groups.csv``.

    python research/pm_factors/state_keys.py --state /data/prod/h/HS300/p1_intrading_dcp/20261106.jz --assign data/reports/stage6_r17/ic_assign.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # noqa: E402 -- state file reader


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--state", type=Path, required=True)
    ap.add_argument("--assign", type=Path, required=True)
    ap.add_argument("--fingerprint", type=Path, default=Path("data/reports/stage5/fingerprint.parquet"))
    ap.add_argument("--out", type=Path, default=Path("data/reports/state_keys.csv"))
    ap.add_argument("--hstats", type=Path, help="hstats/<universe> dir; its family directories split the open columns into groups")
    ap.add_argument("--max-keys", type=int, default=40, help="keys printed per stem (all are in --out)")
    a = ap.parse_args()
    assign = json.loads(a.assign.read_text())
    fp = pl.read_parquet(a.fingerprint).sort("gain_share", descending=True).unique("feature", maintain_order=True)
    feats = fp["feature"].to_list()
    gain = dict(zip(feats, fp["gain_share"].to_list(), strict=True))
    stems: dict[str, list[str]] = {}
    for f in feats:
        stems.setdefault(f.split(".")[0], []).append(f)
    open_stems = {s: fs for s, fs in stems.items() if any(f not in assign for f in fs)}
    st = B.load_state(a.state)
    n_sid = st["sid"].n_unique()
    st = st.with_columns(pl.col("key").str.extract(r"^([^.]+)", 1).alias("stem"))
    rows = []
    for stem, g in st.filter(pl.col("stem").is_in(list(open_stems))).group_by("stem", "key"):
        v = g["pm"].to_numpy()
        v = v[np.isfinite(v)]
        if not v.size:
            continue
        q = np.quantile(v, [0.1, 0.5, 0.9])
        rows.append({"stem": stem[0], "key": stem[1], "stocks": g["sid"].n_unique() / n_sid, "p10": q[0], "p50": q[1], "p90": q[2],
                     "zero": float(np.mean(v == 0)), "integer": bool(np.all(v == np.round(v))), "distinct": int(np.unique(v).size)})
    df = pl.DataFrame(rows).sort("stem", "key") if rows else pl.DataFrame()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        df.write_csv(a.out)
    have = set(df["stem"].unique().to_list()) if rows else set()
    print(f"{a.state.name}: {n_sid} stocks; {len(open_stems)} stems with an unassigned column, {len(have)} of them with state keys -> {a.out}")
    order = sorted(open_stems, key=lambda s: -sum(gain.get(f, 0) for f in open_stems[s] if f not in assign))
    for stem in order:
        fs = open_stems[stem]
        open_gain = sum(gain.get(f, 0) for f in fs if f not in assign)
        cols = ", ".join(f"{f}{'' if f not in assign else ' = ' + assign[f]}" for f in fs)
        if stem not in have:
            print(f"\n{stem} (open gain {open_gain:.4f}): no state keys -- a stateless family (ticks / daily inputs only). Columns: {cols}")
            continue
        g = df.filter(pl.col("stem") == stem)
        print(f"\n{stem} (open gain {open_gain:.4f}, {g.height} keys). Columns: {cols}")
        for r in g.head(a.max_keys).iter_rows(named=True):
            print(f"    {r['key']:<28} stocks {r['stocks']:.2f}  p10/50/90 {r['p10']:.6g} / {r['p50']:.6g} / {r['p90']:.6g}  "
                  f"zero {r['zero']:.2f}{'  integer' if r['integer'] else ''}  distinct {r['distinct']}")

    if a.hstats:
        import re

        hs = {d.name.lower() for d in a.hstats.iterdir() if d.is_dir()}
        base = lambda x: re.sub(r"\d+$", "", x)  # noqa: E731 -- zzro2 -> zzro: a variant family may live on its base's state
        all_st = set(st["stem"].unique().to_list())
        jz_base = {base(x) for x in all_st}
        hs_base = {base(x) for x in hs}

        def where(s: str) -> str:
            if s in all_st:
                return "state"
            if base(s) in jz_base:
                return "state_base"
            if s.lower() in hs:
                return "hstats"
            return "hstats_base" if base(s.lower()) in hs_base else "neither"

        grp = pl.DataFrame([{"feature": f, "stem": s, "gain": gain.get(f, 0.0), "group": where(s)}
                            for s, fs in open_stems.items() for f in fs if f not in assign],
                           schema={"feature": pl.String, "stem": pl.String, "gain": pl.Float64, "group": pl.String})
        grp = grp.sort("group", "gain", descending=[False, True])
        grp.write_csv(a.out.with_name(a.out.stem + "_groups.csv"))
        print(f"\nopen columns by where their inputs can live ({len(hs)} hstats families):")
        for (name,), g in grp.group_by("group", maintain_order=True):
            print(f"  {name:<8} {g.height:4d} columns, gain {g['gain'].sum():.4f}, families {g['stem'].n_unique()}: "
                  f"{', '.join(g['stem'].unique(maintain_order=True).to_list()[:30])}")


if __name__ == "__main__":
    main()
