#!/usr/bin/env bash
# Round 102 on 97, foreground, read-only on inputs; writes data/side_r102, data/implied_scan_r102. ~10 min.
# zzeg as one value per day for every stock (a market-level daily input, e.g. a futures basis from CED's cfd): sweep the
# constant 0.95-1.05 on every day (open columns NaN, as R100b) and print each day's gain curve -- if zzeg is such an input,
# each day has its own best constant, to be compared with the candidate daily series.
#   bash research/r102.sh 2>&1 | tee data/reports/r102.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
DAYS=20260430,20261102,20261103,20261104,20261105,20261106,20261109,20261110,20261111,20261112,20261113
mkdir -p data/side_r102
python - <<'PY'
import numpy as np, polars as pl
cs = np.round(np.concatenate([np.arange(0.95, 0.98, 0.005), np.arange(0.98, 1.02, 0.001), np.arange(1.02, 1.0501, 0.005)]), 4)
for d in "20260430,20261102,20261103,20261104,20261105,20261106,20261109,20261110,20261111,20261112,20261113".split(","):
    k = pl.read_parquet(f"data/reports/stage6_r99_{d[4:]}/x.parquet", columns=["date", "sid", "mark"]).filter(pl.col("date") == int(d)).drop("date")
    k.with_columns([pl.lit(float(c)).alias(f"c{c:.4f}") for c in cs]).write_parquet(f"data/side_r102/const_{d}.parquet")
print(f"{cs.size} constants {cs.min()}..{cs.max()}")
PY
python -u research/featgen/implied_scan.py --days $DAYS --target zzeg --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r102/const_{date}.parquet" --only-extra --transforms id --rank-day 20260430 --models $DATA/models_gen/B --top 3 \
  --work data/implied_scan_r102 2>&1 | grep -v RuntimeWarning | cut -c1-300
python - <<'PY'
import polars as pl
g = pl.read_csv("data/implied_scan_r102/gains_zzeg.csv").with_columns(pl.col("cand").str.replace("^c", "").cast(pl.Float64).alias("c")).sort("c")
days = [c for c in g.columns if c.isdigit()]
print("gain vs NaN by constant (rows) and day (columns):")
print("  c       " + " ".join(f"{d[4:]:>7}" for d in days))
for r in g.iter_rows(named=True):
    print(f"  {r['c']:.4f} " + " ".join(f"{r[d]:+7.4f}" for d in days))
print("  best c  " + " ".join(f"{g.sort(d, descending=True)['c'][0]:7.3f}" for d in days))
PY
echo "[$(date +%T)] done"
