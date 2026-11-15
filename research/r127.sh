#!/usr/bin/env bash
# Round 127 on 97, foreground, read-only on inputs; writes data/full_0430/extra_20260430.parquet, data/reports. ~10 min.
# R126: on 0430's full 3 s grid the residual (PM - ours, open columns NaN) concentrates in the first 30 minutes (corr 0.41)
# and near the day's high (PM above ours by +0.34 z at drawdown < 0.2%). The leads, one fill at a time, scored on the 1.4M
# rows (20x the 240 marks) and inside those slices (research/featgen/single_fill.py).
#   bash research/r127.sh 2>&1 | tee data/reports/r127.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
python - <<'PY'
import polars as pl
b = pl.read_parquet("data/full_0430/base_20260430.parquet")
print("base engine keys:", [c for c in b.columns if c.startswith(("zzmh", "zzeg"))])
r = "zzeg.c.rEma|c0.sum"
ex = b.select("sid", "mark", (pl.col("zzmh.r.sum").sqrt() / 2).alias("mh.dd"), pl.col("zzmh.r.sum").sqrt().alias("mh.2dd"),
              (pl.col("zzeg.c.span") / pl.col(r)).alias("eg.span/rema"), (pl.col(r) / pl.col("zzeg.c.span")).alias("eg.rema/span"))
ex.write_parquet("data/full_0430/extra_20260430.parquet")
print(ex.describe())
PY
FILLS="zzmh:id:x.mh.dd,zzmh:id:x.mh.2dd,zzmh:4b2:x.mh.dd,zzeg:id:x.eg.span/rema,zzeg:id:x.eg.rema/span,zzjx3.b:-dm:ret90,zzab2:id:ret120,zzpe.e:-srank:lpe50,zzpe.e:-srank:rete30,zzop.i:abs4:ret300,zzel.a:-dm:microprice_minus_mid_rel"
python -u research/featgen/single_fill.py --day 20260430 --x-dir data/full_0430 --base "data/full_0430/base_{date}.parquet" --extra "data/full_0430/extra_{date}.parquet" \
  --fills "$FILLS" --models $DATA/models_gen/B --work data/single_fill_r127 2>&1 | grep -v RuntimeWarning | cut -c1-250
echo "[$(date +%T)] done"
