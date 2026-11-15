#!/usr/bin/env bash
# Round 107 on 97, foreground, read-only on inputs; writes data/side_r107, data/implied_scan_r107, data/reports. ~60 min.
# The user: read what the open columns are from PM's .jz state. The .jz keys carry family names (zzba.p / zzba.r, zzmh.hm,
# zzab2.r, zzxo.last, zzgh.*, zzpk.*, zzxu.*, zzvx.*, zzbd.*, zzeb.p, zzqj.*, zzup.*), and the settled state columns are plain
# ratios of keys and px (zzup = zzup.pf / zzup.ps, zzxu.b = (px - zzxu.a) / zzxu.last ...). So: every such ratio
# (research/featgen/state_formulas.py, from our engine's marks, which match PM's .jz) put raw into the open columns.
#   bash research/r107.sh 2>&1 | tee data/reports/r107.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
python -u research/featgen/state_formulas.py --date 20260430 --engine data/engine_m7/marks_20260430.parquet --raw $DATA/feitu_raw --out data/side_r107 2>&1 | grep -vE "UserWarning|return grid" | cut -c1-600
for D in 20261112 20261113; do
  python -u research/featgen/state_formulas.py --date $D --engine data/engine/marks_$D.parquet --raw $DATA/dysl2_raw --out data/side_r107 2>&1 | grep -vE "UserWarning|return grid" | tail -1 | cut -c1-300
done
python -u research/featgen/implied_scan.py --days 20260430,20261112,20261113 \
  --target zzba,zzmh,zzab2,zzxo,zzgh.b,zzeg,zzel.a,zzel.b,zzop.i,zzzl2.c,zzpe.e,zzjx3.b,zzbt.b,zzaa.b,zzpp.b,zzop.i,zzhj \
  --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" --extra "data/side_r107/state_{date}.parquet" --only-extra \
  --transforms "id,-id" --rank-day 20260430 --models $DATA/models_gen/B --top 8 --work data/implied_scan_r107 2>&1 | grep -v RuntimeWarning | cut -c1-260
echo "[$(date +%T)] done"
