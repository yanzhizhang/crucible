#!/usr/bin/env bash
# Round 105b on 97, foreground, after R105; read-only on inputs, writes data/implied_scan_r105b and two links in
# data/side_r105. ~60-90 min. Same 9 open columns and days as R105, now with the formula prefixes their settled siblings
# use (baseline_r41c: zzjx3.a dm:rpe5400, zzpe.c/d srank:..., zzop.a/j |x|*1e4) as raw transforms -- dm, -dm, srank,
# -srank, abs4 -- and with PM's engine state keys (zzmh.hm, zzba.p/r, zzab2.r ...) as candidates too.
#   bash research/r105b.sh 2>&1 | tee data/reports/r105b.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
ln -sfn /work/crucible/data/engine_m7/marks_20260430.parquet data/side_r105/engine_20260430.parquet
for D in 20261112 20261113; do ln -sfn /work/crucible/data/engine/marks_$D.parquet data/side_r105/engine_$D.parquet; done
ls -l data/side_r105/ | sed 's/^/  /'
python -u research/featgen/implied_scan.py --days 20260430,20261112,20261113 --target zzba,zzel.a,zzel.b,zzop.i,zzzl2.c,zzmh,zzpe.e,zzjx3.b,zzab2 \
  --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r105/ratio_{date}.parquet,data/side_r105/engine_{date}.parquet" \
  --transforms "dm,-dm,srank,-srank,abs4" --rank-day 20260430 --models $DATA/models_gen/B --top 10 --work data/implied_scan_r105b 2>&1 | grep -v RuntimeWarning | cut -c1-260
echo "[$(date +%T)] done"
