#!/usr/bin/env bash
# Round 131 on 97, foreground, read-only, writes only data/reports. ~5-10 min, no model scoring; parallel-safe.
# Source-first search, step 1 (the audit's "known source but stand-in" columns): the profile families zzwa / zzqo / zzal /
# zzxc (settled AND open columns) against PM's own start-of-day hstats cells at the marks (side_r95, the D-1 file PM loads)
# -- units fingerprint only: does any cell (as is, negated, inverted, log) sit on a column's thresholds / root CDF points?
#   bash research/r131.sh 2>&1 | tee data/reports/r131.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
T=zzwa.a,zzwa.b,zzwa.c,zzwa.d,zzwa.e,zzwa.f,zzwa.g,zzwa.h,zzwa.i,zzwa.j,zzwa.k,zzqo.a,zzqo.b,zzqo.c,zzqo.d,zzqo.e,zzqo.f,zzal.a,zzal.b,zzal.c,zzal.d,zzal.e,zzal.f,zzal.g,zzal.h,zzxc.a,zzxc.b,zzxc.c,zzxc.d
ls -l data/side_r95/hstats_prev_20260430.parquet data/side_r95/hstats_prev_20261112.parquet 2>&1 | sed 's/^/  /'
python -u research/featgen/fingerprint.py --days 20260430,20261112 --target $T --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r95/hstats_prev_{date}.parquet" --transforms=id,-id,1/b,log,abs4 --models $DATA/models_gen/B --top 5 2>&1 | grep -v RuntimeWarning | cut -c1-200
echo "[$(date +%T)] done"
