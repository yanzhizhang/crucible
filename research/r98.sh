#!/usr/bin/env bash
# Round 98 on 97, foreground, read-only on inputs; writes data/implied_scan_r98 and data/reports. About 20-40 minutes.
# zzeg = the open column with the most split gain (2.6%), never scanned before. 1: what PM's trees say about it (range,
# thresholds, depth, root CDF, neighbours) and whether R96's implied zzeg is a per-stock constant. 2: an implied scan of
# zzeg over every candidate source we have at once (tick bases, CED / Barra / stsfile daily, engine keys, start-of-day
# hstats), calibrated onto the root CDF -- or onto R96's implied values when the trees give too few root points.
#   bash research/r98.sh 2>&1 | tee data/reports/r98.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
IMP="data/invert_r96_1112/implied_{date}.parquet"
for D in 20261112 20260430; do
  echo "[$(date +%T)] 1 col_probe zzeg, generation of $D"
  python -u research/featgen/col_probe.py --col zzeg --date $D --models $DATA/models_gen/B --implied "$IMP" --days 20261112,20260430 2>&1 | cut -c1-400
done
echo "[$(date +%T)] 2 implied scan of zzeg, all sources"
python -u research/featgen/implied_scan.py --days 20261112,20260430 --target zzeg \
  --x-dir "data/reports/stage6_r70_{mmdd}" --base "data/featgen_r67/base_{date}.parquet" \
  --extra "data/side_r90/daily_{date}.parquet,data/side_r90/engine_{date}.parquet,data/side_r95/hstats_prev_{date}.parquet" \
  --cal-implied "$IMP" --models $DATA/models_gen/B --top 25 --work data/implied_scan_r98 2>&1 | cut -c1-220
echo "[$(date +%T)] done"
