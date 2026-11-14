#!/usr/bin/env bash
# Round 85 on 97, foreground, after R84 (both use every core). Read-only on R67/R70 outputs and the model; writes data/reports
# and data/implied_scan_work only. About 30-40 minutes.
# A fast implied scan through the C++ predictor for the open columns that R83 found move today's scores most:
# every C++ base quantity (5 transforms, calibrated onto PM's training CDF) in the column, kept when it raises the
# similarity on both 1112 and 0430.
#   bash research/r85.sh 2>&1 | tee data/reports/r85.txt
set -u
cd /work/crucible
echo "[$(date +%T)] implied scan"
python -u research/featgen/implied_scan.py --days 20261112,20260430 --target zzbt.b,zzaa.b,zzxo,zzpp.b,zzpe.e,zzop.i,zzhj \
  --x-dir "data/reports/stage6_r70_{mmdd}" --base "data/featgen_r67/base_{date}.parquet" --models /work/crucible_data/models_gen/B --top 8 2>&1 | cut -c1-200
echo "[$(date +%T)] done"
