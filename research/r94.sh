#!/usr/bin/env bash
# Round 94 on 97, foreground, after R93. Read-only; writes data/implied_scan_work and data/reports. About 10 minutes.
# R92 found no hstats cells for 0430 (hstats_view starts later), so the hstats scan of zzbt.b / zzaa.b / zzxo / zzpp.b runs on
# 1112 alone (152 cells, R92's parquet). One day is weak evidence (R87: gains on 1112 often turn into losses on 0430); a
# column whose best cell is far above every tick / daily candidate is still worth a second day.
#   bash research/r94.sh 2>&1 | tee data/reports/r94.txt
set -u
cd /work/crucible
echo "[$(date +%T)] implied scan, hstats cells, 1112 only"
python -u research/featgen/implied_scan.py --days 20261112 --target zzbt.b,zzaa.b,zzxo,zzpp.b,zzhj,zzop.i \
  --x-dir "data/reports/stage6_r70_{mmdd}" --base "data/featgen_r67/base_{date}.parquet" --extra "data/side_r92/hstats_{date}.parquet" \
  --only-extra --transforms "b,-b" --models /work/crucible_data/models_gen/B --top 6 2>&1 | cut -c1-200
echo "[$(date +%T)] done"
