#!/usr/bin/env bash
# Round 92 on 97, foreground, after R91. Read-only on inputs; writes data/side_r92, data/pm_export_all (cache of exported
# samplerS cells, as ic_score already does) and data/reports. Perhaps 30-60 minutes, depending on the number of cells.
# R85 / R90: for the open columns that move today's scores most (zzbt.b, zzaa.b, zzxo, zzpp.b) neither the tick bases nor the
# CED / stsfile daily values nor the engine's keys beat NaN. The one source left untried for them is PM's own cross-day
# intraday profiles (hstats samplerS cells): every family's cells at the scoring rows as candidates (b and -b).
#   bash research/r92.sh 2>&1 | tee data/reports/r92.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in 20261112 20260430; do
  echo "[$(date +%T)] hstats cells $D"
  python -u research/featgen/hstats_extra.py --date $D --x data/reports/stage6_r70_${D:4}/x.parquet --hstats $DATA/hstats_view/HS300 --out data/side_r92 2>&1 | tail -1 | cut -c1-300
done
echo "[$(date +%T)] implied scan, hstats cells"
python -u research/featgen/implied_scan.py --days 20261112,20260430 --target zzbt.b,zzaa.b,zzxo,zzpp.b \
  --x-dir "data/reports/stage6_r70_{mmdd}" --base "data/featgen_r67/base_{date}.parquet" --extra "data/side_r92/hstats_{date}.parquet" \
  --only-extra --transforms "b,-b" --models $DATA/models_gen/B --top 8 2>&1 | cut -c1-200
echo "[$(date +%T)] done"
