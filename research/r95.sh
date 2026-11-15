#!/usr/bin/env bash
# Round 95 on 97, foreground. Read-only on inputs; writes data/side_r95, data/pm_export_all (samplerS cache) and data/reports.
# 0430 has no samplerS on 97 or 2.23 (0429 then 0901), but a samplerS.nc is written at 20:01 the evening of its day, so the
# file PM had on the morning of D is the latest one BEFORE D: 0430 <- 0429, 1112 <- 1111. R94's same-day cells for 1112 were
# look-ahead. Same scan as R94 with --prev cells, both days; then 1112 same-day again for comparison.
#   bash research/r95.sh 2>&1 | tee data/reports/r95.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in 20261112 20260430; do
  echo "[$(date +%T)] hstats cells (previous day's file) $D"
  python -u research/featgen/hstats_extra.py --date $D --prev --x data/reports/stage6_r70_${D:4}/x.parquet --hstats $DATA/hstats_view/HS300 --out data/side_r95 2>&1 | tail -1 | cut -c1-400
done
T=zzaa.b,zzhj,zzbt.b,zzxo,zzpp.b,zzop.i
echo "[$(date +%T)] implied scan, previous-day hstats cells, both days"
python -u research/featgen/implied_scan.py --days 20261112,20260430 --target $T \
  --x-dir "data/reports/stage6_r70_{mmdd}" --base "data/featgen_r67/base_{date}.parquet" --extra "data/side_r95/hstats_prev_{date}.parquet" \
  --only-extra --transforms "b,-b" --models $DATA/models_gen/B --top 8 --work data/implied_scan_r95 2>&1 | cut -c1-200
echo "[$(date +%T)] implied scan, 1112 previous-day cells alone (to set against R94's same-day result)"
python -u research/featgen/implied_scan.py --days 20261112 --target zzaa.b,zzhj \
  --x-dir "data/reports/stage6_r70_{mmdd}" --base "data/featgen_r67/base_{date}.parquet" --extra "data/side_r95/hstats_prev_{date}.parquet" \
  --only-extra --transforms "b,-b" --models $DATA/models_gen/B --top 5 --work data/implied_scan_r95 2>&1 | cut -c1-200
echo "[$(date +%T)] done"
