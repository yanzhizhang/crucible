#!/usr/bin/env bash
# Round 115 on 97, foreground, read-only on inputs; writes data/side_r115, data/implied_scan_r115, data/reports. ~5-10 min.
# zzhj.p (PM .jz) looks like an average of quarter-hour volume sampled at 30 s before each quarter end (t ~14190, wgt
# ~14.9, value ~ a day's volume / 16). Check that against the .jz on 1106 / 1112 / 1113, then test "this quarter's
# volume vs the average quarter" as the zzhj column (range -0.97..1.59, 93% of training rows <= 0.390).
#   bash research/r115.sh 2>&1 | tee data/reports/r115.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in 20261106 20261112 20261113; do
  if [ $D = 20261106 ]; then JZ=/data/prod/h/HS300/p1_intrading_dcp/$D.jz; else JZ=$DATA/seed_probe/HS300/$D.jz; fi
  echo "[$(date +%T)] $D"
  python -u research/featgen/zzhj_quarter.py --date $D --x data/reports/stage6_r99_${D:4}/x.parquet --raw $DATA/dysl2_raw --jz $JZ --out data/side_r115 2>&1 | cut -c1-400
done
python -u research/featgen/zzhj_quarter.py --date 20260430 --x data/reports/stage6_r99_0430/x.parquet --raw $DATA/feitu_raw --out data/side_r115 2>&1 | tail -1
python -u research/featgen/fingerprint.py --days 20260430,20261112,20261113 --target zzhj --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r115/zzhj_{date}.parquet" --transforms "id,-id,1/b" --models $DATA/models_gen/B --top 10 2>&1 | grep -v RuntimeWarning | cut -c1-200
python -u research/featgen/implied_scan.py --days 20260430,20261112,20261113 --target zzhj --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r115/zzhj_{date}.parquet" --only-extra --transforms "id,-id" --rank-day 20260430 --models $DATA/models_gen/B --top 10 \
  --work data/implied_scan_r115 2>&1 | grep -v RuntimeWarning | cut -c1-260
echo "[$(date +%T)] done"
