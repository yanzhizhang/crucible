#!/usr/bin/env bash
# Round 105 on 97, foreground, read-only on inputs; writes data/side_r105, data/implied_scan_r105, data/reports. ~45 min.
# R104 shapes of the next 9 open columns: most have no (or < 8) root-CDF points, so earlier scans (calibrated candidates
# only) skipped them; and their ranges read like plain units -- zzel.a ~ tick / price or spread / mid, zzmh ~ 1 - last / high,
# zzba ~ (high - low) / pre, zzop.i ~ seconds (<= 12562) or lots, zzzl2.c / zzjx3.b / zzab2 ~ returns. So: raw values (id,
# -id, 1/b) of our tick bases plus new stock quantities (ratio_extra.py), on 0430 (trusted) + 1112 / 1113 (no restarts).
#   bash research/r105.sh 2>&1 | tee data/reports/r105.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in 20260430 20261112 20261113; do
  if [ $D = 20260430 ]; then RAW="--raw $DATA/feitu_raw --index-raw $DATA/pmmp_raw"; else RAW="--raw $DATA/dysl2_raw"; fi
  python -u research/featgen/ratio_extra.py --date $D --x data/reports/stage6_r99_${D:4}/x.parquet $RAW --out data/side_r105 2>&1 | grep -viE "warning|return grid" | tail -1 | cut -c1-200
done
python -u research/featgen/implied_scan.py --days 20260430,20261112,20261113 --target zzba,zzel.a,zzel.b,zzop.i,zzzl2.c,zzmh,zzpe.e,zzjx3.b,zzab2 \
  --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" --extra "data/side_r105/ratio_{date}.parquet" \
  --transforms "id,-id,1/b" --rank-day 20260430 --models $DATA/models_gen/B --top 10 --work data/implied_scan_r105 2>&1 | grep -v RuntimeWarning | cut -c1-260
echo "[$(date +%T)] done"
