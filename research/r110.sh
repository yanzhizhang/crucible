#!/usr/bin/env bash
# Round 110 on 97, foreground, read-only on inputs; writes data/side_r110, data/implied_scan_r110, data/reports. ~5 min.
# R109: PM's zzeg.c is the 4-day channel of snapshot highs / lows (N = 4 matches the .jz on ~100% of stocks every day),
# span = (high + low) / 2. So zzeg (the open column with the most gain, thresholds 0.985-1.030) is now testable from its own
# state: px / span and the like (research/featgen/zzeg_channel.py) -- units fingerprint first, then the similarity scan.
#   bash research/r110.sh 2>&1 | tee data/reports/r110.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in 20260430 20261112 20261113; do
  if [ $D = 20260430 ]; then RAW=$DATA/feitu_raw; else RAW=$DATA/dysl2_raw; fi
  python -u research/featgen/zzeg_channel.py --date $D --x data/reports/stage6_r99_${D:4}/x.parquet --raw $RAW --out data/side_r110 2>&1 | grep -vE "UserWarning|return grid"
done
python -u research/featgen/fingerprint.py --days 20260430,20261112,20261113 --target zzeg,zzeg2 --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r110/zzeg_{date}.parquet" --transforms "id,-id,1/b" --models $DATA/models_gen/B --top 10 2>&1 | grep -v RuntimeWarning | cut -c1-200
python -u research/featgen/implied_scan.py --days 20260430,20261112,20261113 --target zzeg --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r110/zzeg_{date}.parquet" --only-extra --transforms "id,-id,1/b" --rank-day 20260430 --models $DATA/models_gen/B --top 12 \
  --work data/implied_scan_r110 2>&1 | grep -v RuntimeWarning | cut -c1-260
echo "[$(date +%T)] done"
