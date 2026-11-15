#!/usr/bin/env bash
# Round 100b on 97, foreground, read-only on inputs (reuses data/side_r100 from R100); writes data/implied_scan_r100b. ~5 min.
# R100's top 30 were all +0.0000: zzeg's NaN goes left of every threshold (missing type none -> 0), so any candidate below
# 0.985 equals the baseline and floats to the top, while the ratio candidates sat lower (some day negative). Rerun on the
# ratio columns only, listed by 0430 (the one trusted day: PM's production alpha); every gain goes to gains_zzeg.csv.
#   bash research/r100b.sh 2>&1 | tee data/reports/r100b.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
DAYS=20260430,20261102,20261103,20261104,20261105,20261106,20261109,20261110,20261111,20261112,20261113
python -u research/featgen/implied_scan.py --days $DAYS --target zzeg --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r100/ratio_{date}.parquet" --only-extra --transforms "id,1/b" --rank-day 20260430 --models $DATA/models_gen/B --top 40 \
  --work data/implied_scan_r100b 2>&1 | grep -v RuntimeWarning | cut -c1-400
echo "[$(date +%T)] 0430 alone (all 39 ratio columns of that day)"
python -u research/featgen/implied_scan.py --days 20260430 --target zzeg --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r100/ratio_{date}.parquet" --only-extra --transforms "id,1/b" --models $DATA/models_gen/B --top 40 \
  --work data/implied_scan_r100b0430 2>&1 | grep -v RuntimeWarning | cut -c1-200
echo "[$(date +%T)] done"
