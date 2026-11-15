#!/usr/bin/env bash
# Round 114 on 97, foreground, read-only on inputs; writes data/joint_fill_r114 (scratch) and data/reports. ~15-30 min.
# Units fingerprint picks per open column (R108 raw / dm / srank / |x|*1e4, R113 logs, R111b zzeg = span / rEma at 0.003)
# filled ALL AT ONCE and scored on 0430 (trusted) + 1112 / 1113, then leave-one-out and a cumulative curve
# (research/featgen/joint_fill.py) -- single-column scans are too weak, the joint fill is the test that can move a day.
#   bash research/r114.sh 2>&1 | tee data/reports/r114.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for S in 0.01 0.03; do
  echo "[$(date +%T)] max fingerprint score $S"
  python -u research/featgen/joint_fill.py --days 20260430,20261112,20261113 --reports data/reports/r108.txt,data/reports/r113.txt,data/reports/r111b.txt \
    --max-score $S --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
    --extra "data/side_r105/ratio_{date}.parquet,data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet,data/side_r111/zzrema_{date}.parquet" \
    --models $DATA/models_gen/B --work data/joint_fill_r114 2>&1 | grep -v RuntimeWarning | cut -c1-220
done
echo "[$(date +%T)] done"
