#!/usr/bin/env bash
# Round 96 on 97, foreground. Read-only on inputs; writes data/invert_r96* and data/reports. Perhaps 2-3 hours.
# Walk our inputs toward PM's (research/featgen/invert_cols.py): stage 1 = per-row implied values of all open columns
# (upper bound: how close PM's own trees get when only the open columns move), stage 2 = per column, a regressor of our
# bases learnt on one day fills it on the other, stage 3 = the whole walk with regressors, fitted on the first day,
# judged on the second. Run twice so each day is held out once.
#   bash research/r96.sh 2>&1 | tee data/reports/r96.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for DS in 20261112,20260430 20260430,20261112; do
  echo "[$(date +%T)] invert_cols, days $DS (stage 3 fits on the first)"
  python -u research/featgen/invert_cols.py --days $DS --x-dir "data/reports/stage6_r70_{mmdd}" --base "data/featgen_r67/base_{date}.parquet" \
    --models $DATA/models_gen/B --out data/invert_r96_${DS:4:4} 2>&1 | cut -c1-220
done
echo "[$(date +%T)] done"
