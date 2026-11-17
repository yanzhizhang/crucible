#!/usr/bin/env bash
# Round 159 on 97, foreground, read-only (.jz files only), ~1 min: (1) is zzvx.trEma the true range against the previous close
# and rEma the average of true range / day range? (2) what band are zzxu.upper / lower / dir (centre and width in units of the
# range average), and which band ratio sits where zzxu.a / zzxu.b's thresholds are.
#   bash research/r159.sh 2>&1 | tee data/reports/r159.txt
set -u
cd /work/crucible
SEED=/work/crucible_data/seed_probe/HS300
DAYS=$(ls $SEED | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | tail -9 | paste -sd, -)
python -u research/pm_factors/jz_bands.py --jz-dir $SEED --days "$DAYS" 2>&1 | cut -c1-260
echo "[$(date +%T)] r159 done"
