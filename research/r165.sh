#!/usr/bin/env bash
# Round 165 on 97, foreground, read-only (.jz files only), ~1 min: does PM's zzxu band carry overnight? For stocks in the same
# trend on consecutive days: is the active band unchanged / only tightened, and what exact value does it equal when it moved.
#   bash research/r165.sh 2>&1 | tee data/reports/r165.txt
set -u
cd /work/crucible
SEED=/work/crucible_data/seed_probe/HS300
DAYS=$(ls $SEED | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | tail -9 | paste -sd, -)
python -u research/pm_factors/jz_band_days.py --jz-dir $SEED --days "$DAYS" 2>&1 | cut -c1-320
echo "[$(date +%T)] r165 done"
