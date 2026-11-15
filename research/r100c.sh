#!/usr/bin/env bash
# Round 100c on 97, foreground, read-only on inputs; writes data/implied_scan_r100c. ~5 min.
# R100b: on 0430 (the trusted day) every zzeg fill loses to NaN, and lower constants lose least -- while all 64 other open
# columns sit at NaN too. That may be the context, not zzeg: here every other open column is first set to its R96 implied
# value (the context in which PM's trees reproduce PM), zzeg starts at NaN, and the same candidates (+ R96's implied zzeg
# itself as the reference) are put in. 0430 and 1112, ranked by 0430.
#   bash research/r100c.sh 2>&1 | tee data/reports/r100c.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
python -u research/featgen/implied_scan.py --days 20260430,20261112 --target zzeg --x-dir "data/reports/stage6_r70_{mmdd}" --base "data/featgen_r67/base_{date}.parquet" \
  --extra "data/side_r100/ratio_{date}.parquet,data/invert_r96_1112/implied_{date}.parquet" --only-extra --transforms "id,1/b" \
  --fill-from "data/invert_r96_1112/implied_{date}.parquet" --target-nan --rank-day 20260430 --models $DATA/models_gen/B --top 25 \
  --work data/implied_scan_r100c 2>&1 | grep -v RuntimeWarning | cut -c1-300
echo "[$(date +%T)] done"
