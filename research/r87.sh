#!/usr/bin/env bash
# Round 87 on 97, foreground, after R85 (same machinery, all cores). Read-only; writes data/reports and data/implied_scan_work.
# The implied scan on the heaviest families without a strong formula (R79's worklist: ZZRO 11% of the gain open, ZZGH 5%;
# ZZZL and ZZWA next): every column of the family, assigned or not, against every C++ base -- for an assigned column the gain is
# over its current baseline formula, so a positive line is a better formula. About 50 minutes.
#   bash research/r87.sh 2>&1 | tee data/reports/r87.txt
set -u
cd /work/crucible
echo "[$(date +%T)] implied scan, families"
python -u research/featgen/implied_scan.py --days 20261112,20260430 --target "zzro*,zzgh*" \
  --x-dir "data/reports/stage6_r70_{mmdd}" --base "data/featgen_r67/base_{date}.parquet" --models /work/crucible_data/models_gen/B --top 5 2>&1 | cut -c1-200
echo "[$(date +%T)] done"
