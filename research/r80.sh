#!/usr/bin/env bash
# Round 80 on 97, foreground. Writes data/backfill_r80 and data/reports only (reads the catalogs, PM's state files, models).
# The backfill driver on three days, every step in C++ but the drivers, each day's engine state checked against PM's .jz:
# 20261110 / 20261111 (DataYes) and 20261009 (feitu) at the 3-second grid; only the scores are kept (vectors ~1 GB a day).
# Needs R79's side_day.py to have passed (the side tables come from it, not from ic_score).
#   bash research/r80.sh 2>&1 | tee data/reports/r80.txt
set -u
cd /work/crucible
B=/work/crucible/research/featgen/cpp/build/cmake
echo "[$(date +%T)] backfill"
python -u research/featgen/backfill_days.py --dates 20261009,20261110,20261111 --grid 3 --keep pred --pq $B/lgbpred_pq --out data/backfill_r80 2>&1 | cut -c1-300
echo "[$(date +%T)] what is kept"
du -sh data/backfill_r80 data/backfill_r80/days/* 2>/dev/null | sed 's/^/  /'
echo "[$(date +%T)] done"
