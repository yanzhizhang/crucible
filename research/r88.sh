#!/usr/bin/env bash
# Round 88 on 97, foreground, after R87 (all cores). Writes data/backfill_seedmap and data/reports only. About 10 minutes.
# Why the DataYes days (1104-1113) match PM's day-end state worse than the feitu days in R84 (13-26 keys fully equal on
# SZSE against 40-54): on those days 224's PM did not start from yesterday's state but from a stale sod link (2.23
# link times, user 2026-11-12). Same days again with that seed map, every carried quantity seeded from what PM read;
# if the explanation holds, the cross-day keys (zzup.ps, zzba.p, zzxu.a, zzbd.pEmaT ...) come back to 1.0.
# Also 0429 again (R84: no CED rows that day, the formula layer now runs without industry / exposures).
#   bash research/r88.sh 2>&1 | tee data/reports/r88.txt
set -u
cd /work/crucible
B=/work/crucible/research/featgen/cpp/build/cmake
MAP="20261104=20260914,20261105=20261030,20261106=20261102,20261109=20261104,20261110=20261104,20261111=20261110,20261112=20261110"
echo "[$(date +%T)] DataYes days with the seed map"
python -u research/featgen/backfill_days.py --dates 20261104,20261105,20261106,20261109,20261110,20261111,20261112 --seed-map $MAP --grid 3 --keep pred \
  --pq $B/lgbpred_pq --out data/backfill_seedmap 2>&1 | cut -c1-260
echo "[$(date +%T)] 0429 again"
python -u research/featgen/backfill_days.py --dates 20260429 --grid 3 --keep pred --pq $B/lgbpred_pq --out data/backfill_3s 2>&1 | cut -c1-260
echo "[$(date +%T)] state engine with the seed map, against R84 on the same days"
python -u research/featgen/verify_days.py data/backfill_seedmap 2>&1 | cut -c1-200 | head -40
echo "[$(date +%T)] done"
