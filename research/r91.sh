#!/usr/bin/env bash
# Round 91 on 97, foreground, after R90. Writes data/backfill_r91_* and data/reports only. About 20 minutes.
# R88's seed map changed nothing because our engine starts every EMA from zero each morning; only the |c<N> variants read
# yesterday's state. PM's live process seeds every one. So the DataYes days again with --seed-all c0 (every EMA carried in
# full from the seed), A = the true previous day, B = the stale sod links 224 read; and 1104-1106 from feitu as well, to
# separate the feed from the seed.
#   bash research/r91.sh 2>&1 | tee data/reports/r91.txt
set -u
cd /work/crucible
B=/work/crucible/research/featgen/cpp/build/cmake
DAYS=20261104,20261105,20261106,20261109,20261110,20261111,20261112
MAP="20261104=20260914,20261105=20261030,20261106=20261102,20261109=20261104,20261110=20261104,20261111=20261110,20261112=20261110"
run() { python -u research/featgen/backfill_days.py --grid 3 --keep pred --pq $B/lgbpred_pq "$@" 2>&1 | grep -E "^20" | cut -c1-230; }
echo "[$(date +%T)] A: true previous day, every EMA seeded (c0)"
run --dates $DAYS --seed-all c0 --out data/backfill_r91_A
echo "[$(date +%T)] B: the stale links, every EMA seeded (c0)"
run --dates $DAYS --seed-all c0 --seed-map $MAP --out data/backfill_r91_B
echo "[$(date +%T)] F: 1104-1106 from feitu, no seeding (as R84 did for the feitu days)"
run --dates 20261104,20261105,20261106 --dysl2-from 20261109 --out data/backfill_r91_F
for v in A B F; do
  echo "[$(date +%T)] state engine, $v"
  python -u research/featgen/verify_days.py data/backfill_r91_$v 2>&1 | head -1 | cut -c1-200
  python -u research/featgen/verify_days.py data/backfill_r91_$v 2>&1 | grep -E "^per day" | cut -c1-260
done
echo "[$(date +%T)] done"
