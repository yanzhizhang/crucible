#!/usr/bin/env bash
# Round 172 on 97, foreground, read-only on PM data: score 20261118 (a live day whose 1.37M PM predictions R171 parsed).
#  1 r99.sh for the one day: full_book + engine marks (new engine code) + ic_score's matrix against PM's prediction + bases
#    -> data/reports/stage6_r99_1118 (x.parquet, pred.parquet), data/featgen_r99/base_20261118.parquet. ~10-20 min.
#    The engine is seeded from the latest PM state file before the day that is on 97 (20261116: 1117's is not on 97), so the
#    carried states are one day stale; the day-end check against PM's 1118 state needs that file and is not part of this round.
#  2 which_model on 1118: our matrix through every model generation, ranked by agreement with PM's prediction.
#   bash research/r172.sh 2>&1 | tee data/reports/r172.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
D=20261118
ls -la $DATA/pm_signals/date=$D.parquet $DATA/seed_probe/HS300 2>&1 | tail -4 | cut -c1-160
ls -d $DATA/hstats_view/HS300/*/$D 2>/dev/null | wc -l | xargs echo "hstats families with $D:"
DAYS=$D bash research/r99.sh 2>&1 | cut -c1-220
grep -E "IC: start|rows|columns|similarity|filled" data/reports/r99_${D}_ic.txt 2>/dev/null | tail -8 | cut -c1-220
python -u research/featgen/which_model.py --days $D --x-dir "data/reports/stage6_r99_{mmdd}" \
  --roots $DATA/models_gen/A,$DATA/models_gen/B,$DATA/models_gen/C,/data/prod/models/tk5m 2>&1 | grep -v Warning | cut -c1-200
echo "[$(date +%T)] r172 done"
