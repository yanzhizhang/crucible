#!/usr/bin/env bash
# Round 39 on 97, after R38: R35b says our prediction now over-carries short-term book imbalance / reversal against PM's. Drop,
# one at a time, every assigned column whose formula uses such a base; keep a drop only when it does not lose on 0430 grid 7
# for EITHER stock half (odd / even sids). Writes under /work only.
#   bash research/r39.sh 2>&1 | tee data/reports/r39.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
BASE=${BASE:-$R/r34g_baseline.json}
[ -e $BASE ] || BASE=$R/stage6_r34_gate/ic_assign_gated.json
M7="--models /data/prod/models/tk5m --target $DATA/pm_target_r22 --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $DATA/hstats_view/HS300 --marks rand:240:7 --engine-dir data/engine_m7"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
python -u research/pm_factors/r39_vars.py --base $BASE --out $R/r39_var
for v in $(ls $R/r39_var/*.json | xargs -n1 basename | sed 's/\.json$//'); do
  for h in 2:0 2:1; do
    CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $R/r39_var/$v.json $SCORE --sid-mod $h \
      --full-book $DATA/store/pm_factors/full_book_m7 $M7 --out $R/stage6_r39v_${v}_h${h#*:} > $R/r39v_${v}_h${h#*:}.txt 2>&1
  done
done
python -u research/pm_factors/r27_pick.py --base $R/r39_var/base.json --var-dir $R/r39_var --prefix $R/r39v_ --days h0,h1 \
  --exclude base,drop_all_hits --min-gain 0.0003 --out $R/r39_pick.json 2>&1 | cut -c1-200
grep -h "IC: start" $R/r39v_drop_all_hits_h*.txt | cut -c1-120
grep -lE "Error|Traceback" $R/r39*.txt 2>/dev/null | head | sed 's/^/  errors in /'
echo done
