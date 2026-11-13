#!/usr/bin/env bash
# Round 43 on 97, foreground, scores only: IC / rank IC / IR of our prediction and of PM's against the realised forward
# 5-minute return, and of ours against PM's, on 0430 (2.23 production target) and 1113 (224 target). Writes under /work only.
#   bash research/r43.sh 2>&1 | tee data/reports/r43.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
HV=$DATA/hstats_view/HS300
B="--models /data/prod/models/tk5m --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV --neutral --no-drop-harmful --max-features 0 --implied 0 --dump-pred"
T="--target $DATA/pm_target_r22 --target-stale 4"
say() { echo "[$(date +%T)] $*"; }
for day in 0430 1113; do
  if [ $day = 0430 ]; then
    D=20260430; SRC=feitu; X="--marks rand:240:7 --engine-dir data/engine_m7 --full-book $DATA/store/pm_factors/full_book_m7 --seed-dir /data/prod/h/HS300/p1_intrading_dcp"
  else
    D=20261113; SRC=dysl2; X="--marks rand:240 --full-book $DATA/store/pm_factors/full_book_dysl2 --seed-dir $DATA/seed_probe/HS300"
  fi
  say "$D forward returns (no target)"
  CRUCIBLE_SOURCE=$SRC python -u research/pm_factors/ic_score.py --dates $D --assign $R/stage6_r41c_gate/ic_assign_gated.json $B $X \
    --out $R/stage6_r43_${day}_fwd > $R/r43_${day}_fwd.txt 2>&1
  grep -E "Error|Traceback|seed state|IC: start" $R/r43_${day}_fwd.txt | cut -c1-140
  for tag in r27both r34g r41c; do
    case $tag in r27both) A=$R/r27_both.json ;; r34g) A=$R/r34g_baseline.json ;; r41c) A=$R/stage6_r41c_gate/ic_assign_gated.json ;; esac
    CRUCIBLE_SOURCE=$SRC python -u research/pm_factors/ic_score.py --dates $D --assign $A $B $T $X --out $R/stage6_r43_${day}_$tag > $R/r43_${day}_$tag.txt 2>&1
    grep -E "Error|Traceback" $R/r43_${day}_$tag.txt | head -2
    python -u research/pm_factors/pred_metrics.py --pred $R/stage6_r43_${day}_$tag --fwd $R/stage6_r43_${day}_fwd --tag "$day $tag" 2>&1 | cut -c1-200
  done
done
grep -lE "Error|Traceback" $R/r43*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
