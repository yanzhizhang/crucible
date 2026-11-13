#!/usr/bin/env bash
# Round 44 on 97, foreground, night batch, scores only. Writes under /work only.
#   1 r41c column by column, 0430 against 1113, from the R43 run that used the right seed (R42 used the stale 1106.jz)
#   2 R43's metrics (IC / rank IC / IR / decile spread against the realised forward 5-minute return) on every other 224 day
#   bash research/r44.sh 2>&1 | tee data/reports/r44.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
HV=$DATA/hstats_view/HS300
DAYS=${DAYS:-"20261106 20261109 20261110 20261111 20261112"}
B="--models /data/prod/models/tk5m --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV --neutral --no-drop-harmful --max-features 0 --implied 0 --dump-pred"
T="--target $DATA/pm_target_r22 --target-stale 4"
say() { echo "[$(date +%T)] $*"; }
last_before() { ls "$1" 2>/dev/null | grep -E '^2[0-9]{7}\.jz$' | sed 's/\.jz$//' | awk -v d="$2" '$1 < d' | sort | tail -1; }
say "1 r41c column by column: 0430 against 1113 (seed 1112.jz)"
grep -E "seed state" $R/r43_1113_r41c.txt | head -1
python -u research/pm_factors/col_days.py --a $R/stage6_r41_audit2 --b $R/stage6_r43_1113_r41c --names 0430,1113 2>&1 | cut -c1-400
python -u research/pm_factors/audit.py --run $R/stage6_r43_1113_r41c --out $R/r44_audit_1113.json 2>&1 | sed -n '1,12p;/== tails/,$p' | cut -c1-200
say "2 forward-return metrics on: $DAYS"
for D in $DAYS; do
  day=${D:4}
  S1=$DATA/seed_probe/HS300; S2=/data/prod/h/HS300/p1_intrading_dcp
  a=$(last_before $S1 $D); b=$(last_before $S2 $D)
  if [ "${a:-0}" \> "${b:-0}" ]; then SEED=$S1; else SEED=$S2; fi
  X="--marks rand:240 --full-book $DATA/store/pm_factors/full_book_dysl2 --seed-dir $SEED"
  say "$D forward returns (no target), seed dir $SEED"
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates $D --assign $R/stage6_r41c_gate/ic_assign_gated.json $B $X \
    --out $R/stage6_r44_${day}_fwd > $R/r44_${day}_fwd.txt 2>&1
  grep -E "Error|Traceback|seed state|IC: start" $R/r44_${day}_fwd.txt | cut -c1-140
  if grep -qE "Error|Traceback" $R/r44_${day}_fwd.txt; then echo "  $D skipped"; continue; fi
  for tag in r27both r34g r41c; do
    case $tag in r27both) A=$R/r27_both.json ;; r34g) A=$R/r34g_baseline.json ;; r41c) A=$R/stage6_r41c_gate/ic_assign_gated.json ;; esac
    CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates $D --assign $A $B $T $X --out $R/stage6_r44_${day}_$tag > $R/r44_${day}_$tag.txt 2>&1
    grep -E "Error|Traceback" $R/r44_${day}_$tag.txt | head -2
    python -u research/pm_factors/pred_metrics.py --pred $R/stage6_r44_${day}_$tag --fwd $R/stage6_r44_${day}_fwd --tag "$day $tag" 2>&1 | cut -c1-200 | tee $R/r44_${day}_${tag}_metrics.txt
  done
done
say "3 rank IC and decile spread, every day"
for f in $R/r43_*_metrics.txt $R/r44_*_metrics.txt; do [ -f "$f" ] && grep -E "^==|rank IC|decile" "$f" | cut -c1-170; done
grep -lE "Error|Traceback" $R/r44*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
