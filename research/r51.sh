#!/usr/bin/env bash
# Round 51 on 97, foreground, scores only. R50: the 224 signal is far closer to model generation B (202601.updated..202608)
# than to C (202609..202612) that ic_score picked for November: r41c .65 against .50 on 1110 / 1112. So every 224 day is
# rescored with generation B forced, and r41c's columns are compared between 0430 and each 224 day again (R49 with C gave a
# rank correlation near 0; with the right generation the days should agree). Writes under /work only.
#   bash research/r51.sh 2>&1 | tee data/reports/r51.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
HV=$DATA/hstats_view/HS300
G=$DATA/models_gen/B
DAYS=${DAYS:-"1106 1109 1110 1111 1112 1113"}
say() { echo "[$(date +%T)] $*"; }
last_before() { ls "$1" 2>/dev/null | grep -E '^2[0-9]{7}\.jz$' | sed 's/\.jz$//' | awk -v d="$2" '$1 < d' | sort | tail -1; }
seed_of() { local a b; a=$(last_before $DATA/seed_probe/HS300 $1); b=$(last_before /data/prod/h/HS300/p1_intrading_dcp $1)
  if [ "${a:-0}" \> "${b:-0}" ]; then echo $DATA/seed_probe/HS300; else echo /data/prod/h/HS300/p1_intrading_dcp; fi; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
fwd_of() { if [ $1 = 0430 ] || [ $1 = 1113 ]; then echo $R/stage6_r43_$1_fwd; else echo $R/stage6_r44_$1_fwd; fi; }
B="--models $G --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV --neutral --no-drop-harmful --max-features 0 --implied 0 --dump-pred --target $DATA/pm_target_r22 --target-stale 4"
ls -l $G/ | grep -- "->" | awk '{print "  ", $(NF-2), $(NF-1), $NF}'
say "1 similarity with the 224 signal, generation B"
printf "%-6s %8s %8s %8s\n" day r27both r34g r41c
for day in $DAYS; do
  X="--marks rand:240 --full-book $DATA/store/pm_factors/full_book_dysl2 --seed-dir $(seed_of 2026$day)"
  line="$(printf '%-6s' $day)"
  for tag in r27both r34g r41c; do
    case $tag in r27both) A=$R/r27_both.json ;; r34g) A=$R/r34g_baseline.json ;; r41c) A=$R/stage6_r41c_gate/ic_assign_gated.json ;; esac
    O=$R/stage6_r50_${day}_${tag}_B
    if [ ! -f $O/pred.parquet ]; then
      CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 2026$day --assign $A $B $X --out $O > $R/r50_${day}_${tag}_B.txt 2>&1
    fi
    line="$line $(printf '%8s' "$(ic $R/r50_${day}_${tag}_B.txt)")"
  done
  echo "$line"
done
say "2 against the forward return, generation B (the 224 signal's own lines are in R43 / R44)"
for day in $DAYS; do
  for tag in r27both r34g r41c; do
    python -u research/pm_factors/pred_metrics.py --pred $R/stage6_r50_${day}_${tag}_B --fwd $(fwd_of $day) --tag "$day $tag genB" 2>&1 \
      | grep -E "^==|ours  vs forward return *rank IC|PM's  vs forward return *rank IC|ours  vs PM.*rank IC|top - bottom" | cut -c1-170
  done
done
say "3 r41c column by column: 0430 against each 224 day, generation B on both"
for day in $DAYS; do
  echo "-- 0430 against $day"
  python -u research/pm_factors/col_days.py --a $R/stage6_r41_audit2 --b $R/stage6_r50_${day}_r41c_B --names 0430,$day 2>&1 | sed -n '/== columns by class/,/== by kind x class/p;/rank correlation/p' | grep -v "^$" | cut -c1-200
done
say "4 the columns that still hurt on 1110 and on 1112 with generation B"
for day in 1110 1112; do
  python -u research/pm_factors/col_days.py --a $R/stage6_r41_audit2 --b $R/stage6_r50_${day}_r41c_B --names 0430,$day 2>&1 | sed -n '/== columns that hurt on/,/== columns that help on both/p' | cut -c1-200 | head -45
done
say "5 tails on 1110 and 1112, generation B"
for day in 1110 1112; do
  python -u research/pm_factors/audit.py --run $R/stage6_r50_${day}_r41c_B --out $R/r51_audit_$day.json 2>&1 | sed -n '1,12p;/== tails/,$p' | cut -c1-200
done
grep -lE "Error|Traceback" $R/r50_*_B.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
