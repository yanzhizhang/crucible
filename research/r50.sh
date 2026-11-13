#!/usr/bin/env bash
# Round 50 on 97, foreground, scores only: which model generation does each target come from? tk5m holds three distinct
# generations (R45): A = 202509..202601, B = 202601.updated..202608, C = 202609..202612. ic_score takes the latest month not
# after the day, so 0430 is scored with B and the 224 days with C. R49: the r41c columns that lower the similarity with the
# 224 signal are the same on every 224 day -- a systematic cause. One candidate: 224 does not run generation C.
# Each day is scored with every generation forced (a models dir holding one month link). Writes under /work only.
#   bash research/r50.sh 2>&1 | tee data/reports/r50.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
HV=$DATA/hstats_view/HS300
G=$DATA/models_gen
DAYS=${DAYS:-"0430 1110 1112 1106"}
say() { echo "[$(date +%T)] $*"; }
for g in A:202512 B:202604 C:202611; do
  mkdir -p $G/${g%%:*}
  ln -sfn /data/prod/models/tk5m/${g##*:} $G/${g%%:*}/202501
done
ls -l $G/*/ | grep -- "->" | awk '{print "  ", $(NF-2), $(NF-1), $NF}'
last_before() { ls "$1" 2>/dev/null | grep -E '^2[0-9]{7}\.jz$' | sed 's/\.jz$//' | awk -v d="$2" '$1 < d' | sort | tail -1; }
seed_of() { local a b; a=$(last_before $DATA/seed_probe/HS300 $1); b=$(last_before /data/prod/h/HS300/p1_intrading_dcp $1)
  if [ "${a:-0}" \> "${b:-0}" ]; then echo $DATA/seed_probe/HS300; else echo /data/prod/h/HS300/p1_intrading_dcp; fi; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
B="--drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV --neutral --no-drop-harmful --max-features 0 --implied 0 --dump-pred --target $DATA/pm_target_r22 --target-stale 4"
printf "%-6s %-8s %8s %8s %8s\n" day set genA genB genC
for day in $DAYS; do
  if [ $day = 0430 ]; then SRC=feitu; X="--marks rand:240:7 --engine-dir data/engine_m7 --full-book $DATA/store/pm_factors/full_book_m7 --seed-dir /data/prod/h/HS300/p1_intrading_dcp"
  else SRC=dysl2; X="--marks rand:240 --full-book $DATA/store/pm_factors/full_book_dysl2 --seed-dir $(seed_of 2026$day)"; fi
  for tag in r41c r27both; do
    case $tag in r27both) A=$R/r27_both.json ;; r41c) A=$R/stage6_r41c_gate/ic_assign_gated.json ;; esac
    line="$(printf '%-6s %-8s' $day $tag)"
    for g in A B C; do
      CRUCIBLE_SOURCE=$SRC python -u research/pm_factors/ic_score.py --dates 2026$day --assign $A --models $G/$g $B $X --out $R/stage6_r50_${day}_${tag}_$g > $R/r50_${day}_${tag}_$g.txt 2>&1
      line="$line $(printf '%8s' "$(ic $R/r50_${day}_${tag}_$g.txt)")"
    done
    echo "$line"
  done
done
grep -h "model generation" $R/r50_*_r41c_?.txt | sort -u | cut -c1-140
say "forward-return metrics of r41c per generation (the fwd runs of R43 / R44 give the return)"
for day in $DAYS; do
  if [ $day = 0430 ] || [ $day = 1113 ]; then F=$R/stage6_r43_${day}_fwd; else F=$R/stage6_r44_${day}_fwd; fi
  for g in A B C; do
    [ -d $F ] && python -u research/pm_factors/pred_metrics.py --pred $R/stage6_r50_${day}_r41c_$g --fwd $F --tag "$day r41c gen$g" 2>&1 | grep -E "^==|ours  vs forward return *rank IC|ours  vs PM.*rank IC|ours +top" | cut -c1-170
  done
done
grep -lE "Error|Traceback" $R/r50*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
