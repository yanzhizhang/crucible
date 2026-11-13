#!/usr/bin/env bash
# Round 52 on 97, foreground. R50 / R51: with model generation B the 224 signal is a usable target (r41c .57-.68 a day), so
# there are now seven target days instead of one. Experiments; the baseline file is not replaced. Writes under /work only.
#   A r41c column by column over the seven days (similarity loss per day, t over days): which columns help on every day,
#     which hurt on every fit day (0430 1106 1110 1112) and whether they still hurt on the test days (1109 1111 1113)
#   B r41c without the columns hurting on every fit day, scored on the test days
#   C the open + stateful columns scanned on TWO 224 days (1110 + 1112, even sids), gated on 1113 (odd sids), then judged
#     where nothing was fitted: 0430 (2.23 production, other feed), 1106, 1109, 1111
#   bash research/r52.sh 2>&1 | tee data/reports/r52.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
HV=$DATA/hstats_view/HS300
G=$DATA/models_gen/B
C1=$R/stage6_r41c_gate/ic_assign_gated.json
NO=${NO:-40}
say() { echo "[$(date +%T)] $*"; }
last_before() { ls "$1" 2>/dev/null | grep -E '^2[0-9]{7}\.jz$' | sed 's/\.jz$//' | awk -v d="$2" '$1 < d' | sort | tail -1; }
seed_of() { local a b; a=$(last_before $DATA/seed_probe/HS300 $1); b=$(last_before /data/prod/h/HS300/p1_intrading_dcp $1)
  if [ "${a:-0}" \> "${b:-0}" ]; then echo $DATA/seed_probe/HS300; else echo /data/prod/h/HS300/p1_intrading_dcp; fi; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
fwd_of() { if [ $1 = 0430 ] || [ $1 = 1113 ]; then echo $R/stage6_r43_$1_fwd; else echo $R/stage6_r44_$1_fwd; fi; }
CORE="--models $G --drop-gap 1 --state-fallback --hstats $HV --neutral --target $DATA/pm_target_r22 --target-stale 4"
MB="$CORE --max-cdf 1 --max-score 99"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
FBD=$DATA/store/pm_factors/full_book_dysl2
X0430="--marks rand:240:7 --engine-dir data/engine_m7 --full-book $DATA/store/pm_factors/full_book_m7 --seed-dir /data/prod/h/HS300/p1_intrading_dcp"
# score <tag> <assign> <day>: similarity with the target (generation B) and the forward-return metrics
score() {
  local tag=$1 A=$2 day=$3 src=dysl2 X
  if [ $day = 0430 ]; then src=feitu; X=$X0430; else X="--marks rand:240 --full-book $FBD --seed-dir $(seed_of 2026$day)"; fi
  CRUCIBLE_SOURCE=$src python -u research/pm_factors/ic_score.py --dates 2026$day --assign $A $MB $SCORE --dump-pred $X --out $R/stage6_r52_${tag}_$day > $R/r52_${tag}_$day.txt 2>&1
  grep -E "Error|Traceback" $R/r52_${tag}_$day.txt | head -2
  echo "  $tag $day similarity $(ic $R/r52_${tag}_$day.txt)"
  [ -d "$(fwd_of $day)" ] && python -u research/pm_factors/pred_metrics.py --pred $R/stage6_r52_${tag}_$day --fwd $(fwd_of $day) --tag "$day $tag" 2>&1 \
    | grep -E "ours  vs forward return *rank IC|ours +top - bottom" | cut -c1-170
}

say "A r41c column by column over seven target days (generation B)"
RUNS="0430=$R/stage6_r41_audit2"
for day in 1106 1109 1110 1111 1112 1113; do [ -f $R/stage6_r50_${day}_r41c_B/ic_evidence.parquet ] && RUNS="$RUNS,$day=$R/stage6_r50_${day}_r41c_B"; done
python -u research/pm_factors/col_fwd.py --runs "$RUNS" --fit 0430,1106,1110,1112 --assign $C1 --out $R/r52_pruned.json 2>&1 | cut -c1-250
echo "  (in this table 'loss' is the similarity with the target lost without the column, not the forward return)"

say "B r41c without the columns hurting on every fit day: test days"
echo "  r41c itself, generation B: 1109 $(ic $R/r50_1109_r41c_B.txt) 1111 $(ic $R/r50_1111_r41c_B.txt) 1113 $(ic $R/r50_1113_r41c_B.txt) 0430 .6282"
for day in 1109 1111 1113 0430; do score pruned $R/r52_pruned.json $day; done

say "C open + stateful columns scanned on 1110 + 1112 (even sids), generation B"
U=$(python -c "import json; a=json.load(open('$R/r41_audit2.json')); print(','.join(a['open'][:$NO]))")
SD=$DATA/seed_probe/HS300
CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261110,20261112 --assign $C1 --calibrate --siblings --state-pool --scan-state --scan-feats "$U" \
  --scan-include 'gen\.sd_' --implied 0 --no-drop-harmful --sid-mod 2:0 --marks rand:240 --full-book $FBD --seed-dir $SD $MB --out $R/stage6_r52c > $R/r52c_scan.txt 2>&1
grep -E "Error|Traceback" $R/r52c_scan.txt | head -3; grep -E "^scan-state|IC: start|seed state" $R/r52c_scan.txt | cut -c1-200
if [ -f $R/stage6_r52c/ic_assign.json ]; then
  python -u research/pm_factors/assign_diff.py $C1 $R/stage6_r52c/ic_assign.json | cut -c1-200 | head -70
  say "  gate on 1113 (odd sids) against r41c"
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261113 --assign $R/stage6_r52c/ic_assign.json --gate-against $C1 --gate-t 1 --sid-mod 2:1 $SCORE \
    --marks rand:240 --full-book $FBD --seed-dir $SD $MB --out $R/stage6_r52c_gate > $R/r52c_gate.txt 2>&1
  grep -E "Error|Traceback" $R/r52c_gate.txt | head -3; sed -n '/== gate/,/^gated/p' $R/r52c_gate.txt | cut -c1-230 | head -80
  C2=$R/stage6_r52c_gate/ic_assign_gated.json
  if [ -f $C2 ]; then
    say "  judged on days nothing was fitted on (r41c with generation B: 0430 .6282, 1106 $(ic $R/r50_1106_r41c_B.txt), 1109 $(ic $R/r50_1109_r41c_B.txt), 1111 $(ic $R/r50_1111_r41c_B.txt))"
    for day in 0430 1106 1109 1111; do score r52c $C2 $day; done
    echo "  and on the days it was fitted / gated on (r41c: 1110 $(ic $R/r50_1110_r41c_B.txt), 1112 $(ic $R/r50_1112_r41c_B.txt), 1113 $(ic $R/r50_1113_r41c_B.txt))"
    for day in 1110 1112 1113; do score r52c $C2 $day; done
  else
    echo "  no gated assignment"
  fi
else
  echo "  no scan result"
fi
grep -lE "Error|Traceback" $R/r52*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
