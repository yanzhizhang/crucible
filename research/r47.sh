#!/usr/bin/env bash
# Round 47 on 97, foreground, AFTER R44 (it reads R43 / R44 outputs). Experiments only: no baseline is replaced. /work only.
# The realised forward 5-minute return as a second referee, column by column.
#   A col_fwd: r41c's columns against the forward return on 0430 + every 224 day R44 scored; the columns that hurt on every
#     fit day (0430 1106 1110 1112) and how they behave on the test days (1109 1111 1113)
#   B r41c without those columns: similarity with PM on 0430 (grid 7), forward-return metrics on 0430 and the test days.
#     If dropping them costs no similarity and helps the return out of sample, those formulas are wrong.
#   C the open columns scanned against the forward return on 1106 (no PM target at all), then judged by the OTHER ruler:
#     similarity with PM's production prediction on 0430, and the forward return on the other days. A formula found by the
#     return on one day that raises the similarity with PM on another is very likely PM's.
#   bash research/r47.sh 2>&1 | tee data/reports/r47.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
HV=$DATA/hstats_view/HS300
C1=$R/stage6_r41c_gate/ic_assign_gated.json
FIT=${FIT:-"0430,1106,1110,1112"}
TEST=${TEST:-"1109 1111 1113"}
NO=${NO:-40}
B="--models /data/prod/models/tk5m --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV --neutral --no-drop-harmful --implied 0"
SC="$B --max-features 0 --dump-pred"
T="--target $DATA/pm_target_r22 --target-stale 4"
X0430="--marks rand:240:7 --engine-dir data/engine_m7 --full-book $DATA/store/pm_factors/full_book_m7 --seed-dir /data/prod/h/HS300/p1_intrading_dcp"
say() { echo "[$(date +%T)] $*"; }
last_before() { ls "$1" 2>/dev/null | grep -E '^2[0-9]{7}\.jz$' | sed 's/\.jz$//' | awk -v d="$2" '$1 < d' | sort | tail -1; }
seed_of() { local a b; a=$(last_before $DATA/seed_probe/HS300 $1); b=$(last_before /data/prod/h/HS300/p1_intrading_dcp $1)
  if [ "${a:-0}" \> "${b:-0}" ]; then echo $DATA/seed_probe/HS300; else echo /data/prod/h/HS300/p1_intrading_dcp; fi; }
xday() { echo "--marks rand:240 --full-book $DATA/store/pm_factors/full_book_dysl2 --seed-dir $(seed_of 2026$1)"; }
fwd_of() { if [ $1 = 0430 ] || [ $1 = 1113 ]; then echo $R/stage6_r43_$1_fwd; else echo $R/stage6_r44_$1_fwd; fi; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | cut -c1-120; }
# score <tag> <assign> <day>: a --target run with pred.parquet, then the forward-return metrics
score() {
  local tag=$1 A=$2 day=$3 src=dysl2 X
  if [ $day = 0430 ]; then src=feitu; X=$X0430; else X=$(xday $day); fi
  CRUCIBLE_SOURCE=$src python -u research/pm_factors/ic_score.py --dates 2026$day --assign $A $SC $T $X --out $R/stage6_r47_${tag}_$day > $R/r47_${tag}_$day.txt 2>&1
  grep -E "Error|Traceback" $R/r47_${tag}_$day.txt | head -2
  echo "  $tag $day vs PM: $(ic $R/r47_${tag}_$day.txt)"
  [ -d "$(fwd_of $day)" ] && python -u research/pm_factors/pred_metrics.py --pred $R/stage6_r47_${tag}_$day --fwd $(fwd_of $day) --tag "$day $tag" 2>&1 | cut -c1-200
}

say "A r41c column by column against the forward return"
RUNS=""
for day in 0430 1106 1109 1110 1111 1112 1113; do d=$(fwd_of $day); [ -f $d/ic_evidence.parquet ] && RUNS="$RUNS,$day=$d"; done
RUNS=${RUNS#,}
echo "  runs: $RUNS"
python -u research/pm_factors/col_fwd.py --runs "$RUNS" --pm $R/stage6_r41_audit2 --fit "$FIT" --assign $C1 --out $R/r47_pruned.json 2>&1 | cut -c1-250

say "B r41c without the hurting columns: 0430 and the test days"
if [ -f $R/r47_pruned.json ] && ! cmp -s <(python -c "import json;print(sorted(json.load(open('$R/r47_pruned.json'))))") <(python -c "import json;print(sorted(json.load(open('$C1'))))"); then
  for day in 0430 $TEST; do score pruned $R/r47_pruned.json $day; done
else
  echo "  nothing pruned, step skipped"
fi

say "C open columns scanned against the forward return on 1106 (no target)"
U=$(python -c "import json; a=json.load(open('$R/r41_audit2.json')); print(','.join(a['open'][:$NO]))")
echo "  open columns: $U"
CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261106 --assign $C1 --calibrate --siblings --state-pool --scan-feats "$U" $B $(xday 1106) \
  --out $R/stage6_r47c > $R/r47c_scan.txt 2>&1
grep -E "Error|Traceback" $R/r47c_scan.txt | head -3; grep -E "IC: start" $R/r47c_scan.txt | cut -c1-200
if [ -f $R/stage6_r47c/ic_assign.json ]; then
  python -u research/pm_factors/assign_diff.py $C1 $R/stage6_r47c/ic_assign.json | cut -c1-200 | head -60
  for day in 0430 $TEST 1110 1112; do score fwdscan $R/stage6_r47c/ic_assign.json $day; done
else
  echo "  no scan result"
fi
say "for comparison: r41c on the same days"
for f in $R/r44_*_r41c_metrics.txt; do [ -f "$f" ] && grep -E "^==|rank IC|decile" "$f" | cut -c1-170; done
grep -lE "Error|Traceback" $R/r47*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
