#!/usr/bin/env bash
# Round 53 on 97, foreground, AFTER R52. Experiments; the baseline file is not replaced. Writes under /work only.
# R52 C: formulas picked on 1110 + 1112 and gated on 1113 gain on those days and on 1106, and lose on 0430, 1109, 1111.
# The per-column losses split the days in two groups ({0430, 1109} and {1106, 1110, 1112, 1113}), so one group cannot
# vouch for the other. Here every change of R52 C has to pass a gate on a day of EACH group before it is kept:
#   0 which PM state file each day starts from (the one thing known to differ by day), for the two-group question
#   1 r52c_gate (156 columns) gated against r41c on 0430, the result on 1109, the result on 1106 (all sids, t >= 1)
#   2 what survives, scored on all seven days; 1111 is the day nothing was fitted or gated on
#   3 the same chain started from r52_pruned-style cleaning: the 5 hurting columns of R52 A removed from the survivor
#   bash research/r53.sh 2>&1 | tee data/reports/r53.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
HV=$DATA/hstats_view/HS300
G=$DATA/models_gen/B
C1=$R/stage6_r41c_gate/ic_assign_gated.json
C2=$R/stage6_r52c_gate/ic_assign_gated.json
say() { echo "[$(date +%T)] $*"; }
last_before() { ls "$1" 2>/dev/null | grep -E '^2[0-9]{7}\.jz$' | sed 's/\.jz$//' | awk -v d="$2" '$1 < d' | sort | tail -1; }
seed_of() { local a b; a=$(last_before $DATA/seed_probe/HS300 $1); b=$(last_before /data/prod/h/HS300/p1_intrading_dcp $1)
  if [ "${a:-0}" \> "${b:-0}" ]; then echo $DATA/seed_probe/HS300; else echo /data/prod/h/HS300/p1_intrading_dcp; fi; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
fwd_of() { if [ $1 = 0430 ] || [ $1 = 1113 ]; then echo $R/stage6_r43_$1_fwd; else echo $R/stage6_r44_$1_fwd; fi; }
MB="--models $G --drop-gap 1 --state-fallback --hstats $HV --neutral --target $DATA/pm_target_r22 --target-stale 4 --max-cdf 1 --max-score 99"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
FBD=$DATA/store/pm_factors/full_book_dysl2
X0430="--marks rand:240:7 --engine-dir data/engine_m7 --full-book $DATA/store/pm_factors/full_book_m7 --seed-dir /data/prod/h/HS300/p1_intrading_dcp"
xof() { if [ $1 = 0430 ]; then echo "$X0430"; else echo "--marks rand:240 --full-book $FBD --seed-dir $(seed_of 2026$1)"; fi; }
srcof() { if [ $1 = 0430 ]; then echo feitu; else echo dysl2; fi; }
score() {
  local tag=$1 A=$2 day=$3
  CRUCIBLE_SOURCE=$(srcof $day) python -u research/pm_factors/ic_score.py --dates 2026$day --assign $A $MB $SCORE --dump-pred $(xof $day) --out $R/stage6_r53_${tag}_$day > $R/r53_${tag}_$day.txt 2>&1
  grep -E "Error|Traceback" $R/r53_${tag}_$day.txt | head -2
  echo "  $tag $day similarity $(ic $R/r53_${tag}_$day.txt)"
  [ -d "$(fwd_of $day)" ] && python -u research/pm_factors/pred_metrics.py --pred $R/stage6_r53_${tag}_$day --fwd $(fwd_of $day) --tag "$day $tag" 2>&1 \
    | grep -E "ours  vs forward return *rank IC|ours +top - bottom" | cut -c1-170
}
# gate <assign> <day> <tag>: keep the changes against r41c that gain on <day> (t >= 1); prints the table, echoes nothing else
gate() {
  local A=$1 day=$2 tag=$3
  CRUCIBLE_SOURCE=$(srcof $day) python -u research/pm_factors/ic_score.py --dates 2026$day --assign $A --gate-against $C1 --gate-t 1 $SCORE $(xof $day) $MB \
    --out $R/stage6_r53_$tag > $R/r53_$tag.txt 2>&1
  grep -E "Error|Traceback" $R/r53_$tag.txt | head -3
  sed -n '/== gate/,/^gated/p' $R/r53_$tag.txt | cut -c1-230 | head -60
}

say "0 PM state file each day starts from"
for day in 0430 1106 1109 1110 1111 1112 1113; do
  d=$(seed_of 2026$day); p=$(last_before $d 2026$day)
  echo "  $day: $d/$p.jz $(ls -l --time-style=+%Y-%m-%dT%H:%M $d/$p.jz 2>/dev/null | awk '{print $5, $6}')"
done

[ -f $C2 ] || { say "no $C2: stop"; exit 1; }
say "1 chain of gates on r52c_gate: 0430, then 1109, then 1106"
gate $C2 0430 g1
G1=$R/stage6_r53_g1/ic_assign_gated.json
[ -f $G1 ] || { say "gate 0430 gave nothing: stop"; exit 1; }
gate $G1 1109 g2
G2=$R/stage6_r53_g2/ic_assign_gated.json
[ -f $G2 ] || { say "gate 1109 gave nothing: stop"; exit 1; }
gate $G2 1106 g3
G3=$R/stage6_r53_g3/ic_assign_gated.json
[ -f $G3 ] || G3=$G2
python -u research/pm_factors/assign_diff.py $C1 $G3 | cut -c1-200 | head -40

say "2 the survivor on all seven days (r41c gen B: 0430 .6282 1106 .5690 1109 .6207 1110 .6470 1111 .4727 1112 .6558 1113 .6789)"
for day in 1111 0430 1109 1106 1110 1112 1113; do score surv $G3 $day; done

say "3 survivor without the 5 columns hurting on every fit day (R52 A)"
python - <<PY
import json
a = json.load(open("$G3"))
p = set(json.load(open("$R/r52_pruned.json")))
b = json.load(open("$C1"))
drop = [f for f in b if f not in p]
k = {f: e for f, e in a.items() if f not in drop}
json.dump(k, open("$R/r53_surv_pruned.json", "w"), indent=1)
print(f"  dropped {drop}; {len(k)} of {len(a)} columns -> $R/r53_surv_pruned.json")
PY
for day in 1111 0430 1109 1113; do score survp $R/r53_surv_pruned.json $day; done
grep -lE "Error|Traceback" $R/r53*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
