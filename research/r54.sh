#!/usr/bin/env bash
# Round 54 on 97, foreground, AFTER R53. Experiments; the baseline file is not replaced. Writes under /work only.
# R53's survivor = r41c + two new columns (zzeg gen.rpe120, zzsa2.a crank:-gen.ldh30): up on five 224 days, down on 0430
# (.628 -> .612) although each passed the 0430 gate inside the 156-column set. Here each column is added to r41c alone
# and scored on all seven days, to see which one costs 0430.
#   bash research/r54.sh 2>&1 | tee data/reports/r54.txt
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
  CRUCIBLE_SOURCE=$(srcof $day) python -u research/pm_factors/ic_score.py --dates 2026$day --assign $A $MB $SCORE --dump-pred $(xof $day) --out $R/stage6_r54_${tag}_$day > $R/r54_${tag}_$day.txt 2>&1
  grep -E "Error|Traceback" $R/r54_${tag}_$day.txt | head -2
  echo "  $tag $day similarity $(ic $R/r54_${tag}_$day.txt)"
  [ -d "$(fwd_of $day)" ] && python -u research/pm_factors/pred_metrics.py --pred $R/stage6_r54_${tag}_$day --fwd $(fwd_of $day) --tag "$day $tag" 2>&1 \
    | grep -E "ours  vs forward return *rank IC|ours +top - bottom" | cut -c1-170
}
# gate <assign> <day> <tag>: keep the changes against r41c that gain on <day> (t >= 1); prints the table, echoes nothing else
gate() {
  local A=$1 day=$2 tag=$3
  CRUCIBLE_SOURCE=$(srcof $day) python -u research/pm_factors/ic_score.py --dates 2026$day --assign $A --gate-against $C1 --gate-t 1 $SCORE $(xof $day) $MB \
    --out $R/stage6_r54_$tag > $R/r54_$tag.txt 2>&1
  grep -E "Error|Traceback" $R/r54_$tag.txt | head -3
  sed -n '/== gate/,/^gated/p' $R/r54_$tag.txt | cut -c1-230 | head -60
}

G3=$R/stage6_r53_g3/ic_assign_gated.json
[ -f $G3 ] || { say "no $G3: stop"; exit 1; }
python - <<PY
import json
b = json.load(open("$C1")); g = json.load(open("$G3"))
for f, tag in (("zzeg", "eg"), ("zzsa2.a", "sa")):
    k = dict(b); k[f] = g[f]
    json.dump(k, open(f"$R/r54_{tag}.json", "w"), indent=1)
    print(f"  r41c + {f} = {g[f]} -> $R/r54_{tag}.json ({len(k)} columns)")
PY
say "r41c gen B: 0430 .6282 1106 .5690 1109 .6207 1110 .6470 1111 .4727 1112 .6558 1113 .6789"
for tag in eg sa; do
  say "r41c + $tag"
  for day in 0430 1106 1109 1110 1111 1112 1113; do score $tag $R/r54_$tag.json $day; done
done
grep -lE "Error|Traceback" $R/r54*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
