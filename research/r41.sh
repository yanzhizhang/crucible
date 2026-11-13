#!/usr/bin/env bash
# Round 41 on 97, foreground: how right is r40g, and the three ways left to push it. 0430 only, strict split as R40 (scan on
# even sids x grid 0, gate on odd sids x grid 7, report both halves on grid 7). Writes under /work only.
#   0 audit r40g: each column graded against PM's training CDF before cal: (units / shape / proxy / idle), the model gain behind
#     each grade and behind the open columns, top / bottom decile overlap with PM
#   1 strict rescan: the proxy columns are emptied and rescanned with the CDF gates ON (--max-cdf .05 --max-score 1.5, no cal:);
#     gated against r40g, so a proxy comes back unless the CDF-fitting formula holds up            -> r41b
#   2 stateful + open columns: every stateful stem's columns may now take the shared pool (--state-pool; until now they saw only
#     their own atoms, so the gen.sd_* seed candidates of R36 never reached zzeg), plus the top open columns again  -> r41c
#   3 r40g / r41b / r41c on grid 7, both halves;  4 audit r41c;  5 --implied on the open columns (report only: what PM's model
#     wants in them), input for new candidate quantities
#   bash research/r41.sh 2>&1 | tee data/reports/r41.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBF7=$DATA/store/pm_factors/full_book_m7
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
SEED=${SEED:-/data/prod/h/HS300/p1_intrading_dcp}
CORE="--models /data/prod/models/tk5m --target $TGT --target-stale 4 --drop-gap 1 --state-fallback --hstats $HV --neutral --seed-dir $SEED"
MB="$CORE --max-cdf 1 --max-score 99"
M0="$MB --marks rand:240"
M7="$MB --marks rand:240:7 --engine-dir data/engine_m7"
S0="$CORE --marks rand:240"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
BASE=${BASE:-$R/r40g_baseline.json}
NP=${NP:-40}
NO=${NO:-40}
run() { CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 "$@"; }
errs() { grep -E "Error|Traceback" "$1" | head -3; }

say "0 audit $BASE: all stocks, grid 7"
run --assign $BASE $SCORE --dump-pred --full-book $FBF7 $M7 --out $R/stage6_r41_audit > $R/r41_audit_run.txt 2>&1
errs $R/r41_audit_run.txt; grep "IC: start" $R/r41_audit_run.txt | cut -c1-160
python -u research/pm_factors/audit.py --run $R/stage6_r41_audit --out $R/r41_audit.json 2>&1 | cut -c1-230

say "1 strict rescan of the top $NP proxy columns (CDF gates on, no cal:): even sids x grid 0"
P=$(python - <<PY
import json
a = json.load(open("$R/r41_audit.json")); b = json.load(open("$BASE"))
p = a["proxy"][:$NP]
for k in p:
    b.pop(k, None)
json.dump(b, open("$R/r41_strict_base.json", "w"), indent=1)
print(",".join(p))
PY
)
echo "  proxies: $P"
if [ -n "$P" ]; then
  run --assign $R/r41_strict_base.json --siblings --scan-feats "$P" --implied 0 --no-drop-harmful --sid-mod 2:0 --full-book $FBF $S0 \
    --out $R/stage6_r41b > $R/r41b_scan.txt 2>&1
  errs $R/r41b_scan.txt; grep -E "^pass 0 |IC: start" $R/r41b_scan.txt | cut -c1-200 | head -60
  run --assign $R/stage6_r41b/ic_assign.json --gate-against $BASE --gate-t 1 --sid-mod 2:1 $SCORE --full-book $FBF7 $M7 \
    --out $R/stage6_r41b_gate > $R/r41b_gate.txt 2>&1
  errs $R/r41b_gate.txt; sed -n '/== gate/,/^gated/p' $R/r41b_gate.txt | cut -c1-230 | head -70
  B1=$R/stage6_r41b_gate/ic_assign_gated.json
else
  B1=$BASE
fi
[ -e $B1 ] || B1=$BASE

say "2 stateful columns with the shared pool + the top $NO open columns: even sids x grid 0, from $B1"
U=$(python -c "import json; a=json.load(open('$R/r41_audit.json')); print(','.join(a['open'][:$NO]))")
run --assign $B1 --calibrate --siblings --state-pool --scan-state --scan-feats "$U" --scan-include 'gen\.sd_' --implied 0 --no-drop-harmful \
  --sid-mod 2:0 --full-book $FBF $M0 --out $R/stage6_r41c > $R/r41c_scan.txt 2>&1
errs $R/r41c_scan.txt; grep -E "^scan-state|IC: start" $R/r41c_scan.txt | cut -c1-200
python -u research/pm_factors/assign_diff.py $B1 $R/stage6_r41c/ic_assign.json | cut -c1-200 | head -60
run --assign $R/stage6_r41c/ic_assign.json --gate-against $B1 --gate-t 1 --sid-mod 2:1 $SCORE --full-book $FBF7 $M7 \
  --out $R/stage6_r41c_gate > $R/r41c_gate.txt 2>&1
errs $R/r41c_gate.txt; sed -n '/== gate/,/^gated/p' $R/r41c_gate.txt | cut -c1-230 | head -70
C1=$R/stage6_r41c_gate/ic_assign_gated.json
[ -e $C1 ] || C1=$B1

say "3 grid 7, both halves"
printf "%-10s %8s %8s %8s\n" stocks r40g r41b r41c
for h in 2:1 2:0; do
  line="$(printf '%-10s' "sid%$h")"
  for tag in r40g r41b r41c; do
    case $tag in r40g) A=$BASE ;; r41b) A=$B1 ;; r41c) A=$C1 ;; esac
    run --assign $A $SCORE --sid-mod $h --full-book $FBF7 $M7 --out $R/stage6_r41h_${tag}_${h/:/_} > $R/r41h_${tag}_${h/:/_}.txt 2>&1
    line="$line $(printf '%8s' "$(ic $R/r41h_${tag}_${h/:/_}.txt)")"
  done
  echo "$line"
done

say "4 audit $C1: all stocks, grid 7"
run --assign $C1 $SCORE --dump-pred --full-book $FBF7 $M7 --out $R/stage6_r41_audit2 > $R/r41_audit2_run.txt 2>&1
errs $R/r41_audit2_run.txt
python -u research/pm_factors/audit.py --run $R/stage6_r41_audit2 --out $R/r41_audit2.json 2>&1 | cut -c1-230 | sed -n '1,14p;/== tails/,$p'

say "5 implied: what PM's model wants in the open columns (report only)"
run --assign $C1 --no-drop-harmful --max-features 0 --implied 30 --full-book $FBF $M0 --out $R/stage6_r41_implied > $R/r41_implied.txt 2>&1
errs $R/r41_implied.txt; sed -n '/implied/,$p' $R/r41_implied.txt | cut -c1-230 | head -120
grep -lE "Error|Traceback" $R/r41*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
