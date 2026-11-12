#!/usr/bin/env bash
# Round 36 on 97, foreground, after R34 (it reuses R34's second mark grid for 0430). The carried state as candidates:
# ic_score --seed-dir adds gen.sd_* from the .jz PM started the day from (0430 <- 0429.jz, the true seed): the zzeg channel
# (c.high / c.low carried across days, also widened by today's range), its EMA, yesterday's late prices. Scan the stateful
# columns still open at r27_both on 0430 grid 0, gate on grid 7. Writes under /work only.
#   bash research/r36.sh 2>&1 | tee data/reports/r36.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBF7=$DATA/store/pm_factors/full_book_m7
SEED=${SEED:-/data/prod/h/HS300/p1_intrading_dcp}
BASE=${BASE:-$R/r27_both.json}
COLS=${COLS:-zzeg,zzgh.b,zzba,zzpk,zzab2,zzmh,zzcb,zzxo,zzhj,zzvm,zzvx.a,zzeg2,zzab}
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
MB="--models /data/prod/models/tk5m --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV --seed-dir $SEED"
M0="$MB --marks rand:240"
M7="$MB --marks rand:240:7 --engine-dir data/engine_m7"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
[ -e data/engine_m7/marks_20260430.parquet ] || { say "no grid-7 engine state: run research/r34.sh first"; exit 1; }

say "1 scan $COLS on 0430 grid 0 with seed candidates"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $BASE --calibrate --siblings --scan-feats "$COLS" --implied 0 \
  --full-book $FBF $M0 --out $R/stage6_r36 > $R/r36_scan.txt 2>&1
grep -E "Error|Traceback|seed state|IC: start" $R/r36_scan.txt | cut -c1-200
grep -E "^pass |-> " $R/r36_scan.txt | grep -E "sd_|$(echo $COLS | sed 's/,/|/g')" | cut -c1-230 | head -60
python -u research/pm_factors/assign_diff.py $BASE $R/stage6_r36/ic_assign.json | cut -c1-200 | head -40

say "2 gate on 0430 grid 7 against the base"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $R/stage6_r36/ic_assign.json --gate-against $BASE --gate-t 1 \
  $SCORE --full-book $FBF7 $M7 --out $R/stage6_r36_gate > $R/r36_gate.txt 2>&1
grep -E "Error|Traceback|== gate|^gated" $R/r36_gate.txt
sed -n '/== gate/,/^gated/p' $R/r36_gate.txt | cut -c1-230 | head -60

say "3 base / R36 / R36 gated on 0430 grid 7"
for tag in base r36 r36g; do
  case $tag in base) A=$BASE ;; r36) A=$R/stage6_r36/ic_assign.json ;; r36g) A=$R/stage6_r36_gate/ic_assign_gated.json ;; esac
  CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $A $SCORE --full-book $FBF7 $M7 \
    --out $R/stage6_r36h_$tag > $R/r36h_$tag.txt 2>&1
  echo "  $tag: $(ic $R/r36h_$tag.txt)"
done
grep -lE "Error|Traceback" $R/r36*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
