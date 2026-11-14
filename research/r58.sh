#!/usr/bin/env bash
# Round 58 on 97, foreground, AFTER R56. Experiments; the baseline file is not replaced. Writes under /work only.
# R57b: PM loads the PREVIOUS trading day's hstats at the start of a day (2.23 hstats/sod/<D> == hstats/HS300/<day before D>,
# byte-equal), while our view gave every 224 day its own same-day file (built after the close, so it holds that day's
# data). R58 = R56 with the right files, and with 0430 back as a gate (its start-of-day hstats are on 97 now):
#   0 build /work/crucible_data/hstats_sod/HS300/<family>/<D> -> the view's file of the trading day before D (symlinks);
#     0430 -> /work/crucible_data/hstats_0430 (already the start-of-day version). Own cell cache: data/pm_export_sod
#   1 scan the 30 columns named after hstats families on 1110 + 1112 (even sids), generation B
#   2 gates against r41c: 0430, 1109, 1106, 1113 (odd sids)
#   3 the survivor on all seven days; 1111 is never fitted or gated on
#   bash research/r58.sh 2>&1 | tee data/reports/r58.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
HV0=$DATA/hstats_view/HS300
HV=$DATA/hstats_sod/HS300
G=$DATA/models_gen/B
C1=$R/stage6_r41c_gate/ic_assign_gated.json
C2=$R/stage6_r52c_gate/ic_assign_gated.json
say() { echo "[$(date +%T)] $*"; }
last_before() { ls "$1" 2>/dev/null | grep -E '^2[0-9]{7}\.jz$' | sed 's/\.jz$//' | awk -v d="$2" '$1 < d' | sort | tail -1; }
seed_of() { local a b; a=$(last_before $DATA/seed_probe/HS300 $1); b=$(last_before /data/prod/h/HS300/p1_intrading_dcp $1)
  if [ "${a:-0}" \> "${b:-0}" ]; then echo $DATA/seed_probe/HS300; else echo /data/prod/h/HS300/p1_intrading_dcp; fi; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
fwd_of() { if [ $1 = 0430 ] || [ $1 = 1113 ]; then echo $R/stage6_r43_$1_fwd; else echo $R/stage6_r44_$1_fwd; fi; }
HP=$HV
hof() { if [ $1 = 0430 ]; then echo $HP; else echo $HV; fi; }
MB="--models $G --pm-cache data/pm_export_sod --drop-gap 1 --state-fallback --neutral --target $DATA/pm_target_r22 --target-stale 4 --max-cdf 1 --max-score 99"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
FBD=$DATA/store/pm_factors/full_book_dysl2
X0430="--marks rand:240:7 --engine-dir data/engine_m7 --full-book $DATA/store/pm_factors/full_book_m7 --seed-dir /data/prod/h/HS300/p1_intrading_dcp"
xof() { if [ $1 = 0430 ]; then echo "$X0430 --hstats $HP"; else echo "--marks rand:240 --full-book $FBD --seed-dir $(seed_of 2026$1) --hstats $HV"; fi; }
srcof() { if [ $1 = 0430 ]; then echo feitu; else echo dysl2; fi; }
score() {
  local tag=$1 A=$2 day=$3
  CRUCIBLE_SOURCE=$(srcof $day) python -u research/pm_factors/ic_score.py --dates 2026$day --assign $A $MB $SCORE --dump-pred $(xof $day) --out $R/stage6_r58_${tag}_$day > $R/r58_${tag}_$day.txt 2>&1
  grep -E "Error|Traceback" $R/r58_${tag}_$day.txt | head -2
  echo "  $tag $day similarity $(ic $R/r58_${tag}_$day.txt)"
  [ -d "$(fwd_of $day)" ] && python -u research/pm_factors/pred_metrics.py --pred $R/stage6_r58_${tag}_$day --fwd $(fwd_of $day) --tag "$day $tag" 2>&1 \
    | grep -E "ours  vs forward return *rank IC|ours +top - bottom" | cut -c1-170
}
# gate <assign> <day> <tag>: keep the changes against r41c that gain on <day> (t >= 1); prints the table, echoes nothing else
gate() {
  local A=$1 day=$2 tag=$3
  CRUCIBLE_SOURCE=$(srcof $day) python -u research/pm_factors/ic_score.py --dates 2026$day --assign $A --gate-against $C1 --gate-t 1 $SCORE $(xof $day) $MB \
    --out $R/stage6_r58_$tag > $R/r58_$tag.txt 2>&1
  grep -E "Error|Traceback" $R/r58_$tag.txt | head -3
  sed -n '/== gate/,/^gated/p' $R/r58_$tag.txt | cut -c1-230 | head -60
}

FALL="ZZAL,ZZDS,ZZGW,ZZHL,ZZMS,ZZQI,ZZQO,ZZSQ,ZZTS,ZZUG,ZZVC,ZZWA,ZZXC"
SAFE="ZZAL,ZZGW,ZZHL,ZZMS,ZZQI,ZZQO,ZZTS,ZZVC,ZZWA,ZZXC"
MATCH='^zz(al|ds|gw|hl|ms|qi|qo|sq|ts|ug|vc|wa|xc|ac|cr)[0-9]*(\.|$)'
INC='ced\.(lr\.[a-z0-9]+\.)?ZZ[A-Z]+\.'

say "0 start-of-day hstats view: $HV"
mkdir -p $HV
for f in $(ls $HV0); do
  mkdir -p $HV/$f
  for D in 20261106 20261109 20261110 20261111 20261112 20261113; do
    P=$(ls $HV0/$f | grep -E '^2[0-9]{7}$' | awk -v d=$D '$1 < d' | sort | tail -1)
    [ -n "$P" ] && ln -sfn $(readlink -f $HV0/$f/$P) $HV/$f/$D
  done
  [ -d $DATA/hstats_0430/HS300/$f/20260430 ] && ln -sfn $DATA/hstats_0430/HS300/$f/20260430 $HV/$f/20260430
done
for D in 20260430 20261106 20261109 20261110 20261111 20261112 20261113; do
  echo "  $D: ZZAL -> $(readlink -f $HV/ZZAL/$D) md5 $(md5sum $HV/ZZAL/$D/samplerS.nc 2>/dev/null | cut -c1-8)"
done
echo "  (2.23: start of day 1110 = 210c676d, 1111 = e4427b6c, 1112 = ea725fdd, 1113 = 7873d4bb)"
python - <<PY 2>&1 | cut -c1-400
import csv, re
names = [r[-1] if len(r) == 1 else r[1] for r in csv.reader(open("/data/prod/models/lgb_fn_v1_209_cte.csv")) if r]
m = [x for x in names if re.search(r'$MATCH', x)]
print(f"  model columns named after an hstats family: {len(m)}: {' '.join(m)}")
PY

DAYS=""
for day in 1110 1112; do [ -e $HV/ZZAL/2026$day/samplerS.nc ] && DAYS="$DAYS,2026$day"; done
DAYS=${DAYS#,}
[ -n "$DAYS" ] || { [ -e $HV/ZZAL/20261106/samplerS.nc ] && DAYS=20261106; }
[ -n "$DAYS" ] || { say "no scan day has hstats in the view: stop"; exit 1; }
[ -e $HV/ZZAL/20260430/samplerS.nc ] || { say "no 0430 hstats in $HV: stop"; exit 1; }
SD=$(seed_of ${DAYS%%,*})
say "1 scan on $DAYS (even sids), generation B; seed dir of the first day: $SD"
for P in $SAFE ZZWA,ZZAL,ZZQO,ZZXC,ZZTS; do
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates $DAYS --assign $C1 --calibrate --siblings --profiles $P --max-features 0 --scan-match "$MATCH" \
    --scan-include "$INC" --implied 0 --no-drop-harmful --sid-mod 2:0 --marks rand:240 --full-book $FBD --seed-dir $SD --hstats $HV $MB --out $R/stage6_r58 > $R/r58_scan.txt 2>&1
  [ -f $R/stage6_r58/ic_assign.json ] && { echo "  families used: $P"; break; }
  echo "  scan with $P failed:"; grep -E "Error|Traceback" -A3 $R/r58_scan.txt | tail -8 | cut -c1-200
done
PF=$P
grep -E "^scan-state|IC: start|seed state|^scan" $R/r58_scan.txt | head -8 | cut -c1-200
[ -f $R/stage6_r58/ic_assign.json ] || { say "no scan result: stop"; exit 1; }
python -u research/pm_factors/assign_diff.py $C1 $R/stage6_r58/ic_assign.json | cut -c1-200 | head -70

SCORE="$SCORE --profiles $PF"
say "2 gates against r41c: 0430, 1109, 1106, 1113 (odd sids)"
A=$R/stage6_r58/ic_assign.json
gate $A 0430 g0; [ -f $R/stage6_r58_g0/ic_assign_gated.json ] && A=$R/stage6_r58_g0/ic_assign_gated.json
gate $A 1109 g1; [ -f $R/stage6_r58_g1/ic_assign_gated.json ] && A=$R/stage6_r58_g1/ic_assign_gated.json
gate $A 1106 g2; [ -f $R/stage6_r58_g2/ic_assign_gated.json ] && A=$R/stage6_r58_g2/ic_assign_gated.json
if [ -e $HV/ZZAL/20261113/samplerS.nc ]; then
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261113 --assign $A --gate-against $C1 --gate-t 1 --sid-mod 2:1 $SCORE $(xof 1113) $MB --out $R/stage6_r58_g3 > $R/r58_g3.txt 2>&1
  grep -E "Error|Traceback" $R/r58_g3.txt | head -3; sed -n '/== gate/,/^gated/p' $R/r58_g3.txt | cut -c1-230 | head -60
  [ -f $R/stage6_r58_g3/ic_assign_gated.json ] && A=$R/stage6_r58_g3/ic_assign_gated.json
else echo "  1113 has no hstats: gate skipped"; fi
echo "  survivor: $A"
python -u research/pm_factors/assign_diff.py $C1 $A | cut -c1-200 | head -40

say "3 the survivor on all seven days (r41c gen B: 0430 .6282 1106 .5690 1109 .6207 1110 .6470 1111 .4727 1112 .6558 1113 .6789)"
if cmp -s <(python -c "import json;print(sorted(json.load(open('$A')).items()))") <(python -c "import json;print(sorted(json.load(open('$C1')).items()))"); then
  echo "  nothing survived the gates: no score runs"
else
  for day in 1111 0430 1109 1106 1110 1112 1113; do score surv $A $day; done
fi
grep -lE "Error|Traceback" $R/r58*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
