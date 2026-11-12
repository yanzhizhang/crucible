#!/usr/bin/env bash
# Round 32 on 97, foreground: rescan from r27_both on the cleaner days, with the SOFT family lock (off-class formulas are
# rescanned in class but kept unless beaten; the gate treats every change alike). Writes under /work only.
#   1 scan the columns open at r27_both + the soft-lock rescans on 1110 + 1112 (DataYes; 1112 = first unified-log day)
#   2 gate each change on 0430 (PM production alpha, true seed) against r27_both, t >= 1
#   3 r27_both / R32 / R32 gated on 1109, 1111 (DataYes) and 1104-1106 (feitu)
#   bash research/r32.sh 2>&1 | tee data/reports/r32.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBD=$DATA/store/pm_factors/full_book_dysl2
BASE=$R/r27_both.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
M="--models /data/prod/models/tk5m --marks rand:240 --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
src() { case $1 in 2026110[4-6]|20260430) echo feitu ;; *) echo dysl2 ;; esac; }
fb() { [ "$(src $1)" = feitu ] && echo $FBF || echo $FBD; }
ls $TGT/date=20261112.parquet >/dev/null || { say "no 1112 target: run add_day.sh 20261112 first"; exit 1; }

say "1 scan on 1110 + 1112 from r27_both, soft family lock"
U=$(python -c "import json, polars as pl; a=json.load(open('$BASE')); f=pl.read_parquet('$R/stage5/fingerprint.parquet').sort('gain_share', descending=True)['feature'].unique(maintain_order=True).to_list(); print(','.join(x for x in f if x not in a))")
echo "  open at r27_both: $(echo $U | tr , '\n' | wc -l)"
CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261110,20261112 --assign $BASE --family-lock --family-lock-soft --calibrate \
  --siblings --scan-feats "$U" --implied 0 --full-book $FBD $M --out $R/stage6_r32 > $R/r32_scan.txt 2>&1
grep -E "Error|Traceback|^family lock|  family lock|^families with|IC: start" $R/r32_scan.txt | cut -c1-240 | head -60
python -u research/pm_factors/assign_diff.py $BASE $R/stage6_r32/ic_assign.json | cut -c1-200 | head -80

say "2 gate on 0430 against r27_both"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $R/stage6_r32/ic_assign.json --family-lock --family-lock-soft \
  --gate-against $BASE --gate-t 1 $SCORE --full-book $FBF $M --out $R/stage6_r32_gate > $R/r32_gate.txt 2>&1
grep -E "Error|Traceback|== gate|^gated" $R/r32_gate.txt
sed -n '/== gate/,/^gated/p' $R/r32_gate.txt | cut -c1-230 | head -120

say "3 held out"
printf "%-10s %8s %8s %8s\n" day r27both r32 r32g
for d in 20261109 20261111 20261104 20261105 20261106; do
  line="$(printf '%-10s' $d)"
  for tag in r27both r32 r32g; do
    case $tag in r27both) A=$BASE ;; r32) A=$R/stage6_r32/ic_assign.json ;; r32g) A=$R/stage6_r32_gate/ic_assign_gated.json ;; esac
    CRUCIBLE_SOURCE=$(src $d) python -u research/pm_factors/ic_score.py --dates $d --assign $A $SCORE --full-book $(fb $d) $M \
      --out $R/stage6_r32h_${d}_$tag > $R/r32h_${d}_$tag.txt 2>&1
    line="$line $(printf '%8s' "$(ic $R/r32h_${d}_$tag.txt)")"
  done
  echo "$line"
done
grep -lE "Error|Traceback" $R/r32*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
