#!/usr/bin/env bash
# Round 29 on 97, foreground, after R28. Reads 2.23 only; writes under /work.
#   0 2.23 livev1/rec10 (the other recording dir, next to rec10_p): anything for the November days?
#   1 gen_flags' moved columns (their thresholds moved a whole span between generations A/B and C): dropped one at a time and
#     all together from R27's picked assignment; 0430 runs on generation A, 1110 on C -- a column that helps only after
#     the drop on 0430 means a different quantity in generation A
#   bash research/r29.sh 2>&1 | tee data/reports/r29.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBD=$DATA/store/pm_factors/full_book_dysl2
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
M="--models /data/prod/models/tk5m --marks rand:240 --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
src() { case $1 in 2026110[4-6]|20260430) echo feitu ;; *) echo dysl2 ;; esac; }
fb() { [ "$(src $1)" = feitu ] && echo $FBF || echo $FBD; }

say "0 2.23 livev1/rec10"
ssh -p $P23 -o BatchMode=yes $H23 'find /data/share/data/prod/h/HS300/livev1 -maxdepth 4 -printf "%y %TY-%Tm-%Td %TH:%TM %s %p\n" 2>/dev/null | sort -k5 | tail -40' \
  > $R/r29_livev1.txt 2>&1 || say "  listing failed: $(tail -1 $R/r29_livev1.txt)"
cut -c1-200 $R/r29_livev1.txt

say "1 moved columns dropped from R27's picks, on 0430 (gen A) and 1110 (gen C)"
python -u research/pm_factors/r29_moved.py --base ${BASE:-$R/r27_both.json} --out $R/r29_var
for v in $(ls $R/r29_var/*.json | xargs -n1 basename | sed 's/\.json$//'); do
  for d in 20260430 20261110; do
    CRUCIBLE_SOURCE=$(src $d) python -u research/pm_factors/ic_score.py --dates $d --assign $R/r29_var/$v.json $SCORE --full-book $(fb $d) $M \
      --out $R/stage6_r29v_${v}_$d > $R/r29v_${v}_$d.txt 2>&1
  done
done
python -u research/pm_factors/r27_pick.py --base $R/r29_var/base.json --var-dir $R/r29_var --prefix $R/r29v_ --days 20260430 \
  --exclude base --min-gain 0 --out $R/r29_pick_0430.json 2>&1 | cut -c1-200
echo "  (same changes on 1110, generation C):"
python -u research/pm_factors/r27_pick.py --base $R/r29_var/base.json --var-dir $R/r29_var --prefix $R/r29v_ --days 20261110 \
  --exclude base --min-gain 0 --out $R/r29_pick_1110.json 2>&1 | cut -c1-200
grep -lE "Error|Traceback" $R/r29*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
