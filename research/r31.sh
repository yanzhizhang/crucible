#!/usr/bin/env bash
# Round 31 on 97, foreground, after R30. seg_probe: the count-like columns' training maxima stand 1 : 3 : 21 : 23.7 across
# s1..s4 in both generations = 600 / 1800 / 12600 / 14220 s on the trading clock (09:40 / 10:00 / 14:30 / close), and PM's
# 0430 alpha std steps at 09:40, 10:00 and ~14:30. So score r27_both with each mark predicted by its own segment's
# sub-model (ic_score --sub-by-time) against s1 everywhere, on every day; half-hour tables for 0430 and 1110.
#   bash research/r31.sh 2>&1 | tee data/reports/r31.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBD=$DATA/store/pm_factors/full_book_dysl2
A=${ASSIGN:-$R/r27_both.json}
SEG=${SEG:-s1:600,s2:1800,s3:12600,s4:14220}
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
M="--models /data/prod/models/tk5m --marks rand:240 --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
src() { case $1 in 2026110[4-6]|20260430) echo feitu ;; *) echo dysl2 ;; esac; }
fb() { [ "$(src $1)" = feitu ] && echo $FBF || echo $FBD; }

say "segments $SEG; assignment $A"
printf "%-10s %8s %8s\n" day s1_only by_time
for d in 20260430 20261110 20261109 20261104 20261105 20261106 20261111; do
  line="$(printf '%-10s' $d)"
  for mode in s1 seg; do
    [ $mode = seg ] && F="--sub-by-time $SEG" || F="--subs s1"
    CRUCIBLE_SOURCE=$(src $d) python -u research/pm_factors/ic_score.py --dates $d --assign $A $SCORE $F --full-book $(fb $d) $M \
      --dump-pred --out $R/stage6_r31_${d}_$mode > $R/r31_${d}_$mode.txt 2>&1
    line="$line $(printf '%8s' "$(ic $R/r31_${d}_$mode.txt)")"
  done
  echo "$line"
done
say "half-hour IC, s1 only vs by time"
for d in 20260430 20261110 20261109; do
  for mode in s1 seg; do
    echo "  -- $d $mode"
    python -u research/pm_factors/residual_breakdown.py --pred $R/stage6_r31_${d}_$mode/pred.parquet 2>&1 | sed -n '/per-mark IC by half hour/,/by board/p' | head -14
  done
done
grep -h "sub-models by time" $R/r31_*_seg.txt | sort -u | head -3
grep -lE "Error|Traceback" $R/r31_*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
