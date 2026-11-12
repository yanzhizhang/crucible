#!/usr/bin/env bash
# Round 30 on 97, foreground. ic_score predicts with model_s1 only (--subs s1) at every mark. lgb_label_probe: s1..s4 have
# non-monotone scales (s1 > s2 > s4 > s3), like time-of-day segments, not horizons -- so PM may use s<k> only in its own
# part of the day. Score r27_both with each sub alone on 0430 (production, true seed) and 1110 (224), with the half-hour
# IC table: the sub that agrees best with PM in a half hour is the one PM ran there. Writes under /work only.
#   bash research/r30.sh 2>&1 | tee data/reports/r30.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBD=$DATA/store/pm_factors/full_book_dysl2
A=${ASSIGN:-$R/r27_both.json}
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
M="--models /data/prod/models/tk5m --marks rand:240 --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
src() { case $1 in 2026110[4-6]|20260430) echo feitu ;; *) echo dysl2 ;; esac; }
fb() { [ "$(src $1)" = feitu ] && echo $FBF || echo $FBD; }

printf "%-10s %8s %8s %8s %8s %8s\n" day s1 s2 s3 s4 all4
for d in 20260430 20261110; do
  line="$(printf '%-10s' $d)"
  for s in s1 s2 s3 s4 s1,s2,s3,s4; do
    tag=$(echo $s | tr -d ,); [ "$tag" = s1s2s3s4 ] && tag=all4
    CRUCIBLE_SOURCE=$(src $d) python -u research/pm_factors/ic_score.py --dates $d --assign $A $SCORE --subs $s --full-book $(fb $d) $M \
      --dump-pred --out $R/stage6_r30_${d}_$tag > $R/r30_${d}_$tag.txt 2>&1
    line="$line $(printf '%8s' "$(ic $R/r30_${d}_$tag.txt)")"
  done
  echo "$line"
done
say "half-hour IC per sub (min from 09:30)"
for d in 20260430 20261110; do
  for tag in s1 s2 s3 s4 all4; do
    echo "  -- $d $tag"
    python -u research/pm_factors/residual_breakdown.py --pred $R/stage6_r30_${d}_$tag/pred.parquet 2>&1 | sed -n '/per-mark IC by half hour/,/by board/p' | head -14
  done
done
grep -lE "Error|Traceback" $R/r30_*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
