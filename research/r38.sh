#!/usr/bin/env bash
# Round 38 on 97, foreground, after R37: how much of R34's jump on 0430 (0.377 -> 0.608 on held-out marks) survives on held-out
# STOCKS too. Redo R34 with disjoint halves: scan on even sids at grid 0, gate on odd sids at grid 7 (features still over every
# stock; only the IC is restricted, ic_score --sid-mod). Then r27_both / scan / gated on odd sids x grid 7 and even x grid 7.
#   bash research/r38.sh 2>&1 | tee data/reports/r38.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBF7=$DATA/store/pm_factors/full_book_m7
BASE=${BASE:-$R/r27_both.json}
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
MB="--models /data/prod/models/tk5m --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
M0="$MB --marks rand:240"
M7="$MB --marks rand:240:7 --engine-dir data/engine_m7"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
[ -e data/engine_m7/marks_20260430.parquet ] || { say "no grid-7 engine state: run research/r34.sh first"; exit 1; }

say "1 scan on 0430 grid 0, even sids only, from $BASE (soft family lock)"
U=$(python -c "import json, polars as pl; a=json.load(open('$BASE')); f=pl.read_parquet('$R/stage5/fingerprint.parquet').sort('gain_share', descending=True)['feature'].unique(maintain_order=True).to_list(); print(','.join(x for x in f if x not in a))")
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $BASE --family-lock --family-lock-soft --calibrate --siblings \
  --scan-feats "$U" --implied 0 --sid-mod 2:0 --full-book $FBF $M0 --out $R/stage6_r38 > $R/r38_scan.txt 2>&1
grep -E "Error|Traceback|sid-mod|IC: start" $R/r38_scan.txt | cut -c1-200

say "2 gate on 0430 grid 7, odd sids only"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $R/stage6_r38/ic_assign.json --family-lock --family-lock-soft \
  --gate-against $BASE --gate-t 1 --sid-mod 2:1 $SCORE --full-book $FBF7 $M7 --out $R/stage6_r38_gate > $R/r38_gate.txt 2>&1
grep -E "Error|Traceback|== gate|^gated" $R/r38_gate.txt

say "3 base / scan / gated / R34 gated, on grid 7: odd sids (held out twice) and even sids"
printf "%-12s %8s %8s %8s %8s\n" stocks base r38 r38g r34g
for half in 2:1 2:0; do
  line="$(printf '%-12s' "sid%$half")"
  for tag in base r38 r38g r34g; do
    case $tag in base) A=$BASE ;; r38) A=$R/stage6_r38/ic_assign.json ;; r38g) A=$R/stage6_r38_gate/ic_assign_gated.json ;; r34g) A=$R/stage6_r34_gate/ic_assign_gated.json ;; esac
    CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $A $SCORE --sid-mod $half --full-book $FBF7 $M7 \
      --out $R/stage6_r38h_${tag}_${half/:/_} > $R/r38h_${tag}_${half/:/_}.txt 2>&1
    line="$line $(printf '%8s' "$(ic $R/r38h_${tag}_${half/:/_}.txt)")"
  done
  echo "$line"
done
grep -lE "Error|Traceback" $R/r38*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
