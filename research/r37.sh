#!/usr/bin/env bash
# Round 37 on 97, foreground, after R36: cross-sectional and neutralised candidates, on 0430 (the one correct target).
# ic_score --neutral adds neu:<base> = the base's residual on THAT day's Barra industry + style exposures (the file changes daily),
# next to the z / rank / dm (market demean) / dmi (industry demean) variants already in the pool.
#   0 xs_screen: which open columns' training ranges look like a rank / z-score / demeaned value
#   1 scan every open column on 0430 grid 0 with --neutral (soft family lock), the screened ones first
#   2 gate on grid 7 against the base; 3 base / R37 / R37 gated on grid 7
#   bash research/r37.sh 2>&1 | tee data/reports/r37.txt
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
MB="--models /data/prod/models/tk5m --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV --neutral"
M0="$MB --marks rand:240"
M7="$MB --marks rand:240:7 --engine-dir data/engine_m7"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
[ -e data/engine_m7/marks_20260430.parquet ] || { say "no grid-7 engine state: run research/r34.sh first"; exit 1; }

say "0 xs_screen on $BASE"
python -u research/pm_factors/xs_screen.py --assign $BASE > $R/r37_screen.txt 2>&1
cut -c1-220 $R/r37_screen.txt | head -120
XS=$(grep '^XS_OPEN=' $R/r37_screen.txt | sed 's/^XS_OPEN=//')
U=$(python -c "import json, polars as pl; a=json.load(open('$BASE')); f=pl.read_parquet('$R/stage5/fingerprint.parquet').sort('gain_share', descending=True)['feature'].unique(maintain_order=True).to_list(); print(','.join(x for x in f if x not in a))")
FEATS=$(echo "$XS,$U" | tr , '\n' | awk 'NF && !s[$0]++' | paste -sd,)
echo "  scan order: $(echo $FEATS | tr , '\n' | wc -l) columns, screened first"

say "1 scan on 0430 grid 0 with neutralised candidates"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $BASE --family-lock --family-lock-soft --calibrate --siblings \
  --scan-feats "$FEATS" --implied 0 --full-book $FBF $M0 --out $R/stage6_r37 > $R/r37_scan.txt 2>&1
grep -E "Error|Traceback|IC: start" $R/r37_scan.txt | cut -c1-200
python -u research/pm_factors/assign_diff.py $BASE $R/stage6_r37/ic_assign.json | cut -c1-200 | head -80
echo "  neu: / dmi: / dm: / z: / rank picks:"; grep -oE '"[a-z0-9.]+": "(cal:)?(neu|dmi|dm|z|rank|crank|srank):[^"]*"' $R/stage6_r37/ic_assign.json | head -60

say "2 gate on 0430 grid 7"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $R/stage6_r37/ic_assign.json --family-lock --family-lock-soft \
  --gate-against $BASE --gate-t 1 $SCORE --full-book $FBF7 $M7 --out $R/stage6_r37_gate > $R/r37_gate.txt 2>&1
grep -E "Error|Traceback|== gate|^gated" $R/r37_gate.txt
sed -n '/== gate/,/^gated/p' $R/r37_gate.txt | cut -c1-230 | head -120

say "3 base / R37 / R37 gated on 0430 grid 7"
for tag in base r37 r37g; do
  case $tag in base) A=$BASE ;; r37) A=$R/stage6_r37/ic_assign.json ;; r37g) A=$R/stage6_r37_gate/ic_assign_gated.json ;; esac
  CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $A $SCORE --full-book $FBF7 $M7 \
    --out $R/stage6_r37h_$tag > $R/r37h_$tag.txt 2>&1
  echo "  $tag: $(ic $R/r37h_$tag.txt)"
done
grep -lE "Error|Traceback" $R/r37*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
