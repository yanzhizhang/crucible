#!/usr/bin/env bash
# Round 34 on 97, foreground. Only PM's production on 2.23 is correct (every 224 signal is unreliable), and production gives
# one prediction day: 0430 (livev1 alpha). So scan and gate on 0430 alone, on two DIFFERENT mark grids of that day to
# limit overfitting: scan on rand:240 (seed 0, the grid used so far), gate and test on rand:240:7. Soft family lock.
# The 224 days are printed for reference only. Writes under /work only.
#   0 the second grid for 0430: engine state at the marks (true seed 0429.jz) and whole-book factors
#   1 scan the columns open at r27_both + soft-lock rescans, 0430 grid 0
#   2 gate each change on 0430 grid 7 against r27_both (t >= 1)
#   3 r27_both / R34 / R34 gated on 0430 grid 7 (held-out marks), and 1110 / 1112 (224, reference only)
#   bash research/r34.sh 2>&1 | tee data/reports/r34.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBF7=$DATA/store/pm_factors/full_book_m7
FBD=$DATA/store/pm_factors/full_book_dysl2
DCP=/data/prod/h/HS300/p1_intrading_dcp
BASE=$R/r27_both.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
MB="--models /data/prod/models/tk5m --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
M0="$MB --marks rand:240"
M7="$MB --marks rand:240:7 --engine-dir data/engine_m7"
SCORE="--no-drop-harmful --max-features 0 --implied 0"

say "0 grid rand:240:7 for 0430"
[ -e data/engine_m7/marks_20260430.parquet ] || python -u research/engine/run_cpp.py --date 20260430 --universe pm:$DCP/20260429.jz --prev $DCP/20260429.jz \
  --marks rand:240:7 --threads 32 --out data/engine_m7 --work data/engine_m7/work > $R/r34_engine.txt 2>&1 || say "  engine FAILED: $(tail -3 $R/r34_engine.txt)"
[ -e $FBF7/date=20260430.parquet ] || python -u research/pm_factors/full_book.py --dates 20260430 --signals $DATA/pm_alpha --marks rand:240:7 --workers 1 \
  --out $FBF7 > $R/r34_fullbook.txt 2>&1 || say "  full_book FAILED: $(tail -3 $R/r34_fullbook.txt)"
ls -la data/engine_m7/marks_20260430.parquet $FBF7/date=20260430.parquet 2>&1 | sed 's/^/  /'

say "1 scan on 0430 grid 0 from r27_both, soft family lock"
U=$(python -c "import json, polars as pl; a=json.load(open('$BASE')); f=pl.read_parquet('$R/stage5/fingerprint.parquet').sort('gain_share', descending=True)['feature'].unique(maintain_order=True).to_list(); print(','.join(x for x in f if x not in a))")
echo "  open at r27_both: $(echo $U | tr , '\n' | wc -l)"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $BASE --family-lock --family-lock-soft --calibrate \
  --siblings --scan-feats "$U" --implied 0 --full-book $FBF $M0 --out $R/stage6_r34 > $R/r34_scan.txt 2>&1
grep -E "Error|Traceback|^family lock|  family lock|^families with|IC: start" $R/r34_scan.txt | cut -c1-240 | head -60
python -u research/pm_factors/assign_diff.py $BASE $R/stage6_r34/ic_assign.json | cut -c1-200 | head -80

say "2 gate on 0430 grid 7 against r27_both"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $R/stage6_r34/ic_assign.json --family-lock --family-lock-soft \
  --gate-against $BASE --gate-t 1 $SCORE --full-book $FBF7 $M7 --out $R/stage6_r34_gate > $R/r34_gate.txt 2>&1
grep -E "Error|Traceback|== gate|^gated" $R/r34_gate.txt
sed -n '/== gate/,/^gated/p' $R/r34_gate.txt | cut -c1-230 | head -120

say "3 r27_both / R34 / R34 gated: 0430 grid 7 (held-out marks); 1110 / 1112 = 224, reference only"
printf "%-16s %8s %8s %8s\n" day r27both r34 r34g
for dg in 20260430:7 20261110:0 20261112:0; do
  d=${dg%:*}; g=${dg#*:}
  line="$(printf '%-16s' "$d grid$g")"
  for tag in r27both r34 r34g; do
    case $tag in r27both) A=$BASE ;; r34) A=$R/stage6_r34/ic_assign.json ;; r34g) A=$R/stage6_r34_gate/ic_assign_gated.json ;; esac
    if [ $g = 7 ]; then MM=$M7; FB=$FBF7; SRC=feitu; else MM=$M0; FB=$FBD; SRC=dysl2; fi
    CRUCIBLE_SOURCE=$SRC python -u research/pm_factors/ic_score.py --dates $d --assign $A $SCORE --full-book $FB $MM \
      --out $R/stage6_r34h_${d}_$tag > $R/r34h_${d}_$tag.txt 2>&1
    line="$line $(printf '%8s' "$(ic $R/r34h_${d}_$tag.txt)")"
  done
  echo "$line"
done
grep -lE "Error|Traceback" $R/r34*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
