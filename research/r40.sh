#!/usr/bin/env bash
# Round 40 on 97, foreground: consolidate and refine on 0430 with the strict split (scan on even sids x grid 0, gate on odd
# sids x grid 7, report both halves on grid 7). Writes under /work only.
#   0 base = R37b gated (141 columns, neutralisation working) minus R39's two drops (zzko, zzav2)
#   1 rescan the columns R39 showed carry most (the short-term imbalance / microprice / reversal ones) for in-family variants:
#     multi-level / depth-weighted / order-count imbalances, other windows, z / rank / dm / dmi / neu -- kept only if better
#   2 gate on odd sids x grid 7;  3 base / scan / gated on grid 7, both halves
#   bash research/r40.sh 2>&1 | tee data/reports/r40.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
TGT=$DATA/pm_target_r22
HV=$DATA/hstats_view/HS300
FBF=$DATA/store/pm_factors/full_book
FBF7=$DATA/store/pm_factors/full_book_m7
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
MB="--models /data/prod/models/tk5m --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV --neutral"
M0="$MB --marks rand:240"
M7="$MB --marks rand:240:7 --engine-dir data/engine_m7"
SCORE="--no-drop-harmful --max-features 0 --implied 0"
COLS=${COLS:-zzfb2.a,zzpz.a,zzwa.h,zzyg2.0.b,zzeu2,zzeg2,zzqf.q.c3,zzxc.b,zzpz.d,zzwa.g,zzpe.c,zzja,zzfb.b,zzuc.d,zzhx.d,zzqo.e,zzxt3,zzvm2,zzyj3.c,zzpz.b,zzlf}

say "0 base = R37b gated minus zzko, zzav2"
python - <<'EOF'
import json
a = json.load(open("data/reports/stage6_r37_gate/ic_assign_gated.json"))
for k in ("zzko", "zzav2"):
    a.pop(k, None)
json.dump(a, open("data/reports/r40_base.json", "w"), indent=1)
print(f"  r40_base: {len(a)} columns")
EOF
BASE=$R/r40_base.json

say "1 rescan $COLS: even sids x grid 0"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $BASE --calibrate --siblings --scan-feats "$COLS" --implied 0 \
  --sid-mod 2:0 --full-book $FBF $M0 --out $R/stage6_r40 > $R/r40_scan.txt 2>&1
grep -E "Error|Traceback|IC: start" $R/r40_scan.txt | cut -c1-200
python -u research/pm_factors/assign_diff.py $BASE $R/stage6_r40/ic_assign.json | cut -c1-200 | head -40

say "2 gate: odd sids x grid 7"
CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $R/stage6_r40/ic_assign.json --gate-against $BASE --gate-t 1 \
  --sid-mod 2:1 $SCORE --full-book $FBF7 $M7 --out $R/stage6_r40_gate > $R/r40_gate.txt 2>&1
grep -E "Error|Traceback|^gated" $R/r40_gate.txt
sed -n '/== gate/,/^gated/p' $R/r40_gate.txt | cut -c1-230 | head -50

say "3 grid 7, both halves"
printf "%-10s %8s %8s %8s %8s\n" stocks r34g r40base r40 r40g
for h in 2:1 2:0; do
  line="$(printf '%-10s' "sid%$h")"
  for tag in r34g r40base r40 r40g; do
    case $tag in r34g) A=$R/r34g_baseline.json ;; r40base) A=$BASE ;; r40) A=$R/stage6_r40/ic_assign.json ;; r40g) A=$R/stage6_r40_gate/ic_assign_gated.json ;; esac
    CRUCIBLE_SOURCE=feitu python -u research/pm_factors/ic_score.py --dates 20260430 --assign $A $SCORE --sid-mod $h --full-book $FBF7 $M7 \
      --out $R/stage6_r40h_${tag}_${h/:/_} > $R/r40h_${tag}_${h/:/_}.txt 2>&1
    line="$line $(printf '%8s' "$(ic $R/r40h_${tag}_${h/:/_}.txt)")"
  done
  echo "$line"
done
cp -n $R/stage6_r40_gate/ic_assign_gated.json $R/r40g_baseline.json 2>/dev/null
grep -lE "Error|Traceback" $R/r40*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
