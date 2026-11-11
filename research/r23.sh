#!/usr/bin/env bash
# Round 23 on 97, foreground, from the R17 assignment (the last one no later round beat out of sample). Train on
# 1104-1106 (feitu), gate on 1109, test on 1110 (DataYes, the 224 days) and 0430 (feitu, PM's alpha):
#   0 history stats for 0430 / 1109 / 1110 from 2.23 (read-only) into a /work view next to /data/prod's
#   1 do the model generations mean the same by each column (gen_compare); PM state keys of the open families (state_keys)
#   2 engine state at the marks for 1109 / 1110 (1104-1106 and 0430 have theirs)
#   3 invert PM's model for the 15 open columns with the most gain (--implied) + residual breakdown, train days and 224 days
#   4 scan the columns open at R17, siblings' formulas included (--siblings)
#   5 gate each change on 1109 (kept only when it beats R17's formula there, t >= 1)
#   6 R17 / R23 / R23 gated on 1110 and 0430
#   bash research/r23.sh 2>&1 | tee data/reports/r23.txt
set -u
cd /work/crucible
R=data/reports
D=/work/crucible_data
DCP=/data/prod/h/HS300/p1_intrading_dcp
TGT=$D/pm_target_r22  # 0430 -> PM's alpha, the November days -> the 224 signals
HV=$D/hstats_view/HS300
FBF=$D/store/pm_factors/full_book
FBD=$D/store/pm_factors/full_book_dysl2
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
R17=$R/stage6_r17/ic_assign.json
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\1/'; }
M="--models /data/prod/models/tk5m --marks rand:240 --target $TGT --target-stale 4 --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV"
SCORE="--no-drop-harmful --max-features 0 --implied 0"

say "0 history stats view: /data/prod's days linked, 0430 / 1109 / 1110 copied from 2.23"
mkdir -p $HV $D/hstats_extra/HS300
for fam in $(ls /data/prod/hstats/HS300); do mkdir -p $HV/$fam; for x in /data/prod/hstats/HS300/$fam/*; do ln -sfn $x $HV/$fam/; done; done
for day in 20260430 20261109 20261110; do
  rsync -aLm -e "ssh -p $P23 -o BatchMode=yes" --include="*/" --include="$day/***" --exclude="*" $H23:/data/share/data/prod/hstats/HS300/ $D/hstats_extra/HS300/ \
    > $R/r23_rsync_$day.txt 2>&1 || say "  rsync $day from 2.23 failed: $(tail -1 $R/r23_rsync_$day.txt)"
done
for fam in $(ls $D/hstats_extra/HS300 2>/dev/null); do mkdir -p $HV/$fam; for x in $D/hstats_extra/HS300/$fam/*; do [ -e $HV/$fam/$(basename $x) ] || ln -sfn $x $HV/$fam/; done; done
for day in 20260430 20261104 20261109 20261110; do echo "  $day: $(ls -d $HV/*/$day/samplerS.nc 2>/dev/null | wc -l) families with samplerS.nc"; done

say "1 model generations, open families' state keys"
python -u research/pm_factors/gen_compare.py --models /data/prod/models/tk5m --fit livev1_fit1 --out $R/gen_compare.csv 2>&1 | head -60
python -u research/pm_factors/state_keys.py --state $DCP/20261106.jz --assign $R17 --out $R/state_keys.csv > $R/r23_state_keys.txt 2>&1
head -3 $R/r23_state_keys.txt; echo "  (full list: $R/r23_state_keys.txt, $(wc -l < $R/r23_state_keys.txt) lines)"

say "2 engine state at the marks for 1109 / 1110 (DataYes)"
python -u research/engine/run_cpp.py --date 20261109 --universe pm:$DCP/20261106.jz --prev $DCP/20261106.jz --catalog $D/catalog_dysl2.duckdb \
  --marks rand:240 --threads 32 > $R/r23_engine_1109.txt 2>&1 || say "  engine 1109 FAILED: $(tail -3 $R/r23_engine_1109.txt)"
python -u research/engine/run_cpp.py --date 20261110 --universe pm:$DCP/20261106.jz --prev data/engine/engine_20261109.json.gz --catalog $D/catalog_dysl2.duckdb \
  --marks rand:240 --threads 32 > $R/r23_engine_1110.txt 2>&1 || say "  engine 1110 FAILED: $(tail -3 $R/r23_engine_1110.txt)"
tail -1 $R/r23_engine_1109.txt; tail -1 $R/r23_engine_1110.txt

say "3 implied values of the 15 open columns with the most gain (R17), plus the residual breakdown"
python -u research/pm_factors/ic_score.py --dates 20261104,20261105,20261106 --assign $R17 $SCORE --full-book $FBF $M --implied 15 --dump-pred \
  --out $R/stage6_r23_implied > $R/r23_implied.txt 2>&1
CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261109,20261110 --assign $R17 $SCORE --full-book $FBD $M --implied 15 --dump-pred \
  --out $R/stage6_r23_implied_224 > $R/r23_implied_224.txt 2>&1
grep -hE "Error|Traceback|stocks x .* marks|IC: start" $R/r23_implied.txt $R/r23_implied_224.txt | cut -c1-200
for f in $R/r23_implied.txt $R/r23_implied_224.txt; do echo "  -- $f"; sed -n '/== implied values/,$p' $f | grep -vE "^time " | cut -c1-260 | head -80; done
python -u research/pm_factors/residual_breakdown.py --pred $R/stage6_r23_implied/pred.parquet 2>&1 | cut -c1-200
python -u research/pm_factors/residual_breakdown.py --pred $R/stage6_r23_implied_224/pred.parquet 2>&1 | cut -c1-200

say "4 scan the columns open at R17 on 1104-1106, siblings' formulas included"
U=$(python -c "import json, polars as pl; a=json.load(open('$R17')); f=pl.read_parquet('$R/stage5/fingerprint.parquet').sort('gain_share', descending=True)['feature'].unique(maintain_order=True).to_list(); print(','.join(x for x in f if x not in a))")
echo "  open at R17: $(echo $U | tr , '\n' | wc -l)"
python -u research/pm_factors/ic_score.py --dates 20261104,20261105,20261106 --assign $R17 --calibrate --siblings --scan-feats "$U" --implied 0 \
  --full-book $FBF $M --out $R/stage6_r23 > $R/r23_scan.txt 2>&1
grep -E "Error|Traceback|^dropped|IC: start" $R/r23_scan.txt | cut -c1-240
python -c "import json; a=json.load(open('$R/stage6_r23/ic_assign.json')); b=json.load(open('$R17')); print('  ', len(b), '->', len(a)); [print('  ', k, ':', b.get(k), '=>', v) for k, v in a.items() if b.get(k) != v]; [print('  ', k, ':', v, '=> dropped') for k, v in b.items() if k not in a]"

say "5 gate on 1109: each change kept only when it beats R17's formula there"
CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261109 --assign $R/stage6_r23/ic_assign.json --gate-against $R17 --gate-t 1 $SCORE \
  --full-book $FBD $M --out $R/stage6_r23_gate > $R/r23_gate.txt 2>&1
grep -E "Error|Traceback|== gate|^gated" $R/r23_gate.txt
sed -n '/== gate/,/^gated/p' $R/r23_gate.txt | cut -c1-230 | head -90

say "6 held out: R17 / R23 / R23 gated on 1110 (DataYes) and 0430 (feitu, PM's alpha)"
for tag in r17 r23 r23g; do
  case $tag in r17) A=$R17 ;; r23) A=$R/stage6_r23/ic_assign.json ;; r23g) A=$R/stage6_r23_gate/ic_assign_gated.json ;; esac
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261110 --assign $A $SCORE --full-book $FBD $M --out $R/stage6_r23_1110_$tag > $R/r23_1110_$tag.txt 2>&1
  python -u research/pm_factors/ic_score.py --dates 20260430 --assign $A $SCORE --full-book $FBF $M --out $R/stage6_r23_0430_$tag > $R/r23_0430_$tag.txt 2>&1
  say "  $tag ($(python -c "import json; print(len(json.load(open('$A'))))") columns): 1110 $(ic $R/r23_1110_$tag.txt), 0430 $(ic $R/r23_0430_$tag.txt)"
done
grep -hE "Error|Traceback|profiles for|engine state for" $R/r23_1110_r17.txt $R/r23_0430_r17.txt | cut -c1-200
say "done"
