#!/usr/bin/env bash
# Round 24 on 97, foreground, from R17. Two changes in ic_score:
#   * cal: maps through the training CDF of each day's OWN model generation (0430 = the 202509 models reused to 202604,
#     November = the 202609 ones); before, the CDF points of every probed month were pooled
#   * --family-lock: every column of one PM family (zzro, zzro2 ...) uses one quantity class (ret / ema_dev / ref_px / imb ...);
#     off-class picks are dropped and rescanned in class, the gate never restores an off-class formula
# Steps:
#   0 history stats view (as R23; skipped when built) + what 2.23:/data/share/s/ (224's stsfile) holds, listing only
#   1 engine state at the marks for 1109 / 1110 (skipped when R23 built it)
#   2 calibration A/B: R17 scored on 0430 / 1109 / 1110 with the pooled CDF and with the per-generation one
#   3 scan the open columns + the family-lock drops on 1104-1106 (feitu)
#   4 gate each change on 1109 against R17 (in class only)
#   5 R17 / R24 / R24 gated on 1110 and 0430
#   bash research/r24.sh 2>&1 | tee data/reports/r24.txt
set -u
cd /work/crucible
R=data/reports
D=/work/crucible_data
DCP=/data/prod/h/HS300/p1_intrading_dcp
TGT=$D/pm_target_r22
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
src() { case $1 in 2026110[4-6]|20260430) echo feitu ;; *) echo dysl2 ;; esac; }
fb() { [ "$(src $1)" = feitu ] && echo $FBF || echo $FBD; }

say "0 history stats view; 2.23:/data/share/s/ listing"
if [ ! -d $HV ]; then
  mkdir -p $HV $D/hstats_extra/HS300
  for fam in $(ls /data/prod/hstats/HS300); do mkdir -p $HV/$fam; for x in /data/prod/hstats/HS300/$fam/*; do ln -sfn $x $HV/$fam/; done; done
  for day in 20260430 20261109 20261110; do
    rsync -aLm -e "ssh -p $P23 -o BatchMode=yes" --include="*/" --include="$day/***" --exclude="*" $H23:/data/share/data/prod/hstats/HS300/ $D/hstats_extra/HS300/ \
      > $R/r24_rsync_$day.txt 2>&1 || say "  rsync $day from 2.23 failed: $(tail -1 $R/r24_rsync_$day.txt)"
  done
  for fam in $(ls $D/hstats_extra/HS300 2>/dev/null); do mkdir -p $HV/$fam; for x in $D/hstats_extra/HS300/$fam/*; do [ -e $HV/$fam/$(basename $x) ] || ln -sfn $x $HV/$fam/; done; done
fi
for day in 20260430 20261104 20261109 20261110; do echo "  $day: $(ls -d $HV/*/$day/samplerS.nc 2>/dev/null | wc -l) families with samplerS.nc"; done
ssh -p $P23 -o BatchMode=yes $H23 'ls -la /data/share/s | head -30; echo ...; find /data/share/s -maxdepth 3 | head -40; du -sh /data/share/s 2>/dev/null' \
  > $R/r24_sts_listing.txt 2>&1 || say "  listing 2.23:/data/share/s failed: $(tail -1 $R/r24_sts_listing.txt)"
head -45 $R/r24_sts_listing.txt

say "1 engine state for 1109 / 1110"
[ -e data/engine/marks_20261109.parquet ] || python -u research/engine/run_cpp.py --date 20261109 --universe pm:$DCP/20261106.jz --prev $DCP/20261106.jz \
  --catalog $D/catalog_dysl2.duckdb --marks rand:240 --threads 32 > $R/r24_engine_1109.txt 2>&1 || say "  engine 1109 FAILED: $(tail -3 $R/r24_engine_1109.txt)"
[ -e data/engine/marks_20261110.parquet ] || python -u research/engine/run_cpp.py --date 20261110 --universe pm:$DCP/20261106.jz --prev data/engine/engine_20261109.json.gz \
  --catalog $D/catalog_dysl2.duckdb --marks rand:240 --threads 32 > $R/r24_engine_1110.txt 2>&1 || say "  engine 1110 FAILED: $(tail -3 $R/r24_engine_1110.txt)"
ls -la data/engine/marks_2026110[9].parquet data/engine/marks_20261110.parquet 2>&1

say "2 calibration A/B on R17: pooled CDF vs each day's own generation"
for day in 20260430 20261109 20261110; do
  for cv in pooled gen; do
    [ $cv = pooled ] && F=--no-cdf-by-generation || F=--cdf-by-generation
    CRUCIBLE_SOURCE=$(src $day) python -u research/pm_factors/ic_score.py --dates $day --assign $R17 $SCORE $F --full-book $(fb $day) $M \
      --out $R/stage6_r24_cal_${day}_$cv > $R/r24_cal_${day}_$cv.txt 2>&1
  done
  say "  $day: pooled $(ic $R/r24_cal_${day}_pooled.txt), own generation $(ic $R/r24_cal_${day}_gen.txt)"
done
grep -hE "Error|Traceback|model generation|training CDF by generation" $R/r24_cal_*_gen.txt | sort -u | cut -c1-200

say "3 scan on 1104-1106: the columns open at R17 + the family-lock drops, in class"
U=$(python -c "import json, polars as pl; a=json.load(open('$R17')); f=pl.read_parquet('$R/stage5/fingerprint.parquet').sort('gain_share', descending=True)['feature'].unique(maintain_order=True).to_list(); print(','.join(x for x in f if x not in a))")
echo "  open at R17: $(echo $U | tr , '\n' | wc -l)"
python -u research/pm_factors/ic_score.py --dates 20261104,20261105,20261106 --assign $R17 --family-lock --calibrate --siblings --scan-feats "$U" --implied 0 \
  --full-book $FBF $M --out $R/stage6_r24 > $R/r24_scan.txt 2>&1
grep -E "Error|Traceback|^family lock|  family lock|^families with|^  zz|IC: start" $R/r24_scan.txt | cut -c1-240
python -c "import json; a=json.load(open('$R/stage6_r24/ic_assign.json')); b=json.load(open('$R17')); print('  ', len(b), '->', len(a)); [print('  ', k, ':', b.get(k), '=>', v) for k, v in a.items() if b.get(k) != v]; [print('  ', k, ':', v, '=> dropped') for k, v in b.items() if k not in a]"

say "4 gate on 1109 against R17, in class only"
CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/ic_score.py --dates 20261109 --assign $R/stage6_r24/ic_assign.json --family-lock --gate-against $R17 --gate-t 1 \
  $SCORE --full-book $FBD $M --out $R/stage6_r24_gate > $R/r24_gate.txt 2>&1
grep -E "Error|Traceback|== gate|^gated|^families with" $R/r24_gate.txt
sed -n '/== gate/,/^gated/p' $R/r24_gate.txt | cut -c1-230 | head -90

say "5 held out: R17 / R24 / R24 gated on 1110 and 0430 (own-generation CDF)"
for tag in r17 r24 r24g; do
  case $tag in r17) A=$R17 ;; r24) A=$R/stage6_r24/ic_assign.json ;; r24g) A=$R/stage6_r24_gate/ic_assign_gated.json ;; esac
  for day in 20261110 20260430; do
    CRUCIBLE_SOURCE=$(src $day) python -u research/pm_factors/ic_score.py --dates $day --assign $A $SCORE --full-book $(fb $day) $M \
      --out $R/stage6_r24_${day}_$tag > $R/r24_${day}_$tag.txt 2>&1
  done
  say "  $tag ($(python -c "import json; print(len(json.load(open('$A'))))") columns): 1110 $(ic $R/r24_20261110_$tag.txt), 0430 $(ic $R/r24_20260430_$tag.txt)"
done
grep -hE "Error|Traceback" $R/r24_2026*_r*.txt | sort -u | head
say "done"
