#!/usr/bin/env bash
# Round 100 on 97, foreground, after R99. Writes data/side_r100, data/implied_scan_r100, data/reports; for 0430 also
# data/reports/stage6_r99_0430 + data/featgen_r99/base_20260430.parquet (links to R70 / R67) and, only if missing,
# PM's 0430 index mp (pm_mp/20260430/*_index.mp, read-only rsync from 2.23, a few MB) decoded into pmmp_raw. 30-60 min.
# zzeg (R98): training range [0.867, 1.097], 124 thresholds 0.985-1.030, never NaN in training, never a root split --
# a price ratio near 1; every calibrated / ranked candidate failed and a constant helped 1112. Now plain ratios
# (indices and stocks, research/featgen/ratio_extra.py) put in as they are, over every 224 day R99 built plus 0430
# (feitu stocks, the index from PM's own mp -- feitu_raw has no 0430 index stream).
#   bash research/r100.sh 2>&1 | tee data/reports/r100.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
H23=${H23:-root@172.21.2.23}
SSH="ssh -p ${P23:-60777} -o BatchMode=yes"
say() { echo "[$(date +%T)] $*"; }

say "0 0430: PM's index mp"
D=20260430
if [ ! -d $DATA/pmmp_raw/kind=index/date=$D ]; then
  SRC=$DATA/pm_mp_index; mkdir -p $SRC/$D  # index files only, so the decoder touches nothing else
  if ls /data/prod/mp/$D/*_index.mp >/dev/null 2>&1; then ln -sfn /data/prod/mp/$D/*_index.mp $SRC/$D/; else
    rsync -aL -e "$SSH" --include="*_index.mp" --exclude="*" $H23:/data/share/data/prod/mp/$D/ $SRC/$D/ 2>&1 | tail -2
  fi
  ls -l $SRC/$D/*_index.mp 2>&1 | awk '{print "  ", $5, $NF}'
  python -u research/decode_pm_mp.py --root $SRC --date $D --out $DATA/pmmp_raw --workers 4 2>&1 | tail -3 | cut -c1-200
fi
ls $DATA/pmmp_raw/kind=index/date=$D 2>&1 | sed 's/^/  /'
ln -sfn /work/crucible/data/reports/stage6_r70_0430 data/reports/stage6_r99_0430
mkdir -p data/featgen_r99 && ln -sfn /work/crucible/data/featgen_r67/base_$D.parquet data/featgen_r99/base_$D.parquet

DAYS=""
for D in 20260430 20261102 20261103 20261104 20261105 20261106 20261109 20261110 20261111 20261112 20261113; do
  O=data/reports/stage6_r99_${D:4}
  if [ -e $O/x.parquet ] && [ -e $O/pred.parquet ] && [ -e data/featgen_r99/base_$D.parquet ]; then
    say "ratios $D"
    if [ $D = 20260430 ]; then RAW="--raw $DATA/feitu_raw --index-raw $DATA/pmmp_raw"; else RAW="--raw $DATA/dysl2_raw"; fi
    python -u research/featgen/ratio_extra.py --date $D --x $O/x.parquet $RAW --out data/side_r100 2>&1 | grep -v UserWarning | grep -v "return grid" | tail -4 | cut -c1-300
    [ -e data/side_r100/ratio_$D.parquet ] && DAYS="$DAYS,$D"
  else
    echo "  $D: no matrix, skipped"
  fi
done
DAYS=${DAYS#,}
[ -n "$DAYS" ] || { say "no ratio file for any day: stop"; exit 1; }
say "implied scan of zzeg, raw ratios, days $DAYS"
python -u research/featgen/implied_scan.py --days $DAYS --target zzeg --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r100/ratio_{date}.parquet" --transforms "id,1/b" --models $DATA/models_gen/B --top 30 --work data/implied_scan_r100 2>&1 | cut -c1-600
say "done"
