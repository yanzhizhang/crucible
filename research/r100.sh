#!/usr/bin/env bash
# Round 100 on 97, foreground, after R99. Writes data/side_r100, data/implied_scan_r100, data/reports. About 30-60 minutes.
# zzeg (R98): training range [0.867, 1.097], 124 thresholds 0.985-1.030, never NaN in training, never a root split --
# a price ratio near 1; every calibrated / ranked candidate failed and a constant helped 1112. Now plain ratios
# (indices and stocks, research/featgen/ratio_extra.py) put in as they are (no calibration), over every 224 day R99 built.
#   bash research/r100.sh 2>&1 | tee data/reports/r100.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
DAYS=""
for D in 20261102 20261103 20261104 20261105 20261106 20261109 20261110 20261111 20261112 20261113; do
  O=data/reports/stage6_r99_${D:4}
  if [ -e $O/x.parquet ] && [ -e $O/pred.parquet ] && [ -e data/featgen_r99/base_$D.parquet ]; then
    echo "[$(date +%T)] ratios $D"
    python -u research/featgen/ratio_extra.py --date $D --x $O/x.parquet --out data/side_r100 2>&1 | tail -2 | cut -c1-300
    [ -e data/side_r100/ratio_$D.parquet ] && DAYS="$DAYS,$D"
  else
    echo "  $D: no R99 matrix, skipped"
  fi
done
DAYS=${DAYS#,}
echo "[$(date +%T)] implied scan of zzeg, raw ratios, days $DAYS"
python -u research/featgen/implied_scan.py --days $DAYS --target zzeg --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r100/ratio_{date}.parquet" --transforms "id,1/b" --models $DATA/models_gen/B --top 30 --work data/implied_scan_r100 2>&1 | cut -c1-400
echo "[$(date +%T)] done"
