#!/usr/bin/env bash
# Round 132 on 97, foreground, read-only, writes only data/reports. ~10-20 min, no model scoring; parallel-safe.
# Model probe of the 8 "settled" columns R130's range check showed in the wrong units (our values far outside PM's training
# range): 1 what PM's trees say about each (range, thresholds, depth, root CDF, neighbours; featgen/col_probe.py),
# 2 the units fingerprint of every candidate we have against them, raw transforms incl. dm / srank / |x|*1e4 / logs.
#   bash research/r132.sh 2>&1 | tee data/reports/r132.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
COLS="zzqf.q.c3 zzrg zzwa.g zzag2.e zzag2.a zzro2.a zzop2.b zzqf7.q.c2"
for C in $COLS; do
  python -u research/featgen/col_probe.py --col $C --date 20261112 --models $DATA/models_gen/B 2>&1 | grep -vE "model files" | cut -c1-330
done
python -u research/featgen/fingerprint.py --days 20260430,20261112 --target "$(echo $COLS | tr ' ' ',')" --x-dir "data/reports/stage6_r99_{mmdd}" \
  --base "data/featgen_r99/base_{date}.parquet" --extra "data/side_r105/ratio_{date}.parquet,data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet,data/side_r95/hstats_prev_{date}.parquet" \
  --transforms=id,-id,1/b,dm,-dm,srank,abs4,log --models $DATA/models_gen/B --top 6 2>&1 | grep -v RuntimeWarning | cut -c1-200
echo "[$(date +%T)] done"
