#!/usr/bin/env bash
# Round 111 on 97, foreground, read-only on inputs; writes data/side_r111, data/implied_scan_r111, data/reports. ~10-20 min.
# R110: px / span is wider than zzeg's thresholds (10% of rows outside 0.985-1.030), so zzeg is not px against the raw
# channel. PM keeps zzeg.c.rEma = a decayed average of the span (sum / wgt within cents of the span at the close): fit its
# clock to the .jz (zzeg_rema.py: half-life x start x seed, day-end wgt and value), then rEma / span, px / rEma ...
# against zzeg (fingerprint, then scan; 0430 uses the half-life / start fitted on the 224 days).
#   bash research/r111.sh 2>&1 | tee data/reports/r111.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
P1=/data/prod/h/HS300/p1_intrading_dcp
fit() {  # day, jz, prev jz
  python -u research/featgen/zzeg_rema.py --date $1 --x data/reports/stage6_r99_${1:4}/x.parquet --raw $DATA/dysl2_raw --jz $2 --prev-jz $3 --out data/side_r111 2>&1 | grep -vE "UserWarning|return grid"
}
echo "[$(date +%T)] fit on 1105, 1106 (PM state in $P1) and 1112, 1113 (seed_probe)"
fit 20261105 $P1/20261105.jz $P1/20261104.jz | tee data/reports/r111_1105.txt
fit 20261106 $P1/20261106.jz $P1/20261105.jz
fit 20261112 $SP/20261112.jz $SP/20261111.jz
fit 20261113 $SP/20261113.jz $SP/20261112.jz | tee data/reports/r111_1113.txt
B=$(grep -o "best half [0-9]*s start [0-9]*" data/reports/r111_1113.txt | head -1)
H=$(echo "$B" | awk '{print $3}' | tr -d s); S=$(echo "$B" | awk '{print $5}')
echo "[$(date +%T)] 0430 with half ${H}s start $S (fitted on 1113)"
python -u research/featgen/zzeg_rema.py --date 20260430 --x data/reports/stage6_r99_0430/x.parquet --raw $DATA/feitu_raw --halves ${H:-240} --start ${S:-0930} \
  --out data/side_r111 2>&1 | grep -vE "UserWarning|return grid"
python -u research/featgen/fingerprint.py --days 20260430,20261112,20261113 --target zzeg,zzeg2 --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r111/zzrema_{date}.parquet,data/side_r110/zzeg_{date}.parquet" --transforms "id,-id,1/b" --models $DATA/models_gen/B --top 10 2>&1 | grep -v RuntimeWarning | cut -c1-200
python -u research/featgen/implied_scan.py --days 20260430,20261112,20261113 --target zzeg,zzeg2 --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r111/zzrema_{date}.parquet" --only-extra --transforms "id,-id,1/b" --rank-day 20260430 --models $DATA/models_gen/B --top 10 \
  --work data/implied_scan_r111 2>&1 | grep -v RuntimeWarning | cut -c1-260
echo "[$(date +%T)] done"
