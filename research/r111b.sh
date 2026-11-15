#!/usr/bin/env bash
# Round 111b on 97, foreground, read-only on inputs; rewrites data/side_r111/zzrema_*.parquet, writes data/implied_scan_r111b,
# data/reports. ~10 min. R111 fixed rEma = decayed average of the span, half 300 s, from 09:30 (value equal 0.98-1.00 on
# 1105 / 1106 / 1112 / 1113); rema / span sits within 0.1% of 1 -- the shape of zzeg's thresholds (half of them within
# 0.9986-1.001, tails to 0.985 / 1.03). The morning is where it moves (the channel drops a day at the open, the average
# carries over), so the marks now carry yesterday's average (PM's .jz where we have it, else yesterday's span) next to a
# cold start; 0430 had crashed on a stock without history (fixed).
#   bash research/r111b.sh 2>&1 | tee data/reports/r111b.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
for D in 20261112 20261113; do
  P=$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1)
  python -u research/featgen/zzeg_rema.py --date $D --x data/reports/stage6_r99_${D:4}/x.parquet --raw $DATA/dysl2_raw --prev-jz $SP/$P.jz --jz $SP/$D.jz \
    --halves 300 --out data/side_r111 2>&1 | grep -vE "UserWarning|return grid" | tail -1
done
python -u research/featgen/zzeg_rema.py --date 20260430 --x data/reports/stage6_r99_0430/x.parquet --raw $DATA/feitu_raw --halves 300 --start 0930 \
  --out data/side_r111 2>&1 | grep -vE "UserWarning|return grid" | tail -1
python -u research/featgen/fingerprint.py --days 20260430,20261112,20261113 --target zzeg,zzeg2 --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r111/zzrema_{date}.parquet" --transforms "id,-id,1/b" --models $DATA/models_gen/B --top 10 2>&1 | grep -v RuntimeWarning | cut -c1-200
python -u research/featgen/implied_scan.py --days 20260430,20261112,20261113 --target zzeg,zzeg2 --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r111/zzrema_{date}.parquet" --only-extra --transforms "id,-id,1/b" --rank-day 20260430 --models $DATA/models_gen/B --top 10 \
  --work data/implied_scan_r111b 2>&1 | grep -v RuntimeWarning | cut -c1-260
echo "[$(date +%T)] done"
