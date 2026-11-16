#!/usr/bin/env bash
# Round 143 on 97, foreground, read-only on inputs: exact rules for the averages R138 e8 found proportional
# (zzui.r ~ EMA(mid) h~3 s, zzba.p ~ EMA(vwap_snap) h~3100 s, zzvm.c.re ~ EMA(price | micro) h~3900 s, carried?) and the
# four still open (zzcb.r, zzeb.r, zzhj.p, zztx.r). ema_fit by value (clock x start x half x input, rel 1e-3), now with
# vwap_snap / micro / half_spread inputs and a finer half grid around 1 h; with yesterday's .jz for the carried variant.
# Days 1112 / 1113 (1116 had a factor restart, so its state is checked separately). ~10-20 min.
#   bash research/r143.sh 2>&1 | tee data/reports/r143.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
K=zzui.r,zzba.p,zzvm.c.re,zzcb.r,zzeb.r,zzhj.p,zztx.r
for D in 20261112 20261113; do
  P=$(ls $DATA/seed_probe/HS300 | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1)
  echo "== $D (prev $P)"
  python -u research/pm_factors/ema_fit.py --date $D --jz $DATA/seed_probe/HS300/$D.jz --prev-jz $DATA/seed_probe/HS300/$P.jz --raw $DATA/dysl2_raw \
    --keys $K --by value --top 4 2>&1 | grep -vE "Warning|warn"
done
echo "[$(date +%T)] r143 done"
