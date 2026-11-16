#!/usr/bin/env bash
# Round 145 on 97, foreground, read-only on inputs. Two independent parts, one tab each:
#  audit  node_audit: send our filled matrix down every PM tree and compare, node by node, our left share with PM's stored
#         counts; per column the weighted share error at its own splits. The 4 PM-state columns calibrate the scale;
#         settled stand-ins high on every day are the first to re-derive. 0430 full grid + 1112 + 1113. ~5-15 min.
#         (synthetic check: right column 0.004, corr-0.85 proxy 0.013, unrelated decoy 0.067)
#  wgt    ema_fit by WEIGHT for the 4 averages no snapshot-EMA explains (zzcb.r zzeb.r zzhj.p zztx.r): the weight depends
#         only on the event clock and the half-life, so new event clocks (bbo change, spread change, volume change) are
#         tried before any input. 1112 + 1113. ~10-20 min.
#   bash research/r145.sh audit 2>&1 | tee data/reports/r145_audit.txt
#   bash research/r145.sh wgt   2>&1 | tee data/reports/r145_wgt.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
case "${1:?audit|wgt}" in
audit)
  python -u research/featgen/node_audit.py --days 20260430 --x-dir data/full_0430 --models $DATA/models_gen/B --out data/reports/node_audit_0430 2>&1 | grep -v Warning
  python -u research/featgen/node_audit.py --days 20261112,20261113 --x-dir "data/reports/stage6_r99_{mmdd}" --models $DATA/models_gen/B --out data/reports/node_audit_nov 2>&1 | grep -v Warning ;;
wgt)
  for D in 20261112 20261113; do
    P=$(ls $DATA/seed_probe/HS300 | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1)
    echo "== $D (prev $P)"
    python -u research/pm_factors/ema_fit.py --date $D --jz $DATA/seed_probe/HS300/$D.jz --prev-jz $DATA/seed_probe/HS300/$P.jz --raw $DATA/dysl2_raw \
      --keys zzcb.r,zzeb.r,zzhj.p,zztx.r --by wgt --top 6 2>&1 | grep -vE "Warning|warn"
  done ;;
esac
echo "[$(date +%T)] r145 $1 done"
