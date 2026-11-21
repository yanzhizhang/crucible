#!/usr/bin/env bash
# Round 203 on 97, foreground, read-only; prints only. ~10 min, light, no engine. R201 follow-up on DataYesSharesL2 only:
# (1) weights on the EXCHANGE clock too (PM decays on its own receive time, ms away from ours; on exchange stamps the jitter
# goes away if that is what PM uses), (2) zzhj.p window / slot amounts (3 .. 60 s, the current / last 15 / 20 / 30 / 60 s slot),
# (3) more price-scaled inputs for zzcb.r / zzeb.r; inputs scored at R200's half-lives, averaged on the local then the exchange clock.
#   bash research/r203.sh 2>&1 | tee data/reports/r203.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in ${DAYS:-20261119 20261120}; do
  for VC in snap "snap, exchange clock"; do
    echo "[$(date +%T)] $D dysl2, inputs averaged on: $VC"
    CRUCIBLE_SOURCE=dysl2 python -u research/engine/ema_probe.py --date $D --jz $DATA/seed_probe/HS300/$D.jz --catalog $DATA/catalog_dysl2.duckdb --keys zzhj.p,zzui.r,zzcb.r,zzeb.r --value-clock "$VC" 2>&1 | grep -v Warning | cut -c1-220
  done
done
echo "[$(date +%T)] r203 done"
