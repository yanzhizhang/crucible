#!/usr/bin/env bash
# Round 189 on 97, foreground, read-only (PM's chrono logs under pm_224, 224's stsfile copy); prints only.
# runS's execution target, exact: decoded GTp (band + poscst clamp + sellable caps) then cFtp with nmv / bp / sp carried
# across all stocks in log order. ~3-6 min per day (one pass over the day's log in python), light.
#   bash research/r189.sh 2>&1 | tee data/reports/r189.txt          (DAYS="20261118 20261119 20261120" by default)
set -u
cd /work/crucible
DATA=/work/crucible_data
for D in ${DAYS:-20261118 20261119 20261120}; do
  L=$DATA/pm_224/$D/chrono_logs
  [ -d $L ] || { echo "== $D: no $L"; continue; }
  echo "[$(date +%T)] $D"
  python -u research/pm_factors/runs_ftp.py --logs $L --date $D --sts $DATA/sts_224 --show 20 2>&1 | grep -v -E "Warning|warn|stsfile" | cut -c1-330
done
echo "[$(date +%T)] r189 done"
