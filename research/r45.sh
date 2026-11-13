#!/usr/bin/env bash
# Round 45 on 97, foreground, light and read-only: are s1..s4 different stock pools? Text of the model files and directory
# names only; nothing is written except pm_lgb_meta.csv (if missing) and the tee'd report under /work.
#   bash research/r45.sh 2>&1 | tee data/reports/r45.txt
set -u
cd /work/crucible
say() { echo "[$(date +%T)] $*"; }
say "1 pools visible under /data/prod (names only)"
for d in /data/prod /data/prod/h /data/prod/models /data/prod/models/tk5m; do echo "-- $d"; ls $d 2>&1 | head -60 | tr "\n" " "; echo; done
for d in /data/prod/h/*/; do echo "-- $d: $(ls $d 2>/dev/null | head -30 | tr "\n" " ")"; done
ls -d /data/prod/*/*500* /data/prod/*/*1000* /data/prod/*/*ZZ* /data/prod/*/*zz* /data/prod/*/*/*500* /data/prod/*/*/*1000* 2>/dev/null | head -40
say "2 rows and ranges per sub-model"
[ -f data/reports/pm_lgb_meta.csv ] || python -u research/pm_factors/pm_lgb_meta.py --models /data/prod/models/tk5m --out data/reports/pm_lgb_meta.csv | tail -3
python -u research/pm_factors/pool_probe.py --models /data/prod/models/tk5m --meta data/reports/pm_lgb_meta.csv 2>&1 | cut -c1-260
say "done"
