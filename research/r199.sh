#!/usr/bin/env bash
# Round 199 on 97, foreground, read-only; prints only. ~2 min per day (three group-bys on one day's ticks), moderate, no engine run.
# A profile of every PM state key we cannot rebuild, on 224's own feed (s224): what the weight says about the clock and the
# half-life, where the stamp sits, what the value tracks (day statistics, our engine's keys, yesterday's value) -- to see
# which keys have a lead and which have none, before guessing formulas again.
#   bash research/r199.sh 2>&1 | tee data/reports/r199.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
E=data/engine_r194/dysl2; [ -d $E ] || E=data/engine_r187/dysl2
python -u research/engine/key_profile.py --dates ${DAYS:-20261118,20261119,20261120} --jz $DATA/seed_probe/HS300 --catalog $DATA/catalog_s224.duckdb --engine $E \
  --prefix zzcb,zzeb.r,zztx,zzhj,zzvm.c.re,zzui,zzbd.drSum,zzgh2.nb,zzgh.nb,zzxu.upper,zzxu.lower,zzxu.dir 2>&1 | grep -v Warning | cut -c1-330
echo "[$(date +%T)] r199 done"
