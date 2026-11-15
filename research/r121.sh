#!/usr/bin/env bash
# Round 121 on 97, foreground, read-only on inputs; writes data/engine_r121 and data/reports. ~20-40 min.
# R119 (value match, rel 1e-3): zzup.ps = snapshot price average 600 s (1.000 all days -- the engine's rule; its 1e-6 misses
# are feed noise), zzba.p ~ trade-clock price average 300-360 s, zzui.r ~ snapshot price 45-90 s, zzvm.c.re only 0.3 at
# 7200 s (longer / carried?). 1: the new zzba.p / zzui.r variants in the C++ engine, verified at 1e-6 and 1e-4 against PM.
# 2: ema_fit with half-lives to 2 days and the carried (c0) average for the keys nothing fits yet.
#   bash research/r121.sh 2>&1 | tee data/reports/r121.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
P1=/data/prod/h/HS300/p1_intrading_dcp
SP=$DATA/seed_probe/HS300
O=data/reports/r121; mkdir -p $O
python -u research/engine/selftest.py --seed 1 2>&1 | tail -1
for D in 20261106 20261112 20261113; do
  case $D in 20261106) JZ=$P1/$D.jz; PV=$P1/20261105.jz ;; 20261112) JZ=$SP/$D.jz; PV=$SP/20261111.jz ;; 20261113) JZ=$SP/$D.jz; PV=$SP/20261112.jz ;; esac
  echo "[$(date +%T)] $D engine"
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --marks rand:240 --threads 32 \
    --out data/engine_r121 --work data/engine_r121/work 2>&1 | tail -1 | cut -c1-200
  python -u research/engine/verify.py --date $D --engine data/engine_r121/engine_$D.json.gz --pm $JZ --catalog $DATA/catalog_dysl2.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
    || echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"
  python -c "import polars as pl; t=pl.read_csv('$O/verify_$D.csv'); print(t.filter(pl.col('key').str.contains('zzba.p|zzui.r|zzup.ps|zzeg')).sort('key'))"
  echo "[$(date +%T)] $D ema_fit (value, carried)"
  python -u research/pm_factors/ema_fit.py --date $D --jz $JZ --prev-jz $PV --raw $DATA/dysl2_raw --by value --top 5 \
    --keys zzvm.c.re,zzmh.r,zzcb.r,zzeb.r,zztx.r,zzhj.p 2>&1 | grep -v "best clocks by wgt" | cut -c1-400
done
echo "[$(date +%T)] done"
