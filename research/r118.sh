#!/usr/bin/env bash
# Round 118 on 97, foreground, read-only on inputs; writes data/engine_r118 (engine day-end + marks, the cache for these days)
# and data/reports. ~10-20 min. zzeg.c is now in the C++ engine (user: write it in and cache; revisit only on wide
# mismatches): chan.bin = the previous 3 days' snapshot high / low (featgen/zzeg_channel.day_hl, cached per day under
# /work/crucible_data/cache/daily_hl), rule "zzeg.c.rEma|c0" = span average, half 300 s, carried from yesterday's .jz.
# Selftest first (C++ vs Python, synthetic, must be 0 mismatches), then 1106 / 1112 / 1113 against PM's .jz.
#   bash research/r118.sh 2>&1 | tee data/reports/r118.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
P1=/data/prod/h/HS300/p1_intrading_dcp
SP=$DATA/seed_probe/HS300
O=data/reports/r118; mkdir -p $O
python -u research/engine/selftest.py --seed 1 2>&1 | tail -3
python -u research/engine/selftest.py --seed 2 2>&1 | tail -1
for D in 20261106 20261112 20261113; do
  case $D in 20261106) JZ=$P1/$D.jz; PV=$P1/20261105.jz ;; 20261112) JZ=$SP/$D.jz; PV=$SP/20261111.jz ;; 20261113) JZ=$SP/$D.jz; PV=$SP/20261112.jz ;; esac
  echo "[$(date +%T)] $D"
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --marks rand:240 --threads 32 \
    --out data/engine_r118 --work data/engine_r118/work 2>&1 | tail -3 | cut -c1-200
  python -u research/engine/verify.py --date $D --engine data/engine_r118/engine_$D.json.gz --pm $JZ --catalog $DATA/catalog_dysl2.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
    || echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"
  grep -E "zzeg|^key|SZSE" $O/verify_$D.txt | head -12
  python -c "import polars as pl; t=pl.read_csv('$O/verify_$D.csv'); print(t.filter(pl.col('key').str.starts_with('zzeg')))"
done
echo "[$(date +%T)] done"
