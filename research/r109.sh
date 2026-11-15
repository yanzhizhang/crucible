#!/usr/bin/env bash
# Round 109 on 97, foreground, read-only on inputs; writes data/reports/r109 only. ~5-10 min.
# The user wants the complete list -- every state key in PM's .jz and every one of the 209 features -- for a bit-exact copy.
# 1 engine vs PM per key (engine/verify.py --csv) on 1106 (PM's .jz in /data/prod/h) and 1113 (seed_probe), from the
#   engine day-end files already in data/engine; 2 inventory.md per day (research/inventory.py); 3 the zzeg.c / zzxu
#   channel window (pm_factors/channel_probe.py).
#   bash research/r109.sh 2>&1 | tee data/reports/r109.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
O=data/reports/r109; mkdir -p $O
for D in 20261106 20261113; do
  if [ $D = 20261106 ]; then JZ=/data/prod/h/HS300/p1_intrading_dcp/$D.jz; else JZ=$DATA/seed_probe/HS300/$D.jz; fi
  ls -l data/engine/engine_$D.json.gz $JZ 2>&1 | sed 's/^/  /'
  if [ -e data/engine/engine_$D.json.gz ]; then
    python -u research/engine/verify.py --date $D --engine data/engine/engine_$D.json.gz --pm $JZ --catalog $DATA/catalog_dysl2.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
      || echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"
  fi
  python -u research/inventory.py --jz $JZ --verify-csv $O/verify_$D.csv --models $DATA/models_gen/B --date $D --out $O/$D 2>&1 | tail -2
done
python -u research/pm_factors/channel_probe.py --days 20261104,20261105,20261106,20261112,20261113 2>&1 | cut -c1-900 | tee $O/channel.txt
echo "[$(date +%T)] done"
