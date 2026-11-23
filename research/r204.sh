#!/usr/bin/env bash
# Round 204 on 97, foreground, read-only on PM data; writes data/engine_r204, data/reports/r204. ONE ENGINE RUN AT A TIME.
# zzvm.c.re promoted into the engine (trade clock, half 300 s, running VWAP from 09:30, R200-R201). Self-test first, then
# 1118 / 1119 / 1120 on dysl2: the zzvm rows of the verify table (zzvm.c.re should be ~1.0) and a check that nothing else moved.
#   bash research/r204.sh 2>&1 | tee data/reports/r204.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
O=data/reports/r204; mkdir -p $O
python -u research/engine/selftest.py --stocks 20 --events 4000 > $O/selftest.txt 2>&1; RC=$?
tail -3 $O/selftest.txt | cut -c1-220
[ $RC -eq 0 ] || { echo "SELFTEST FAILED (rc $RC): not running the engine"; exit 1; }
S=dysl2
for D in ${DAYS:-20261118 20261119 20261120}; do
  PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
  echo "[$(date +%T)] == $D"
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_$S.duckdb --chan-hist "${S}_raw,dysl2_raw,feitu_raw" --threads 32 \
    --out data/engine_r204/$S --work data/engine_r204/$S/work > $O/engine_$D.txt 2>&1 || { echo "  engine FAILED: $(tail -2 $O/engine_$D.txt)"; continue; }
  CRUCIBLE_SOURCE=$S python -u research/engine/verify.py --date $D --engine data/engine_r204/$S/engine_$D.json.gz --pm $SP/$D.jz --catalog $DATA/catalog_$S.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
    || { echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"; continue; }
  head -1 $O/verify_$D.csv; grep -E '^zzvm\.' $O/verify_$D.csv | cut -c1-200
  if [ -e data/reports/r194/verify_$D.csv ]; then
    python - <<PY 2>&1 | cut -c1-240
import polars as pl
a = pl.read_csv("data/reports/r194/verify_$D.csv").select("key", pl.col("all").alias("r194"))
b = pl.read_csv("$O/verify_$D.csv").select("key", pl.col("all").alias("r204"))
j = a.join(b, on="key", how="full", coalesce=True).filter(((pl.col("r194") - pl.col("r204")).abs() > 1e-9) | pl.col("r194").is_null() | pl.col("r204").is_null())
print("   keys whose share moved against R194 (new keys show r194 null):", j.sort("key").rows())
PY
  fi
done
echo "[$(date +%T)] r204 done"
