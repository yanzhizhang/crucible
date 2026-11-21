#!/usr/bin/env bash
# Round 194 on 97, foreground, read-only on PM data and on the pulls.
#  so    ~4 min, light, prints only: R192's SO context again with the decision lines counted properly, plus, for every SO
#        line printed inside another stock's block, the latest block of the ORDER'S OWN stock (its decision / nullptr lines,
#        how long before the SO stamp) -- where do the ~1,580 orders without a flagged decision come from.
#  eng   ~3 min/day, ONE ENGINE RUN AT A TIME: zzmh.hm is now the snapshot high from 09:30:00 with no end (R193, promoted;
#        variants removed). Self-test first, then 1118 / 1119 / 1120 on dysl2: the zzmh rows (hm should be 1.0, r should follow).
#  dec   HEAVY, alone (96 workers), writes s224_raw: re-decode the SNAPSHOTS of 224's feed with the level sizes x 10000
#        (R192: the TDF files hold them divided by 10000), day by day, then the level-1 check of R190c on both stores.
#   bash research/r194.sh so 2>&1 | tee data/reports/r194_so.txt     (then eng, then dec)
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
O=data/reports/r194; mkdir -p $O
say() { echo "[$(date +%T)] $*"; }
case "${1:?so|eng|dec}" in
so)
  python -u research/pm_factors/runs_soctx.py --logs $DATA/pm_224/20261118/chrono_logs --date 20261118 --show 2 2>&1 | grep -v -E "Warning|warn" | cut -c1-240 ;;
eng)
  python -u research/engine/selftest.py --stocks 20 --events 4000 > $O/selftest.txt 2>&1; RC=$?
  tail -2 $O/selftest.txt | cut -c1-220
  [ $RC -eq 0 ] || { echo "SELFTEST FAILED (rc $RC): not running the engine"; exit 1; }
  S=dysl2
  for D in ${DAYS:-20261118 20261119 20261120}; do
    PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
    say "== $D"
    python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_$S.duckdb --chan-hist "${S}_raw,dysl2_raw,feitu_raw" --threads 32 \
      --out data/engine_r194/$S --work data/engine_r194/$S/work > $O/engine_$D.txt 2>&1 || { echo "  engine FAILED: $(tail -2 $O/engine_$D.txt)"; continue; }
    CRUCIBLE_SOURCE=$S python -u research/engine/verify.py --date $D --engine data/engine_r194/$S/engine_$D.json.gz --pm $SP/$D.jz --catalog $DATA/catalog_$S.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
      || { echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"; continue; }
    head -1 $O/verify_$D.csv; grep -E '^zzmh\.' $O/verify_$D.csv | cut -c1-200
    python -u research/engine/gap_probe.py --date $D --engine data/engine_r194/$S/engine_$D.json.gz --pm $SP/$D.jz --who zzmh.hm,zzmh.r 2>&1 | grep -v "PM keys equal" | cut -c1-240
  done ;;
dec)
  for D in ${DAYS:-20261118 20261119 20261120}; do
    say "s224 snapshots $D, level sizes x 10000"
    python -u research/decode_tdf_day.py --root $DATA/tdf_224 --date $D --out $DATA/s224_raw --kinds quotation --level-vol-scale 10000 --workers 96 2>&1 | grep -v Warning | tail -4 | cut -c1-200
  done
  for S in s224 dysl2; do
    python - <<PY 2>&1 | cut -c1-260
import duckdb
c = duckdb.connect("$DATA/catalog_$S.duckdb", read_only=True)
print("$S 600000 at the last snapshot before 10:00 on 20261118 (bid_vol[1:3], ask_vol[1:3], total_volume):", c.sql("SELECT bid_vol[1:3], ask_vol[1:3], total_volume FROM raw_quotation WHERE date = 20261118 AND symbol_id = 600000 AND ((time / 1e9 + 28800) % 86400) < 36000 ORDER BY time DESC LIMIT 1").fetchall())
print("   rows per day:", c.sql("SELECT date, count(*), avg((bid_vol[1] = 0)::INT)::DECIMAL(6,4) FROM raw_quotation WHERE date >= 20261118 GROUP BY 1 ORDER BY 1").fetchall())
PY
  done ;;
esac
say "r194 $1 done"
