#!/usr/bin/env bash
# Round 193 on 97, foreground, read-only on PM data; writes data/engine_r193, data/reports/r193.
# Follow-up of R191 (hm part only; nb unchanged). ONE ENGINE RUN AT A TIME on 97 (shared binary): do not start next to another engine round.
#  hm   ~3 min/day: engine on dysl2 with three more zzmh.hm variants (R191: PM = the last price on every stock we miss, all SH):
#       "|ge930" = snapshot high from 09:30:00 with no end, "|ge930t" = that with trade prices too, "|last" = max(session high, last trade price).
#  nb   ~2 min/day, no engine: zzgh.nb as the amount imbalance over [start, end) for every end on a 1 s grid 14:50..15:01
#       and several starts, exchange and local clock; the gap PM - ours in yuan and its sign by exchange.
#   bash research/r193.sh hm 2>&1 | tee data/reports/r193_hm.txt      (then nb)
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
O=data/reports/r193; mkdir -p $O
S=dysl2
if [ "${1:-}" = hm ]; then  # the engine was edited (two zzmh.hm variants): python and C++ must still agree before any day is run
  python -u research/engine/selftest.py --stocks 20 --events 4000 > $O/selftest.txt 2>&1; RC=$?
  tail -4 $O/selftest.txt | cut -c1-220
  [ $RC -eq 0 ] || { echo "SELFTEST FAILED (rc $RC): not running the engine"; exit 1; }
fi
for D in ${DAYS:-20261118 20261119 20261120}; do
  [ -e $SP/$D.jz ] || { echo "no $SP/$D.jz"; continue; }
  PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
  echo "[$(date +%T)] == $D $1"
  case "${1:?hm|nb}" in
  hm)
    python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_$S.duckdb --chan-hist "${S}_raw,dysl2_raw,feitu_raw" --threads 32 \
      --out data/engine_r193/$S --work data/engine_r193/$S/work > $O/engine_$D.txt 2>&1 || { echo "  engine FAILED: $(tail -2 $O/engine_$D.txt)"; continue; }
    CRUCIBLE_SOURCE=$S python -u research/engine/verify.py --date $D --engine data/engine_r193/$S/engine_$D.json.gz --pm $SP/$D.jz --catalog $DATA/catalog_$S.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
      || { echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"; continue; }
    head -1 $O/verify_$D.csv; grep -E '^zz(mh|pk)\.' $O/verify_$D.csv | cut -c1-200
    python -u research/engine/gap_probe.py --date $D --engine data/engine_r193/$S/engine_$D.json.gz --pm $SP/$D.jz --who zzmh.hm,zzmh.r 2>&1 | cut -c1-240 ;;
  nb)
    E=data/engine_r187/$S/engine_$D.json.gz; [ -e data/engine_r193/$S/engine_$D.json.gz ] && E=data/engine_r193/$S/engine_$D.json.gz
    CRUCIBLE_SOURCE=$S python -u research/engine/gap_probe.py --date $D --engine $E --pm $SP/$D.jz --catalog $DATA/catalog_$S.duckdb --who zzgh.nb 2>&1 | grep -v Warning | sed -n '/^nb sweep/,$p' | cut -c1-260 ;;
  esac
done
echo "[$(date +%T)] r193 $1 done"
