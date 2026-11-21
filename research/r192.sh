#!/usr/bin/env bash
# Round 192 on 97, foreground, read-only; prints only, light (one pass over one day's log + three unzip heads).
#  tdf    the raw snapshot rows of 224's TDF for three stocks: are the level sizes in the FILE already tiny (a scale
#         applied by the shm -> TDF conversion), or does decode_tdf_day.py truncate them? (R190c: s224 sizes ~ dysl2 / 10000, cut to integers)
#  so     what precedes each SO line in the chrono log: evaluation / decision / nullptr lines of its snapshot block,
#         SO lines per block, same stock or another -- why only half of the 3,063 orders have a decision flagged "nullptr: false".
#   bash research/r192.sh tdf 2>&1 | tee data/reports/r192_tdf.txt ; bash research/r192.sh so 2>&1 | tee data/reports/r192_so.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
D=${D:-20261118}
case "${1:?tdf|so}" in
tdf)
  for C in 600000.SH 000001.SZ 300750.SZ; do
    Z=$DATA/tdf_224/$C/0/$D.zip
    [ -e $Z ] || { echo "no $Z; have: $(ls $DATA/tdf_224 | head -3) ..."; continue; }
    echo "== $Z: $(unzip -l $Z | tail -1)"
    unzip -p $Z | head -1 | cut -c1-1500
    unzip -p $Z | awk -F, 'NR > 1 && NR % 1500 == 0' | head -3 | cut -c1-1500
  done
  python - <<PY 2>&1 | cut -c1-300
import duckdb
for s in ("s224", "dysl2"):
    c = duckdb.connect("$DATA/catalog_%s.duckdb" % s, read_only=True)
    print(s, "600000 at the last snapshot before 10:00: total_volume, total_amount, total_bid_volume, total_ask_volume, bid_vol[1:3], ask_vol[1:3]:",
          c.sql("SELECT total_volume, total_amount, total_bid_volume, total_ask_volume, bid_vol[1:3], ask_vol[1:3] FROM raw_quotation WHERE date = $D AND symbol_id = 600000 AND ((time / 1e9 + 28800) % 86400) < 36000 ORDER BY time DESC LIMIT 1").fetchall())
PY
  ;;
so)
  python -u research/pm_factors/runs_soctx.py --logs $DATA/pm_224/$D/chrono_logs --date $D --show 2 2>&1 | grep -v -E "Warning|warn" | cut -c1-240 ;;
esac
echo "[$(date +%T)] r192 $1 done"
