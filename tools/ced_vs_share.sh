#!/usr/bin/env bash
# Rebuild CED with research/ced for a date range and compare with the production CED .xr files.
#
#   fill DB_USER / DB_PASSWORD in research/ced/db.py (or export CRUCIBLE_DB_USER / CRUCIBLE_DB_PASSWORD)
#   bash tools/ced_vs_share.sh 20260401 20260430 [/data/share/CED]
#
# Every step runs even if an earlier one fails. Log: $CRUCIBLE_DATA/logs/ced.log (default /work/crucible_data); report:
# data/reports/ced_vs_share/<timestamp>/summary.csv, cells.parquet.
set -u
START=${1:-20260401}
END=${2:-20260430}
CED_ROOT=${3:-/data/share/CED}
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
python -c "import sys; sys.path.insert(0, 'research'); from ced import db; import sqlalchemy as sa; [print(k, 'ok' if db.engine(k).connect().execute(sa.text('SELECT 1')).scalar() == 1 else '?') for k in (db.WIND, db.JY, db.ZY)]"   || echo "!!! database connection test failed (see above); continuing"
[ -d "$CED_ROOT" ] || { echo "no CED root at $CED_ROOT"; exit 1; }

run() { echo; echo "=== $*"; python research/run_ced.py "$@" || echo "!!! failed (rc $?): $*"; }

run calendar
run all-hist --start "$START" --end "$END" --overwrite      # hist SOD / index weights + every other dataset
run daily-sod --start "$START" --end "$END" --overwrite     # live SOD (what production writes pre-open)
run index-universe --start "$START" --end "$END" --overwrite
echo; echo "=== compare with $CED_ROOT"
python research/compare_ced.py --ced "$CED_ROOT" --start "$START" --end "$END"
