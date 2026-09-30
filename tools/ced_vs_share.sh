#!/usr/bin/env bash
# Rebuild CED with research/ced for a date range and compare with the production CED .xr files.
#
#   export CRUCIBLE_WIND_URL='mssql+pymssql://USER:PWD@HOST:1433/WindDB?charset=utf8&tds_version=7.0'
#   export CRUCIBLE_JY_URL='mssql+pymssql://USER:PWD@HOST:1433/JYDB?charset=utf8&tds_version=7.0'
#   export CRUCIBLE_ZY_URL='mssql+pymssql://USER:PWD@HOST:1433/Zyyx2.0?charset=utf8&tds_version=7.0'
#   bash tools/ced_vs_share.sh 20260401 20260430 [/data/share/CED]
#
# Every step runs even if an earlier one fails (a missing JY/ZY URL only costs sw-industry /
# zy-div). Log: $CRUCIBLE_DATA/logs/ced.log (default /work/crucible_data); report:
# data/reports/ced_vs_share/<timestamp>/summary.csv, cells.parquet.
set -u
START=${1:-20260401}
END=${2:-20260430}
CED_ROOT=${3:-/data/share/CED}
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src"
: "${CRUCIBLE_WIND_URL:?set CRUCIBLE_WIND_URL (see the top of this script)}"
for v in CRUCIBLE_JY_URL CRUCIBLE_ZY_URL; do
  [ -n "${!v:-}" ] || echo "WARNING: $v not set -- the datasets that need it will be missing"
done
[ -d "$CED_ROOT" ] || { echo "no CED root at $CED_ROOT"; exit 1; }

run() { echo; echo "=== $*"; python research/run_ced.py "$@" || echo "!!! failed (rc $?): $*"; }

run calendar
run all-hist --start "$START" --end "$END" --overwrite      # hist SOD / index weights + every other dataset
run daily-sod --start "$START" --end "$END" --overwrite     # live SOD (what production writes pre-open)
run index-universe --start "$START" --end "$END" --overwrite
echo; echo "=== compare with $CED_ROOT"
python research/compare_ced.py --ced "$CED_ROOT" --start "$START" --end "$END"
