#!/usr/bin/env bash
# Intranet (97): rebuild CED with research/ced for April 2026 and compare with /data/share/CED.
#
#   bash tools/ced_vs_share.sh
#
# Everything is the intranet default: account in research/ced/db.py (DB_USER / DB_PASSWORD),
# conda env "crucible", data under /work/crucible_data, production CED at /data/share/CED.
# Every step runs even if an earlier one fails. Everything printed also goes to
# ced_vs_share_<timestamp>.log in the crucible root; the report is
# data/reports/ced_vs_share/<timestamp>/summary.csv + cells.parquet.
set -u
START=20260401
END=20260430
CED_ROOT=/data/share/CED
ENV_NAME=crucible

cd "$(dirname "$0")/.."
LOG="$PWD/ced_vs_share_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

if [ "${CONDA_DEFAULT_ENV:-}" != "$ENV_NAME" ]; then
  for sh in "$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh" /opt/anaconda3/etc/profile.d/conda.sh; do
    [ -f "$sh" ] && { . "$sh"; break; }
  done
  conda activate "$ENV_NAME" || { echo "cannot activate conda env $ENV_NAME"; exit 1; }
fi
export PYTHONPATH="$PWD/src"
echo "python $(command -v python), range $START..$END, CED $CED_ROOT, log $LOG"
[ -d "$CED_ROOT" ] || { echo "no CED root at $CED_ROOT"; exit 1; }

echo "=== database connections"
python - <<'EOF' || echo "!!! database connection test failed (see above); continuing"
import sys
sys.path.insert(0, "research")
import sqlalchemy as sa
from ced import db
for k in (db.WIND, db.JY, db.ZY):
    with db.engine(k).connect() as c:
        print(k, "ok" if c.execute(sa.text("SELECT 1")).scalar() == 1 else "?")
EOF

run() { echo; echo "=== $*"; python research/run_ced.py "$@" || echo "!!! failed (rc $?): $*"; }

run calendar
run all-hist --start "$START" --end "$END" --overwrite      # hist SOD / index weights + every other dataset
run daily-sod --start "$START" --end "$END" --overwrite     # live SOD (what production writes pre-open)
run index-universe --start "$START" --end "$END" --overwrite
echo; echo "=== compare with $CED_ROOT"
python research/compare_ced.py --ced "$CED_ROOT" --start "$START" --end "$END"
echo; echo "log: $LOG"
