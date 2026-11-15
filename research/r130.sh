#!/usr/bin/env bash
# Round 130 on 97, foreground, read-only, writes only data/reports/audit. ~1 min. The 209-column source audit
# (research/audit_sources.py): which PM factor each column's family has and whether our formula reads it.
#   bash research/r130.sh 2>&1 | tee data/reports/r130.txt
set -u
cd /work/crucible
python -u research/audit_sources.py --jz /work/crucible_data/seed_probe/HS300/20261113.jz --verify data/reports/r125/verify_20261113.csv \
  --hstats /work/crucible_data/hstats_view/HS300 --models /work/crucible_data/models_gen/B --date 20261113 --out data/reports/audit 2>&1 | cut -c1-250
python -u research/range_check.py --x data/reports/stage6_r99_1113/x.parquet --models /work/crucible_data/models_gen/B --date 20261113 2>&1 | cut -c1-250
echo "[$(date +%T)] done; full table in data/reports/audit/audit.md (and audit.csv)"
