#!/usr/bin/env bash
# Round 117b on 97, foreground, read-only on inputs; writes data/greedy_fill_r117b (scratch) and data/reports. ~30-60 min.
# R114: fills interact (zzeg +0.07..+0.12 inside a set, ~0 alone); only zzab2 <- ret120 helped all three days in both
# sets. So choose fills in context: backward then greedy on 0430 + 1112 + 1113, each step the option (top 4 fingerprint rows of every open
# column, R108 / R113 / R111b / R115) that raises the worst day most (research/featgen/greedy_fill.py).
#   bash research/r117.sh 2>&1 | tee data/reports/r117b.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
python -u research/featgen/greedy_fill.py --days 20260430,20261112,20261113 \
  --reports data/reports/r108.txt,data/reports/r113.txt,data/reports/r111b.txt,data/reports/r115.txt \
  --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r105/ratio_{date}.parquet,data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet,data/side_r111/zzrema_{date}.parquet,data/side_r115/zzhj_{date}.parquet" \
  --models $DATA/models_gen/B --work data/greedy_fill_r117b --backward --min-gain 0.0002 2>&1 | grep -v RuntimeWarning | cut -c1-300
echo "[$(date +%T)] done"
