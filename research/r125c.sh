#!/usr/bin/env bash
# Round 125c on 97, foreground, read-only on inputs; writes data/engine_r125 (0430 redone), data/implied_scan_r125, data/reports.
# ~10 min. R125: zzmh.r is in the engine (0.987-0.99 vs PM). The zzmh column test lacked the drawdown on 0430 (the old
# R125b found 0 candidates: the 0430 scoring matrix uses marks rand:240:7 (r70 args_of), not rand:240, so the engine rows never joined.
# engine_m7 marks), so: the engine for 0430 too (feitu catalog), then zzmh against the engine's own zzmh.r at the marks
# (sqrt = 2 x drawdown), fingerprint + scan. (--transforms=... : a list starting with "-" needs the "=" form.)
#   bash research/r125b.sh 2>&1 | tee data/reports/r125c.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
python -u research/engine/run_cpp.py --date 20260430 --universe pm:/data/prod/h/HS300/p1_intrading_dcp/20261104.jz --catalog $DATA/catalog.duckdb \
  --marks rand:240:7 --threads 32 --out data/engine_r125 --work data/engine_r125/work 2>&1 | tail -2 | cut -c1-200
EX="data/engine_r125/marks_{date}.parquet"
python -u research/featgen/fingerprint.py --days 20260430,20261112,20261113 --target zzmh --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "$EX" --transforms=sqrt,id,2b,4b2 --models $DATA/models_gen/B --top 10 2>&1 | grep -v RuntimeWarning | cut -c1-200
python -u research/featgen/implied_scan.py --days 20260430,20261112,20261113 --target zzmh --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "$EX" --only-extra --transforms=sqrt,id --rank-day 20260430 --models $DATA/models_gen/B --top 8 \
  --work data/implied_scan_r125 2>&1 | grep -v RuntimeWarning | cut -c1-260
echo "[$(date +%T)] done"
