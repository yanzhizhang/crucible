#!/usr/bin/env bash
# R33b on 97: pm_orders fixed for the 1112 plugin build (lines matched by file + keys, not line numbers; nullptr true/false;
# the new alg.cpp evaluation line mp fa vpen vr poscst cpos asq hs folded into each decision), then raw samples for reading
# runS's rule. Writes data/reports and /work/crucible_data/pm_orders only.
#   bash research/r33b.sh 20261112 2>&1 | tee data/reports/r33b.txt
set -u
cd /work/crucible
D=${1:-20261112}
python -u research/pm_factors/pm_orders.py --logs /work/crucible_data/chrono_logs --dates $D --out /work/crucible_data/pm_orders > data/reports/r33b_orders_$D.txt 2>&1 \
  || echo "pm_orders FAILED: $(tail -3 data/reports/r33b_orders_$D.txt)"
cut -c1-230 data/reports/r33b_orders_$D.txt | head -300
python -u research/pm_factors/runs_sample.py --orders /work/crucible_data/pm_orders --date $D 2>&1 | cut -c1-250
