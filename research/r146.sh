#!/usr/bin/env bash
# Round 146 on 97, foreground, read-only (model files only, ~1 min): where do PM's training windows lie?
# Per model root (generations A / B / C and production tk5m), the month list; for the last 3 months of each, per file:
# trees, root row counts (= training rows), tree0 leaf mean (label mean), file mtime, the PM-state columns' training
# ranges per month, and the 'parameters:' block (data paths, bagging, bin sampling, seeds) once per root.
#   bash research/r146.sh 2>&1 | tee data/reports/r146.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
ls -la --time-style=long-iso /data/prod/models/ 2>&1 | head -20 | sed 's/^/  /'
python -u research/featgen/model_meta.py --roots $DATA/models_gen/A,$DATA/models_gen/B,$DATA/models_gen/C,/data/prod/models/tk5m --months 3 2>&1 | cut -c1-600
echo "[$(date +%T)] r146 done"
