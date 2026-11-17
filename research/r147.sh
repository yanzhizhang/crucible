#!/usr/bin/env bash
# Round 147 on 97, foreground, read-only, model files only (~1 min). The user's two points (2026-11-17):
#  units     pin each column's SCALE from the tree model itself: training range + split thresholds -> a scale class
#            (rank [-1,1], share [0,1], ratio ~1, integer code, bp / count, z-like, small signed ...), listed per class
#            with the current formula next to it, and the families that mix classes.
#  famlinks  use families to find related factors: family x family adjacency lift in the trees; for every family with no
#            .jz key and no hstats profile, the sourced families it splits next to most (a renamed / combined factor
#            should sit next to its ingredients). Both for the production generation C and for B.
#   bash research/r147.sh 2>&1 | tee data/reports/r147.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for G in C B; do
  echo "######## generation $G"
  python -u research/featgen/model_atlas.py units --gens $DATA/models_gen --gen $G --date 20261113 --out data/reports/atlas_r147_$G 2>&1 | cut -c1-230
  python -u research/featgen/model_atlas.py famlinks --gens $DATA/models_gen --gen $G --date 20261113 --out data/reports/atlas_r147_$G 2>&1 | cut -c1-260
done
echo "[$(date +%T)] r147 done"
