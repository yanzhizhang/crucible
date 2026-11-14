#!/usr/bin/env bash
# Round 76 on 97, foreground. Read-only on the model text files; writes data/reports only.
# How much of PM's model rests on the 144 columns we reproduce: split gain per feature, settled vs the 65 not settled,
# for the model we score with (generation B) and for the latest production model month.
#   bash research/r76.sh 2>&1 | tee data/reports/r76.txt
set -u
cd /work/crucible
echo "[$(date +%T)] generation B (the one we score with)"
python -u research/featgen/gain_share.py --models /work/crucible_data/models_gen/B --date 20261112 --top 40 2>&1 | cut -c1-260
echo "[$(date +%T)] production model month for 20261113"
python -u research/featgen/gain_share.py --models /data/prod/models/tk5m --date 20261113 --top 20 2>&1 | cut -c1-260
echo "[$(date +%T)] done"
