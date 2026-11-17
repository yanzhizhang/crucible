#!/usr/bin/env bash
# Round 157 on 97, foreground, read-only on inputs, ~5 min. The "exact" columns zzvx.a and zzxu.a (and probably zzxu.b) do not
# match the model's own training ranges / thresholds (R148 unit_check: thresholds live 0.00 / 0.07; zzvx.a's thresholds are a
# ratio just above 1, ours is a few percent). Re-derive them where they were never searched: seg_fit over the state-formula
# grammar of their OWN families' keys (zzvx, zzxu, zzup, zzpk: K / px, (px - K) / K, K1 / K2 ...), no other candidates,
# against the range + threshold + root fingerprint. zzup is the control (its formula should come first). Generations C and B.
#   bash research/r157.sh 2>&1 | tee data/reports/r157.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
for G in C B; do
  echo "######## generation $G"
  python -u research/featgen/seg_fit.py --days 20261112,20261113 --target zzup,zzvx.a,zzxu.a,zzxu.b --x-dir "data/reports/stage6_r99_{mmdd}" \
    --base "data/featgen_r99/base_{date}.parquet" --no-base --cand-re "zz(vx|xu|up|pk)\." --extra "data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet" \
    --models $DATA/models_gen/$G --transforms=id,-id,1/b,log,sqrt,abs4 --scales 1,100,1e4,0.01 --top 10 2>&1 | grep --line-buffered -v Warning | stdbuf -oL cut -c1-230
done
echo "[$(date +%T)] r157 done"
