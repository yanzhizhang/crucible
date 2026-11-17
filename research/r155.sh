#!/usr/bin/env bash
# Round 155 on 97, foreground, read-only on inputs. The marginal fingerprint (R153) only names a shape class; this is the
# conditional test with EVERY column of our matrix trusted on the paths (cond_counts --trusted all): candidates for one
# column are compared on the same rows at every node that splits on it, so a proxy with the right marginal but the wrong
# joint behaviour falls behind (synthetic: true 0.0009, corr-0.85 proxy 0.0025, marginal twin 0.0123).
# Three independent parts, one tab each:
#  cal  calibration: the 4 PM-state columns as targets, pool = every base / state formula in range. The matrix column
#       (exact) and its own state formula must come first; how far behind the next candidate is = the test's power.
#  a    the columns R153 passed or nearly passed: is the fingerprint's candidate also first here, and does it beat the stand-in?
#  b    node_audit's worst stand-ins with no fingerprint candidate.
# Day 20261112, production generation C. ~10-40 min per part (32 workers).
#   bash research/r155.sh cal 2>&1 | tee data/reports/r155_cal.txt     (same for a, b)
set -u
cd /work/crucible
DATA=/work/crucible_data
case "${1:?cal|a|b}" in
cal) T=zzup,zzvx.a,zzxu.a,zzxu.b ;;
a)   T=zzqf7.q.c2,zzqf5.q.c1,zzpe.b,zzop.j,zzqf.q.c3,zzal.h,zzrg,zzgh2.b ;;
b)   T=zzag2.c,zzpp.e,zzhi2.b,zzwa.h,zzwa.e,zzkp,zzlk,zzxt3 ;;
esac
python -u research/featgen/cond_counts.py --day 20261112 --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
  --extra "data/side_r105/ratio_{date}.parquet,data/side_r107/state_{date}.parquet,data/side_r105/engine_{date}.parquet" \
  --target $T --trusted all --pool --models $DATA/models_gen/C --rows 150000 --max-nodes 2500 --workers 32 --top 8 2>&1 | grep -v Warning | cut -c1-200
echo "[$(date +%T)] r155 $1 done"
