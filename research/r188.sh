#!/usr/bin/env bash
# Round 188 on 97, foreground, read-only on PM data; writes data/reports/r188 only. Three parts, none heavy (no engine run,
# no decode), so they can run next to another session's round, each in its own tab.
#  sim    ~3 min: PM's alpha against our rebuild (pred.parquet of R177 / R187 score) under every definition of "similar",
#         per day and feed, to see which one reads ~0.20 next to our 0.59-0.64; then 224's feed (s224) against the full
#         one (dysl2): are the residuals larger on the (stock, mark) rows with a snapshot missing from 224's feed.
#  band   ~5 min per day: per-snapshot trace of the zzxu band inputs against PM's day-end upper / lower / dir: at which
#         snapshot the active band took PM's value, whether it ratchets after, and a multiplier search over centres / widths.
#  gaps   ~2 min per day: zzmh.hm / zzmh.r / zzgh.nb / zzba.p mismatches (which other key carries PM's value), and a
#         brute-force search for zzgh.nb over the trade stream (quantity x window x side source x denominator).
#   bash research/r188.sh sim 2>&1 | tee data/reports/r188_sim.txt        (same for band, gaps)
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
O=data/reports/r188; mkdir -p $O
DAYS=${DAYS:-"20261118 20261119 20261120"}
say() { echo "[$(date +%T)] $*"; }
prev_of() { ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$1 '$1 < d' | tail -1; }
case "${1:?sim|band|gaps}" in
sim)
  for D in $DAYS; do
    M=${D:4:4}
    say "== $D"
    python -u research/pm_factors/alpha_agree.py --pred s224=data/reports/stage6_s224_$M/pred.parquet dysl2=data/reports/stage6_r99_$M/pred.parquet \
      --full $DATA/catalog_dysl2.duckdb --sub $DATA/catalog_s224.duckdb 2>&1 | grep -v Warning | cut -c1-260
  done ;;
band)
  for D in $DAYS; do
    P=$(prev_of $D)
    [ -e $SP/$D.jz ] && [ -n "$P" ] || { echo "no .jz pair for $D"; continue; }
    S=dysl2; [ -d $DATA/dysl2_raw/kind=quotation/date=$D ] || S=s224
    say "== $D (prev $P, feed $S)"
    CRUCIBLE_SOURCE=$S python -u research/engine/band_trace.py --date $D --pm $SP/$D.jz --prev $SP/$P.jz --catalog $DATA/catalog_$S.duckdb --show 8 2>&1 | grep -v Warning | cut -c1-260
  done ;;
gaps)
  for D in $DAYS; do
    for S in dysl2 s224; do
      E=data/engine_r187/$S/engine_$D.json.gz
      [ -e $E ] || { echo "== $D $S: no $E"; continue; }
      say "== $D $S"
      C=""; [ $S = dysl2 ] && C="--catalog $DATA/catalog_$S.duckdb"
      CRUCIBLE_SOURCE=$S python -u research/engine/gap_probe.py --date $D --engine $E --pm $SP/$D.jz $C 2>&1 | grep -v Warning | cut -c1-260
    done
  done ;;
esac
say "r188 $1 done"
