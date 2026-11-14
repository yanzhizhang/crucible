#!/usr/bin/env bash
# Round 60 on 97, foreground. Writes under /work only; no remote machine is touched.
# The C++ base-quantity program (research/featgen: the 89 quantities behind the baseline's 144 formulas, from snapshots,
# trades, orders and yesterday's state) on real days, against the Python definitions it was ported from:
#   0 the synthetic self-test (Python against C++), to see the build works on 97
#   1 20261112 (DataYes) at the scoring marks, 8 stocks compared value by value with Python
#   2 20260430 (feitu) at its scoring marks, 8 stocks compared
#   3 20261112 on a 3-second grid (4740 times x 300 stocks): the time the per-snapshot base takes
#   bash research/r60.sh 2>&1 | tee data/reports/r60.txt
set -u
cd /work/crucible
R=data/reports
SEED=/work/crucible_data/seed_probe/HS300
P1=/data/prod/h/HS300/p1_intrading_dcp
say() { echo "[$(date +%T)] $*"; }
say "0 self-test"
python -u research/featgen/selftest.py --work data/featgen_selftest > $R/r60_selftest.txt 2>&1; echo "  exit $?"; tail -2 $R/r60_selftest.txt | cut -c1-200
grep -E "^  OFF" $R/r60_selftest.txt | cut -c1-200 | head -20
say "1 20261112 at rand:240, 8 stocks against Python"
CRUCIBLE_SOURCE=dysl2 python -u research/featgen/run_day.py --date 20261112 --state $SEED/20261112.jz --seed-dir $SEED --marks rand:240 --check 8 2>&1 | cut -c1-230
say "2 20260430 at rand:240:7, 8 stocks against Python"
CRUCIBLE_SOURCE=feitu python -u research/featgen/run_day.py --date 20260430 --state $P1/20260430.jz --seed-dir $P1 --marks rand:240:7 --check 8 2>&1 | cut -c1-230
say "3 20261112 every 3 s"
CRUCIBLE_SOURCE=dysl2 python -u research/featgen/run_day.py --date 20261112 --state $SEED/20261112.jz --seed-dir $SEED --grid 3 --out data/featgen_3s 2>&1 | cut -c1-230
ls -la data/featgen_3s/ 2>&1 | sed 's/^/  /'
say "done"
