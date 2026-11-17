#!/usr/bin/env bash
# Round 152 on 97, foreground, read-only on inputs. Two independent parts.
#  poscst  runs_poscst: the per-stock constant poscst against the start-of-day table: what separates it from 5 x mb. ~1 min.
#  model   which_model: our 1112 / 1113 matrix through every model generation and fit (A, B, C, production tk5m months),
#          against PM's logged prediction -- the generation 224 really ran should agree clearly better. ~5-10 min.
#   bash research/r152.sh poscst 2>&1 | tee data/reports/r152_poscst.txt
#   bash research/r152.sh model  2>&1 | tee data/reports/r152_model.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
case "${1:?poscst|model}" in
poscst)
  DAYS=$(ls -d $DATA/pm_orders/date=* 2>/dev/null | sed 's/.*date=//' | tail -2 | paste -sd, -)
  python -u research/pm_factors/runs_poscst.py --orders $DATA/pm_orders --dates "$DAYS" --sts $DATA/pm_stsfile 2>&1 | grep -v -E "Warning|warn" | cut -c1-240 ;;
model)
  python -u research/featgen/which_model.py --days 20261112,20261113 --x-dir "data/reports/stage6_r99_{mmdd}" \
    --roots $DATA/models_gen/A,$DATA/models_gen/B,$DATA/models_gen/C,/data/prod/models/tk5m 2>&1 | grep -v Warning | cut -c1-200 ;;
esac
echo "[$(date +%T)] r152 $1 done"
