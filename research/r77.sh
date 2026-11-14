#!/usr/bin/env bash
# Round 77 on 97, foreground. Read-only; writes data/reports only.
# The open columns ranked by gain (R76: the 65 not settled carry 38% of the model's gain) with the closest candidate the
# last scans saw, so the next feature rounds aim at the heaviest ones: model_status over the baseline's assignment, and
# which ic_score reports mention the top open columns at all.
#   bash research/r77.sh 2>&1 | tee data/reports/r77.txt
set -u
cd /work/crucible
R=data/reports
echo "[$(date +%T)] model_status on the baseline (r41c gate)"
python -u research/pm_factors/model_status.py --stage5 $R/stage5 --assign $R/stage6_r41c_gate/ic_assign_gated.json --top 30 2>&1 | cut -c1-260 | tail -70
echo "[$(date +%T)] where the top open columns appear in the scan outputs (count of lines per file, top files)"
for k in zzeg zzop.i zzel.a zzel.b zzmh zzzl2.c zzpe.e zzhj zzba zzjx3.b; do
  echo "  $k: $(grep -rlwF "$k" $R --include=*.txt --include=*.json 2>/dev/null | wc -l) files; latest $(grep -rlwF "$k" $R --include=*.txt --include=*.json 2>/dev/null | xargs -r ls -t | head -3 | tr '\n' ' ')"
done
echo "[$(date +%T)] done"
