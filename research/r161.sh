#!/usr/bin/env bash
# Round 161 on 97, foreground, read-only on inputs; writes data/engine_r161 and data/reports/r161 (new dirs). ~5 min.
# Second set of candidates for the zzvx / zzxu state, checked value for value against PM's day-end .jz on 1112 and 1113.
# R160: trEma* = the snapshot's own high - low is exact; trDEma* is NARROWER on ~9% and rEma* > 1 there. Candidates:
#   zzvx.trDEmaD|sp     range of the snapshot PRICES in the continuous session   |spall  of all snapshot prices of the day
#   zzvx.trDEmaD|cont   range of the continuous session's trades
#   zzvx.rEmaD|rsp / |rspall / |rcont   the average of (day range / that narrower range)
#   zzxu.upper / lower / dir |st1..st6  six band rules (multiplier 3 around (high + low) / 2, width = the 300 s range average):
#     st1 simple, st2 classic, st3 classic with the current price, st4 classic starting down, st5 classic on the mid, st6 simple on the mid
# Each "|tag" row is scored against the PM key before the bar. 1 selftest. 2 per day: engine run, verify, the zzvx / zzxu rows.
#   bash research/r161.sh 2>&1 | tee data/reports/r161.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
O=data/reports/r161; mkdir -p $O
python -u research/engine/selftest.py --seed 1 2>&1 | tail -2
for D in 20261112 20261113; do
  PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --threads 32 \
    --out data/engine_r161 --work data/engine_r161/work 2>&1 | tail -1 | cut -c1-160
  python -u research/engine/verify.py --date $D --engine data/engine_r161/engine_$D.json.gz --pm $SP/$D.jz --catalog $DATA/catalog_dysl2.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
    || echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"
  python - <<PY
import polars as pl
t = pl.read_csv("$O/verify_$D.csv").filter(pl.col("key").str.contains(r"^zz(vx|xu)\.")).sort("key")
print("== $D  key: n, equal share at 1e-6 (all / SZSE), at 1e-4 (all / SZSE), median relative error")
for r in t.iter_rows(named=True):
    print(f"   {r['key']:<24} n {r['n']:>4}  {r['all']:.4f} / {(r['SZSE'] if r['SZSE'] is not None else float('nan')):.4f}   {r['all_1e4']:.4f} / {(r['SZSE_1e4'] if r['SZSE_1e4'] is not None else float('nan')):.4f}   mre {(r['mre'] if r['mre'] is not None else float('nan')):.3g}")
PY
done
echo "[$(date +%T)] r161 done"
