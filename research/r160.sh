#!/usr/bin/env bash
# Round 160 on 97, foreground, read-only on inputs; writes data/engine_r160 and data/reports/r160 (new dirs). ~5 min.
# Candidate fixes for the zzvx / zzxu state, checked value for value against PM's day-end .jz on 1112 and 1113:
#   zzvx.trEmaD|pc      true range against the previous close        zzvx.trEmaD|trade   the trades' day range
#   zzvx.trEmaD|tro     the trades' range against the open           zzvx.rEmaD|tr       average of true range / day range
#   zzvx.rEmaD|tratio   average of trades' range / snapshot range
#   zzxu.upper|st / lower|st / dir|st   a trend band, multiplier 3 around (high + low) / 2, width = the 300 s range average
# Each "|tag" row is scored against the PM key before the bar; the untagged rows are today's rules (trEmaD, rEmaD: right
# only where PM's two range averages agree). 1 selftest. 2 per day: engine run, verify, the zzvx / zzxu rows.
#   bash research/r160.sh 2>&1 | tee data/reports/r160.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
O=data/reports/r160; mkdir -p $O
python -u research/engine/selftest.py --seed 1 2>&1 | tail -2
for D in 20261112 20261113; do
  PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --threads 32 \
    --out data/engine_r160 --work data/engine_r160/work 2>&1 | tail -1 | cut -c1-160
  python -u research/engine/verify.py --date $D --engine data/engine_r160/engine_$D.json.gz --pm $SP/$D.jz --catalog $DATA/catalog_dysl2.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
    || echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"
  python - <<PY
import polars as pl
t = pl.read_csv("$O/verify_$D.csv").filter(pl.col("key").str.contains(r"^zz(vx|xu)\.")).sort("key")
print("== $D  key: n, equal share at 1e-6 (all / SZSE), at 1e-4 (all / SZSE), median relative error")
for r in t.iter_rows(named=True):
    print(f"   {r['key']:<24} n {r['n']:>4}  {r['all']:.4f} / {(r['SZSE'] if r['SZSE'] is not None else float('nan')):.4f}   {r['all_1e4']:.4f} / {(r['SZSE_1e4'] if r['SZSE_1e4'] is not None else float('nan')):.4f}   mre {(r['mre'] if r['mre'] is not None else float('nan')):.3g}")
PY
done
echo "[$(date +%T)] r160 done"
