#!/usr/bin/env bash
# Round 162 on 97, foreground, read-only on inputs; writes data/engine_r162 and data/reports/r162 (new dirs). ~5 min.
# Third set of candidates for the zzvx state against PM's day-end .jz on 1112 and 1113, and a breakdown of the band misses.
# R161: no narrower range explains trDEma*; its misses are mostly 1e-6..1e-4 relative with the same weight and clock, the
# footprint of a DOUBLE average (the 30 s average of the 30 s range average, lagging a range that only grows). Candidates:
#   zzvx.trDEmaD|dema   the 30 s average of trEma's value
#   zzvx.rEmaD|rte      average of range / trEma     |rtd  of range / the double average     |red  of trEma / the double average
# Band (unchanged rules st1..st6): for st1, the match of upper / lower split by PM's direction and ours, to see which state misses.
#   bash research/r162.sh 2>&1 | tee data/reports/r162.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
O=data/reports/r162; mkdir -p $O
python -u research/engine/selftest.py --seed 1 2>&1 | tail -2
for D in 20261112 20261113; do
  PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --threads 32 \
    --out data/engine_r162 --work data/engine_r162/work 2>&1 | tail -1 | cut -c1-160
  python -u research/engine/verify.py --date $D --engine data/engine_r162/engine_$D.json.gz --pm $SP/$D.jz --catalog $DATA/catalog_dysl2.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
    || echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"
  python - <<PY
import polars as pl
t = pl.read_csv("$O/verify_$D.csv").filter(pl.col("key").str.contains(r"^zz(vx|xu)\.")).sort("key")
print("== $D  key: n, equal share at 1e-6 (all / SZSE), at 1e-4 (all / SZSE), median relative error")
for r in t.iter_rows(named=True):
    print(f"   {r['key']:<24} n {r['n']:>4}  {r['all']:.4f} / {(r['SZSE'] if r['SZSE'] is not None else float('nan')):.4f}   {r['all_1e4']:.4f} / {(r['SZSE_1e4'] if r['SZSE_1e4'] is not None else float('nan')):.4f}   mre {(r['mre'] if r['mre'] is not None else float('nan')):.3g}")
PY
  python - <<PY
import gzip, json
e = json.load(gzip.open("data/engine_r162/engine_$D.json.gz")); pm = json.load(gzip.open("$SP/$D.jz"))
eq = lambda a, b: a is not None and b is not None and abs(a - b) <= max(1e-6, 1e-9 * abs(b))
cells = {}
for sid, kv in e.items():
    p = pm.get(sid)
    if not p or p.get("zzxu.dir") is None:
        continue
    k = (int(p["zzxu.dir"]), int(kv["zzxu.dir|st1"]))
    c = cells.setdefault(k, [0, 0, 0])
    c[0] += 1
    c[1] += eq(kv["zzxu.upper|st1"], p.get("zzxu.upper"))
    c[2] += eq(kv["zzxu.lower|st1"], p.get("zzxu.lower"))
print("   band st1 by (PM dir, our dir): stocks, upper equal, lower equal")
for k in sorted(cells):
    print(f"      PM {k[0]:+d} ours {k[1]:+d}: {cells[k][0]:>4}  {cells[k][1]:>4}  {cells[k][2]:>4}")
PY
done
echo "[$(date +%T)] r162 done"
