#!/usr/bin/env bash
# Round 164 on 97, foreground, read-only on inputs; writes data/engine_r164 and data/reports/r164. ~5 min.
# zzxu band: where does its state come from? PM's active band always sits beyond the day's low / high, which a band started
# fresh at the open (range average ~0) cannot do. Four variants of the simple rule against PM's day-end .jz on 1112 / 1113:
#   b1 fresh width, fresh band (= R162's st1)      b2 width carried from yesterday (zzxu.a|c0), fresh band
#   b3 width and band carried from yesterday's .jz  b4 fresh width, band carried
# Per day: the zzxu rows of verify, then per variant the (PM dir, our dir) table with upper / lower matches.
#   bash research/r164.sh 2>&1 | tee data/reports/r164.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
O=data/reports/r164; mkdir -p $O
python -u research/engine/selftest.py --seed 1 2>&1 | tail -2
for D in 20261112 20261113; do
  PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --threads 32 \
    --out data/engine_r164 --work data/engine_r164/work 2>&1 | tail -1 | cut -c1-160
  python -u research/engine/verify.py --date $D --engine data/engine_r164/engine_$D.json.gz --pm $SP/$D.jz --catalog $DATA/catalog_dysl2.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
    || echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"
  python - <<PY
import gzip, json
import polars as pl
t = pl.read_csv("$O/verify_$D.csv").filter(pl.col("key").str.contains(r"^zzxu\.") & ~pl.col("key").str.contains(r"\.t(\||$)")).sort("key")
print("== $D (prev $(basename $PV))  zzxu key: n, equal share at 1e-6 (all / SZSE), at 1e-4 (all / SZSE), median relative error")
for r in t.iter_rows(named=True):
    print(f"   {r['key']:<18} n {r['n']:>4}  {r['all']:.4f} / {(r['SZSE'] or 0):.4f}   {r['all_1e4']:.4f} / {(r['SZSE_1e4'] or 0):.4f}   mre {(r['mre'] if r['mre'] is not None else float('nan')):.3g}")
e = json.load(gzip.open("data/engine_r164/engine_$D.json.gz")); pm = json.load(gzip.open("$SP/$D.jz"))
eq = lambda a, b: a is not None and b is not None and abs(a - b) <= max(1e-6, 1e-9 * abs(b))
for tag in ("b1", "b2", "b3", "b4"):
    cells = {}
    for sid, kv in e.items():
        p = pm.get(sid)
        if not p or p.get("zzxu.dir") is None or kv.get(f"zzxu.dir|{tag}") is None:
            continue
        c = cells.setdefault((int(p["zzxu.dir"]), int(kv[f"zzxu.dir|{tag}"])), [0, 0, 0])
        c[0] += 1
        c[1] += eq(kv[f"zzxu.upper|{tag}"], p.get("zzxu.upper"))
        c[2] += eq(kv[f"zzxu.lower|{tag}"], p.get("zzxu.lower"))
    print(f"   {tag} by (PM dir, ours): stocks / upper equal / lower equal:  " + "  |  ".join(f"PM {k[0]:+d} ours {k[1]:+d}: {v[0]} / {v[1]} / {v[2]}" for k, v in sorted(cells.items())))
PY
done
echo "[$(date +%T)] r164 done"
