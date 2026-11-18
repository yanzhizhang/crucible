#!/usr/bin/env bash
# Round 170 on 97, foreground, read-only on inputs; writes data/engine_r170 and data/reports/r170. ~5 min.
# zzxu active band: a wider search on the CARRIED width. R169: when the band starts makes no difference, and our band never
# flips down where PM's does, so PM's active band is tighter than a ratchet of mid - 3 w. For PM's up-trend stocks, PM's lower
# band against 75 running maxima of centre - mult x width; for the down-trend stocks, PM's upper against the 75 running minima
# (centre mid / px / p / hi / lo; mult 1, 1.5, 2, 2.5, 3; width c0 = zzxu.a carried, tr = zzvx.trEma, de = zzvx.trDEma).
# A stock that never flipped has its active band equal to the right one. Also the zzvx rows (should stay 1.0000).
#   bash research/r170.sh 2>&1 | tee data/reports/r170.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
O=data/reports/r170; mkdir -p $O
python -u research/engine/selftest.py --seed 1 2>&1 | tail -2
for D in 20261112 20261113; do
  PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --threads 32 \
    --out data/engine_r170 --work data/engine_r170/work 2>&1 | tail -1 | cut -c1-160
  python -u research/engine/verify.py --date $D --engine data/engine_r170/engine_$D.json.gz --pm $SP/$D.jz --catalog $DATA/catalog_dysl2.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
    || echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"
  python - <<PY
import gzip, json
import polars as pl
t = pl.read_csv("$O/verify_$D.csv").filter(pl.col("key").str.contains(r"^zzvx\.") & ~pl.col("key").str.ends_with(".t")).sort("key")
print("== $D  zzvx key: n, equal share at 1e-6 (all / SZSE), at 1e-4 (all / SZSE)")
for r in t.iter_rows(named=True):
    print(f"   {r['key']:<16} n {r['n']:>4}  {r['all']:.4f} / {(r['SZSE'] or 0):.4f}   {r['all_1e4']:.4f} / {(r['SZSE_1e4'] or 0):.4f}")
w = pl.read_csv("$O/verify_$D.csv").sort("all").head(12)
print("   the 12 worst keys of the whole table: " + ", ".join(f"{r['key']} {r['all']:.3f}" for r in w.iter_rows(named=True)))
e = json.load(gzip.open("data/engine_r170/engine_$D.json.gz")); pm = json.load(gzip.open("$SP/$D.jz"))
eq = lambda a, b, tol=1e-6: a is not None and b is not None and abs(a - b) <= max(tol, 1e-9 * abs(b))
for side, pdir in (("lower", 1), ("upper", -1)):
    cnt, n = {}, 0
    for sid, kv in e.items():
        p = pm.get(sid)
        if not p or p.get("zzxu.dir") != pdir or p.get(f"zzxu.{side}") is None:
            continue
        n += 1
        for k, v in kv.items():
            if k.startswith(f"zzxu.{side}|"):
                c = cnt.setdefault(k, [0, 0, []])
                c[0] += eq(v, p[f"zzxu.{side}"])
                c[1] += eq(v, p[f"zzxu.{side}"], 1e-4 * abs(p[f"zzxu.{side}"]))
                if v is not None:
                    c[2].append(abs(v - p[f"zzxu.{side}"]) / abs(p[f"zzxu.{side}"]))
    print(f"   PM dir {pdir:+d}: {n} stocks; PM's {side} band equal to (share at 1e-6, at 1e-4 relative, median relative error), best 10:")
    for k, c in sorted(cnt.items(), key=lambda kv: -kv[1][0])[:10]:
        med = sorted(c[2])[len(c[2]) // 2] if c[2] else float("nan")
        print(f"      {k:<28} {c[0] / max(n, 1):.3f}  {c[1] / max(n, 1):.3f}  {med:.3g}")
PY
done
echo "[$(date +%T)] r170 done"
