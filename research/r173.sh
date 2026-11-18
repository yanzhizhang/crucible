#!/usr/bin/env bash
# Round 173 on 97, foreground, read-only on inputs; writes data/engine_r173, data/reports/r173. ~3 min.
# The whole engine state against PM's day-end file on the fresh day 20261118 (seeded from 20261117.jz, both now on 97), and on
# 1113 for comparison: every key, its match share at 1e-6 and 1e-4, per exchange. This is the status table of the engine:
# which keys are equal to PM, which are close, which are not built. Candidate keys of the zzxu band ("|m....") are left out.
#   bash research/r173.sh 2>&1 | tee data/reports/r173.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
O=data/reports/r173; mkdir -p $O
for D in 20261118 20261113; do
  PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
  [ -e $SP/$D.jz ] || { echo "no $SP/$D.jz"; continue; }
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --threads 32 \
    --out data/engine_r173 --work data/engine_r173/work 2>&1 | tail -1 | cut -c1-160
  python -u research/engine/verify.py --date $D --engine data/engine_r173/engine_$D.json.gz --pm $SP/$D.jz --catalog $DATA/catalog_dysl2.duckdb --csv $O/verify_$D.csv > $O/verify_$D.txt 2>&1 \
    || echo "  verify FAILED: $(tail -2 $O/verify_$D.txt)"
  python - <<PY
import gzip, json, re
import polars as pl
t = pl.read_csv("$O/verify_$D.csv").filter(~pl.col("key").str.contains(r"\|m\."))
print("== $D (prev $(basename $PV)): %d keys; equal at 1e-6: all / SZSE | at 1e-4: all / SZSE | n" % t.height)
for r in t.sort("key").iter_rows(named=True):
    f = lambda v: "  -   " if v is None else f"{v:.4f}"
    print(f"   {r['key']:<24} {f(r['all'])} / {f(r['SZSE'])} | {f(r['all_1e4'])} / {f(r['SZSE_1e4'])} | {r['n']}")
pm = json.load(gzip.open("$SP/$D.jz"))
keys = sorted({re.sub(r"\.\d+$", ".<n>", k) for kv in pm.values() for k in kv})
ours = set(pl.read_csv("$O/verify_$D.csv")["key"].str.replace(r"\|.*$", "").to_list())
base = lambda k: re.sub(r"\.(sum|wgt|t)$", "", k)
miss = sorted({base(k) for k in keys} - {base(k) for k in ours} - {k for k in ours})
print(f"   PM keys (grids folded): {len(keys)}; PM key stems the engine does not produce at all: {miss}")
PY
done
echo "[$(date +%T)] r173 done"
