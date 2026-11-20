#!/usr/bin/env bash
# Round 187 on 97, foreground: bring 20261119 / 20261120 in on both feeds and check the engine against PM's state on them.
# Read-only on PM data and on the pulls; writes the usual per-day stores plus data/engine_r187, data/reports/r187.
#  dec    HEAVY, alone (96 workers): 224's own feed 20261120 (tdf_224) -> s224_raw + catalog_s224; the DataYesSharesL2
#         packages of 20261117 / 20261119 / 20261120 that are on 97 and not decoded yet -> dysl2_raw + catalog_dysl2
#         (1117 is the missing day behind the zzeg channel on 1118).
#  score  ~15-30 min per day, moderate: r177.sh on the s224 feed for 20261119 and 20261120 (builds pm_signals from the
#         pulled chrono logs first), then r99.sh on dysl2 for the same days where dysl2 has them.
#  state  ~3 min: the engine's day-end state against PM's .jz for 20261118 / 19 / 20, on each feed (s224, dysl2): every
#         key below 0.99 at 1e-4, and the zzvx / zzeg / zzxu rows. PM's own feed is neither, so this shows which of our
#         two feeds reproduces PM's state better, and whether zzvx stays at 1.0 on new days.
#   bash research/r187.sh dec 2>&1 | tee data/reports/r187_dec.txt       (then score, state; state needs dec only)
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
DY=${DY:-/mnt/TickDataRepo/DataYesSharesL2}
say() { echo "[$(date +%T)] $*"; }
case "${1:?dec|score|state}" in
dec)
  say "s224 20261120"
  if [ -d $DATA/s224_raw/kind=quotation/date=20261120 ]; then echo "  already decoded"; else
    python -u research/decode_tdf_day.py --root $DATA/tdf_224 --date 20261120 --out $DATA/s224_raw --workers 96 2>&1 | grep -v Warning | tail -6 | cut -c1-200; fi
  CRUCIBLE_SOURCE=s224 python -u research/catalog.py 2>&1 | tail -3 | cut -c1-200
  for D in 20261117 20261119 20261120; do
    say "dysl2 $D"
    if [ -d $DATA/dysl2_raw/kind=quotation/date=$D ]; then echo "  already decoded"; continue; fi
    if [ -e $DY/$D.zip ] || [ -d $DY/$D ]; then
      python -u research/decode_datayes_day.py --root $DY --date $D --out $DATA/dysl2_raw --workers 96 --kinds quotation,order,transaction,index 2>&1 | grep -v Warning | tail -3 | cut -c1-200
    else echo "  no package $DY/$D(.zip) on 97"; fi
  done
  CRUCIBLE_SOURCE=dysl2 python -u research/catalog.py 2>&1 | tail -3 | cut -c1-200
  for S in s224 dysl2; do
    CRUCIBLE_SOURCE=$S python -c "import sys; sys.path.insert(0,'research/pm_factors'); import duckdb, dcp_build as B; c=duckdb.connect(str(B.CATALOG), read_only=True); print('  $S quotes per day:', c.sql('SELECT date, count(*), count(DISTINCT symbol_id) FROM raw_quotation WHERE date >= 20261117 GROUP BY 1 ORDER BY 1').fetchall())" 2>&1 | tail -1 | cut -c1-260
  done ;;
score)
  DAYS="20261119 20261120" bash research/r177.sh 2>&1 | cut -c1-230
  say "dysl2 for the same days"
  DAYS="20261119 20261120" bash research/r99.sh 2>&1 | cut -c1-230 ;;
state)
  O=data/reports/r187; mkdir -p $O
  for D in 20261118 20261119 20261120; do
    [ -e $SP/$D.jz ] || { echo "no $SP/$D.jz"; continue; }
    PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
    for S in s224 dysl2; do
      [ -d $DATA/${S}_raw/kind=quotation/date=$D ] || { echo "== $D $S: no ticks"; continue; }
      python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_$S.duckdb --chan-hist "${S}_raw,dysl2_raw,feitu_raw" --threads 32 \
        --out data/engine_r187/$S --work data/engine_r187/$S/work > $O/engine_${D}_$S.txt 2>&1 || { echo "== $D $S: engine FAILED: $(tail -2 $O/engine_${D}_$S.txt)"; continue; }
      CRUCIBLE_SOURCE=$S python -u research/engine/verify.py --date $D --engine data/engine_r187/$S/engine_$D.json.gz --pm $SP/$D.jz --catalog $DATA/catalog_$S.duckdb --csv $O/verify_${D}_$S.csv > $O/verify_${D}_$S.txt 2>&1 \
        || echo "  verify FAILED: $(tail -2 $O/verify_${D}_$S.txt)"
    done
    python - <<PY 2>&1 | cut -c1-250
import polars as pl
from pathlib import Path
t = {}
for s in ("s224", "dysl2"):
    f = Path("$O/verify_${D}_%s.csv" % s)
    if f.exists():
        t[s] = pl.read_csv(f).filter(~pl.col("key").str.contains(r"\|"))
print("== $D (prev $(basename $PV)); feeds with a result:", list(t))
if t:
    j = None
    for s, x in t.items():
        x = x.select("key", pl.col("all").alias(f"{s}_1e6"), pl.col("all_1e4").alias(f"{s}_1e4"))
        j = x if j is None else j.join(x, on="key", how="full", coalesce=True)
    val = j.filter(~pl.col("key").str.ends_with(".t")).sort("key")
    cols = [c for c in j.columns if c != "key"]
    f = lambda v: "  -   " if v is None else f"{v:.4f}"
    print("   key: " + " | ".join(cols))
    low = val.filter(pl.any_horizontal([pl.col(c) < 0.99 for c in cols if c.endswith("1e4")]) | pl.col("key").str.contains(r"^zz(vx|eg|xu)\."))
    for r in low.iter_rows(named=True):
        print(f"   {r['key']:<22} " + " | ".join(f(r[c]) for c in cols))
    for c in cols:
        v = val[c].drop_nulls()
        print(f"   {c}: {v.len()} value keys, mean {v.mean():.4f}, at 1.0 on {int((v >= 0.9999).sum())}, below 0.99 on {int((v < 0.99).sum())}")
PY
  done ;;
esac
say "r187 $1 done"
