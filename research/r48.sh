#!/usr/bin/env bash
# Round 48 on 97, foreground, AFTER R47: one 224 day on PM's own market data. 2.23 keeps the HS300 mp files of 20261112 (the
# input PM's production read); until now every 224 day ran on DataYes. Same day, same stocks, same marks, two feeds:
#   0 pull prod/mp/20261112 from 2.23 (read-only there, ~0.94 GB) into /work/crucible_data/pm_mp
#   1 decode (decode_pm_mp.py, --check) into pmmp_raw, catalog_pmmp.duckdb          (CRUCIBLE_SOURCE=pmmp)
#   2 day-end state against PM's 20261112.jz (dcp_build) on the mp feed, next to the same on DataYes
#   3 full_book + engine state at the marks on the mp feed (own directories: nothing of the dysl2 run is overwritten)
#   4 r41c scored on the mp feed: against the 224 signal and the forward return; our prediction mp against dysl2 per mark
# Every write is under /work.   bash research/r48.sh 2>&1 | tee data/reports/r48.txt
set -u
D=${D:-20261112}
day=${D:4}
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
HV=$DATA/hstats_view/HS300
SEED=$DATA/seed_probe/HS300
H23=${H23:-root@172.21.2.23}
P23=${P23:-60777}
SSH="ssh -p $P23 -o BatchMode=yes"
C1=$R/stage6_r41c_gate/ic_assign_gated.json
ENG=data/engine_pmmp
FBP=$DATA/store/pm_factors/full_book_pmmp
say() { echo "[$(date +%T)] $*"; }

say "0 pull mp/$D from 2.23"
mkdir -p $DATA/pm_mp/$D
rsync -aL -e "$SSH" --include="*HS300*.mp" --exclude="*" $H23:/data/share/data/prod/mp/$D/ $DATA/pm_mp/$D/ > $R/r48_rsync.txt 2>&1 || say "  rsync failed: $(tail -1 $R/r48_rsync.txt)"
ls -l $DATA/pm_mp/$D | awk '{print "  ", $5, $9}'
[ "$(ls $DATA/pm_mp/$D/*.mp 2>/dev/null | wc -l)" -ge 6 ] || { say "mp files missing: stop"; exit 1; }

say "1 decode + catalog (source pmmp)"
python -u research/decode_pm_mp.py --root $DATA/pm_mp --date $D --out $DATA/pmmp_raw --workers 8 --check > $R/r48_decode.txt 2>&1 || say "  decode FAILED"
tail -30 $R/r48_decode.txt | cut -c1-200
CRUCIBLE_SOURCE=pmmp python -u research/catalog.py > $R/r48_catalog.txt 2>&1 || say "  catalog FAILED: $(tail -2 $R/r48_catalog.txt)"
python - <<PY 2>&1 | cut -c1-200
import duckdb
for src, cat in (("pmmp", "$DATA/catalog_pmmp.duckdb"), ("dysl2", "$DATA/catalog_dysl2.duckdb")):
    try:
        c = duckdb.connect(cat, read_only=True)
        r = [c.sql(f"SELECT count(*) FROM raw_{k} WHERE date = '$D'").fetchone()[0] for k in ("quotation", "order", "transaction")]
        print(f"  {src}: quotation {r[0]:,} order {r[1]:,} transaction {r[2]:,} rows on $D (pmmp = HS300 only, dysl2 = whole market)")
    except Exception as e:
        print(f"  {src}: {type(e).__name__}: {str(e)[:150]}")
PY

P=$(ls $SEED | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1)
echo "  PM state before $D: ${P:-none}; $D.jz $([ -e $SEED/$D.jz ] && echo present || echo missing)"

say "2 day-end state against PM's $D.jz, mp feed"
CRUCIBLE_SOURCE=pmmp python -u research/pm_factors/dcp_build.py --dates $D --dcp-dir $SEED --out $R/stage4_pmmp > $R/r48_dcp_pmmp.txt 2>&1 || say "  dcp_build FAILED: $(tail -3 $R/r48_dcp_pmmp.txt)"
tail -40 $R/r48_dcp_pmmp.txt | cut -c1-200
say "  the same on DataYes"
if [ ! -f $R/add_${D}_dcp.txt ]; then
  CRUCIBLE_SOURCE=dysl2 python -u research/pm_factors/dcp_build.py --dates $D --dcp-dir $SEED --out $R/stage4_r48_dysl2 > $R/add_${D}_dcp.txt 2>&1 || say "  dcp_build (dysl2) FAILED"
fi
tail -40 $R/add_${D}_dcp.txt | cut -c1-200

say "3 full_book + engine on the mp feed"
CRUCIBLE_SOURCE=pmmp python -u research/pm_factors/full_book.py --dates $D --marks rand:240 --workers 1 > $R/r48_fullbook.txt 2>&1 || say "  full_book FAILED: $(tail -3 $R/r48_fullbook.txt)"
ls -la $FBP/date=$D.parquet 2>&1 | sed 's/^/  /'
python -u research/engine/run_cpp.py --date $D --universe pm:$SEED/$P.jz --prev $SEED/$P.jz --catalog $DATA/catalog_pmmp.duckdb \
  --marks rand:240 --threads 32 --out $ENG --work $ENG/work > $R/r48_engine.txt 2>&1 || say "  engine FAILED: $(tail -3 $R/r48_engine.txt)"
ls -la $ENG/marks_$D.parquet 2>&1 | sed 's/^/  /'

say "4 r41c on the mp feed"
B="--models /data/prod/models/tk5m --drop-gap 1 --max-cdf 1 --max-score 99 --state-fallback --hstats $HV --neutral --no-drop-harmful --max-features 0 --implied 0 --dump-pred"
X="--marks rand:240 --full-book $FBP --engine-dir $ENG --seed-dir $SEED"
CRUCIBLE_SOURCE=pmmp python -u research/pm_factors/ic_score.py --dates $D --assign $C1 $B $X --out $R/stage6_r48_fwd > $R/r48_fwd.txt 2>&1
grep -E "Error|Traceback|seed state|IC: start" $R/r48_fwd.txt | cut -c1-160
grep -cE "^drop " $R/r48_fwd.txt | sed 's/^/  columns dropped at load: /'
CRUCIBLE_SOURCE=pmmp python -u research/pm_factors/ic_score.py --dates $D --assign $C1 $B $X --target $DATA/pm_target_r22 --target-stale 4 \
  --out $R/stage6_r48_r41c > $R/r48_r41c.txt 2>&1
grep -E "Error|Traceback|IC: start" $R/r48_r41c.txt | cut -c1-160
python -u research/pm_factors/pred_metrics.py --pred $R/stage6_r48_r41c --fwd $R/stage6_r48_fwd --tag "$day r41c mp feed" 2>&1 | cut -c1-200
echo "  for comparison, DataYes feed:"; cat $R/r44_${day}_r41c_metrics.txt 2>/dev/null | cut -c1-200
python - <<PY 2>&1 | cut -c1-200
import numpy as np, polars as pl
try:
    a = pl.read_parquet("$R/stage6_r48_r41c/pred.parquet").select("sid", "mark", pl.col("ours").alias("mp"), pl.col("pm").alias("pm_mp"))
    b = pl.read_parquet("$R/stage6_r44_${day}_r41c/pred.parquet").select("sid", "mark", pl.col("ours").alias("dy"), pl.col("pm").alias("pm_dy"))
    j = a.join(b, on=["sid", "mark"], how="inner").filter(pl.col("mp").is_finite() & pl.col("dy").is_finite())
    v = [g.select(pl.corr("mp", "dy", method="spearman")).item() for _, g in j.group_by("mark") if g.height >= 30]
    v = np.array([x for x in v if x is not None and np.isfinite(x)])
    print(f"  our prediction, mp feed against DataYes feed: {j.height:,} rows, {v.size} marks, rank correlation per mark mean {v.mean():.4f} min {v.min():.4f} p10 {np.quantile(v, .1):.4f}")
    print(f"  rows of mp run {a.height:,}, of DataYes run {b.height:,}; target equal on joined rows: {float((j['pm_mp'] - j['pm_dy']).abs().max()):.3g}")
except Exception as e:
    print(f"  compare failed: {type(e).__name__}: {str(e)[:150]}")
PY
grep -lE "Error|Traceback" $R/r48*.txt 2>/dev/null | head | sed 's/^/  errors in /'
say "done"
