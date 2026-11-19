#!/usr/bin/env bash
# Round 177 on 97, foreground, read-only on PM data (an experiment the user asked for, low priority): the 224 alpha rebuilt
# from 224's OWN feed. Same pipeline as R172 (r99.sh), but every market-data step reads the s224 source (R176's decode):
# full_book, engine marks, ic_score's matrix against PM's prediction (pm_signals = the alpha.parquet stream), bases.
# Outputs go to their own places, nothing of the dysl2 run is overwritten:
#   store/pm_factors/full_book_s224, data/engine_s224, data/reports/stage6_s224_<mmdd>, data/featgen_s224
# Then, per day: the similarity on s224 next to the dysl2 one, and column by column how much the feed changes our
# matrix (correlation and share of equal values between the two x.parquet). ~15-30 min per day; engine runs one at a time.
#   bash research/r177.sh 2>&1 | tee data/reports/r177.txt            (DAYS="20261118 20261119" by default)
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
SEED=$DATA/seed_probe/HS300
SRC=s224
FBD=$DATA/store/pm_factors/full_book_$SRC
TGT=$DATA/pm_target_r22
C1=$R/stage6_r41c_gate/ic_assign_gated.json
G=$DATA/models_gen/B
DAYS=${DAYS:-20261118 20261119}
say() { echo "[$(date +%T)] $*"; }
ic() { grep -E "IC: start" "$1" 2>/dev/null | tail -1 | sed -E 's/.*IC: start ([0-9.]+) -> end ([0-9.]+).*/\2/'; }
SCORE="--no-drop-harmful --max-features 0 --implied 0 --drop-gap 1 --state-fallback --hstats $DATA/hstats_view/HS300 --neutral --target $TGT --target-stale 4 --max-cdf 1 --max-score 99 --models $G --dump-x --dump-pred"
for D in $DAYS; do
  say "$D"
  if [ ! -e $DATA/pm_signals/date=$D.parquet ]; then
    L=$DATA/pm_224/$D/chrono_logs
    [ -d $L ] || { echo "  no pm_signals and no $L for $D: skipped"; continue; }
    python -u research/pm_factors/pm_signals.py --logs $L --dates $D --out $DATA/pm_signals > $R/r177_${D}_signals.txt 2>&1 || { echo "  pm_signals FAILED: $(tail -2 $R/r177_${D}_signals.txt)"; continue; }
    grep -E "signals,|->" $R/r177_${D}_signals.txt | cut -c1-200 | sed 's/^/  /'
  fi
  [ -e $TGT/date=$D.parquet ] || ln -sf $DATA/pm_signals/date=$D.parquet $TGT/
  [ -d $DATA/${SRC}_raw/kind=quotation/date=$D ] || { echo "  no $SRC ticks for $D: skipped"; continue; }
  P=$(ls $SEED | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1)
  [ -n "$P" ] || { echo "  no PM state before $D: skipped"; continue; }
  echo "  seed $P.jz; hstats families with $D: $(ls -d $DATA/hstats_view/HS300/*/$D 2>/dev/null | wc -l)"
  if [ ! -e $FBD/date=$D.parquet ]; then
    CRUCIBLE_SOURCE=$SRC python -u research/pm_factors/full_book.py --dates $D --marks rand:240 --workers 1 > $R/r177_${D}_fullbook.txt 2>&1 \
      || { echo "  full_book FAILED: $(tail -2 $R/r177_${D}_fullbook.txt)"; continue; }
  fi
  python -u research/engine/run_cpp.py --date $D --universe pm:$SEED/$P.jz --prev $SEED/$P.jz --catalog $DATA/catalog_$SRC.duckdb \
    --chan-hist "${SRC}_raw,dysl2_raw,feitu_raw" --marks rand:240 --threads 32 --out data/engine_$SRC --work data/engine_$SRC/work \
    > $R/r177_${D}_engine.txt 2>&1 || { echo "  engine FAILED: $(tail -2 $R/r177_${D}_engine.txt)"; continue; }
  ST=$SEED/$D.jz; [ -e $ST ] || ST=$SEED/$P.jz
  O=$R/stage6_${SRC}_${D:4}
  CRUCIBLE_SOURCE=$SRC python -u research/pm_factors/ic_score.py --dates $D --assign $C1 $SCORE --marks rand:240 --engine-dir data/engine_$SRC \
    --full-book $FBD --seed-dir $SEED --out $O > $R/r177_${D}_ic.txt 2>&1
  echo "  similarity on $SRC: $(ic $R/r177_${D}_ic.txt)   (dysl2, R172 / r99: $(ic $R/r99_${D}_ic.txt))  $(grep -E 'Error|Traceback' $R/r177_${D}_ic.txt | head -1 | cut -c1-120)"
  CRUCIBLE_SOURCE=$SRC python -u research/featgen/run_day.py --date $D --state $ST --seed-dir $SEED --marks rand:240 --out data/featgen_$SRC 2>&1 | tail -1 | cut -c1-200
  python - <<PY 2>&1 | grep -v Warning | cut -c1-220
import numpy as np, polars as pl
from pathlib import Path
a, b = Path("$O/x.parquet"), Path("$R/stage6_r99_${D:4}/x.parquet")
if not (a.exists() and b.exists()):
    print("  no pair of matrices to compare:", a.exists(), b.exists())
else:
    A = pl.read_parquet(a).with_columns(pl.col("sid").cast(pl.Int64)); B = pl.read_parquet(b).with_columns(pl.col("sid").cast(pl.Int64))
    k = [c for c in ("date", "sid", "mark") if c in A.columns and c in B.columns]
    J = A.join(B, on=k, how="inner", suffix="__d")
    cols = [c for c in A.columns if c not in k and f"{c}__d" in J.columns]
    rows = []
    for c in cols:
        x, y = J[c].cast(pl.Float64).to_numpy(), J[f"{c}__d"].cast(pl.Float64).to_numpy()
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 1000:
            continue
        eq = float(np.mean(np.abs(x[m] - y[m]) <= 1e-9 * np.maximum(np.abs(y[m]), 1e-12)))
        r = float(np.corrcoef(x[m], y[m])[0, 1]) if x[m].std() > 0 and y[m].std() > 0 else float("nan")
        rows.append((r, eq, c))
    rows.sort(key=lambda t: (t[0] if t[0] == t[0] else 2))
    r_ = np.array([t[0] for t in rows]); e_ = np.array([t[1] for t in rows])
    print(f"  matrices s224 vs dysl2: {J.height:,} shared rows of {A.height:,} / {B.height:,}; {len(rows)} columns; correlation q10 / q50 / q90 "
          + " / ".join(f"{v:.4f}" for v in np.nanquantile(r_, [.1, .5, .9])) + f"; equal-value share q10 / q50 / q90 " + " / ".join(f"{v:.3f}" for v in np.quantile(e_, [.1, .5, .9])))
    print("  the 12 columns the feed changes most (correlation, equal share): " + ", ".join(f"{c} {r:.3f} {e:.2f}" for r, e, c in rows[:12]))
PY
done
say "r177 done"
