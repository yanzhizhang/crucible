#!/usr/bin/env bash
# Round 166 on 97, foreground, read-only on inputs; writes data/engine_r166, data/single_fill_r166. ~10 min.
# zzvx.a: its training thresholds never go below 1 and start at 1.0000005; zzvx.rEma (the average of trEma / trDEma, now
# equal to PM's state) is 1 once the range has settled and above 1 while it widens. Is the column rEma itself?
#  1 engine state at the 240 marks of 1112 / 1113 with the new zzvx rules
#  2 seg_fit: zzvx.a against the zzvx keys at the marks (range, thresholds, root points), generations C and B
#  3 single_fill: replace our zzvx.a by rEma in the scoring matrix; change in agreement with PM's logged prediction
#   bash research/r166.sh 2>&1 | tee data/reports/r166.txt
set -u
cd /work/crucible
DATA=/work/crucible_data
SP=$DATA/seed_probe/HS300
for D in 20261112 20261113; do
  PV=$SP/$(ls $SP | grep -E '^[0-9]{8}\.jz$' | sed 's/\.jz//' | awk -v d=$D '$1 < d' | tail -1).jz
  python -u research/engine/run_cpp.py --date $D --universe pm:$PV --prev $PV --catalog $DATA/catalog_dysl2.duckdb --marks rand:240 --threads 32 \
    --out data/engine_r166 --work data/engine_r166/work 2>&1 | tail -2 | cut -c1-160
done
python - <<'PY'
import polars as pl
for d in ("20261112", "20261113"):
    t = pl.read_parquet(f"data/engine_r166/marks_{d}.parquet")
    r = t["zzvx.rEmaD.sum"].drop_nulls().drop_nans()
    print(f"{d}: marks {t.height} rows; zzvx columns {[k for k in t.columns if k.startswith('zzvx.')]}")
    print(f"   rEmaD at the marks: share == 1 (1e-9) {float(((r - 1).abs() < 1e-9).mean()):.3f}; q1 / q50 / q90 / q99 / max "
          + " / ".join(f"{r.quantile(x):.6g}" for x in (0.01, 0.5, 0.9, 0.99)) + f" / {r.max():.6g}   (zzvx.a thresholds q1 / q50 / q99: 1.0000005 / 1.0044 / 1.139; range to 2.49)")
PY
for G in C B; do
  echo "######## seg_fit, generation $G"
  python -u research/featgen/seg_fit.py --days 20261112,20261113 --target zzvx.a --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" \
    --no-base --cand-re "zzvx" --extra "data/engine_r166/marks_{date}.parquet" --models $DATA/models_gen/$G --transforms=id,1/b,log,sqrt --scales 1,100 --top 8 2>&1 \
    | grep --line-buffered -v Warning | stdbuf -oL cut -c1-230
done
for G in C B; do
  for D in 20261112 20261113; do
    echo "######## single_fill $D, generation $G"
    python -u research/featgen/single_fill.py --day $D --x-dir "data/reports/stage6_r99_{mmdd}" --base "data/featgen_r99/base_{date}.parquet" --extra "data/engine_r166/marks_{date}.parquet" \
      --fills "zzvx.a:id:x.zzvx.rEmaD.sum,zzvx.a:id:x.zzvx.rEmaO.sum,zzvx.a:1/b:x.zzvx.rEmaD.sum,zzvx.a:id:x.zzvx.trDEmaD.sum,zzvx.a:id:x.zzvx.trEmaD.sum" \
      --models $DATA/models_gen/$G --subs s1,s2,s3,s4 --work data/single_fill_r166 2>&1 | grep -v Warning | cut -c1-220
  done
done
echo "[$(date +%T)] r166 done"
