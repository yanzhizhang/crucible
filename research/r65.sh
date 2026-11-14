#!/usr/bin/env bash
# Round 65 on 97, foreground, read-mostly (writes under /work). One open difference: gen.stoch (column zzro2.a) on 20260430
# differs between the C++ columns and ic_score's matrix on ~3 % of cells (271 stocks, most on 688xxx), though the base
# quantity agreed with Python on the 8 stocks checked. Here:
#   1 the differing cells of the top stocks: mark, ic_score's value, the C++ base value, and the trades around the mark
#   2 featgen against the Python definitions on exactly those stocks
#   bash research/r65.sh 2>&1 | tee data/reports/r65.txt
set -u
cd /work/crucible
R=data/reports
P1=/data/prod/h/HS300/p1_intrading_dcp
python - <<'PY' 2>&1 | cut -c1-230
import numpy as np, polars as pl
x = pl.read_parquet("data/reports/stage6_r64_0430_snap/x.parquet", columns=["sid", "mark", "zzro2.a"]).with_columns(pl.col("sid").cast(pl.Int64))
b = pl.read_parquet("data/featgen_r64/base_20260430.parquet", columns=["sid", "mark", "stoch", "px", "imbv10", "vfrac10"])
j = x.join(b, on=["sid", "mark"]).with_columns((pl.col("zzro2.a") - pl.col("stoch").cast(pl.Float32).cast(pl.Float64)).abs().alias("d"))
off = j.filter(pl.col("d") > 1e-6)
print(f"cells {j.height:,}, differing {off.height:,}; ic_score's value there: {off['zzro2.a'].describe().to_dict(as_series=False)}")
print("by ic_score value / C++ value (rounded to 2 places), top 12 pairs:")
print(off.with_columns(pl.col("zzro2.a").round(2).alias("py"), pl.col("stoch").round(2).alias("cpp")).group_by("py", "cpp").len().sort("len", descending=True).head(12))
print("differing cells by 10-minute bucket of the mark:")
print(off.with_columns((pl.col("mark") // 600).alias("b10")).group_by("b10").len().sort("b10"))
with pl.Config(tbl_rows=40, tbl_cols=-1, tbl_width_chars=220, float_precision=6):
    for sid in (688008, 688041):
        print(f"sid {sid}: every differing cell")
        print(off.filter(pl.col("sid") == sid).sort("mark").select("mark", "zzro2.a", "stoch", "px", "imbv10", "vfrac10").head(45))
PY
echo "== 2 featgen against Python on those stocks"
CRUCIBLE_LPE_SNAP=1 CRUCIBLE_SOURCE=feitu python -u research/featgen/run_day.py --date 20260430 --state $P1/20260430.jz --seed-dir $P1 --marks rand:240:7 \
  --check-sids 688008,688041,688012,688047 --out data/featgen_r65 2>&1 | cut -c1-230
python - <<'PY' 2>&1 | cut -c1-230
import sys
import numpy as np, polars as pl, duckdb
sys.path.insert(0, "research/pm_factors")
import os
os.environ["CRUCIBLE_SOURCE"] = "feitu"
import dcp_build as B, dcp_ema as E
from ic_score import mark_grid
T = mark_grid("rand:240:7")
pm = B.load_state(sorted(p for p in B.DCP.glob("2*.jz") if p.stem < "20260430")[-1])
tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
con = duckdb.connect(str(B.CATALOG), read_only=True)
gs = E.load(con, 20260430, [688008], stop, extra=True)["slow"]
t = gs["t_local"].to_numpy().astype(float)
print(f"sid 688008 trades {gs.height:,}; columns {gs.columns}")
print(f"t_local sorted: {bool(np.all(np.diff(t) >= 0))}; decreasing steps {int((np.diff(t) < 0).sum())}; largest step back {float(-np.diff(t).min()) if t.size > 1 else 0:.6f} s; "
      f"price <= 0 rows {int((gs['price'].to_numpy() <= 0).sum())}; s < 34200 rows {int((gs['s'].to_numpy() < 34200).sum())}")
PY
echo done
