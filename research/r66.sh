#!/usr/bin/env bash
# Round 66 on 97, foreground, read-only (prints only). gen.stoch on 20260430: ic_score's matrix differs from what both the
# C++ program and a fresh call of the Python definition give (R65). Which step of ic_score changes it? For two stocks at a
# few marks: the matrix value, the value in ic_score's cached build (long.parquet), and a fresh generic() with its parts.
#   bash research/r66.sh 2>&1 | tee data/reports/r66.txt
set -u
cd /work/crucible
CRUCIBLE_LPE_SNAP=1 CRUCIBLE_SOURCE=feitu python - <<'PY' 2>&1 | cut -c1-240
import sys, glob, os
import numpy as np, polars as pl, duckdb
sys.path.insert(0, "research/pm_factors")
import dcp_build as B, dcp_ema as E, feature_intraday as FI, ic_score as IC
T = IC.mark_grid("rand:240:7")
SIDS, MK = [688041, 688008], {688041: [72.0, 418.0, 1887.0], 688008: [347.0, 758.0, 1250.0]}
x = pl.read_parquet("data/reports/stage6_r64_0430_snap/x.parquet", columns=["sid", "mark", "zzro2.a"])
print("marks of the grid, first 6:", T[:6].tolist(), "; dtype of x.mark", x["mark"].dtype, "; marks in x not in the grid:", int((~x["mark"].is_in(T.tolist())).sum()))
dirs = sorted(glob.glob("/work/crucible_data/cache/ic_build/*/date=20260430.*"), key=os.path.getmtime)[-4:]
longs = {}
for d in dirs:
    lf = pl.read_parquet(d + "/long.parquet").filter(pl.col("sid").is_in(SIDS) & pl.col("key").str.contains("stoch"))
    longs[d] = lf
    print(f"cache {d.split('ic_build/')[1]} (mtime {os.path.getmtime(d):.0f}): keys with 'stoch' {sorted(lf['key'].unique().to_list())}, rows {lf.height}, "
          f"rows per (sid, key, mark) max {lf.group_by('sid', 'key', 'mark').len()['len'].max() if lf.height else 0}")
pm = B.load_state(sorted(p for p in B.DCP.glob("2*.jz") if p.stem < "20260430")[-1])
tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
stop = (tz + 34200 if tz <= 7200 else tz - 7200 + 46800) + 0.005
sids = sorted(pm["sid"].unique().to_list())
con = duckdb.connect(str(B.CATALOG), read_only=True)
up = E.load(con, 20260430, sids, stop, extra=True)
slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
for sid in SIDS:
    gs, gf = slow[sid], fast[sid]
    gen = IC.generic(gf, gs, T)
    b = E.arrays(gs, "slow")
    t, cont = b["t_local"], gs["s"].to_numpy() >= 34200
    hi, lo = np.fmax.accumulate(np.where(cont, b["price"], np.nan)), np.fmin.accumulate(np.where(cont, b["price"], np.nan))
    hi_all, lo_all = np.fmax.accumulate(b["price"]), np.fmin.accumulate(b["price"])
    print(f"sid {sid}: trades {gs.height:,}, before 09:30 exchange time {int((~cont).sum())}, t_local of the first continuous trade {t[cont][0]:.3f}, "
          f"trades with t_local <= 0 and s >= 34200: {int((cont & (t <= 0)).sum())}; first 3 (t_local, s, price): "
          f"{[(round(float(a), 3), float(c), float(p)) for a, c, p in zip(t[:3], gs['s'].to_numpy()[:3], b['price'][:3])]}")
    for m in MK[sid]:
        j = int(np.flatnonzero(T == m)[0])
        k = int(np.searchsorted(t, m, side="right")) - 1
        row = [f"mark {m:.0f}", f"matrix {x.filter((pl.col('sid') == sid) & (pl.col('mark') == m))['zzro2.a'].to_list()}"]
        for d, lf in longs.items():
            row.append(f"cache..{d[-17:-9]} {[round(v, 6) for v in lf.filter((pl.col('sid') == sid) & (pl.col('mark') == m) & (pl.col('key') == 'gen.stoch'))['pm'].to_list()]}")
        st_all = (b["price"][k] - lo_all[k]) / (hi_all[k] - lo_all[k])
        row.append(f"fresh generic {gen['stoch'][j]:.6f} (px {b['price'][k]}, lo {lo[k]}, hi {hi[k]}; whole-day range lo {lo_all[k]}, hi {hi_all[k]} -> {st_all:.6f})")
        print("  " + " | ".join(row))
PY
echo done
