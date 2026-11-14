#!/usr/bin/env bash
# Round 68 on 97, foreground. Local only (writes under /work; the engine files the baseline reads are NOT overwritten).
# The state engine's event export had the same flaw as the trade loader (R66 / R67): ORDER BY arrival stamp and kind only,
# so messages of one arrival batch came in a different order each run. Now: arrival stamp, kind, sequence number, time.
#   1 how common equal arrival stamps are, 0430 (feitu) and 1112 (DataYes): trades sharing a stamp with another trade of the
#     stock, and trades sharing a stamp with a snapshot of the stock (those have no common sequence number at all)
#   2 the engine on 0430 twice with the new export: are the two outputs identical? and against the marks file the
#     baseline has been using (data/engine_m7/marks_20260430.parquet): which state keys move, and by how much
#   bash research/r68.sh 2>&1 | tee data/reports/r68.txt
set -u
cd /work/crucible
R=data/reports
DATA=/work/crucible_data
P1=/data/prod/h/HS300/p1_intrading_dcp
say() { echo "[$(date +%T)] $*"; }
say "1 equal arrival stamps"
for pair in "feitu 20260430" "dysl2 20261112"; do
  set -- $pair
  CRUCIBLE_SOURCE=$1 python - $2 <<'PY' 2>&1 | cut -c1-230
import sys, duckdb
sys.path.insert(0, "research/pm_factors")
import dcp_build as B
d = int(sys.argv[1])
con = duckdb.connect(str(B.CATALOG), read_only=True)
con.execute("SET threads TO 16")
hs = "symbol_id IN (SELECT DISTINCT symbol_id FROM raw_quotation WHERE date = %d AND symbol_id IN (600000, 600519, 601318, 688008, 688041, 1, 2, 858, 300750, 2594))" % d
t = con.sql(f"""WITH tr AS (SELECT symbol_id, spider_ts, count(*) AS n FROM raw_transaction WHERE date = {d} AND trade_type = 1 AND {hs} GROUP BY 1, 2),
    q AS (SELECT DISTINCT symbol_id, spider_ts FROM raw_quotation WHERE date = {d} AND {hs})
    SELECT sum(n) AS trades, sum(CASE WHEN n > 1 THEN n ELSE 0 END) AS in_shared_stamp, max(n) AS largest_batch,
           sum(CASE WHEN q.spider_ts IS NOT NULL THEN n ELSE 0 END) AS same_stamp_as_snapshot
    FROM tr LEFT JOIN q USING (symbol_id, spider_ts)""").fetchone()
print(f"  {d} ({B.CATALOG.name}), 10 stocks: trades {t[0]:,}; sharing an arrival stamp with another trade {t[1]:,} ({t[1] / max(t[0], 1):.2%}), "
      f"largest batch {t[2]}; sharing a stamp with a snapshot {t[3]:,} ({t[3] / max(t[0], 1):.4%})")
PY
done
say "2 engine on 0430, twice"
for X in a b; do
  CRUCIBLE_SOURCE=feitu python -u research/engine/run_cpp.py --date 20260430 --universe pm:$P1/20260429.jz --prev $P1/20260429.jz \
    --marks rand:240:7 --threads 32 --out data/engine_r68$X --work data/engine_r68$X/work > $R/r68_engine_$X.txt 2>&1 || say "  engine $X FAILED: $(tail -3 $R/r68_engine_$X.txt | cut -c1-200)"
  tail -1 $R/r68_engine_$X.txt | cut -c1-200
done
python - <<'PY' 2>&1 | cut -c1-230
import numpy as np, polars as pl
def load(p):
    return pl.read_parquet(p).sort("sid", "mark")
a, b = load("data/engine_r68a/marks_20260430.parquet"), load("data/engine_r68b/marks_20260430.parquet")
def diff(u, v, tol):
    keys = [c for c in u.columns if c in v.columns and c not in ("sid", "mark")]
    j = u.join(v, on=["sid", "mark"], suffix="__o")
    out = []
    for c in keys:
        x, y = j[c].to_numpy().astype(float), j[c + "__o"].to_numpy().astype(float)
        f = np.isfinite(x) & np.isfinite(y)
        n = int((np.isfinite(x) != np.isfinite(y)).sum()) + int((np.abs(x[f] - y[f]) > tol * np.maximum(np.abs(y[f]), 1e-12)).sum())
        if n:
            out.append((n, c, float(np.max(np.abs(x[f] - y[f]) / np.maximum(np.abs(y[f]), 1e-12))) if f.any() else float("nan")))
    return len(keys), j.height, sorted(out, reverse=True)
k, h, o = diff(a, b, 0.0)
print(f"  two runs: {k} keys x {h:,} rows; keys with any differing cell: {len(o)} {[(c, n) for n, c, _ in o[:8]]}")
try:
    m = load("data/engine_m7/marks_20260430.parquet")
    k, h, o = diff(a, m, 1e-9)
    print(f"  new export against data/engine_m7 (the baseline's file): {k} common keys x {h:,} rows; keys moved: {len(o)}")
    for n, c, w in o[:25]:
        print(f"    {c:<28} cells {n:>7} ({n / h:.2%})  max rel diff {w:.3g}")
    need = ["zzup.pf.sum", "zzup.ps.sum", "zzvx.trDEmaD.sum", "zzxu.a.sum", "zzxu.low", "zzxu.last"]
    print("  the six names the baseline's state columns read:", [(c, next((n for n, cc, _ in o if cc == c), 0)) for c in need])
except Exception as e:
    print(f"  compare with engine_m7 failed: {type(e).__name__}: {str(e)[:150]}")
PY
say "done"
