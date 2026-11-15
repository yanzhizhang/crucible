#!/usr/bin/env bash
# Round 103 on 97, foreground, read-only, writes nothing. ~1 min.
# R102: zzeg's best constant differs by day (1.011 on 1105/1106, 1.009 on 1112/1113, 0.99 on 1109/1110, ~flat 1104/1111,
# and on 0430 nothing >= 0.985 helps). Print each day's daily index numbers next to it -- previous close, open and close
# per index, and the day-over-day ratios -- to see which daily series (if any) lines up with the best constants.
#   bash research/r103.sh 2>&1 | tee data/reports/r103.txt
set -u
cd /work/crucible
python - <<'PY'
from pathlib import Path
import polars as pl
DATA = Path("/work/crucible_data")
best = {"20260430": "<0.985", "20261102": "1.011", "20261103": "<0.985", "20261104": "0.985", "20261105": "1.011", "20261106": "1.011",
        "20261109": "0.990", "20261110": "0.990", "20261111": "0.985", "20261112": "1.009", "20261113": "1.008"}
rows = []
for d in best:
    src = "pmmp_raw" if d == "20260430" else "dysl2_raw"
    parts = list((DATA / src).glob(f"kind=index/date={d}/part-*.parquet"))
    if not parts:
        continue
    ix = pl.read_parquet(parts, columns=["symbol_id", "market_id", "time", "last_price", "pre_close_price"]).filter(pl.col("last_price") > 0).sort("time")
    for code in (300, 905, 852, 1, 16):
        g = ix.filter((pl.col("symbol_id") == code) & (pl.col("market_id") == 3553))
        if g.height:
            t = g["time"] % (86400 * 10**9)
            o = g.filter(t >= 5400 * 10**9)  # first print at or after 09:30 CST (01:30 UTC)
            rows.append({"date": d, "code": code, "pre": g["pre_close_price"][-1], "open": o["last_price"][0] if o.height else None, "close": g["last_price"][-1]})
df = pl.DataFrame(rows).sort("code", "date")
df = df.with_columns((pl.col("open") / pl.col("pre")).alias("open/pre"), (pl.col("close") / pl.col("pre")).alias("close/pre"),
                     (pl.col("pre") / pl.col("pre").shift(1).over("code")).alias("pre/prev_pre"))
for code, g in df.group_by("code", maintain_order=True):
    print(f"index {code[0]:06d}:  date  best_c  pre_close  open/pre  close/pre  pre/prev_pre(=D-1 return)")
    for r in g.iter_rows(named=True):
        f = lambda v: f"{v:.4f}" if v is not None else "  -   "  # noqa: E731
        print(f"  {r['date']}  {best[r['date']]:>6}  {r['pre']:10.3f}  {f(r['open/pre'])}  {f(r['close/pre'])}  {f(r['pre/prev_pre'])}")
PY
