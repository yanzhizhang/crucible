#!/usr/bin/env bash
# Round 55 on 97, read-only, a few seconds: which of the baseline's columns use (a) a neutralised / industry-demeaned
# quantity (neu: dm: dmi:), (b) a CED static field (ced.*), (c) an hstats cell (@), and what each is worth on the
# seven target days (similarity lost without the column). Also the hstats families on disk with their file kinds.
#   bash research/r55.sh 2>&1 | tee data/reports/r55.txt
set -u
cd /work/crucible
R=data/reports
echo "== hstats families on disk (newest day of each): files"
for d in /data/prod/hstats/HS300/*/; do l=$(ls $d | tail -1); echo "  $(basename $d) $l: $(ls $d/$l 2>/dev/null | tr '\n' ' ')"; done
python - <<PY 2>&1 | cut -c1-220
import json, re, os
import polars as pl
a = json.load(open("$R/stage6_r41c_gate/ic_assign_gated.json"))
def form(e):
    return e if isinstance(e, str) else str(e.get("formula", e) if isinstance(e, dict) else e)
rows = []
for f, e in a.items():
    s = form(e)
    k = []
    if re.search(r"(^|:)neu:", s): k.append("neu")
    if re.search(r"(^|:)dmi?:", s): k.append("dm")
    if "ced." in s: k.append("ced")
    if "@" in s: k.append("hstats")
    if re.search(r"(^|:)(s|c)?rank:|(^|:)z:", s): k.append("xs")
    rows.append({"feature": f, "formula": s[:70], "kind": "+".join(k) or "plain"})
t = pl.DataFrame(rows)
runs = {"0430": "$R/stage6_r41_audit2"}
runs.update({d: f"$R/stage6_r50_{d}_r41c_B" for d in ("1106", "1109", "1110", "1111", "1112", "1113")})
for n, d in runs.items():
    p = d + "/ic_evidence.parquet"
    if os.path.exists(p):
        t = t.join(pl.read_parquet(p).select("feature", pl.col("ic_loss").alias("l_" + n)), on="feature", how="left")
lc = [c for c in t.columns if c.startswith("l_")]
t = t.with_columns(pl.mean_horizontal(lc).alias("mean"), pl.sum_horizontal([(pl.col(c) > 0).cast(pl.Int32) for c in lc]).alias("days_pos"))
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=220, fmt_str_lengths=70, float_precision=4):
    print("== columns per kind (neu = neutralised, dm = industry-demeaned, ced = CED static field, hstats = hstats cell, xs = rank / z over stocks)")
    print(t.group_by("kind").agg(pl.len().alias("columns"), pl.col("mean").sum().alias("sum_mean_loss"), pl.col("days_pos").mean()).sort("columns", descending=True))
    print("== every column that is not plain / xs only")
    print(t.filter(~pl.col("kind").is_in(["plain", "xs"])).sort("mean", descending=True).select(["feature", "formula", "kind"] + lc + ["mean", "days_pos"]))
PY
