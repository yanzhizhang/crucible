#!/usr/bin/env bash
# Round 49 on 97, seconds, read-only on earlier outputs: are the r41c columns that lower the similarity with the 224 signal the
# same on every 224 day (a systematic cause: feed or model generation) or different each day (noise / a broken day)?
#   bash research/r49.sh 2>&1 | tee data/reports/r49.txt
set -u
cd /work/crucible
R=data/reports
run_of() { if [ $1 = 0430 ]; then echo $R/stage6_r41_audit2; elif [ $1 = 1113 ]; then echo $R/stage6_r43_1113_r41c; else echo $R/stage6_r44_$1_r41c; fi; }
echo "== 1 r41c column by column, 0430 (2.23 target) against each 224 day (its signal)"
for d in 1106 1109 1110 1111 1112; do
  echo "-- 0430 against $d"
  python -u research/pm_factors/col_days.py --a $(run_of 0430) --b $(run_of $d) --names 0430,$d 2>&1 | sed -n '/== columns by class/,/== by kind x class/p;/rank correlation/p' | grep -v "^$" | cut -c1-200
done
echo "== 2 224 days against each other"
for p in "1106 1110" "1110 1113" "1106 1113" "1109 1112" "1111 1112"; do
  set -- $p
  echo "-- $1 against $2"
  python -u research/pm_factors/col_days.py --a $(run_of $1) --b $(run_of $2) --names $1,$2 2>&1 | sed -n '/== columns by class/,/== by kind of formula/p;/rank correlation/p' | grep -v "^$" | cut -c1-200
done
echo "== 3 per-column similarity loss on all days, hurting on at least 4 of the 6 224 days"
python - <<PY 2>&1 | cut -c1-250
import polars as pl
runs = {"0430": "$R/stage6_r41_audit2", "1106": "$R/stage6_r44_1106_r41c", "1109": "$R/stage6_r44_1109_r41c", "1110": "$R/stage6_r44_1110_r41c",
        "1111": "$R/stage6_r44_1111_r41c", "1112": "$R/stage6_r44_1112_r41c", "1113": "$R/stage6_r43_1113_r41c"}
j = None
for k, d in runs.items():
    e = pl.read_parquet(d + "/ic_evidence.parquet").select("feature", "formula", pl.col("ic_loss").alias("l_" + k), pl.col("t_loss").alias("t_" + k))
    j = e if j is None else j.join(e.drop("formula"), on="feature", how="inner")
d224 = [k for k in runs if k != "0430"]
j = j.with_columns(pl.sum_horizontal([(pl.col("t_" + k) <= -2).cast(pl.Int32) for k in d224]).alias("days_hurt"),
                   pl.sum_horizontal([(pl.col("t_" + k) >= 2).cast(pl.Int32) for k in d224]).alias("days_help"),
                   pl.mean_horizontal(["l_" + k for k in d224]).alias("mean224"))
with pl.Config(tbl_rows=-1, tbl_cols=-1, tbl_width_chars=250, fmt_str_lengths=48, float_precision=4):
    print(j.group_by("days_hurt").agg(pl.len().alias("columns"), pl.col("l_0430").sum(), pl.col("mean224").sum()).sort("days_hurt"))
    print(j.filter(pl.col("days_hurt") >= 4).sort("mean224").select(["feature", "formula"] + ["l_" + k for k in runs] + ["days_hurt"]))
    print("helping on at least 5 of the 6 224 days and on 0430:")
    print(j.filter((pl.col("days_help") >= 5) & (pl.col("t_0430") >= 2)).sort("mean224", descending=True).select(["feature", "formula"] + ["l_" + k for k in runs]))
print("hurt on >= 4 days: " + ",".join(j.filter(pl.col("days_hurt") >= 4)["feature"].to_list()))
PY
