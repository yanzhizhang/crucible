r"""Stage 4: decode PM's day-end factor state (``h/HS300/p1_intrading_dcp/<D>.jz``) against our data.

The file is gzip JSON ``{sid: {key: float}}``, 300 stocks x ~7500 keys, the state the multi-day
factors carry into the next day. Its price fields are already proven exact (``zzpk.high/low/open``,
``zzxu.high/low``, ``zzvx.high/low`` = our 0429 daily values for 300 of 300 stocks). Most of the
rest are decaying weighted averages stored as ``<prefix>.sum`` / ``.t`` / ``.wgt`` (value =
sum / wgt), with weights that differ per stock (so not time-only). This screens every such prefix:

* **value** (sum / wgt) against day-D quantities (last, vwap, open, high, low, return, range, ...)
  and their exponentially weighted multi-day versions (half-life in trading days, ``HALF``);
* **wgt** against decayed sums of volume, amount, trade count and plain day count -- the best
  candidate has the smallest spread of log(wgt / candidate) across stocks (a constant ratio = the
  right weight, the ratio being its unit);
* **t** against the stock's last trade time on day D (rank correlation).

Plain scalar keys (no triple, no histogram) are scored like values. Uses every decoded trade day
up to ``--date`` (decode more days to sharpen the long half-lives).

    python research/pm_factors/diag_dcp.py --date 20260429 \
        --dcp /data/prod/h/HS300/p1_intrading_dcp/20260429.jz
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sampler_r as R  # catalog, side codes

HALF = (0.5, 1, 2, 3, 5, 7, 10, 15, 20, 40)


def load_state(path: Path) -> pl.DataFrame:
    """Load the dcp file as a long frame (sid, key, value), histogram bins dropped."""
    with gzip.open(path) as f:
        d = json.loads(f.read())
    rows = [(int(s), k, float(v)) for s, kv in d.items() for k, v in kv.items() if not re.search(r"\.c\.\d+$", k)]
    return pl.DataFrame(rows, schema={"sid": pl.Int64, "key": pl.String, "value": pl.Float64}, orient="row")


def daily(con: duckdb.DuckDBPyConnection, days: list[str], sids: list[int]) -> pl.DataFrame:
    """Per (date, sid) trade statistics with return and range."""
    con.register("want", pl.DataFrame({"sid": sids}))
    dl = ", ".join(days)
    d = con.sql(f"""
      SELECT date, symbol_id AS sid, sum(volume)::DOUBLE AS vol, sum(volume * price) AS amt, count(*)::DOUBLE AS n,
             arg_max(price, time) AS last, arg_min(price, time) AS open, max(price) AS high, min(price) AS low,
             (sum(volume) FILTER (dir = {R.BUY}))::DOUBLE AS bvol, (sum(volume) FILTER (dir = {R.SELL}))::DOUBLE AS svol,
             max(time) AS last_t
      FROM raw_transaction r JOIN want w ON w.sid = r.symbol_id
      WHERE r.date IN ({dl}) AND r.trade_type = 1 GROUP BY ALL""").pl()
    return (d.sort("sid", "date").with_columns(
        (pl.col("amt") / pl.col("vol")).alias("vwap"),
        (pl.col("last") / pl.col("last").shift(1).over("sid")).log().alias("ret"),
        (pl.col("high") / pl.col("low")).log().alias("range"),
        ((pl.col("bvol") - pl.col("svol")) / pl.col("vol")).alias("imb")))


def candidates(dd: pl.DataFrame, date: int) -> tuple[pl.DataFrame, list[str], list[str]]:
    """Wide (sid -> candidate) frame; value candidates and weight candidates."""
    ages = {d: i for i, d in enumerate(sorted(dd["date"].unique().to_list(), reverse=True))}
    dd = dd.with_columns(pl.col("date").replace_strict(ages).alias("age"))
    day = dd.filter(pl.col("date") == date).select(
        "sid", *[pl.col(c).alias(f"D:{c}") for c in ("last", "vwap", "open", "high", "low", "ret", "range", "imb", "vol", "amt", "n")],
        ((pl.col("high") + pl.col("low")) / 2).alias("D:hl_mid"))
    out = [day]
    for h in HALF:
        w = 0.5 ** (pl.col("age") / h)
        out.append(dd.group_by("sid").agg(
            ((w * pl.col("amt")).sum() / (w * pl.col("vol")).sum()).alias(f"ew{h}:vwap"),
            ((w * pl.col("last")).sum() / w.sum()).alias(f"ew{h}:last"),
            ((w * pl.col("ret")).filter(pl.col("ret").is_not_null()).sum() / w.filter(pl.col("ret").is_not_null()).sum()).alias(f"ew{h}:ret"),
            ((w * pl.col("ret").abs()).filter(pl.col("ret").is_not_null()).sum() / w.filter(pl.col("ret").is_not_null()).sum()).alias(f"ew{h}:absret"),
            ((w * pl.col("range")).sum() / w.sum()).alias(f"ew{h}:range"),
            ((w * pl.col("imb")).sum() / w.sum()).alias(f"ew{h}:imb"),
            (w * pl.col("vol")).sum().alias(f"W{h}:vol"), (w * pl.col("amt")).sum().alias(f"W{h}:amt"),
            (w * pl.col("n")).sum().alias(f"W{h}:n"), w.sum().alias(f"W{h}:days")))
    wide = out[0]
    for f in out[1:]:
        wide = wide.join(f, on="sid", how="full", coalesce=True)
    cols = [c for c in wide.columns if c != "sid"]
    wcands = [c for c in cols if c.startswith("W") or c in ("D:vol", "D:amt", "D:n")]
    return wide, [c for c in cols if c not in wcands], wcands


def score_value(v: pl.DataFrame, wide: pl.DataFrame, cands: list[str]) -> list[dict]:
    """(sid, x) vs each candidate: rank corr, median ratio, share within 1 % raw and after the ratio."""
    j = v.join(wide, on="sid")
    res = []
    for c in cands:
        k = j.filter(pl.col("x").is_finite() & pl.col(c).is_finite() & (pl.col(c) != 0))
        if k.height < 30:
            continue
        r = k.select(pl.corr("x", c, method="spearman").alias("rho"), (pl.col("x") / pl.col(c)).median().alias("ratio")).row(0, named=True)
        w1 = k.select(((pl.col("x") / pl.col(c) - 1).abs() <= 0.01).mean()).item()
        w1s = k.select(((pl.col("x") / (pl.col(c) * r["ratio"]) - 1).abs() <= 0.01).mean()).item() if r["ratio"] else 0.0
        mre = k.select((pl.col("x") / pl.col(c) - 1).abs().median()).item()  # median relative error, unscaled
        res.append({"cand": c, "n": k.height, "rho": r["rho"], "ratio": r["ratio"], "mre": mre, "w1": w1, "w1_scaled": w1s})
    return res


def score_weight(w: pl.DataFrame, wide: pl.DataFrame, cands: list[str]) -> list[dict]:
    """(sid, x=wgt) vs each weight candidate: spread of log(wgt / cand) (0 = proportional) and the ratio."""
    j = w.join(wide, on="sid").filter(pl.col("x") > 0)
    res = []
    for c in cands:
        k = j.filter(pl.col(c) > 0).select((pl.col("x") / pl.col(c)).log().alias("lr"))
        if k.height < 30:
            continue
        res.append({"cand": c, "lr_sd": k["lr"].std(), "ratio": float(np.exp(k["lr"].median())),
                    "distinct_wgt": j["x"].round(9).n_unique()})
    return res


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--dcp", type=Path, required=True)
    ap.add_argument("--memory", default="200GB")
    a = ap.parse_args()
    st = load_state(a.dcp)
    sids = sorted(st["sid"].unique().to_list())
    con = duckdb.connect(str(R.CATALOG), read_only=True)
    con.execute(f"SET memory_limit='{a.memory}'")
    days = [str(d) for d in con.sql(f"SELECT DISTINCT date FROM raw_transaction WHERE date <= {int(a.date)} ORDER BY date").pl()["date"]]
    print(f"state: {len(sids)} stocks, {st['key'].n_unique()} non-histogram keys; trade days used: {days}")
    dd = daily(con, days, sids)
    wide, vcands, wcands = candidates(dd, int(a.date))
    last_t = dd.filter(pl.col("date") == int(a.date)).select("sid", pl.col("last_t").cast(pl.Float64).alias("lt"))

    keys = st["key"].unique().to_list()
    prefixes = sorted({k[:-4] for k in keys if k.endswith(".sum") and f"{k[:-4]}.wgt" in keys})
    in_triple = {f"{p}.{s}" for p in prefixes for s in ("sum", "t", "wgt")}
    pv = st.pivot(on="key", index="sid", values="value")
    rows, detail = [], []
    for p in prefixes:
        v = pv.select("sid", (pl.col(f"{p}.sum") / pl.col(f"{p}.wgt")).alias("x"))
        sv = sorted(score_value(v, wide, vcands), key=lambda r: (r["mre"], -abs(r["rho"] or 0)))
        sw = sorted(score_weight(pv.select("sid", pl.col(f"{p}.wgt").alias("x")), wide, wcands),
                    key=lambda r: (round(r["lr_sd"], 6), abs(np.log(r["ratio"])) if r["ratio"] > 0 else 9e9))
        rt = None
        if f"{p}.t" in pv.columns:
            tj = pv.select("sid", pl.col(f"{p}.t").alias("t")).join(last_t, on="sid").filter(pl.col("t") > 0)
            rt = tj.select(pl.corr("t", "lt", method="spearman")).item() if tj.height > 30 else None
        b, bw = (sv[0] if sv else {}), (sw[0] if sw else {})
        rows.append({"prefix": p, "value_best": b.get("cand"), "rho": b.get("rho"), "ratio": b.get("ratio"), "mre": b.get("mre"),
                     "w1": b.get("w1"), "w1_scaled": b.get("w1_scaled"), "wgt_best": bw.get("cand"), "wgt_lr_sd": bw.get("lr_sd"),
                     "wgt_ratio": bw.get("ratio"), "wgt_distinct": bw.get("distinct_wgt"), "rho_t_lasttrade": rt})
        detail += [{"prefix": p, "kind": "value", **r} for r in sv[:10]] + [{"prefix": p, "kind": "wgt", **r} for r in sw[:10]]
    scal = [k for k in keys if k not in in_triple]
    srows = []
    for k in sorted(scal):
        sv = sorted(score_value(pv.select("sid", pl.col(k).alias("x")), wide, vcands), key=lambda r: (r["mre"], -abs(r["rho"] or 0)))
        if sv:
            srows.append({"key": k, **{f"best_{x}": sv[0][x] for x in ("cand", "rho", "ratio", "mre", "w1", "w1_scaled")}})
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=260, fmt_float="mixed", fmt_str_lengths=40):
        print(f"\n== {len(prefixes)} weighted averages (value = sum / wgt): best value candidate, best weight candidate")
        print(pl.DataFrame(rows))
        print(f"\n== {len(srows)} plain scalar keys: best candidate")
        print(pl.DataFrame(srows))
    out = Path("data/reports/stage4")
    out.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(detail).write_parquet(out / f"dcp_{a.date}.parquet")
    print(f"top-10 per prefix: {out / f'dcp_{a.date}.parquet'}")


if __name__ == "__main__":
    main()
