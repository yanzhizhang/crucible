r"""Stage 4c: fit PM's multi-day weighted averages (``<p>.sum / .wgt / .t``) as a one-day recursion.

The state file of day D holds each average after D's update; the file of the previous trade day
(or ``sod/D``, PM's copy of it at the open) holds it before. So for every prefix p the day's update
can be fitted exactly, with no long history:

    wgt_D = a * wgt_base + c * w_D        sum_D = a * sum_base + c * w_D * x_D

* **overnight** (when ``sod/D`` exists): ``sod/D`` against the file of the previous trade day --
  is the open copy the same state, or already decayed?
* **weight**: for each candidate day weight w (volume, amount, trade count, snapshot count, 1;
  whole day from 09:30 or continuous session only), a least-squares fit of ``a`` and ``c`` across
  the 300 stocks; the right w leaves a residual of ~0 and the same ``a`` every day;
* **value**: with the best weight, the implied ``x_D = (sum_D - a sum_base) / (c w_D)`` against the
  day's quantities (vwap, twap, last, open, high, low, returns, range).

Prints the inventory, the overnight check, the best-fitting prefixes and a per-family summary;
every (prefix, candidate, day) fit goes to ``ema_fits.csv``.

    python research/pm_factors/diag_ema.py --dates 20260923,20260924,20260928,20260929,20260930,20261009
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

import dcp_build as B  # catalog path, time expressions, state loader

WEIGHTS = ("vol", "amt", "n", "vol_cont", "amt_cont", "n_cont", "n_snap", "n_snap_cont", "one")
VALUES = ("vwap", "vwap_cont", "twap", "last", "open", "high", "low", "ret_oc", "logret_oc", "range")


def triples(pm: pl.DataFrame) -> pl.DataFrame:
    """(sid, p, sum, wgt, t) for every prefix that has both ``.sum`` and ``.wgt``."""
    x = pm.filter(pl.col("key").str.contains(r"\.(sum|wgt|t)$")).with_columns(
        pl.col("key").str.extract(r"^(.*)\.(sum|wgt|t)$", 1).alias("p"),
        pl.col("key").str.extract(r"\.(sum|wgt|t)$", 1).alias("f"))
    w = x.pivot(on="f", index=["sid", "p"], values="pm")
    for f in ("sum", "wgt", "t"):
        if f not in w.columns:
            w = w.with_columns(pl.lit(None, pl.Float64).alias(f))
    return w.filter(pl.col("sum").is_not_null() & pl.col("wgt").is_not_null())


def day_inputs(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> pl.DataFrame:
    """Candidate weights and values per stock for one day."""
    S, L = B.SOD, B.LSOD
    cont = f"{S} >= 34200 AND {S} < 53820"
    con.register("want", pl.DataFrame({"sid": sids}))
    tr = con.sql(f"""SELECT symbol_id AS sid,
        (sum(volume) FILTER ({S} >= 34200))::DOUBLE AS vol, sum(volume * price) FILTER ({S} >= 34200) AS amt,
        (count(*) FILTER ({S} >= 34200))::DOUBLE AS n,
        (sum(volume) FILTER ({cont}))::DOUBLE AS vol_cont, sum(volume * price) FILTER ({cont}) AS amt_cont,
        (count(*) FILTER ({cont}))::DOUBLE AS n_cont,
        arg_max(price, time) AS last, arg_min(price, time) AS open, max(price) AS high, min(price) AS low
        FROM raw_transaction r JOIN want w ON w.sid = r.symbol_id
        WHERE r.date = {date} AND r.trade_type = 1 AND {L} <= {stop} GROUP BY 1""").pl()
    sn = con.sql(f"""SELECT symbol_id AS sid, (count(*) FILTER ({L} <= {stop}))::DOUBLE AS n_snap,
        (count(*) FILTER ({cont}))::DOUBLE AS n_snap_cont, avg(price) FILTER (price > 0 AND {cont}) AS twap
        FROM raw_quotation q JOIN want w ON w.sid = q.symbol_id WHERE q.date = {date} GROUP BY 1""").pl()
    con.unregister("want")
    d = tr.with_columns(pl.col("sid").cast(pl.Int64)).join(sn.with_columns(pl.col("sid").cast(pl.Int64)), on="sid", how="left")
    return d.with_columns(
        (pl.col("amt") / pl.col("vol")).alias("vwap"), (pl.col("amt_cont") / pl.col("vol_cont")).alias("vwap_cont"),
        (pl.col("last") / pl.col("open") - 1).alias("ret_oc"), (pl.col("last") / pl.col("open")).log().alias("logret_oc"),
        (pl.col("high") / pl.col("low") - 1).alias("range"), pl.lit(1.0).alias("one"))


def fit_day(cur: pl.DataFrame, base: pl.DataFrame, inp: pl.DataFrame, date: int) -> pl.DataFrame:
    """One row per (prefix, weight candidate): fitted a, c, residual and the best value candidate."""
    j = cur.join(base.select("sid", "p", pl.col("sum").alias("sum0"), pl.col("wgt").alias("wgt0")), on=["sid", "p"]).join(inp, on="sid")
    rows = []
    for p in j["p"].unique().sort().to_list():
        g = j.filter(pl.col("p") == p)
        for wn in WEIGHTS:
            h = g.filter(pl.col(wn).is_finite() & (pl.col(wn) > 0) & pl.col("wgt0").is_finite() & pl.col("wgt").is_finite())
            if h.height < 20:
                continue
            y, b0, w = h["wgt"].to_numpy(), h["wgt0"].to_numpy(), h[wn].to_numpy()
            (a, c), *_ = np.linalg.lstsq(np.column_stack([b0, w]), y, rcond=None)
            mre = float(np.median(np.abs(y - a * b0 - c * w) / np.maximum(np.abs(y), 1e-300)))
            row = {"date": date, "p": p, "w": wn, "n": h.height, "a": float(a), "c": float(c), "mre": mre,
                   "x": None, "x_mre": None}
            dw = c * w
            if np.median(np.abs(dw) / np.maximum(np.abs(y), 1e-300)) > 1e-6:  # the day added weight: read its value
                x = (h["sum"].to_numpy() - a * h["sum0"].to_numpy()) / dw
                best = min(((vn, float(np.nanmedian(np.abs(x - h[vn].to_numpy()) / np.maximum(np.abs(h[vn].to_numpy()), 1e-12))))
                            for vn in VALUES), key=lambda t: t[1])
                row["x"], row["x_mre"] = best
            rows.append(row)
    return pl.DataFrame(rows, schema={"date": pl.Int64, "p": pl.String, "w": pl.String, "n": pl.Int64, "a": pl.Float64,
                                      "c": pl.Float64, "mre": pl.Float64, "x": pl.String, "x_mre": pl.Float64})


def overnight(sod: pl.DataFrame, prev: pl.DataFrame) -> pl.DataFrame:
    """Per family: whether sod/D is the previous day's state unchanged or already decayed."""
    j = sod.join(prev, on=["sid", "p"], suffix="_prev").with_columns(
        (pl.col("wgt") / pl.col("wgt_prev")).alias("rw"), (pl.col("sum") / pl.col("sum_prev")).alias("rs"),
        pl.col("p").str.extract(r"^([^.]+)").alias("fam"))
    return j.group_by("fam").agg(pl.col("p").n_unique().alias("prefixes"), pl.col("rw").median().alias("wgt_ratio"),
                                 (pl.col("rw").quantile(0.9) - pl.col("rw").quantile(0.1)).alias("p90_p10"),
                                 ((pl.col("rs") / pl.col("rw") - 1).abs().median()).alias("sum_vs_wgt"),
                                 (pl.col("t") == pl.col("t_prev")).mean().alias("t_same")).sort("fam")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4"))
    a = ap.parse_args()
    files = sorted(p.stem for p in a.dcp_dir.glob("*.jz"))
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    fits = []
    for d in [x.strip() for x in a.dates.split(",") if x.strip()]:
        prev = [f for f in files if f < d]
        if d not in files or not prev:
            print(f"skip {d}: no state file for it or for a day before it")
            continue
        pm = B.load_state(a.dcp_dir / f"{d}.jz")
        cur, before = triples(pm), triples(B.load_state(a.dcp_dir / f"{prev[-1]}.jz"))
        sod_f = a.dcp_dir / "sod" / f"{d}.jz"
        base = before
        if sod_f.exists():
            sod = triples(B.load_state(sod_f))
            with pl.Config(tbl_rows=60, tbl_width_chars=160, float_precision=6):
                print(f"\n== {d}: sod/{d} vs {prev[-1]} (wgt_ratio 1 and t_same 1 = an unchanged copy)")
                print(overnight(sod, before))
            base = sod
        t = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        stop = (t + 34200 if t <= 7200 else t - 7200 + 46800) + 0.005
        inp = day_inputs(con, int(d), sorted(pm["sid"].unique().to_list()), stop)
        if inp.is_empty():
            print(f"skip {d}: no decoded ticks")
            continue
        fits.append(fit_day(cur, base, inp, int(d)))
        print(f"done {d} (base {'sod/' + d if base is not before else prev[-1]}, {cur['p'].n_unique()} prefixes)")
    if not fits:
        return
    f = pl.concat(fits)
    a.out.mkdir(parents=True, exist_ok=True)
    f.write_csv(a.out / "ema_fits.csv")
    # per (prefix, weight): median over days; the best weight per prefix is the one with the smallest residual
    agg = f.group_by("p", "w").agg(pl.len().alias("days"), pl.col("mre").median(), pl.col("a").median().alias("a"),
                                   (pl.col("a").max() - pl.col("a").min()).alias("a_rng"), pl.col("c").median(),
                                   pl.col("x").mode().first(), pl.col("x_mre").median())
    ranked = agg.sort("mre").group_by("p", maintain_order=True)
    best = ranked.first().join(ranked.agg(pl.col("w").slice(1, 1).first().alias("w2"), pl.col("mre").slice(1, 1).first().alias("mre2")),
                               on="p").with_columns(pl.col("p").str.extract(r"^([^.]+)").alias("fam")).sort("mre")
    with pl.Config(tbl_rows=60, tbl_cols=12, tbl_width_chars=200, fmt_str_lengths=30, float_precision=5):
        print(f"\n== best weight per prefix, 45 best of {best.height} (mre: median |resid| / wgt; a_rng: spread of a over days; w2: runner-up)")
        print(best.head(45).select("p", "w", "days", "mre", "a", "a_rng", "c", "x", "x_mre", "w2", "mre2"))
        print("\n== per family: prefixes, how many fit (mre < 1e-4), most common best weight / value")
        print(best.group_by("fam").agg(pl.len().alias("prefixes"), (pl.col("mre") < 1e-4).sum().alias("fit"),
                                       pl.col("mre").median().alias("mre_med"), pl.col("w").mode().first().alias("w"),
                                       pl.col("a").median(), pl.col("x").mode().first().alias("x"),
                                       pl.col("x_mre").median()).sort("fam"))
    print(f"all fits: {a.out / 'ema_fits.csv'}")


if __name__ == "__main__":
    main()
