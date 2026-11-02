r"""Which samplerS cells are carried unchanged from the previous day, and whether that follows time, stock, chance or our data.

Per (file, V) over consecutive export days: the share of cells with pm_d == pm_prev (same), its spread over bars of the
day and over stocks (min / max of the per-bar and per-stock shares), the persistence P(same_d | same_{d-1}) against the
base rate (1 = the same cells every day, base = independent draws), and, when the basics table has the day, the same
share in bars where our data has no trade vs at least one trade.

    python research/pm_factors/same_cells.py --dates 20260923,20260924,20260929,20260930,20261009
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

from basic_match import BASICS
from rank_candidates import pm_frame


def load(exports: Path, name: str, d: str) -> tuple[pl.DataFrame, str] | None:
    f = exports / f"{name}_{d}.parquet"
    if not f.exists():
        return None
    pm, vals, kind = pm_frame(f, None, None, None, None, None)
    return pm.select("code", "minute", "V", pl.col(vals[0]).alias("x")), kind


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
    ap.add_argument("--basics", type=Path, default=BASICS)
    a = ap.parse_args()
    dates = sorted(x for x in a.dates.split(",") if x)
    names = sorted({f.stem.rsplit("_", 1)[0] for f in a.exports.glob("*.parquet")})
    out = []
    for name in names:
        frames = {d: r for d in dates if (r := load(a.exports, name, d))}
        days = sorted(frames)
        if len(days) < 2:
            continue
        kind = frames[days[0]][1]
        table = "m5" if kind == "time" else "d1"
        pairs = []
        for d0, d1 in itertools.pairwise(days):
            j = (frames[d1][0].join(frames[d0][0].rename({"x": "xp"}), on=["code", "minute", "V"])
                 .filter(pl.col("x").is_finite() & pl.col("xp").is_finite())
                 .with_columns((pl.col("x") == pl.col("xp")).alias("same"), pl.lit(d1).alias("day")))
            g = a.basics / f"date={d1}" / f"{table}.parquet"
            if g.exists():
                b = pl.read_parquet(g, columns=["symbol_id", "minute", "n"]).rename({"symbol_id": "code", "n": "trades"}).with_columns(
                    pl.col("code").cast(pl.Int64), pl.col("minute").cast(pl.Int64))
                j = j.join(b.drop("minute") if kind != "time" else b, on=["code"] if kind != "time" else ["code", "minute"], how="left")
            else:
                j = j.with_columns(pl.lit(None, pl.Float64).alias("trades"))
            pairs.append(j)
        t = pl.concat(pairs)
        for V in sorted(t["V"].unique().to_list()):
            s = t.filter(pl.col("V") == V)
            base = float(s["same"].mean())
            by_bar = s.group_by("minute").agg(pl.col("same").mean())["same"]
            by_stock = s.group_by("code").agg(pl.col("same").mean())["same"]
            # persistence: same on day d given same on the previous pair, over cells present in both pairs
            per = (s.select("code", "minute", "day", "same").sort("day").group_by("code", "minute")
                   .agg(pl.col("same").alias("seq")).with_columns(pl.col("seq").list.len().alias("k")).filter(pl.col("k") >= 2))
            cond = None
            if per.height:
                seqs = per["seq"].to_list()
                num = sum(1 for q in seqs for i in range(1, len(q)) if q[i - 1] and q[i])
                den = sum(1 for q in seqs for i in range(1, len(q)) if q[i - 1])
                cond = num / den if den else None
            tr = s.filter(pl.col("trades").is_not_null() & (pl.col("trades") > 0))
            dec = {}
            if tr.height >= 1000:  # same share by quintile of our trade count in the bar: a Bernoulli-per-event sampler gives (1-p)^n
                q = tr.with_columns((pl.col("trades").rank("ordinal") * 5 / tr.height).ceil().clip(1, 5).cast(pl.Int32).alias("q5"))
                g = q.group_by("q5").agg(pl.col("same").mean(), pl.col("trades").median().alias("n")).sort("q5")
                dec = {f"same_q{r['q5']}": float(r["same"]) for r in g.iter_rows(named=True)}
                dec["n_q1"], dec["n_q5"] = float(g["n"][0]), float(g["n"][-1])
            out.append({"file": name, "V": V, "pairs": len(pairs), "same": base, "bar_min": float(by_bar.min()), "bar_max": float(by_bar.max()),
                        "stock_min": float(by_stock.min()), "stock_max": float(by_stock.max()), "p_same_given_same": cond, **dec})
    with pl.Config(tbl_rows=300, tbl_cols=-1, tbl_width_chars=200, float_precision=3):
        print("== cells carried unchanged (pm_d == pm_prev): base share, spread over bars / stocks, persistence, same share by quintile of our "
              "trade count in the bar (q1 fewest .. q5 most; n_q1 / n_q5 = median trades)")
        print(pl.DataFrame(out, infer_schema_length=None))


if __name__ == "__main__" and "--by-stock" not in sys.argv:
    main()


def by_stock(exports: Path, basics: Path, dates: list[str], files: list[str]) -> None:
    """Per-stock rate of unchanged cells: its distribution, and how it lines up with market, price, amount and code."""
    rows = []
    for name in files:
        frames = {d: r for d in dates if (r := load(exports, name, d))}
        days = sorted(frames)
        if len(days) < 2:
            continue
        V = sorted(frames[days[0]][0]["V"].unique().to_list())[0]
        pairs = []
        for d0, d1 in itertools.pairwise(days):
            j = (frames[d1][0].filter(pl.col("V") == V).join(frames[d0][0].filter(pl.col("V") == V).rename({"x": "xp"}), on=["code", "minute", "V"])
                 .filter(pl.col("x").is_finite() & pl.col("xp").is_finite()).with_columns((pl.col("x") == pl.col("xp")).alias("same")))
            pairs.append(j.select("code", "same"))
        rate = pl.concat(pairs).group_by("code").agg(pl.col("same").mean().alias("rate"), pl.len().alias("cells"))
        attrs = []
        for d in days:
            g = basics / f"date={d}" / "m5.parquet"
            if g.exists():
                attrs.append(pl.read_parquet(g, columns=["symbol_id", "n", "amt", "vwap"]).group_by("symbol_id")
                             .agg(pl.col("n").sum().alias("trades"), pl.col("amt").sum().alias("amount"), pl.col("vwap").mean().alias("price")))
        if attrs:
            at = pl.concat(attrs).group_by("symbol_id").agg(pl.all().mean()).rename({"symbol_id": "code"}).with_columns(pl.col("code").cast(pl.Int64))
            rate = rate.join(at, on="code", how="left")
        rate = rate.with_columns(pl.when(pl.col("code") >= 600000).then(pl.lit("SSE")).otherwise(pl.lit("SZSE")).alias("mkt"))
        r = rate["rate"]
        row = {"file": name, "V": V, "stocks": rate.height, "rate<0.05": float((r < 0.05).mean()), "0.05-0.5": float(((r >= 0.05) & (r < 0.5)).mean()),
               "rate>=0.5": float((r >= 0.5).mean()), "SSE": float(rate.filter(pl.col("mkt") == "SSE")["rate"].mean()),
               "SZSE": float(rate.filter(pl.col("mkt") == "SZSE")["rate"].mean())}
        for c in ("trades", "amount", "price"):
            if c in rate.columns:
                row[f"rho_{c}"] = float(rate.select(pl.corr("rate", c, method="spearman")).item())
        row["rho_code"] = float(rate.select(pl.corr("rate", pl.col("code").cast(pl.Float64), method="spearman")).item())
        top = rate.sort("rate", descending=True).head(5)
        row["top5"] = " ".join(f"{c}:{x:.2f}" for c, x in zip(top["code"].to_list(), top["rate"].to_list(), strict=True))
        rows.append(row)
    with pl.Config(tbl_rows=100, tbl_cols=-1, tbl_width_chars=220, float_precision=3, fmt_str_lengths=80):
        print("\n== per-stock rate of unchanged cells (first V of each file): distribution, by market, rank correlation with activity / price / code, top stocks")
        print(pl.DataFrame(rows))


if __name__ == "__main__":
    import sys as _s
    if "--by-stock" in _s.argv:
        _s.argv.remove("--by-stock")
        _ap = argparse.ArgumentParser()
        _ap.add_argument("--dates", required=True)
        _ap.add_argument("--exports", type=Path, default=Path("data/pm_export"))
        _ap.add_argument("--basics", type=Path, default=BASICS)
        _a = _ap.parse_args()
        by_stock(_a.exports, _a.basics, sorted(_a.dates.split(",")), ["ZZAL", "ZZMS", "ZZQO", "ZZTS", "ZZXC", "ZZDS", "ZZUGSS", "ZZVC"])
