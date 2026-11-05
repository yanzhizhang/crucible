r"""Where does a matched rolling median miss? One PM column against one basic at a fixed window, cell by cell.

``window_match.py`` found the window (median of the last N trading days, d included); this asks why the exact share
stops short of 1. For each (family, V, basic) it scores the base window and breaks the miss down by PM day, exchange
and bar, then tries the window with one day left out (a day PM did not have, or had different data for, jumps the
exact share when dropped) and with each non-trading basics day (e.g. a Saturday folder) added.

    python research/pm_factors/window_probe.py --days 20261020,20261021,20261022,20261023,20261026,20261027

A basic written ``cm:vol+oa_vol|ca_vol`` is the running sum over the day's bars of per-bar medians (median of the
summed columns per ``|`` term, terms added) instead of the median of a running total.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from basic_match import BASICS
from rank_candidates import pm_frame
from window_match import trading_days

COMBOS = "ZZTS:q:b_amt,ZZXC:n:amt,ZZXC:q:vol,ZZMS:n:cum_amt,ZZMS:q:cum_vol,ZZDS:h:high,ZZDS:l:low,ZZDS:c:mid,ZZDS:c:last"


UNIT = False  # --unit: score at ratio 1 instead of the median ratio
BARS = [*range(575, 691, 5), *range(785, 901, 5)]  # 5-minute bar labels


def hit(y: np.ndarray, x: np.ndarray, tol: float = 1e-6) -> tuple[np.ndarray, float]:
    """Per-cell exact flag after scaling x by the median ratio, or by 1 with ``--unit`` (cells with no value are dropped)."""
    m = np.isfinite(y) & np.isfinite(x) & (x != 0)
    if m.sum() < 20:
        return np.zeros(0, bool), np.nan
    r = 1.0 if UNIT else np.median(y[m] / x[m])
    out = np.full(y.shape, np.nan)
    out[m] = np.abs(y[m] - r * x[m]) <= tol * np.abs(y[m]) + 1e-12
    return out, r


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--combos", default=COMBOS, help="family:V:basic,...")
    ap.add_argument("--days", required=True, help="PM days to score")
    ap.add_argument("--n", type=int, default=22)
    ap.add_argument("--excl", action="store_true", help="window [d-N, d-1] instead of [d-N+1, d]")
    ap.add_argument("--hstats", type=Path, default=Path("/data/prod/hstats/HS300"))
    ap.add_argument("--cache", type=Path, default=Path("data/pm_export_all"))
    ap.add_argument("--basics", type=Path, default=BASICS)
    ap.add_argument("--unit", action="store_true", help="exact = equal at ratio 1 (a basic already in PM's unit)")
    a = ap.parse_args()
    global UNIT
    UNIT = a.unit
    have = sorted(p.name.split("=")[1] for p in a.basics.glob("date=*"))
    tabs: dict[tuple[str, str], pl.DataFrame] = {}

    def basic(x: str, table: str, keys: list[str], cells: pl.DataFrame, col: str) -> np.ndarray:
        if (x, table) not in tabs:
            t = pl.read_parquet(a.basics / f"date={x}" / f"{table}.parquet").rename({"symbol_id": "code"})
            tabs[x, table] = t.with_columns(pl.col("code").cast(pl.Int64), pl.col("minute").cast(pl.Int64))
        t = tabs[x, table]
        if col in t.columns:
            v = pl.col(col)
        elif any(ch in col for ch in "+-*/()"):  # an expression over the table's columns, e.g. "ordvolrc_sum - (ordvolrc_sum - ordvolc_sum) * 2 / 5"
            v = pl.sql_expr(col)
        else:
            return np.full(cells.height, np.nan)
        try:
            return cells.select(keys).join(t.select(*keys, v.cast(pl.Float64).alias("_v")), on=keys, how="left")["_v"].to_numpy()
        except pl.exceptions.ColumnNotFoundError:
            return np.full(cells.height, np.nan)

    def median(stack: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN cells
            return np.nanmedian(stack, axis=0)

    N = a.n
    for combo in a.combos.split(","):
        fam, V, col = combo.split(":", 2)
        tdays = trading_days(a.hstats, fam)
        base, by_day, parts, drops, adds = [], {}, [], {}, {}
        for d in a.days.split(","):
            f = a.cache / f"{fam}_{d}.parquet"
            if not f.exists() or d not in tdays:
                continue
            pm, vals, kind = pm_frame(f, None, None, None, None, None)
            table = "d1" if kind == "daily" else "m5"
            keys = ["code"] if table == "d1" else ["code", "minute"]
            cells = pm.filter(pl.col("V") == V).select(*keys, pl.col(vals[0]).alias("y")).filter(pl.col("y").is_finite())
            cells = cells.with_columns(pl.col(k).cast(pl.Int64) for k in keys)
            i = tdays.index(d)
            span = tdays[i - N - (0 if not a.excl else 1): i + 1]  # N + 1 (or N + 2) trading days, oldest first
            cm = col.startswith("cm:")  # cumulative over the day's bars of per-bar medians (the volume-profile convention)
            src, c = (table, col[3:]) if cm else col.split(".", 1) if col.startswith("book.") else (table, col)  # "book.aamt_x" reads date=D/book.parquet
            miss = [x for x in span if not (a.basics / f"date={x}" / f"{src}.parquet").exists()]
            if miss:
                print(f"{combo} {d}: no {src}.parquet for {len(miss)} days {miss[0]}..{miss[-1]}")
                continue
            y = cells["y"].to_numpy()
            if cm:  # "cm:vol+oa_vol|ca_vol": terms split by | take separate medians, columns joined by + are summed first
                grid = cells.select("code").unique().join(pl.DataFrame({"minute": BARS}), how="cross")
                terms = [t.split("+") for t in c.split("|")]

                def term_day(x: str, cols: list[str]) -> np.ndarray:
                    s = np.stack([basic(x, src, keys, grid, z) for z in cols])
                    return np.where(np.isnan(s).all(axis=0), np.nan, np.nansum(s, axis=0))

                vals_by = {x: [term_day(x, t) for t in terms] for x in span}

                def value(w: list[str]) -> np.ndarray:
                    tot = sum(np.nan_to_num(median(np.stack([vals_by[x][k] for x in w]))) for k in range(len(terms)))
                    g = grid.with_columns(pl.Series("v", tot)).sort("code", "minute").with_columns(pl.col("v").cum_sum().over("code"))
                    return cells.select(keys).join(g, on=keys, how="left")["v"].to_numpy()
            else:
                vals_by = {x: basic(x, src, keys, cells, c) for x in span}

                def value(w: list[str]) -> np.ndarray:
                    return median(np.stack([vals_by[x] for x in w]))
            win = span[1:] if not a.excl else span[1:-1]
            med = value(win)
            h, r = hit(y, med)
            if h.size == 0:
                continue
            by_day[d] = (np.nanmean(h), r)
            base.append(h)
            pos = k = eq = np.full(y.shape, np.nan)
            if not cm:  # where PM sits among our window's day values
                st = np.stack([vals_by[x] for x in win])
                ok = np.isfinite(st).all(axis=0) & np.isfinite(y)
                s = np.sort(st, axis=0)
                lo, hi = s[(len(win) - 1) // 2], s[len(win) // 2]
                with np.errstate(divide="ignore", invalid="ignore"):
                    pos = np.where(ok & (hi > lo), (y - lo) / (hi - lo), np.nan)  # 0 = lower middle value, 1 = upper
                    eq = np.where(ok, (np.abs(st - y) <= 1e-6 * np.abs(y) + 1e-12).any(axis=0), np.nan)  # PM equals one of our day values
                k = np.where(ok, (st < y - 1e-6 * np.abs(y)).sum(axis=0), np.nan)  # our days strictly below PM
            parts.append(cells.select("code", *(["minute"] if "minute" in keys else [])).with_columns(
                pl.Series("hit", h), pl.Series("rel", y / med - 1.0), pl.Series("pos", pos), pl.Series("k", k), pl.Series("eq", eq)))
            for x in win:  # leave x out, take one more day from the front
                w = [z for z in span if z != x][-N:] if not a.excl else [z for z in span[:-1] if z != x][-N:]
                drops.setdefault(x, []).append(np.nanmean(hit(y, value(w))[0]))
            for e in [z for z in have if not cm and z not in tdays and win[0] < z < d and (a.basics / f"date={z}" / f"{src}.parquet").exists()]:  # a basics day the calendar skips
                if e not in vals_by:
                    vals_by[e] = basic(e, src, keys, cells, c)
                w = sorted([*win, e])[-N:]
                adds.setdefault(e, []).append(np.nanmean(hit(y, median(np.stack([vals_by[z] for z in w])))[0]))
        if not base:
            print(f"{combo}: nothing scored")
            continue
        allh = pl.concat(parts).filter(pl.col("hit").is_not_nan())
        ex = allh["hit"].mean()
        mk = allh.group_by(pl.when(pl.col("code") >= 600000).then(pl.lit("SH")).otherwise(pl.lit("SZ")).alias("m")).agg(
            pl.col("hit").mean(), *[pl.col("rel").filter(pl.col("rel").is_finite()).quantile(q).alias(f"r{int(q * 100)}") for q in (0.1, 0.5, 0.9)]).sort("m")
        dd = sorted(((x, float(np.mean(v)) - ex) for x, v in drops.items()), key=lambda t: -t[1])
        print(f"== {fam} {V} ~ median_{N}{'excl' if a.excl else ''}({col}): exact {ex:.3f}, ratio {np.median([r for _, r in by_day.values()]):.4f}")
        print("  day   " + " ".join(f"{d[4:]} {e:.3f}" for d, (e, _) in by_day.items()))
        print("  exch  " + " ".join(f"{m} {h:.3f}" for m, h, *_ in mk.iter_rows()))
        print("  PM/ours-1 p10/p50/p90  " + " ".join(f"{m} {q1:+.4f}/{q5:+.4f}/{q9:+.4f}" for m, _, q1, q5, q9 in mk.iter_rows()))
        rk = allh.filter(pl.col("k").is_not_nan()).group_by(pl.when(pl.col("code") >= 600000).then(pl.lit("SH")).otherwise(pl.lit("SZ")).alias("m")).agg(
            pl.col("pos").filter(pl.col("pos").is_not_nan()).is_between(0, 1).mean().alias("inmid"),
            *[pl.col("pos").filter(pl.col("pos").is_not_nan()).quantile(q).alias(f"p{int(q * 100)}") for q in (0.1, 0.5, 0.9)],
            *[pl.col("k").quantile(q).alias(f"k{int(q * 100)}") for q in (0.1, 0.5, 0.9)], pl.col("eq").mean().alias("eq")).sort("m")
        if rk.height:  # pos: 0 / 1 = our lower / upper middle day value; k: how many of our days lie below PM
            def g(v: float | None, spec: str) -> str:
                return "-" if v is None else format(v, spec)
            print("  rank   " + " | ".join(f"{m} between middles {g(im, '.3f')} pos {g(a1, '+.2f')}/{g(a5, '+.2f')}/{g(a9, '+.2f')} "
                                         f"days below {g(b1, '.0f')}/{g(b5, '.0f')}/{g(b9, '.0f')} = a day value {g(e, '.3f')}"
                                         for m, im, a1, a5, a9, b1, b5, b9, e in rk.iter_rows()))
        if "minute" in allh.columns and allh["minute"].n_unique() > 1:
            rb = allh.filter(pl.col("rel").is_finite()).group_by(pl.when(pl.col("code") >= 600000).then(pl.lit("SH")).otherwise(pl.lit("SZ")).alias("m"), "minute").agg(
                pl.col("rel").median().alias("r")).with_columns(pl.col("r").abs().alias("a")).sort("m", "a", descending=[False, True])
            print("  off bars (p50 PM/ours-1) " + " | ".join(m + " " + " ".join(f"{mi // 60:02d}{mi % 60:02d} {r:+.4f}" for _, mi, r, _ in g.head(5).iter_rows())
                                                          for (m,), g in rb.group_by("m", maintain_order=True)))
            bm = allh.group_by("minute").agg(pl.col("hit").mean()).sort("hit")
            print("  worst bars " + " ".join(f"{m // 60:02d}{m % 60:02d} {h:.2f}" for m, h in bm.head(6).iter_rows())
                  + " | best " + " ".join(f"{m // 60:02d}{m % 60:02d} {h:.2f}" for m, h in bm.tail(3).iter_rows()))
        sc = allh.group_by("code").agg(pl.col("hit").mean()).sort("hit")
        print(f"  stocks: {sc.filter(pl.col('hit') == 1).height} all-exact, {sc.filter(pl.col('hit') == 0).height} never, of {sc.height};"
              " worst " + " ".join(f"{c:06d} {h:.2f}" for c, h in sc.head(5).iter_rows()))
        print("  drop day (exact change) " + " ".join(f"{x[4:]} {v:+.3f}" for x, v in dd[:4]) + " ... " + " ".join(f"{x[4:]} {v:+.3f}" for x, v in dd[-2:]))
        if adds:
            print("  add non-trading day " + " ".join(f"{e} {float(np.mean(v)) - ex:+.3f}" for e, v in adds.items()))


if __name__ == "__main__":
    main()
