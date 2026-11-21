r"""A profile of PM state keys we cannot rebuild: what their weight, clock stamp and value say, before guessing a formula.

For every key under the given prefixes in PM's day-end ``.jz`` (grids ``.<n>`` skipped), per day:
  * shape: an average (``.sum / .wgt / .t``) or a plain value; stocks carrying it;
  * weight: quantiles, how many distinct values (one value for all stocks = a fixed time grid), the half-life it implies if
    the average were fed every 3 s / 1 s without end (w = 1 / (1 - exp(-dt ln2 / half))), its rank correlation with the day's
    event counts per stock (snapshots, snapshots with a volume change, trades, orders, cancels) and the ratio weight / count;
  * stamp ``.t`` (trading-clock seconds): quantiles and the share of stocks at the latest stamp;
  * value (sum / wgt): quantiles, share negative / zero, its ratio to the last price, rank correlation with the day's
    simple statistics (last price, return, range, volume, amount, trade count, average trade size, buy share);
  * against our engine: the five engine keys with the highest |rank correlation|, and any engine key equal on >= 20 % of stocks;
  * against yesterday: rank correlation of the value with the previous day's value, and of today's sum with yesterday's sum.
A final table sorts the keys by the strongest lead found. Read-only.

    python research/engine/key_profile.py --dates 20261118,20261119,20261120 --jz /work/crucible_data/seed_probe/HS300 \
        --catalog /work/crucible_data/catalog_s224.duckdb --engine data/engine_r194/dysl2 --prefix zzcb,zzeb.r,zztx,zzhj,zzvm.c.re,zzui,zzbd.drSum,zzgh2.nb,zzgh.nb,zzxu.upper,zzxu.lower,zzxu.dir
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path

import numpy as np

SOD = "((time / 1e9 + 28800) % 86400)"
GRID = re.compile(r"\.\d+$")


def load(path: Path) -> dict[int, dict[str, float]]:
    """A state file as it is: sid -> key -> number."""
    with gzip.open(path) as f:
        d = json.load(f)
    return {int(s): {k: float(v) for k, v in kv.items() if isinstance(v, (int, float))} for s, kv in d.items()}


def fold(kv: dict[str, float]) -> dict[str, float]:
    """<p>.sum / <p>.wgt -> <p>; other keys kept."""
    out = {}
    for k, v in kv.items():
        if k.endswith(".sum"):
            w = kv.get(k[:-4] + ".wgt")
            out[k[:-4]] = v / w if w else math.nan
        elif not k.endswith(".wgt"):
            out[k] = v
    return out


def rank(x: np.ndarray) -> np.ndarray:
    o = np.argsort(x, kind="stable")
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r


def spear(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation on the rows where both are finite."""
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 20 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return math.nan
    return float(np.corrcoef(rank(a[m]), rank(b[m]))[0, 1])


def q(v: np.ndarray, qs=(1, 10, 50, 90, 99)) -> str:
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.5g}" for x in np.percentile(v, qs)) if v.size else "-"


def day_stats(con, date: int, sids: list[int]) -> dict[str, dict[int, float]]:
    """Per stock: event counts and simple statistics of the day from the tick store."""
    import polars as pl

    con.register("want_kp", pl.DataFrame({"sid": sids}))
    out: dict[str, dict[int, float]] = {}
    qd = con.sql(f"""SELECT symbol_id AS sid, count(*) FILTER ({SOD} >= 34200 AND {SOD} <= 54000)::DOUBLE AS n_snap,
        (count(DISTINCT total_volume) FILTER ({SOD} >= 34200 AND {SOD} <= 54000))::DOUBLE AS n_snap_vol,
        arg_max(price, time)::DOUBLE AS last, max(pre_close)::DOUBLE AS pc, max(high)::DOUBLE AS hi, min(low) FILTER (low > 0)::DOUBLE AS lo,
        max(total_volume)::DOUBLE AS volume, max(total_amount)::DOUBLE AS amount
        FROM raw_quotation q JOIN want_kp w ON w.sid = q.symbol_id WHERE q.date = {date} GROUP BY 1""").pl()
    td = con.sql(f"""SELECT symbol_id AS sid, count(*) FILTER (trade_type = 1)::DOUBLE AS n_trade, count(*) FILTER (trade_type <> 1)::DOUBLE AS n_cancel,
        (sum(volume * price) FILTER (trade_type = 1 AND dir = 1063) / nullif(sum(volume * price) FILTER (trade_type = 1 AND dir IN (1063, 1550)), 0))::DOUBLE AS buy_share
        FROM raw_transaction r JOIN want_kp w ON w.sid = r.symbol_id WHERE r.date = {date} AND {SOD} >= 34200 GROUP BY 1""").pl()
    od = con.sql(f"""SELECT symbol_id AS sid, count(*)::DOUBLE AS n_order FROM raw_order o JOIN want_kp w ON w.sid = o.symbol_id WHERE o.date = {date} AND {SOD} >= 34200 GROUP BY 1""").pl()
    j = qd.join(td, on="sid", how="left").join(od, on="sid", how="left").with_columns(
        (pl.col("last") / pl.col("pc") - 1).alias("ret"), ((pl.col("hi") - pl.col("lo")) / pl.col("pc")).alias("range"),
        (pl.col("volume") / pl.col("n_trade")).alias("trade_size"), (pl.col("n_snap") + pl.col("n_trade")).alias("n_snap+trade"),
        (pl.col("n_snap") + pl.col("n_trade") + pl.col("n_order").fill_null(0) + pl.col("n_cancel").fill_null(0)).alias("n_all"))
    sid = j["sid"].to_list()
    for c in j.columns:
        if c != "sid":
            out[c] = dict(zip(sid, j[c].to_list(), strict=True))
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--jz", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--engine", type=Path, help="dir of engine_<date>.json.gz")
    ap.add_argument("--prefix", required=True)
    a = ap.parse_args()
    import duckdb

    con = duckdb.connect(str(a.catalog), read_only=True)
    pre = tuple(p for p in a.prefix.split(",") if p)
    days = [d for d in a.dates.split(",") if d]
    allj = sorted(p.stem for p in a.jz.glob("2*.jz"))
    lead: dict[str, list[str]] = {}
    for d in days:
        pm = load(a.jz / f"{d}.jz")
        prev_day = max((x for x in allj if x < d), default=None)
        prev = load(a.jz / f"{prev_day}.jz") if prev_day else {}
        sids = sorted(pm)
        st = day_stats(con, int(d), sids)
        eng = {}
        if a.engine and (a.engine / f"engine_{d}.json.gz").exists():
            eng = {s: fold(kv) for s, kv in load(a.engine / f"engine_{d}.json.gz").items()}
        ekeys = sorted({k for kv in eng.values() for k in kv if not GRID.search(k) and "|" not in k and not k.endswith(".t")})
        emat = {k: np.array([eng.get(s, {}).get(k, math.nan) for s in sids]) for k in ekeys}
        smat = {k: np.array([v.get(s, math.nan) if v.get(s) is not None else math.nan for s in sids], dtype=float) for k, v in st.items()}
        keys = sorted({k for kv in pm.values() for k in kv if k.startswith(pre) and not GRID.search(k)})
        bases = sorted({k[:-4] for k in keys if k.endswith(".sum")} | {k for k in keys if not k.endswith((".sum", ".wgt", ".t")) or k[: k.rfind(".")] + ".sum" not in keys})
        bases = [b for b in bases if not (b.endswith(".t") and b[:-2] + ".sum" in keys)]
        print(f"\n######## {d} (previous file {prev_day}); {len(sids)} stocks; keys under {pre}: {len(bases)}")
        for b in bases:
            tri = f"{b}.sum" in keys
            col = lambda k, src=pm: np.array([src.get(s, {}).get(k, math.nan) for s in sids])  # noqa: E731
            val = col(f"{b}.sum") / col(f"{b}.wgt") if tri else col(b)
            have = int(np.isfinite(val).sum())
            print(f"\n== {b}: {'average (sum / wgt / t)' if tri else 'plain value'}; on {have} stocks")
            notes = []
            if tri:
                w, t = col(f"{b}.wgt"), col(f"{b}.t")
                wf = w[np.isfinite(w)]
                nd = len(np.unique(np.round(wf, 6)))
                med = float(np.median(wf)) if wf.size else math.nan
                hl = lambda dt: dt * math.log(2) / math.log(med / (med - 1)) if med > 1 else math.nan  # noqa: E731
                print(f"   wgt q1/10/50/90/99 {q(w)}; distinct values {nd}; half-life if fed every 3 s {hl(3):.4g} s, every 1 s {hl(1):.4g} s; sum q {q(col(f'{b}.sum'))}")
                cs = []
                for c in ("n_snap", "n_snap_vol", "n_trade", "n_order", "n_cancel", "n_snap+trade", "n_all"):
                    r = spear(w, smat[c])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = w / smat[c]
                    cs.append((abs(r) if r == r else 0, f"{c} {r:+.3f} (wgt / count q10/50/90 {q(ratio, (10, 50, 90))})"))
                print("   wgt vs event counts (rank corr): " + "; ".join(x for _, x in sorted(cs, reverse=True)[:4]))
                tf = t[np.isfinite(t)]
                if tf.size:
                    print(f"   t q1/50/99 {q(t, (1, 50, 99))}; at the latest stamp ({tf.max():.3f}) {float((tf >= tf.max() - 1e-6).mean()):.3f}; whole seconds {float((np.abs(tf - np.round(tf)) < 1e-6).mean()):.3f}")
                if nd <= 3:
                    notes.append("one weight for all stocks: a fixed time grid")
                elif cs and max(cs)[0] > 0.98:
                    notes.append("weight follows " + max(cs)[1].split(" ")[0])
            vf = val[np.isfinite(val)]
            if not vf.size:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = val / smat["last"]
            print(f"   value q1/10/50/90/99 {q(val)}; negative {float((vf < 0).mean()):.3f}, zero {float((vf == 0).mean()):.3f}, distinct {len(np.unique(np.round(vf, 9)))}; value / last price q {q(rel)}")
            ss = sorted(((abs(r), f"{c} {r:+.3f}") for c in smat if (r := spear(val, smat[c])) == r), reverse=True)[:5]
            print("   vs day statistics (rank corr): " + "; ".join(x for _, x in ss))
            if emat:
                es = sorted(((abs(r), k, r) for k in emat if not k.startswith(b) and (r := spear(val, emat[k])) == r), reverse=True)[:5]
                print("   vs engine keys (rank corr): " + "; ".join(f"{k} {r:+.3f}" for _, k, r in es))
                eq = [(float((np.abs(emat[k] - val) <= 1e-6 * np.maximum(1, np.abs(val))).mean()), k) for k in emat]
                eq = [x for x in sorted(eq, reverse=True) if x[0] >= 0.2][:4]
                if eq:
                    print("   engine keys EQUAL on a share of stocks: " + "; ".join(f"{k} {sh:.3f}" for sh, k in eq))
                    notes.append(f"equals engine {eq[0][1]} on {eq[0][0]:.2f}")
                if es and es[0][0] > 0.97:
                    notes.append(f"rank corr {es[0][2]:+.3f} with engine {es[0][1]}")
            if ss and ss[0][0] > 0.9:
                notes.append("rank corr " + ss[0][1])
            if prev:
                pv = col(f"{b}.sum", prev) / col(f"{b}.wgt", prev) if tri else col(b, prev)
                line = f"   vs the previous day's value: rank corr {spear(val, pv):+.3f}"
                if tri:
                    line += f"; sum today vs sum yesterday {spear(col(f'{b}.sum'), col(f'{b}.sum', prev)):+.3f}; wgt today / wgt yesterday q10/50/90 {q(col(f'{b}.wgt') / col(f'{b}.wgt', prev), (10, 50, 90))}"
                print(line)
            if float((vf == 0).mean()) > 0.99:
                notes.append("always 0")
            lead.setdefault(b, []).append("; ".join(notes) or "no lead")
    print("\n######## leads per key (one entry per day)")
    for b, v in sorted(lead.items()):
        print(f"   {b:<18} " + " || ".join(v))


if __name__ == "__main__":
    main()
