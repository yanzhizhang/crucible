r"""Stage 4c: fit PM's day-end weighted averages (``<p>.sum / .wgt / .t``) as intraday EMAs.

``diag_ema`` showed the day-end ``wgt`` does not depend on the previous day's (one-day recursion
fits nothing), and each prefix's ``.t`` is the fast (snapshot) or slow (trade) clock. So each
average is fitted as an EMA over that day's updates:

    wgt = sum_i w_i exp(-(T - t_i) / tau)        sum / wgt = the same average of x_i

* updates: the stock's snapshots (fast prefixes) or trades (slow prefixes) received by PM's stop,
  either all of them or from 09:30 exchange time; T = the last one;
* clock: trading clock (lunch removed), wall clock, or update count;
* weight w: 1, time since the previous update, volume, amount (snapshot deltas for snapshots);
* tau: solved per stock from PM's ``wgt`` on a log grid; the right (updates, clock, weight) gives
  the same tau for every stock, so combos are ranked by the spread of tau across SZSE stocks
  (PM's SZSE ticks are complete but for a few at the open); the winner's tau is then refined and
  scored exactly, and ``sum / wgt`` is matched against candidate values x (price, mid, returns).

    python research/pm_factors/diag_iema.py --dates 20260923,20260929,20261009
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

SZSE = 3554
GRID = np.logspace(0, 7, 141)  # tau candidates: 1 .. 1e7 (seconds, or updates on the count clock)
CHUNK = 20000
EXACT = 1e-6
FIT_STOCKS = 60  # SZSE stocks per day used to refine tau (all are scored)
WEIGHTS = {"fast": ("one", "dt", "dvol", "damt"), "slow": ("one", "dt", "vol", "amt")}
VALUES = {"fast": ("price", "mid", "logp", "ret", "logret", "absret", "ret2", "spread"),
          "slow": ("price", "logp", "ret", "logret", "absret", "ret2")}


def state(pm: pl.DataFrame) -> pl.DataFrame:
    """(sid, p, sum, wgt, t) per prefix."""
    x = pm.filter(pl.col("key").str.contains(r"\.(sum|wgt|t)$")).with_columns(
        pl.col("key").str.extract(r"^(.*)\.(sum|wgt|t)$", 1).alias("p"), pl.col("key").str.extract(r"\.(sum|wgt|t)$", 1).alias("f"))
    return x.pivot(on="f", index=["sid", "p"], values="pm").filter(pl.col("sum").is_not_null() & pl.col("wgt").is_not_null())


def updates(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> dict[str, pl.DataFrame]:
    """Per-stock update streams: snapshots (fast) and trades (slow) received by the stop."""
    S, L = B.SOD, B.LSOD
    con.register("want", pl.DataFrame({"sid": sids}))
    fast = con.sql(f"""SELECT symbol_id AS sid, market_id AS mkt, {B.tc(L)} AS tl, {L} AS wall, {S} AS s, price,
        CASE WHEN q.bid_px[1] > 0 AND q.ask_px[1] > 0 THEN (q.bid_px[1] + q.ask_px[1]) / 2 ELSE price END AS mid,
        CASE WHEN q.bid_px[1] > 0 AND q.ask_px[1] > 0 THEN (q.ask_px[1] - q.bid_px[1]) / ((q.bid_px[1] + q.ask_px[1]) / 2) ELSE 0 END AS spread,
        total_volume::DOUBLE AS cv, total_amount::DOUBLE AS ca
        FROM raw_quotation q JOIN want w ON w.sid = q.symbol_id WHERE q.date = {date} AND {L} <= {stop}
        ORDER BY sid, spider_ts""").pl()
    slow = con.sql(f"""SELECT symbol_id AS sid, market_id AS mkt, {B.tc(L)} AS tl, {L} AS wall, {S} AS s, price,
        volume::DOUBLE AS vol, volume * price AS amt
        FROM raw_transaction r JOIN want w ON w.sid = r.symbol_id
        WHERE r.date = {date} AND r.trade_type = 1 AND {L} <= {stop} ORDER BY sid, spider_ts""").pl()
    con.unregister("want")
    return {"fast": fast, "slow": slow}


def stream(g: pl.DataFrame, kind: str, start: str) -> dict[str, np.ndarray]:
    """Arrays for one stock: times on each clock, weights and values per update."""
    if start == "0930":
        g = g.filter(pl.col("s") >= 34200)
    px = g["price"].to_numpy().astype(float)
    prev = np.concatenate([[px[0]], px[:-1]]) if px.size else px
    with np.errstate(divide="ignore", invalid="ignore"):
        ret = np.where(prev > 0, px / prev - 1, 0.0)
        a = {"tc": g["tl"].to_numpy().astype(float), "wall": g["wall"].to_numpy().astype(float),
             "count": np.arange(px.size, dtype=float), "price": px, "logp": np.log(np.where(px > 0, px, np.nan)),
             "ret": ret, "logret": np.log1p(ret), "absret": np.abs(ret), "ret2": ret * ret, "one": np.ones(px.size)}
    a["dt"] = np.diff(a["tc"], prepend=a["tc"][0]) if px.size else px
    if kind == "fast":
        a["mid"], a["spread"] = g["mid"].to_numpy().astype(float), g["spread"].to_numpy().astype(float)
        a["dvol"] = np.clip(np.diff(g["cv"].to_numpy().astype(float), prepend=0.0), 0, None)
        a["damt"] = np.clip(np.diff(g["ca"].to_numpy().astype(float), prepend=0.0), 0, None)
    else:
        a["vol"], a["amt"] = g["vol"].to_numpy().astype(float), g["amt"].to_numpy().astype(float)
    return a


def wgt_on_grid(t: np.ndarray, ws: np.ndarray) -> np.ndarray:
    """(n_weights, n_grid): sum_i w_i exp(-(T - t_i) / tau) for every tau on the grid, T = t[-1]."""
    out = np.zeros((ws.shape[0], GRID.size))
    age = t[-1] - t
    for i in range(0, t.size, CHUNK):
        out += ws[:, i:i + CHUNK] @ np.exp(-age[i:i + CHUNK, None] / GRID[None, :])
    return out


def ema(a: dict[str, np.ndarray], clock: str, w: str, tau: float, x: str | None = None) -> float:
    """One stock's weight (x None) or weighted average of x at tau."""
    e = a[w] * np.exp(-(a[clock][-1] - a[clock]) / tau)
    return float(e.sum()) if x is None else float((e * a[x]).sum() / e.sum())


def refine_and_score(r: dict, final: list) -> dict:
    """Refine tau for one (prefix, combo) on up to FIT_STOCKS SZSE stocks a day, then score weight and value."""
    pairs = []  # (market, stream, PM wgt, PM sum)
    for _, st, _k, streams in final:
        for sid, y, s_ in st.filter(pl.col("p") == r["p"]).select("sid", "wgt", "sum").iter_rows():
            s = streams.get((r["kind"], sid, r["start"]))
            if s and s[1]["tc"].size >= 2:
                pairs.append((s[0], s[1], y, s_))
    fit = [q for q in pairs if q[0] == SZSE][:FIT_STOCKS * len(final)]

    def err(lt: float) -> float:
        return float(np.median([abs(ema(a, r["clock"], r["w"], np.exp(lt)) / y - 1) for _, a, y, _s in fit]))

    lo, hi = np.log(r["tau_med"]) - 0.2, np.log(r["tau_med"]) + 0.2  # golden section on log tau
    g = (np.sqrt(5) - 1) / 2
    for _ in range(30):
        m1, m2 = hi - g * (hi - lo), lo + g * (hi - lo)
        lo, hi = (lo, m2) if err(m1) < err(m2) else (m1, hi)
    tau = float(np.exp((lo + hi) / 2))
    rw = {True: [], False: []}
    xs = {x: [] for x in VALUES[r["kind"]]}
    for mkt, a, y, s_ in pairs:
        rw[mkt == SZSE].append(abs(ema(a, r["clock"], r["w"], tau) / y - 1))
        if mkt == SZSE:
            for x in xs:
                xs[x].append(abs(ema(a, r["clock"], r["w"], tau, x) - s_ / y) / max(abs(s_ / y), 1e-12))
    xb = min(xs, key=lambda x: np.nanmedian(xs[x]))
    return {"p": r["p"], "kind": r["kind"], "combo": f"{r['start']}/{r['clock']}/{r['w']}", "tau": tau, "disp": r["disp"],
            "feas": r["feas"], "mre_w": float(np.median(rw[True])), "ok_w": float(np.mean(np.array(rw[True]) <= EXACT)),
            "mre_w_sh": float(np.median(rw[False])) if rw[False] else None, "x": xb,
            "mre_x": float(np.nanmedian(xs[xb])), "ok_x": float(np.nanmean(np.array(xs[xb]) <= EXACT))}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=B.DCP)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4"))
    args = ap.parse_args()
    con = duckdb.connect(str(B.CATALOG), read_only=True)
    taus, final = [], []
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        f = args.dcp_dir / f"{d}.jz"
        if not f.exists():
            print(f"skip {d}: no state file")
            continue
        pm = B.load_state(f)
        st = state(pm)
        t = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        stop = (t + 34200 if t <= 7200 else t - 7200 + 46800) + 0.005
        up = updates(con, int(d), sorted(pm["sid"].unique().to_list()), stop)
        if up["slow"].is_empty():
            print(f"skip {d}: no decoded ticks")
            continue
        # which clock each prefix runs on: its .t against our last snapshot / trade (trading clock)
        last = {k: v.group_by("sid").agg(pl.col("tl").max().alias("T"), pl.col("mkt").first()).with_columns(pl.col("sid").cast(pl.Int64))
                for k, v in up.items()}
        kind_of = {}
        for p in st["p"].unique().sort().to_list():
            g = st.filter(pl.col("p") == p)
            share = {k: g.join(last[k], on="sid").select(((pl.col("t") - pl.col("T")).abs() <= B.CLOCK_TOL).mean()).item()
                     for k in last}
            k = max(share, key=lambda k: share[k] or 0)
            if (share[k] or 0) >= 0.5:
                kind_of[p] = k
        print(f"{d}: {len(kind_of)} of {st['p'].n_unique()} prefixes on a clock "
              f"(fast {sum(v == 'fast' for v in kind_of.values())}, slow {sum(v == 'slow' for v in kind_of.values())}); "
              f"not on one: {sorted(set(st['p'].unique()) - set(kind_of))}")
        streams = {}
        for kind, u in up.items():
            for g in u.partition_by("sid"):
                for start in ("all", "0930"):
                    streams[(kind, int(g["sid"][0]), start)] = (int(g["mkt"][0]), stream(g, kind, start))
        # stage 1: tau per stock on the grid, SZSE only
        for kind in ("fast", "slow"):
            ps = [p for p, k in kind_of.items() if k == kind]
            pmw = st.filter(pl.col("p").is_in(ps))
            for (k2, sid, start), (mkt, a) in streams.items():
                if k2 != kind or mkt != SZSE or a["tc"].size < 2:
                    continue
                mine = pmw.filter(pl.col("sid") == sid)
                for clock in ("tc", "wall", "count"):
                    W = wgt_on_grid(a[clock], np.vstack([a[w] for w in WEIGHTS[kind]]))
                    for p, y in zip(mine["p"].to_list(), mine["wgt"].to_list()):
                        for wi, wn in enumerate(WEIGHTS[kind]):
                            curve = W[wi]
                            ok = curve[0] <= y <= curve[-1] and np.all(np.diff(curve) >= 0)
                            tau = float(np.exp(np.interp(y, curve, np.log(GRID)))) if ok else np.nan
                            taus.append((int(d), p, kind, start, clock, wn, sid, tau))
        final.append((int(d), st, kind_of, streams))
    if not taus:
        return
    T = pl.DataFrame(taus, schema=["date", "p", "kind", "start", "clock", "w", "sid", "tau"], orient="row")
    combo = T.group_by("p", "kind", "start", "clock", "w").agg(
        pl.col("tau").is_not_nan().mean().alias("feas"), pl.col("tau").filter(pl.col("tau").is_not_nan()).median().alias("tau_med"),
        (pl.col("tau").filter(pl.col("tau").is_not_nan()).log().quantile(0.75)
         - pl.col("tau").filter(pl.col("tau").is_not_nan()).log().quantile(0.25)).alias("disp"))
    # stage 2: per prefix the few combos with the tightest tau, refined and scored on weight AND value
    # (on a fixed 3 s cadence w=dt is w=1 times 3 with tau / 3, so the weight alone cannot tell them apart)
    cands = combo.filter(pl.col("feas") >= 0.8).with_columns(pl.col("disp").min().over("p").alias("dmin")).filter(
        pl.col("disp") <= pl.col("dmin") + 0.05).sort("disp").group_by("p", maintain_order=True).head(4)
    # fast prefixes also count SSE: PM's SSE snapshots are complete and their uneven cadence splits clock ties
    sc = pl.DataFrame([refine_and_score(r, final) for r in cands.iter_rows(named=True)]).with_columns(
        (pl.col("mre_w") + pl.col("mre_x").fill_null(1.0)
         + pl.when(pl.col("kind") == "fast").then(pl.col("mre_w_sh").fill_null(0.0)).otherwise(0.0)).alias("score")).sort("score")
    g = sc.group_by("p", maintain_order=True)
    rows = g.first().join(g.agg(pl.col("combo").slice(1, 1).first().alias("combo2"),
                                pl.col("score").slice(1, 1).first().alias("score2")), on="p").drop("score").to_dicts()
    missing = sorted(set(combo["p"].unique()) - {r["p"] for r in rows})
    if missing:
        print(f"no combo feasible for 80% of SZSE stocks: {missing}")
    if not rows:
        return
    out = pl.DataFrame(rows).sort("kind", "mre_w")
    args.out.mkdir(parents=True, exist_ok=True)
    T.write_parquet(args.out / "iema_taus.parquet")
    combo.write_csv(args.out / "iema_combos.csv")
    with pl.Config(tbl_rows=60, tbl_cols=16, tbl_width_chars=230, fmt_str_lengths=24, float_precision=5):
        print("\n== per prefix: best (updates/clock/weight) by the spread of tau over SZSE stocks (disp = log IQR),"
              " refined tau, SZSE / SSE weight error, best value x")
        print(out)
    print(f"taus: {args.out / 'iema_taus.parquet'}, combos: {args.out / 'iema_combos.csv'}")


if __name__ == "__main__":
    main()
