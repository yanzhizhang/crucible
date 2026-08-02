"""Full crucible pipeline on real 3-second microstructure data.

Consumes the panel built by :mod:`build_tick_panel` and runs it through every
stage: forge transforms, horizon labels, assay evaluation, sieve admission,
kiln purged walk-forward, ballast/toll/ledger accounting, herald reporting.

The honest framing, stated before any number appears
----------------------------------------------------
At 3-second slots the cost side is brutal and structural. A round trip pays
stamp duty (5 bp, sell-side), commission (2.5 bp each way) and impact -- call it
~10 bp -- while a 30-second forward return has a standard deviation of a few
basis points. **Rebalancing every slot cannot win**, and no factor quality
fixes that; it is arithmetic, not a modelling failure.

So the study reports two things separately:

1. **Does the signal contain information?** -- IC, decay, quantile monotonicity.
   This is what microstructure factors are genuinely good at.
2. **Does any rebalance frequency survive costs?** -- a sweep, because the
   answer is a frequency, not a yes/no.

Reporting only the second would throw away a real finding; reporting only the
first would be the standard way tick research lies to itself.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import polars as pl

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
# `src/` sits one level up in the repo but alongside the script when the
# package is shipped to a data box. Accept either rather than assuming.
for _cand in (_HERE.parent / "src", _HERE / "src"):
    if _cand.is_dir():
        sys.path.insert(0, str(_cand))
        break

from micro_factors import (  # noqa: E402
    BOOK_FACTORS,
    MICRO_FACTORS,
    add_forward_label,
    build_book_factors,
    build_factors,
)

from assay.cross_sectional import decay, ic, ic_summary, quantile_returns, turnover  # noqa: E402
from assay.time_series import performance  # noqa: E402
from ballast.portfolio import Constraints, score_to_weight  # noqa: E402
from crucible.determinism import seed_all  # noqa: E402
from crucible.errors import LeakageError  # noqa: E402
from forge.ops import apply_pipeline  # noqa: E402
from kiln.models import assert_no_feature_leakage, walk_forward  # noqa: E402
from kiln.splits import PurgedWalkForward  # noqa: E402
from ledger.accounting import simulate_cross_sectional  # noqa: E402
from sieve.screen import FactorPool  # noqa: E402
from toll.costs import CostModel, stamp_duty_for  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

SLOTS_PER_DAY = 4800  # 4h of continuous trading at 3s
PERIODS_PER_YEAR = 244.0 * SLOTS_PER_DAY
EQUITY_PREFIX = ("60", "68", "00", "30")


def rule(t: str) -> None:
    print(f"\n{'=' * 78}\n{t}\n{'=' * 78}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("panel", type=Path, help="flow panel from build_tick_panel")
    ap.add_argument("--mbo", type=Path, default=None, help="order-book panel from build_mbo_panel")
    ap.add_argument("--top", type=int, default=300, help="most-traded symbols to keep")
    ap.add_argument("--horizon", type=int, default=10, help="label horizon in slots (10 = 30s)")
    ap.add_argument("--window", type=int, default=20, help="factor lookback in slots")
    ap.add_argument("--out", type=Path, default=Path("data/tick_out"))
    a = ap.parse_args()
    seed_all()
    a.out.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- load
    rule("1. panel")
    panel = pl.read_parquet(a.panel)
    print(f"raw: {panel.height:,} rows  {panel['symbol'].n_unique():,} symbols  "
          f"{panel['slot'].n_unique():,} slots")

    # Equities only: the dump carries funds, bonds and ETFs, whose
    # microstructure is different enough that pooling them into one
    # cross-section compares unlike things.
    panel = panel.filter(pl.col("symbol").str.slice(0, 2).is_in(EQUITY_PREFIX))

    liquid = (
        panel.group_by("symbol")
        .agg(pl.col("amount").sum().alias("amt"), pl.col("n_trades").sum().alias("nt"))
        .filter(pl.col("nt") > 0)
        .sort("amt", descending=True)
        .head(a.top)["symbol"]
        .to_list()
    )
    panel = panel.filter(pl.col("symbol").is_in(liquid))
    print(f"equities, top {a.top} by turnover: {panel.height:,} rows  "
          f"{panel['symbol'].n_unique():,} symbols")

    # ---------------------------------------------------- factors + label
    rule("2. forge + horizon -- microstructure factors and a 30s label")
    df = build_factors(panel, window=a.window)

    if a.mbo is not None and a.mbo.exists():
        book = pl.read_parquet(a.mbo)
        two_sided = book.filter((pl.col("best_bid") > 0) & (pl.col("best_ask") > 0)).height
        print(f"MBO panel: {book.height:,} rows, {two_sided / book.height:.1%} two-sided")
        book = build_book_factors(book, window=a.window)
        keep_cols = ["slot", "symbol", *[c for c in BOOK_FACTORS if c in book.columns]]
        df = df.join(book.select(keep_cols), on=["slot", "symbol"], how="left")
        print(f"joined {len(keep_cols) - 2} book factors")

    df = add_forward_label(df, horizon=a.horizon, entry_lag=1, label="y")
    df = df.rename({"slot": "ts"})

    have = [f for f in (*MICRO_FACTORS, *BOOK_FACTORS) if f in df.columns]
    print(f"factors: {len(have)}")
    print(f"label: forward {a.horizon} slots ({a.horizon * 3}s), entry lagged 1 slot")

    df = df.with_columns(
        [pl.col(f).replace([float("inf"), float("-inf")], None) for f in have]
    ).filter(pl.col("y").is_not_null())

    steps = []
    for f in have:
        steps += [
            {"op": "winsorize", "column": f, "method": "mad", "k": 5.0},
            {"op": "zscore", "column": f},
        ]
    df = apply_pipeline(df, steps)
    print(f"winsorized (MAD-5) + cross-sectionally z-scored per slot")
    print(f"usable rows: {df.height:,}   cross-sections: {df['ts'].n_unique():,}")

    # ----------------------------------------------------------- 3. assay
    rule("3. assay -- information content")
    rows = []
    for f in have:
        r = ic(df, f, "y", min_names=30)
        c = ic_summary(r.series, method=r.method, newey_west_lags=a.horizon - 1)
        rows.append({
            "factor": f, "ic": c.mean, "icir": c.icir, "t_nw": c.t_stat,
            "pos_rate": c.positive_rate, "n": c.n_periods,
        })
    rank = pl.DataFrame(rows).sort(pl.col("ic").abs(), descending=True)
    print(rank.head(18))
    rank.write_csv(a.out / "factor_ranking.csv")

    best = rank["factor"][0]
    print(f"\nstrongest: {best}  IC={rank['ic'][0]:+.5f}  t(NW)={rank['t_nw'][0]:+.2f}")

    q = quantile_returns(df, best, "y", n=10)
    print(f"  {q}")
    t = turnover(df, best)
    print(f"  {t}")

    # Decay across horizons tells you the rebalance frequency the signal can
    # actually support -- the single most useful number in a tick study.
    lab = build_factors(panel, window=a.window)
    for h in (1, 2, 5, 10, 20, 40):
        lab = add_forward_label(lab, horizon=h, entry_lag=1, label=f"y{h}")
    lab = lab.rename({"slot": "ts"})
    lab = apply_pipeline(lab, [{"op": "winsorize", "column": best, "method": "mad", "k": 5.0},
                               {"op": "zscore", "column": best}])
    d = decay(lab, best, {h: f"y{h}" for h in (1, 2, 5, 10, 20, 40)}, min_names=30)
    print(f"\n  {d}")
    print(d.curve)

    # ----------------------------------------------------------- 4. sieve
    rule("4. sieve -- pool admission")
    pool = FactorPool(threshold=0.7, reject_above=0.95)
    order = rank["factor"].to_list()
    work = df
    for f in order[:12]:
        res, work = pool.propose(work, f, min_names=30)
        print(f"  {res}")
    print(f"\nadmitted {len(pool)}: {pool.factors}")
    pool.admission_log().write_csv(a.out / "admission_log.csv")

    # ------------------------------------------------------------ 5. kiln
    rule("5. kiln -- purged walk-forward")
    feats = pool.factors[:10]
    try:
        assert_no_feature_leakage(work, feats, "y", min_names=30)
        print(f"leakage canary: clean ({len(feats)} features)")
    except LeakageError as exc:
        print(f"LEAKAGE: {exc}")
        return

    n_slots = work["ts"].n_unique()
    cv = PurgedWalkForward(
        train_span=max(200, n_slots // 6),
        test_span=max(50, n_slots // 24),
        embargo=a.horizon,
        label_horizon=a.horizon + 1,
    )
    wf = walk_forward(work, feats, "y", cv, num_boost_round=120, seed=20240101)
    print(wf)
    print(wf.fold_scores)

    if wf.predictions.height == 0:
        print("no out-of-sample predictions")
        return
    oos = ic(wf.predictions, "pred", "y", min_names=30)
    oos_nw = ic_summary(oos.series, newey_west_lags=a.horizon - 1)
    print(f"\nout-of-sample IC {oos_nw.mean:+.5f}  t(NW)={oos_nw.t_stat:+.2f}  "
          f"n={oos_nw.n_periods}")

    # ------------------------------------- 6. rebalance-frequency sweep
    rule("6. ballast + toll + ledger -- what frequency survives costs?")
    model = CostModel(stamp_duty=stamp_duty_for("2026-04-21"))
    print(f"stamp duty {model.stamp_duty:.4%} sell-side, commission "
          f"{model.commission:.4%}/side\n")

    px = panel.rename({"slot": "ts"}).select(
        "ts", "symbol", pl.col("vwap").forward_fill().over("symbol").alias("vwap"),
        pl.col("volume").alias("adv"), pl.lit(0.002).alias("sigma"),
    )
    preds = wf.predictions.join(px, on=["ts", "symbol"], how="inner")

    print(f"{'every':>8} {'slots':>7} {'gross Sh':>10} {'net Sh':>9} "
          f"{'cost bp/rb':>11} {'turnover':>9}")
    sweep = []
    for every in (1, 10, 20, 60, 200, 600):
        stamps = sorted(preds["ts"].unique().to_list())[::every]
        sub = preds.filter(pl.col("ts").is_in(stamps))
        if sub["ts"].n_unique() < 8:
            continue
        w = score_to_weight(
            sub, "pred", method="rank",
            constraints=Constraints(max_weight=0.02, min_weight=-0.02,
                                    max_industry=None, gross_exposure=1.0),
        )
        gross = simulate_cross_sectional(w, sub, capital=5e7, costs=None)
        net = simulate_cross_sectional(w, sub, capital=5e7, costs=model)
        ppy = PERIODS_PER_YEAR / every
        pg = performance(gross.pnl, periods_per_year=ppy)
        pn = performance(net.pnl, periods_per_year=ppy)
        sweep.append({
            "every_slots": every, "seconds": every * 3, "n_rebal": sub["ts"].n_unique(),
            "gross_sharpe": pg.sharpe, "net_sharpe": pn.sharpe,
            "cost_bp": net.cost_drag_bps, "turnover": float(net.pnl["turnover"].mean()),
        })
        print(f"{every * 3:>7}s {sub['ts'].n_unique():>7} {pg.sharpe:>10.2f} "
              f"{pn.sharpe:>9.2f} {net.cost_drag_bps:>11.2f} "
              f"{float(net.pnl['turnover'].mean()):>9.3f}")

    if sweep:
        sw = pl.DataFrame(sweep)
        sw.write_csv(a.out / "rebalance_sweep.csv")
        bestrow = sw.sort("net_sharpe", descending=True).head(1)
        print(f"\nbest net: rebalance every {bestrow['seconds'][0]}s  "
              f"net Sharpe {bestrow['net_sharpe'][0]:+.2f}")

    rule("done")
    print(f"artifacts -> {a.out}")


if __name__ == "__main__":
    main()
