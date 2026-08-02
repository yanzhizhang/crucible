"""End-to-end factor study on real A-share data, through the crucible pipeline.

Runs the full stack against the store built by ``build_store.py``:

    quarry -> almanac -> horizon -> assay -> sieve -> kiln -> ballast/toll/ledger

Every factor here was produced by ``research/factors.py``, the prism stand-in.
The pipeline consumes it through the real loader and the real fingerprint gate,
so the producer/consumer contract is genuinely exercised rather than bypassed.

Read the caveats before the numbers
-----------------------------------
* **Selection bias.** The universe is today's largest names by market cap, so
  it is conditioned on having survived and grown. Levels are optimistic;
  *relative* comparisons between factors on the same panel are what this study
  is for.
* **Short sample.** ~2.5 years is a few hundred cross-sections. Results carry
  ``is_thin`` and the t-statistics are Newey-West corrected, but a single
  regime dominates.
* **Nothing here is tradable.** A factor that scores well must still be ported
  to prism in C++ and reconciled through the parity gate before it means
  anything live.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import polars as pl

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

STORE = ROOT / "data" / "store"
OUT = ROOT / "data" / "out"


def rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def main() -> None:  # noqa: PLR0915
    from almanac.masks import apply_masks, build_masks
    from assay.cross_sectional import decay, ic, ic_summary, quantile_returns, turnover
    from horizon.labels import forward_return
    from quarry.db import open_db
    from quarry.loaders import dump_fingerprint, load_daily, load_factor_frame
    from quarry.schema import SchemaFingerprint

    OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- load
    rule("1. quarry -- load through the producer gate")
    conn = open_db(STORE)
    sidecar = STORE / "factor_frame" / "_fingerprint.json"
    expected = SchemaFingerprint.from_json(sidecar.read_text(encoding="utf-8"))
    actual = dump_fingerprint(conn, "factor_frame", sidecar=str(sidecar))
    print(f"producer          : {expected.producer}")
    print(f"factors declared  : {len(expected)}")
    print(f"fingerprint       : {expected.digest()[:16]}...")

    daily = load_daily(conn)
    factors = load_factor_frame(conn, expected=actual, sidecar=str(sidecar))
    names = [c for c in factors.columns if c not in ("ts", "symbol")]
    print(f"daily rows        : {daily.height:,}")
    print(f"factor rows       : {factors.height:,}  ({len(names)} factors)")
    print(f"span              : {daily['ts'].min()} .. {daily['ts'].max()}")
    print(f"symbols           : {daily['symbol'].n_unique()}")

    # ---------------------------------------------------------------- masks
    rule("2. almanac -- tradability masks")
    masks = build_masks(daily)
    cov = masks.coverage()
    totals = {
        f: int(cov[f].sum()) for f in ("suspended", "limit_up", "limit_down", "one_word_board")
    }
    print(f"observations      : {len(masks):,}")
    for k, v in totals.items():
        print(f"  {k:<16}: {v:>6,}  ({v / len(masks):.2%})")

    # Adjusted VWAP is the label basis; masks apply to features and labels
    # together, in one call, so a label can never survive a masked feature row.
    daily = daily.with_columns(adj_vwap=pl.col("vwap") * pl.col("adj_factor"))
    daily = apply_masks(daily, masks, columns=["close", "adj_close", "adj_vwap", "volume"])

    # ---------------------------------------------------------------- labels
    rule("3. horizon -- tradable forward returns")
    lab = daily
    for h in (1, 2, 5, 10, 20):
        lab = forward_return(lab, n=h, price="adj_vwap", entry_lag=1, label=f"y{h}")
    print("VWAP-to-VWAP, entry_lag=1 (decide at t, enter over t+1, exit t+1+n)")
    for h in (1, 5, 20):
        col = lab[f"y{h}"]
        print(f"  y{h:<3}: {col.drop_nulls().len():>7,} non-null   mean={col.mean():+.5f}")

    panel = lab.join(factors, on=["ts", "symbol"], how="inner")
    print(f"panel             : {panel.height:,} rows")

    # ---------------------------------------------------------------- leakage
    rule("4. kiln -- leakage canary on every factor")
    from kiln.models import assert_no_feature_leakage

    assert_no_feature_leakage(panel, names, "y1", max_abs_ic=0.95, min_names=20)
    print(f"all {len(names)} factors passed |IC| < 0.95 against y1")

    # ---------------------------------------------------------------- assay
    rule("5. assay -- single-factor evaluation (ranked by |ICIR|)")
    from factors import FACTORS

    rows = []
    for f in names:
        r = ic(panel, f, "y1", method="spearman", min_names=20)
        # y1 with entry_lag=1 does not overlap, so no NW correction is needed
        # here; longer horizons below do overlap and are corrected.
        t = turnover(panel, f, min_names=20)
        q = quantile_returns(panel, f, "y1", n=5)
        rows.append(
            {
                "factor": f,
                "family": FACTORS[f][0],
                "ic": r.mean,
                "icir": r.icir,
                "t_stat": r.t_stat,
                "pos_rate": r.positive_rate,
                "periods": r.n_periods,
                "breadth": r.mean_breadth,
                "turnover": t.mean,
                "q_spread": q.spread_mean,
                "monotonic": q.monotonicity,
            }
        )

    table = pl.DataFrame(rows).with_columns(abs_icir=pl.col("icir").abs()).sort(
        "abs_icir", descending=True
    )
    with pl.Config(tbl_rows=60, tbl_width_chars=200, float_precision=4):
        print(
            table.select(
                "factor", "family", "ic", "icir", "t_stat", "pos_rate",
                "turnover", "q_spread", "monotonic", "periods",
            )
        )
    table.write_csv(OUT / "factor_ranking.csv")
    print(f"\nwrote {OUT / 'factor_ranking.csv'}")

    thin = table["periods"].min()
    print(f"\nsample: {thin} cross-sections -- {'THIN' if thin < 250 else 'adequate'}")

    # ---------------------------------------------------------------- decay
    rule("6. assay -- decay of the strongest factor")
    best = str(table["factor"][0])
    d = decay(panel, best, {1: "y1", 2: "y2", 5: "y5", 10: "y10", 20: "y20"}, min_names=20)
    print(f"factor: {best}")
    with pl.Config(float_precision=4):
        print(d.curve)
    hl = "not reached" if d.half_life == float("inf") else f"{d.half_life:.1f} periods"
    print(f"peak IC {d.peak_ic:+.4f} at horizon {d.peak_horizon}; half-life {hl}")
    tv = turnover(panel, best, min_names=20)
    print(f"implied holding {tv.implied_holding_periods:.1f} periods vs half-life {hl}")
    print("t-stats above are Newey-West corrected per horizon for label overlap.")

    # ---------------------------------------------------------------- sieve
    rule("7. sieve -- factor pool admission")
    from sieve.screen import FactorPool, correlation_matrix

    pool = FactorPool(threshold=0.7, reject_above=0.95)
    ordered = [str(f) for f in table["factor"].to_list()]
    # Compute the full matrix once; screening each candidate against a freshly
    # recomputed matrix is O(k) redundant passes over the panel.
    cmat = correlation_matrix(panel, ordered, min_names=20)
    working = panel
    for cand in ordered:
        res, working = pool.propose(working, cand, min_names=20, corr=cmat)
        mark = {"admit": "+", "orthogonalize": "~", "reject": "-"}[res.verdict]
        print(f"  {mark} {cand:<16} {res.verdict:<14} max|rho|={res.max_abs_corr:.3f} vs {res.most_correlated}")
    print(f"\nadmitted {len(pool)} of {len(ordered)}")
    pool.admission_log().write_csv(OUT / "admission_log.csv")

    # ---------------------------------------------------------------- kiln
    rule("8. kiln -- purged walk-forward")
    from kiln.splits import PurgedWalkForward

    feats = pool.factors[:12]
    model_panel = working.drop_nulls(["y1", *feats])
    n_ts = model_panel["ts"].n_unique()
    cv = PurgedWalkForward(
        train_span=max(60, n_ts // 4), test_span=20, embargo=2, label_horizon=2
    )
    print(f"features   : {len(feats)}")
    print(f"timestamps : {n_ts}  (train={cv.train_span} test={cv.test_span} embargo={cv.embargo})")

    try:
        from kiln.models import walk_forward

        wf = walk_forward(model_panel, feats, "y1", cv, num_boost_round=120, seed=20240101)
        print(f"\n{wf}")
        with pl.Config(float_precision=4):
            print(wf.fold_scores)
        preds = wf.predictions
    except ImportError as exc:
        print(f"\nlightgbm unavailable ({exc}); falling back to an equal-weight composite")
        preds = _equal_weight_composite(model_panel, feats)

    if preds.height == 0:
        print("no out-of-sample predictions; stopping before portfolio construction")
        conn.close()
        return

    oos_ic = ic(preds, "pred", "y1", min_names=20)
    print(f"\nout-of-sample IC: {oos_ic}")

    # ------------------------------------------------------- portfolio + costs
    rule("9. ballast + toll + ledger -- portfolio, costs, PnL")
    from ballast.portfolio import Constraints, score_to_weight
    from ledger.accounting import simulate_cross_sectional
    from toll.costs import CostModel, stamp_duty_for

    px = daily.select("ts", "symbol", "adj_vwap", "amount", "close").with_columns(
        adv=pl.col("amount").rolling_mean(20, min_samples=5).over("symbol"),
        sigma=pl.lit(0.02),
    )
    # ALIGNMENT. The label was built with entry_lag=1, so pred[t] forecasts the
    # return from t+1 to t+2. simulate_cross_sectional marks weight[t] against
    # t -> t+1. Feeding pred[t] in as weight[t] therefore trades every signal
    # one period early; in a reversal-dominated market that flips the sign and
    # manufactures a large, stable NEGATIVE Sharpe that looks like a finding.
    # Advance the signal to the slot it is actually executable in.
    slots = sorted(preds["ts"].unique().to_list())
    nxt = dict(zip(slots[:-1], slots[1:]))
    preds_exec = (
        preds.with_columns(
            # replace_strict with Python datetimes yields microsecond
            # precision; the panel is nanosecond, and polars refuses to join
            # mismatched temporal dtypes rather than coercing silently.
            pl.col("ts")
            .replace_strict(nxt, default=None)
            .cast(pl.Datetime("ns"))
            .alias("_exec")
        )
        .drop_nulls("_exec")
        .select(pl.col("_exec").alias("ts"), "symbol", score="pred")
    )
    scored = preds_exec.join(px, on=["ts", "symbol"], how="inner").drop_nulls("adj_vwap")

    constraints = Constraints(
        max_weight=0.05, min_weight=-0.05, max_industry=None,
        gross_exposure=1.0, net_exposure=0.0,
    )
    weights = score_to_weight(scored, "score", method="rank", constraints=constraints)

    last_day = daily["ts"].max()
    model = CostModel(stamp_duty=stamp_duty_for(last_day.date()))  # type: ignore[union-attr]
    print(f"stamp duty in effect: {model.stamp_duty:.4%} (sell side)")

    res = simulate_cross_sectional(
        weights,
        scored.select("ts", "symbol", vwap="adj_vwap", adv="adv", sigma="sigma"),
        capital=1e8,
        costs=model,
        price_col="vwap",
    )
    print(f"\n{res}")

    from assay.time_series import performance

    gross = simulate_cross_sectional(
        weights,
        scored.select("ts", "symbol", vwap="adj_vwap", adv="adv", sigma="sigma"),
        capital=1e8,
        costs=None,
        price_col="vwap",
    )
    p_net = performance(res.pnl, column="net_return", periods_per_year=244)
    p_gross = performance(gross.pnl, column="net_return", periods_per_year=244)
    print(f"gross : {p_gross}")
    print(f"net   : {p_net}")
    print(f"cost drag: {res.cost_drag_bps:.2f} bp per rebalance")
    res.pnl.write_csv(OUT / "pnl.csv")

    # ---------------------------------------------------------------- herald
    rule("10. herald -- tearsheets")
    try:
        from herald.report import Caption, factor_tearsheet, strategy_tearsheet

        cap = Caption(
            universe="top-80 A-share by market cap (SELECTION BIASED), ex-suspended/limit",
            sample_start=daily["ts"].min(),  # type: ignore[arg-type]
            sample_end=daily["ts"].max(),  # type: ignore[arg-type]
            n_names=daily["symbol"].n_unique(),
            n_periods=daily["ts"].n_unique(),
            note="REAL DATA -- selection-biased universe, ~2.5y sample",
        )
        raw_ic = ic(panel, best, "y1", min_names=20)
        factor_tearsheet(
            best, cap,
            ic=ic_summary(raw_ic.series, newey_west_lags=0),
            quantiles=quantile_returns(panel, best, "y1", n=5),
            decay=d, turnover=tv, coverage=raw_ic.series,
            out_path=OUT / f"factor_{best}.html",
        )
        strategy_tearsheet(
            "composite long-short", res.pnl, cap,
            costs=res.trades, out_path=OUT / "strategy.html",
        )
        print(f"wrote {OUT / f'factor_{best}.html'}")
        print(f"wrote {OUT / 'strategy.html'}")
    except ImportError as exc:
        print(f"matplotlib unavailable ({exc}); skipping tearsheets")

    conn.close()
    rule("done")
    print(f"outputs in {OUT}")


def _equal_weight_composite(panel: pl.DataFrame, feats: list[str]) -> pl.DataFrame:
    """Fallback signal when LightGBM is unavailable.

    Equal-weighted mean of cross-sectionally z-scored factors. Deliberately
    simple and **in-sample** -- it is a smoke test for the plumbing below it,
    not a model, and must not be reported as an out-of-sample result.
    """
    z = []
    for f in feats:
        mu = pl.col(f).mean().over("ts")
        sd = pl.col(f).std().over("ts")
        z.append(pl.when(sd > 0).then((pl.col(f) - mu) / sd).otherwise(0.0))
    return panel.with_columns(pred=pl.sum_horizontal(z) / len(feats)).select(
        "ts", "symbol", "y1", "pred"
    )


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        main()
