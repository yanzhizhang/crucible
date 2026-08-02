"""Step 6 -- factor and strategy tearsheets.

Writes self-contained HTML into ``examples/_out/``.

Run: ``uv run python examples/06_tearsheets.py``
"""

from __future__ import annotations

import polars as pl

from _bootstrap import OUT, open_store, quiet_close_to_close, rule
from almanac.masks import apply_masks, build_masks
from assay.cross_sectional import decay, ic, ic_summary, quantile_returns, turnover
from ballast.portfolio import Constraints, score_to_weight
from herald.report import Caption, factor_tearsheet, strategy_tearsheet
from horizon.labels import forward_return
from ledger.accounting import simulate_cross_sectional
from quarry.loaders import load_daily, load_factor_frame
from toll.costs import CostModel

conn, truth = open_store()

daily = load_daily(conn)
masks = build_masks(daily)
daily = apply_masks(daily, masks, columns=["close", "vwap", "volume"])

with quiet_close_to_close():
    labelled = daily
    for h in (1, 2, 3, 5):
        labelled = forward_return(labelled, n=h, price="close", entry_lag=0, label=f"y{h}")

panel = labelled.join(load_factor_frame(conn), on=["ts", "symbol"], how="inner")

caption = Caption(
    universe="synthetic A-share, 60 names, ex-suspended and ex-limit",
    sample_start=truth.dates[0],
    sample_end=truth.dates[-1],
    n_names=len(truth.symbols),
    n_periods=len(truth.dates),
    frequency="daily",
    note="SYNTHETIC DATA -- not a research result",
)

rule("Factor tearsheet")
raw_ic = ic(panel, "alpha_strong", "y1", min_names=20)
html = factor_tearsheet(
    "alpha_strong",
    caption,
    # Horizon 1 with entry_lag 0 does not overlap, so no NW correction is needed.
    ic=ic_summary(raw_ic.series, method=raw_ic.method, newey_west_lags=0),
    quantiles=quantile_returns(panel, "alpha_strong", "y1", n=5),
    decay=decay(panel, "alpha_strong", {1: "y1", 2: "y2", 3: "y3", 5: "y5"}),
    turnover=turnover(panel, "alpha_strong"),
    coverage=raw_ic.series,
    out_path=OUT / "factor_alpha_strong.html",
)
print(f"wrote {OUT / 'factor_alpha_strong.html'}  ({len(html):,} bytes, self-contained)")

rule("Strategy tearsheet")
scored = panel.with_columns(score=pl.col("alpha_strong"), adv=pl.col("volume").mean().over("symbol"), sigma=pl.lit(0.02))
weights = score_to_weight(
    scored, "score", method="rank", constraints=Constraints(max_weight=0.03, min_weight=-0.03)
)
res = simulate_cross_sectional(
    weights,
    scored.select("ts", "symbol", "vwap", "adv", "sigma"),
    capital=1e8,
    costs=CostModel(),
)
positions = weights.join(daily.select("ts", "symbol", "industry"), on=["ts", "symbol"], how="left")

html = strategy_tearsheet(
    "alpha_strong long-short",
    res.pnl,
    caption,
    positions=positions,
    costs=res.trades,
    out_path=OUT / "strategy_alpha_strong.html",
)
print(f"wrote {OUT / 'strategy_alpha_strong.html'}  ({len(html):,} bytes)")
print("\nEvery chart caption carries the sample period and universe definition.")

conn.close()
