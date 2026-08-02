"""Step 5 -- weights, constraints, costs, and the C++ parity gate.

Run: ``uv run python examples/05_portfolio_costs.py``
"""

from __future__ import annotations

import polars as pl

from _bootstrap import open_store, rule
from ballast.portfolio import Constraints, market_neutral, score_to_weight
from crucible.errors import ParityError
from ledger.accounting import parity_check, simulate_cross_sectional
from quarry.loaders import load_daily, load_factor_frame
from toll.costs import CostModel, stamp_duty_for

conn, truth = open_store()

daily = load_daily(conn)
factors = load_factor_frame(conn, factors=["alpha_strong"])
panel = factors.join(
    daily.select("ts", "symbol", "vwap", "close", "volume", "industry"),
    on=["ts", "symbol"],
    how="inner",
).with_columns(
    score=pl.col("alpha_strong"),
    adv=pl.col("volume").mean().over("symbol"),
    sigma=pl.lit(0.02),
)

rule("Score -> weights, with constraints")
constraints = Constraints(
    max_weight=0.03, min_weight=-0.03, max_industry=0.15, gross_exposure=1.0, net_exposure=0.0
)
weights = score_to_weight(panel, "score", method="rank", constraints=constraints)
summary = weights.group_by("ts").agg(
    pl.col("weight").sum().alias("net"),
    pl.col("weight").abs().sum().alias("gross"),
    pl.col("weight").abs().max().alias("max_abs"),
)
print(summary.head(3))
print(f"\nname cap respected: max |w| = {weights['weight'].abs().max():.4f} <= 0.03")

rule("Costs")
model = CostModel(stamp_duty=stamp_duty_for("2024-01-15"))
print(f"stamp duty in effect 2024-01-15: {model.stamp_duty:.4%} (sell side only)")
deadband = model.deadband_threshold(20.0, 10_000, adv=1e6, sigma=0.02)
print(f"analytic round-trip deadband   : {deadband:.4%}")
print("Derived from tax + commission + two-way impact, not a constant.")

rule("Position accounting")
prices = panel.select("ts", "symbol", "vwap", "adv", "sigma")
res = simulate_cross_sectional(weights, prices, capital=1e8, costs=model)
print(res)
print(res.pnl.select("ts", "gross_pnl", "cost", "net_pnl", "turnover", "equity").head(5))

rule("Market-neutral hedge")
last = weights.filter(pl.col("ts") == weights["ts"].max())
hedge = market_neutral(last, capital=1e8, index_price=6000.0, hedge="IM", beta_col=None)
print(hedge)
print("Contracts are integral; the residual is reported rather than hidden.")

rule("Parity gate against the C++ backtester")
print("Replay C++ fills into the accountant, then diff. Fills held constant means")
print("any gap is a weights/cost/accounting bug, never a fill-model difference.\n")

report = parity_check(res.pnl, res.pnl, tolerance=1e-9)
print(f"identical streams  -> {report}")

drifted = res.pnl.with_columns(
    net_pnl=pl.when(pl.int_range(pl.len()) == 4)
    .then(pl.col("net_pnl") * 1.02)
    .otherwise(pl.col("net_pnl"))
)
try:
    parity_check(res.pnl, drifted, tolerance=1e-6)
except ParityError as exc:
    print(f"\nperturbed stream   -> raised as designed:\n  {exc}")

conn.close()
