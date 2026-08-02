"""Step 3 -- transforms and factor-pool admission.

Shows a declared forge pipeline and a duplicate factor being rejected.

Run: ``uv run python examples/03_screen_pool.py``
"""

from __future__ import annotations

import polars as pl

from _bootstrap import open_store, rule
from forge.ops import apply_pipeline
from quarry.loaders import load_daily, load_factor_frame
from sieve.screen import FactorPool, correlation_matrix

conn, truth = open_store()

daily = load_daily(conn)
factors = load_factor_frame(conn).join(
    daily.select("ts", "symbol", "industry", "market_cap"), on=["ts", "symbol"], how="inner"
)

rule("Declared transform pipeline")
cleaned = apply_pipeline(
    factors,
    [
        {"op": "winsorize", "column": "alpha_strong", "method": "mad", "k": 3.0},
        {"op": "neutralize", "column": "alpha_strong", "by": ["industry"], "size_col": "market_cap"},
        {"op": "zscore", "column": "alpha_strong"},
    ],
)
check = cleaned.group_by("ts").agg(
    pl.col("alpha_strong").mean().alias("mean"), pl.col("alpha_strong").std().alias("std")
)
print("after winsorize -> neutralize -> zscore, each cross-section is standardised:")
print(check.head(3))
print("\nThese are research-only transforms. If one must run live, it belongs in prism.")

rule("Cross-sectional correlation, averaged over time")
names = ["alpha_strong", "alpha_mid", "alpha_weak", "alpha_none", "dupe_of_strong"]
print(correlation_matrix(factors, names))

rule("Pool admission")
pool = FactorPool(threshold=0.7, reject_above=0.95)
for candidate in names:
    res, factors = pool.propose(factors, candidate)
    print(f"  {res}")

print(f"\nadmitted: {pool.factors}")
print("\nAdmission log (rejections recorded too):")
print(pool.admission_log().select("factor", "verdict", "max_abs_corr", "most_correlated"))

conn.close()
