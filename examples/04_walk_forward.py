"""Step 4 -- purged walk-forward, the leakage canary, and the noise benchmark.

Run: ``uv run python examples/04_walk_forward.py``
"""

from __future__ import annotations

import numpy as np
import polars as pl

from _bootstrap import open_store, quiet_close_to_close, rule
from assay.time_series import sharpe
from crucible.errors import LeakageError
from horizon.labels import forward_return
from kiln.models import assert_no_feature_leakage, noise_benchmark, walk_forward
from kiln.splits import PurgedWalkForward, assert_no_leakage
from quarry.loaders import load_daily, load_factor_frame

conn, truth = open_store()

daily = load_daily(conn)
with quiet_close_to_close():
    labelled = forward_return(daily, n=1, price="close", entry_lag=0, label="y")
panel = labelled.join(load_factor_frame(conn), on=["ts", "symbol"], how="inner").drop_nulls("y")

FEATURES = ["alpha_strong", "alpha_mid", "alpha_weak", "alpha_none"]

rule("Purged walk-forward splits")
cv = PurgedWalkForward(train_span=8, test_span=3, embargo=1, label_horizon=1)
stamps = panel["ts"].to_numpy()
for i, (tr, te) in enumerate(cv.split(stamps)):
    assert_no_leakage(stamps, tr, te, label_horizon=1, embargo=1)
    print(f"  fold {i}: train={len(tr):>5} rows, test={len(te):>4} rows  [leakage assertion passed]")

rule("Leakage canary")
poisoned = panel.with_columns(cheat=pl.col("y") * -3.0 + 0.01)
try:
    assert_no_feature_leakage(poisoned, ["cheat"], "y")
except LeakageError as exc:
    print(f"caught as intended:\n  {exc}")

assert_no_feature_leakage(panel, FEATURES, "y")
print("\nhonest features pass the canary")

rule("Walk-forward fit")
res = walk_forward(panel, FEATURES, "y", cv, num_boost_round=50)
print(res)
print(res.fold_scores)

rule("Noise benchmark")
index = conn.execute("SELECT ts, ret FROM index ORDER BY ts").pl()
returns = index["ret"].to_numpy()


def strategy_score(series: np.ndarray) -> float:
    """A trend follower -- deliberately one that lives off autocorrelation."""
    d = np.diff(series, prepend=series[0])
    position = np.sign(np.concatenate([[0.0], d[:-1]]))
    return sharpe(position * d, periods_per_year=244)


bench = noise_benchmark(returns, strategy_score, n_trials=100)
print(bench)
print("\nPhase-randomized surrogates keep the same power spectrum -- and therefore")
print("the same autocorrelation -- so a strategy that only exploits that structure")
print("cannot beat them. Failing this test is the correct outcome for such a rule.")

conn.close()
