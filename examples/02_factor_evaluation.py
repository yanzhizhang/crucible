"""Step 2 -- label construction and single-factor evaluation.

Shows IC recovering the planted signal ladder, quantile monotonicity, decay and
turnover.

Run: ``uv run python examples/02_factor_evaluation.py``
"""

from __future__ import annotations

from _bootstrap import open_store, quiet_close_to_close, rule
from almanac.masks import apply_masks, build_masks
from assay.cross_sectional import decay, ic, quantile_returns, turnover
from horizon.labels import forward_return
from quarry.loaders import load_daily, load_factor_frame
from quarry.synth import PLANTED_FACTORS

conn, truth = open_store()

daily = load_daily(conn)
masks = build_masks(daily)
daily = apply_masks(daily, masks, columns=["close", "vwap", "volume"])

# The fixture planted each factor against the t -> t+1 close return, so the
# diagnostic label matches that. entry_lag=0 is NOT tradable -- it is used here
# only to validate the estimator against known truth.
with quiet_close_to_close():
    labelled = daily
    for h in (1, 2, 3, 5):
        labelled = forward_return(labelled, n=h, price="close", entry_lag=0, label=f"y{h}")

panel = labelled.join(load_factor_frame(conn), on=["ts", "symbol"], how="inner")

rule("IC vs planted signal strength")
print(f"{'factor':<16}{'planted':>9}{'measured':>10}{'ICIR':>8}{'t':>7}{'pos%':>7}{'periods':>9}")
for name, planted in PLANTED_FACTORS.items():
    r = ic(panel, name, "y1", min_names=20)
    print(
        f"{name:<16}{planted:>9.2f}{r.mean:>10.4f}{r.icir:>8.2f}"
        f"{r.t_stat:>7.2f}{r.positive_rate:>7.1%}{r.n_periods:>9}"
    )
print("\nMeasured IC is ordered as planted. One month is a thin sample and the")
print("results carry is_thin=True to say so.")

rule("Quantile returns")
q = quantile_returns(panel, "alpha_strong", "y1", n=5)
print(q)
print(q.by_bucket)

noise = quantile_returns(panel, "alpha_none", "y1", n=5)
print(f"\nmonotonicity  alpha_strong={q.monotonicity:+.2f}  alpha_none={noise.monotonicity:+.2f}")

rule("Decay")
d = decay(panel, "alpha_strong", {1: "y1", 2: "y2", 3: "y3", 5: "y5"})
print(d)
print(d.curve)
print("\nt-stats are Newey-West corrected per horizon; longer labels overlap more")
print("and would otherwise look more significant than short ones.")

rule("Turnover")
t = turnover(panel, "alpha_strong")
print(t)
print(f"implied holding {t.implied_holding_periods:.1f} periods vs half-life {d.half_life:.1f}")

conn.close()
