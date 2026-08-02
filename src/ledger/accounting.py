"""Position accounting and the C++ parity gate.

This is a **PnL accountant, not a market simulator**, and the distinction is
the whole design.

Order matching, queue position, partial fills and latency belong to the C++
backtester, because that is the live path. Reimplementing them here would give
two fill models that disagree, and the parity gate would then be forever
"explained" by slippage assumptions instead of catching real bugs. crucible
takes fill prices as **given** -- either replayed from the C++ fill stream, or
one documented naive assumption -- and does the bookkeeping.

That single-sourcing is what makes :func:`parity_check` meaningful. With fills
held constant across both sides, any PnL gap is necessarily a bug in weights,
costs, or position accounting.

Two entry points, split by signal type, because the two fail differently:

* :func:`simulate_cross_sectional` -- rank many names, hold a basket. PnL is
  driven by the return *spread between names*; fills are a cost term. Accurate.
* :func:`simulate_time_series` -- one instrument, entry/exit timing. PnL *is*
  the fill. This function reports how much of the result rests on the assumed
  fill price, and says so loudly when that share is large.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

from crucible.errors import ParityError
from crucible.frames import SYMBOL, TS, Frame, require_columns, to_polars
from toll.costs import CostModel

__all__ = [
    "BacktestResult",
    "TimeSeriesResult",
    "ParityReport",
    "simulate_cross_sectional",
    "simulate_time_series",
    "parity_check",
]


@dataclass(frozen=True)
class BacktestResult:
    """Per-period PnL accounting for a cross-sectional book."""

    pnl: pl.DataFrame
    trades: pl.DataFrame = field(repr=False, default_factory=pl.DataFrame)
    capital: float = 0.0

    @property
    def net_returns(self) -> pl.Series:
        """Net return per period, for :func:`assay.performance`."""
        return self.pnl["net_return"]

    @property
    def total_cost(self) -> float:
        """Total cost paid across the sample, in CNY."""
        return float(self.pnl["cost"].sum()) if self.pnl.height else 0.0

    @property
    def cost_drag_bps(self) -> float:
        """Average per-period cost as basis points of capital.

        The number to compare against the gross edge. A strategy earning 4bp
        per rebalance and paying 6bp is not a marginal strategy, it is a
        negative one, and this makes that immediately legible.
        """
        if not self.pnl.height or self.capital <= 0:
            return 0.0
        return float(self.pnl["cost"].mean() / self.capital * 1e4)

    def __repr__(self) -> str:
        n = self.pnl.height
        tot = float(self.pnl["net_return"].sum()) if n else 0.0
        return (
            f"BacktestResult({n} periods, net={tot:+.2%}, "
            f"cost drag={self.cost_drag_bps:.2f}bp/period)"
        )


@dataclass(frozen=True)
class TimeSeriesResult:
    """Single-instrument PnL, with an explicit fill-assumption diagnostic."""

    pnl: pl.DataFrame
    fill_sensitivity: float
    n_trades: int
    assumed_fill: str

    @property
    def is_fill_dominated(self) -> bool:
        """Whether the assumed fill price drives most of the result.

        True when moving the fill by one half-spread changes PnL by more than
        half its magnitude. At that point the number belongs to the C++
        backtester, not to this one -- Python cannot model queue position.
        """
        return self.fill_sensitivity > 0.5

    def __repr__(self) -> str:
        warn = "  [FILL-DOMINATED -- defer to C++]" if self.is_fill_dominated else ""
        total = float(self.pnl["net_pnl"].sum()) if self.pnl.height else 0.0
        return (
            f"TimeSeriesResult(pnl={total:+,.0f} trades={self.n_trades} "
            f"fill_sensitivity={self.fill_sensitivity:.1%} "
            f"assumed={self.assumed_fill}){warn}"
        )


def simulate_cross_sectional(
    weights: Frame,
    prices: Frame,
    *,
    capital: float = 1e8,
    costs: CostModel | None = None,
    weight_col: str = "weight",
    price_col: str = "vwap",
    fill_col: str | None = None,
    adv_col: str = "adv",
    sigma_col: str = "sigma",
) -> BacktestResult:
    """Account a cross-sectional book period by period.

    Parameters
    ----------
    weights:
        ``ts``, ``symbol``, ``weight`` -- target weights decided at ``ts``.
    prices:
        ``ts``, ``symbol``, and the price columns. Must cover every
        ``(ts, symbol)`` in ``weights``.
    fill_col:
        Column holding **actual fill prices**, normally replayed from the C++
        backtester. When given, it is used instead of ``price_col``, and that
        is the configuration :func:`parity_check` expects -- with fills held
        constant, any residual gap is an accounting bug rather than a fill-model
        difference.
    costs:
        Cost model. When omitted, costs are zero and the result is a *gross*
        PnL, which for any realistic turnover is not a strategy result.

    Returns
    -------
    BacktestResult
        ``pnl`` has one row per period: ``gross_pnl``, ``cost``, ``net_pnl``,
        ``net_return``, ``turnover``, ``gross_exposure``, ``equity``.

    Point-in-time contract
    ----------------------
    Weights stamped ``t`` are executed at ``t``'s fill price and earn the return
    from ``t`` to ``t+1``. A weight must therefore already incorporate any
    decision lag; this function applies none of its own.
    """
    w = to_polars(weights)
    px = to_polars(prices)
    require_columns(w, (TS, SYMBOL, weight_col), where="simulate_cross_sectional")
    require_columns(px, (TS, SYMBOL, price_col), where="simulate_cross_sectional")

    fill = fill_col if fill_col and fill_col in px.columns else price_col
    keep = [c for c in {price_col, fill, adv_col, sigma_col} if c in px.columns]

    panel = (
        w.select(TS, SYMBOL, weight_col)
        .join(px.select(TS, SYMBOL, *keep), on=[TS, SYMBOL], how="left")
        .sort([TS, SYMBOL], maintain_order=True)
    )

    # `costs or CostModel()` would make costs=None unreachable -- None is
    # falsy, so "no costs" silently became "default costs" and the documented
    # gross-PnL path never ran.
    model = costs
    stamps = panel[TS].unique().sort().to_list()

    prev_shares: dict[str, float] = {}
    rows: list[dict[str, object]] = []
    trade_blocks: list[pl.DataFrame] = []
    equity = capital

    for i, stamp in enumerate(stamps):
        cur = panel.filter(pl.col(TS) == stamp)
        syms = cur[SYMBOL].to_list()
        wt = np.nan_to_num(cur[weight_col].to_numpy().astype(float))
        fill_px = np.nan_to_num(cur[fill].to_numpy().astype(float))

        with np.errstate(divide="ignore", invalid="ignore"):
            target = np.where(fill_px > 0, wt * equity / fill_px, 0.0)
        prev = np.array([prev_shares.get(s, 0.0) for s in syms])
        delta = target - prev

        trade = cur.select(TS, SYMBOL).with_columns(
            shares=pl.Series("shares", delta),
            price=pl.Series("price", fill_px),
        )
        for extra in (adv_col, sigma_col):
            if extra in cur.columns:
                trade = trade.with_columns(cur[extra].alias(extra))

        if model is None:
            costed = trade.with_columns(
                shares_filled=pl.col("shares"), cost_total=pl.lit(0.0)
            )
            period_cost = 0.0
        else:
            costed = to_polars(model.apply(trade, adv_col=adv_col, sigma_col=sigma_col))
            period_cost = float(costed["cost_total"].sum())
        filled = costed["shares_filled"].to_numpy().astype(float)
        held = prev + filled
        trade_blocks.append(costed)

        # Mark to the next period's price to realise the holding return.
        gross = 0.0
        if i + 1 < len(stamps):
            nxt = panel.filter(pl.col(TS) == stamps[i + 1]).select(SYMBOL, _next=fill)
            nxt_map = dict(zip(nxt[SYMBOL].to_list(), nxt["_next"].to_list()))
            future = np.array([nxt_map.get(s, np.nan) for s in syms], dtype=float)
            move = np.nan_to_num(future - fill_px)
            gross = float((held * move).sum())

        net = gross - period_cost
        ret = net / equity if equity > 0 else 0.0
        equity += net

        rows.append(
            {
                TS: stamp,
                "gross_pnl": gross,
                "cost": period_cost,
                "net_pnl": net,
                "net_return": ret,
                "turnover": float(np.abs(filled * fill_px).sum() / max(equity, 1e-9)),
                "gross_exposure": float(np.abs(held * fill_px).sum() / max(equity, 1e-9)),
                "net_exposure": float((held * fill_px).sum() / max(equity, 1e-9)),
                "equity": equity,
                "n_names": int((np.abs(held) > 0).sum()),
            }
        )
        prev_shares = dict(zip(syms, held))

    return BacktestResult(
        pnl=pl.DataFrame(rows) if rows else pl.DataFrame(),
        trades=pl.concat(trade_blocks) if trade_blocks else pl.DataFrame(),
        capital=capital,
    )


def simulate_time_series(
    signal: Frame,
    *,
    signal_col: str = "signal",
    price_col: str = "close",
    fill_col: str | None = None,
    spread_col: str | None = "spread",
    contract_size: float = 1.0,
    costs: CostModel | None = None,
    position_scale: float = 1.0,
) -> TimeSeriesResult:
    """Account a single-instrument position stream, honestly.

    Parameters
    ----------
    signal:
        ``ts`` plus a target position in ``signal_col`` (in contracts or
        shares, before ``position_scale``).
    fill_col:
        Actual fill prices when available. Otherwise the next bar's
        ``price_col`` is assumed, which is the naive assumption this function
        is designed to make visible rather than hide.
    spread_col:
        Bid-ask spread per bar. Used to compute :attr:`fill_sensitivity` -- how
        much the PnL moves if every fill were one half-spread worse.

    Returns
    -------
    TimeSeriesResult
        Carrying ``fill_sensitivity``. **Read it before reading the PnL.** When
        it exceeds 0.5 the result is an artefact of the fill assumption, and
        the only trustworthy number comes from the C++ backtester with a real
        book.

    Notes
    -----
    No queue model, no partial fills, no latency. Positions change at the
    assumed price, in full. That is adequate for a daily or hourly signal and
    misleading for anything approaching tick frequency -- which is exactly what
    ``fill_sensitivity`` is there to tell you.
    """
    lf = to_polars(signal).sort(TS, maintain_order=True)
    require_columns(lf, (TS, signal_col, price_col), where="simulate_time_series")

    pos = np.nan_to_num(lf[signal_col].to_numpy().astype(float)) * position_scale
    px = np.nan_to_num(lf[price_col].to_numpy().astype(float))
    fill = (
        np.nan_to_num(lf[fill_col].to_numpy().astype(float))
        if fill_col and fill_col in lf.columns
        else np.concatenate([px[1:], px[-1:]])
    )
    assumed = fill_col if fill_col and fill_col in lf.columns else f"next-bar {price_col}"

    delta = np.diff(pos, prepend=0.0)
    move = np.concatenate([np.diff(px), [0.0]])
    gross = pos * move * contract_size

    model = costs or CostModel()
    notional = np.abs(delta) * fill * contract_size
    comm = np.where(notional > 0, np.maximum(notional * model.commission, model.min_commission), 0.0)
    stamp = np.where(delta < 0, notional * model.stamp_duty, 0.0)
    cost = comm + stamp

    net = gross - cost

    if spread_col and spread_col in lf.columns:
        half = np.nan_to_num(lf[spread_col].to_numpy().astype(float)) / 2.0
    else:
        half = np.zeros_like(px)
    penalty = float((np.abs(delta) * half * contract_size).sum())
    total = float(np.abs(net.sum()))
    sensitivity = penalty / total if total > 0 else (1.0 if penalty > 0 else 0.0)

    pnl = lf.select(TS).with_columns(
        position=pl.Series("position", pos),
        gross_pnl=pl.Series("gross_pnl", gross),
        cost=pl.Series("cost", cost),
        net_pnl=pl.Series("net_pnl", net),
        equity=pl.Series("equity", np.cumsum(net)),
    )
    return TimeSeriesResult(
        pnl=pnl,
        fill_sensitivity=float(min(sensitivity, 1.0)),
        n_trades=int((np.abs(delta) > 0).sum()),
        assumed_fill=assumed,
    )


@dataclass(frozen=True)
class ParityReport:
    """Result of reconciling Python PnL against the C++ backtester."""

    n_periods: int
    max_abs_diff: float
    max_rel_diff: float
    total_python: float
    total_cpp: float
    worst_period: object | None
    tolerance: float
    passed: bool
    diffs: pl.DataFrame = field(repr=False, default_factory=pl.DataFrame)

    def __repr__(self) -> str:
        verdict = "PASS" if self.passed else "FAIL"
        return (
            f"ParityReport({verdict}: max|diff|={self.max_abs_diff:.6g} "
            f"rel={self.max_rel_diff:.2e} tol={self.tolerance:.2e} "
            f"over {self.n_periods} periods)"
        )


def parity_check(
    python_pnl: Frame,
    cpp_pnl: Frame,
    *,
    column: str = "net_pnl",
    tolerance: float = 1e-6,
    relative: bool = True,
    raise_on_fail: bool = True,
) -> ParityReport:
    """Reconcile Python PnL against the C++ backtester on the same signal.

    Parameters
    ----------
    tolerance:
        Maximum acceptable difference. With fills replayed from C++ the two
        sides should agree to floating-point noise, so the default is tight on
        purpose. Loosening it to make a failure go away converts a detectable
        bug into a permanent unexplained residual.
    relative:
        Compare relative to the C++ magnitude rather than in absolute CNY.

    Returns
    -------
    ParityReport
        Naming the worst period, which is where to start debugging.

    Raises
    ------
    ParityError
        When the mismatch exceeds ``tolerance`` and ``raise_on_fail`` is set.
        A mismatch is a build failure, not a rounding note.
    """
    a = to_polars(python_pnl).select(TS, pl.col(column).alias("_py"))
    b = to_polars(cpp_pnl).select(TS, pl.col(column).alias("_cpp"))

    joined = a.join(b, on=TS, how="inner").sort(TS)
    if joined.height == 0:
        raise ParityError(
            "python and C++ PnL share no timestamps. Reconciliation compared nothing, "
            "which is not a pass -- check that both sides ran the same signal stream "
            "on the same calendar."
        )

    missing = max(a.height, b.height) - joined.height
    py = joined["_py"].to_numpy().astype(float)
    cpp = joined["_cpp"].to_numpy().astype(float)
    abs_diff = np.abs(py - cpp)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_diff = np.where(np.abs(cpp) > 0, abs_diff / np.abs(cpp), abs_diff)

    measure = rel_diff if relative else abs_diff
    worst_i = int(np.argmax(measure)) if measure.size else 0
    passed = bool(measure.max() <= tolerance) and missing == 0

    diffs = joined.with_columns(
        abs_diff=pl.Series("abs_diff", abs_diff),
        rel_diff=pl.Series("rel_diff", rel_diff),
    )
    report = ParityReport(
        n_periods=joined.height,
        max_abs_diff=float(abs_diff.max()),
        max_rel_diff=float(rel_diff.max()),
        total_python=float(py.sum()),
        total_cpp=float(cpp.sum()),
        worst_period=joined[TS][worst_i],
        tolerance=tolerance,
        passed=passed,
        diffs=diffs,
    )

    if not passed and raise_on_fail:
        extra = (
            f" {missing} period(s) present on one side only."
            if missing
            else ""
        )
        raise ParityError(
            f"PnL parity FAILED against the C++ backtester. "
            f"max abs diff={report.max_abs_diff:.6g}, max rel diff={report.max_rel_diff:.3e} "
            f"(tolerance {tolerance:.1e}), worst at {report.worst_period}. "
            f"Python total={report.total_python:.2f} vs C++ {report.total_cpp:.2f}.{extra} "
            f"With fills replayed from C++ this can only be a weights, cost or "
            f"position-accounting bug -- do not widen the tolerance."
        )
    return report
