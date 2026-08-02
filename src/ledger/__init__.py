"""ledger -- position accounting and the C++ parity gate.

A PnL accountant, deliberately not a market simulator. Fills come from the C++
backtester or from one documented assumption; this package does the
bookkeeping. That single-sourcing is what lets :func:`parity_check` attribute
any discrepancy to weights, costs or accounting rather than to a fill-model
difference.
"""

from __future__ import annotations

from ledger.accounting import (
    BacktestResult,
    ParityReport,
    TimeSeriesResult,
    parity_check,
    simulate_cross_sectional,
    simulate_time_series,
)

__all__ = [
    "BacktestResult",
    "ParityReport",
    "TimeSeriesResult",
    "parity_check",
    "simulate_cross_sectional",
    "simulate_time_series",
]
