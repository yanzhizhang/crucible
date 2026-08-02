"""Factor definitions -- a **stand-in for prism**, not part of the crucible core.

Read this before anything else in the file
------------------------------------------
crucible's second invariant is that Python never re-implements a production
factor. Nothing here contradicts that, because **nothing here is production**.
This module plays the role prism plays in the real system: it is a *producer*
that emits a ``factor_frame`` dump, which the crucible pipeline then consumes
through exactly the same loader, fingerprint check and masking path it would
use against a genuine C++ dump.

That is why these live in ``research/`` and not in ``forge/``. ``forge`` holds
research-only *transforms* (winsorize, neutralize, rank). This file holds
*factor definitions*, and any of them that survives screening must be ported to
prism in C++ before it can trade. A factor that scores well here and is then
run live from Python is precisely the live/research divergence the invariant
exists to prevent.

Factor families
---------------
``trad``
    Classical cross-sectional equity factors, chosen for A-share relevance.
    Note that A-shares are famously *reversal*-dominated at short horizons
    rather than momentum-dominated, unlike US equities.
``a101``
    WorldQuant "101 Formulaic Alphas" (Kakushadze 2015), restricted to those
    computable from daily OHLCV.
``a191``
    Guotai Junan "191 Alphas" -- the standard formulaic set for Chinese
    equities.
``q158``
    qlib Alpha158-style rolling price/volume features.

All factors are computed **causally**: every window ends at ``t`` inclusive and
no expression reads a future row. That is asserted structurally by the tests,
not just intended.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl

__all__ = ["FACTORS", "FAMILIES", "compute_factors", "factor_specs"]

_S = "symbol"

# --------------------------------------------------------------------------
# operator helpers -- all trailing, all within-symbol
# --------------------------------------------------------------------------


def _d(col: str, n: int) -> pl.Expr:
    """Change over ``n`` periods: ``x[t] - x[t-n]``."""
    return pl.col(col) - pl.col(col).shift(n).over(_S)


def _ret(n: int = 1) -> pl.Expr:
    """Simple return over ``n`` periods on the adjusted close."""
    return pl.col("adj_close") / pl.col("adj_close").shift(n).over(_S) - 1.0


def _mean(e: pl.Expr, n: int) -> pl.Expr:
    return e.rolling_mean(window_size=n, min_samples=n).over(_S)


def _std(e: pl.Expr, n: int) -> pl.Expr:
    return e.rolling_std(window_size=n, min_samples=n).over(_S)


def _sum(e: pl.Expr, n: int) -> pl.Expr:
    return e.rolling_sum(window_size=n, min_samples=n).over(_S)


def _max(e: pl.Expr, n: int) -> pl.Expr:
    return e.rolling_max(window_size=n, min_samples=n).over(_S)


def _min(e: pl.Expr, n: int) -> pl.Expr:
    return e.rolling_min(window_size=n, min_samples=n).over(_S)


def _corr(a: pl.Expr, b: pl.Expr, n: int) -> pl.Expr:
    return pl.rolling_corr(a, b, window_size=n, min_samples=n).over(_S)


def _cs_rank(e: pl.Expr) -> pl.Expr:
    """Cross-sectional rank scaled to [0, 1] within each timestamp."""
    return (e.rank("average").over("ts") - 1) / (pl.len().over("ts") - 1).clip(1, None)


def _ts_rank(e: pl.Expr, n: int) -> pl.Expr:
    """Fraction of the trailing ``n`` window the current value exceeds.

    Implemented via ``rolling_map`` because polars has no native windowed rank.
    Slow, but the alternative -- dropping the operator -- would mean silently
    not implementing the alphas that use it.
    """
    return e.rolling_map(
        lambda s: float((s[-1] > s.to_numpy()).mean()), window_size=n, min_samples=n
    ).over(_S)


def _sign(e: pl.Expr) -> pl.Expr:
    return pl.when(e > 0).then(1.0).when(e < 0).then(-1.0).otherwise(0.0)


# --------------------------------------------------------------------------
# factor definitions
# --------------------------------------------------------------------------

FactorFn = Callable[[], pl.Expr]

#: name -> (family, description, expression builder)
FACTORS: dict[str, tuple[str, str, FactorFn]] = {}


def _reg(name: str, family: str, desc: str) -> Callable[[FactorFn], FactorFn]:
    def deco(fn: FactorFn) -> FactorFn:
        FACTORS[name] = (family, desc, fn)
        return fn

    return deco


# ---- traditional ---------------------------------------------------------


@_reg("rev_5", "trad", "5-day reversal (negated 5-day return)")
def _rev5() -> pl.Expr:
    return -_ret(5)


@_reg("rev_20", "trad", "20-day reversal; the dominant short-horizon A-share effect")
def _rev20() -> pl.Expr:
    return -_ret(20)


@_reg("mom_60_20", "trad", "momentum over t-60..t-20, skipping the recent month")
def _mom() -> pl.Expr:
    c = pl.col("adj_close")
    return c.shift(20).over(_S) / c.shift(60).over(_S) - 1.0


@_reg("vol_20", "trad", "20-day realised volatility of daily returns")
def _vol20() -> pl.Expr:
    return _std(_ret(1), 20)


@_reg("beta_60", "trad", "60-day rolling beta against the equal-weighted market")
def _beta() -> pl.Expr:
    r, m = _ret(1), pl.col("mkt_ret")
    return _corr(r, m, 60) * _std(r, 60) / _std(m, 60)


@_reg("ivol_20", "trad", "idiosyncratic volatility: std of market-residual returns")
def _ivol() -> pl.Expr:
    return _std(_ret(1) - pl.col("mkt_ret"), 20)


@_reg("turn_20", "trad", "mean 20-day turnover; attention / liquidity proxy")
def _turn() -> pl.Expr:
    return _mean(pl.col("turnover"), 20)


@_reg("turn_bias", "trad", "20d turnover relative to 60d; abnormal attention")
def _turnbias() -> pl.Expr:
    return _mean(pl.col("turnover"), 20) / _mean(pl.col("turnover"), 60) - 1.0


@_reg("illiq", "trad", "Amihud illiquidity: mean(|ret| / amount) over 20 days")
def _illiq() -> pl.Expr:
    return _mean(_ret(1).abs() / (pl.col("amount") + 1.0), 20) * 1e9


@_reg("size", "trad", "log market cap; the classic size exposure")
def _size() -> pl.Expr:
    return pl.col("market_cap").log()


@_reg("max_5", "trad", "MAX effect: largest daily return in 5 days (lottery demand)")
def _max5() -> pl.Expr:
    return _max(_ret(1), 5)


@_reg("skew_20", "trad", "20-day return skewness")
def _skew() -> pl.Expr:
    r = _ret(1)
    return _mean((r - _mean(r, 20)) ** 3, 20) / (_std(r, 20) ** 3 + 1e-12)


@_reg("bias_20", "trad", "deviation of price from its own 20-day mean")
def _bias() -> pl.Expr:
    c = pl.col("adj_close")
    return c / _mean(c, 20) - 1.0


@_reg("hl_range_20", "trad", "mean normalised daily high-low range; intraday risk")
def _hl() -> pl.Expr:
    return _mean((pl.col("high") - pl.col("low")) / pl.col("close"), 20)


# ---- WorldQuant Alpha101 -------------------------------------------------


@_reg("a101_001", "a101", "(close-open)/(high-low+0.001) -- Alpha#101, intraday drive")
def _a101_101() -> pl.Expr:
    return (pl.col("close") - pl.col("open")) / (pl.col("high") - pl.col("low") + 0.001)


@_reg("a101_006", "a101", "-corr(open, volume, 10) -- Alpha#6")
def _a101_006() -> pl.Expr:
    return -_corr(pl.col("open"), pl.col("volume"), 10)


@_reg("a101_012", "a101", "sign(dvolume) * -dclose -- Alpha#12")
def _a101_012() -> pl.Expr:
    return _sign(_d("volume", 1)) * (-_d("adj_close", 1))


@_reg("a101_041", "a101", "sqrt(high*low) - vwap -- Alpha#41")
def _a101_041() -> pl.Expr:
    return (pl.col("high") * pl.col("low")).sqrt() - pl.col("vwap")


@_reg("a101_053", "a101", "-delta(((close-low)-(high-close))/(close-low), 9) -- Alpha#53")
def _a101_053() -> pl.Expr:
    inner = ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (
        pl.col("close") - pl.col("low") + 1e-6
    )
    return -(inner - inner.shift(9).over(_S))


@_reg("a101_054", "a101", "-((low-close)*open^5)/((low-high)*close^5) -- Alpha#54")
def _a101_054() -> pl.Expr:
    lo, hi, c, o = pl.col("low"), pl.col("high"), pl.col("close"), pl.col("open")
    return -((lo - c) * o.pow(5)) / ((lo - hi) * c.pow(5) + 1e-12)


@_reg("a101_004", "a101", "-ts_rank(cs_rank(low), 9) -- Alpha#4")
def _a101_004() -> pl.Expr:
    # Reads the pre-computed cross-sectional rank: ts_rank partitions by
    # symbol and cs_rank by ts, and polars silently yields nulls when the two
    # are nested rather than raising.
    return -_ts_rank(pl.col("_csr_low"), 9)


# ---- Guotai Junan Alpha191 ----------------------------------------------


@_reg("a191_001", "a191", "-corr(rank(dlog(volume)), rank((close-open)/open), 6)")
def _a191_001() -> pl.Expr:
    # Reads pre-computed rank columns rather than nesting .over("ts") inside
    # .over("symbol") -- polars cannot compose two different window partitions
    # in one expression, and the attempt yields an all-null column rather than
    # an error. See _RANK_HELPERS.
    return -_corr(pl.col("_csr_dlogv"), pl.col("_csr_retio"), 6)


@_reg("a191_002", "a191", "-delta(((close-low)-(high-close))/(high-low), 1)")
def _a191_002() -> pl.Expr:
    inner = ((pl.col("close") - pl.col("low")) - (pl.col("high") - pl.col("close"))) / (
        pl.col("high") - pl.col("low") + 1e-6
    )
    return -(inner - inner.shift(1).over(_S))


@_reg("a191_014", "a191", "close - close[5]; raw 5-day price change")
def _a191_014() -> pl.Expr:
    return _d("adj_close", 5)


@_reg("a191_018", "a191", "close / close[5]; 5-day price ratio")
def _a191_018() -> pl.Expr:
    return pl.col("adj_close") / pl.col("adj_close").shift(5).over(_S)


@_reg("a191_020", "a191", "6-day rate of change in percent")
def _a191_020() -> pl.Expr:
    c = pl.col("adj_close")
    return (c - c.shift(6).over(_S)) / (c.shift(6).over(_S) + 1e-12) * 100.0


@_reg("a191_053", "a191", "count of up days over 12, as a percentage")
def _a191_053() -> pl.Expr:
    return _sum((_ret(1) > 0).cast(pl.Float64), 12) / 12.0 * 100.0


# ---- qlib Alpha158-style -------------------------------------------------


@_reg("q158_kmid", "q158", "(close-open)/open -- KMID candle body")
def _kmid() -> pl.Expr:
    return (pl.col("close") - pl.col("open")) / pl.col("open")


@_reg("q158_klen", "q158", "(high-low)/open -- KLEN candle length")
def _klen() -> pl.Expr:
    return (pl.col("high") - pl.col("low")) / pl.col("open")


@_reg("q158_kup", "q158", "(high-max(open,close))/open -- upper shadow")
def _kup() -> pl.Expr:
    return (pl.col("high") - pl.max_horizontal("open", "close")) / pl.col("open")


@_reg("q158_rsv_20", "q158", "(close-min_low)/(max_high-min_low) over 20 days")
def _rsv() -> pl.Expr:
    lo, hi = _min(pl.col("low"), 20), _max(pl.col("high"), 20)
    return (pl.col("close") - lo) / (hi - lo + 1e-12)


@_reg("q158_roc_10", "q158", "10-day rate of change")
def _roc10() -> pl.Expr:
    return _ret(10)


@_reg("q158_std_60", "q158", "60-day return volatility")
def _std60() -> pl.Expr:
    return _std(_ret(1), 60)


@_reg("q158_vsty_20", "q158", "volume volatility: 20d std of volume / 20d mean volume")
def _vsty() -> pl.Expr:
    return _std(pl.col("volume"), 20) / (_mean(pl.col("volume"), 20) + 1.0)


@_reg("q158_cntp_20", "q158", "fraction of up days over 20")
def _cntp() -> pl.Expr:
    return _mean((_ret(1) > 0).cast(pl.Float64), 20)


@_reg("q158_corr_pv_20", "q158", "20-day price-volume correlation")
def _corrpv() -> pl.Expr:
    return _corr(pl.col("adj_close"), pl.col("log_volume"), 20)


FAMILIES: dict[str, str] = {
    "trad": "Classical cross-sectional equity factors",
    "a101": "WorldQuant 101 Formulaic Alphas (daily-OHLCV subset)",
    "a191": "Guotai Junan 191 Alphas (daily-OHLCV subset)",
    "q158": "qlib Alpha158-style rolling price/volume features",
}


def factor_specs() -> list[tuple[str, str, str]]:
    """``(name, family, description)`` for every factor, in registration order."""
    return [(n, fam, desc) for n, (fam, desc, _) in FACTORS.items()]


def compute_factors(
    bars: pl.DataFrame,
    *,
    names: list[str] | None = None,
) -> pl.DataFrame:
    """Compute the factor panel from a prepared bar frame.

    Parameters
    ----------
    bars:
        Long frame sorted by ``(symbol, ts)`` carrying at least ``ts``,
        ``symbol``, ``open``, ``high``, ``low``, ``close``, ``adj_close``,
        ``volume``, ``amount``, ``turnover``, ``vwap``, ``log_volume``,
        ``market_cap`` and ``mkt_ret``.

    Returns
    -------
    ``ts``, ``symbol`` and one column per factor, sorted by ``(ts, symbol)``.

    Notes
    -----
    Every expression uses only trailing windows and ``shift(+n)``, so the value
    at ``t`` is knowable at ``t``. Leading rows are null while windows fill --
    that is correct, and filling them would fabricate history.
    """
    wanted = names or list(FACTORS)
    df = bars.sort([_S, "ts"], maintain_order=True)

    # Cross-sectional ranks that time-series operators later consume. They must
    # be materialised in a separate pass: an expression cannot partition by
    # `ts` and by `symbol` at once, and polars returns nulls instead of raising
    # when you try.
    # Pass 1: the within-symbol delta. Cannot be inlined below, because the
    # rank partitions by `ts` while this partitions by `symbol`.
    df = df.with_columns(
        _dlogv=pl.col("log_volume") - pl.col("log_volume").shift(1).over(_S),
        _retio=(pl.col("close") - pl.col("open")) / pl.col("open"),
    )
    # Pass 2: cross-sectional ranks of those columns.
    df = df.with_columns(
        _csr_dlogv=_cs_rank(pl.col("_dlogv")),
        _csr_retio=_cs_rank(pl.col("_retio")),
        _csr_low=_cs_rank(pl.col("low")),
    )

    exprs = []
    for name in wanted:
        if name not in FACTORS:
            raise KeyError(f"unknown factor {name!r}; have {sorted(FACTORS)}")
        _, _, fn = FACTORS[name]
        exprs.append(fn().alias(name))

    out = df.with_columns(exprs)
    # Infinities arise from near-zero denominators in the formulaic alphas.
    # They must become null rather than propagate: a single inf poisons every
    # cross-sectional mean and standard deviation downstream.
    out = out.with_columns(
        [
            pl.when(pl.col(n).is_finite()).then(pl.col(n)).otherwise(None).alias(n)
            for n in wanted
        ]
    )
    return out.select(["ts", _S, *wanted]).sort(["ts", _S], maintain_order=True)
