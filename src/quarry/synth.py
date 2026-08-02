"""Synthetic A-share store generator -- research and test scaffolding only.

This exists so the whole stack is runnable end-to-end without a prism dump. It
is **not** a market simulator and nothing it produces should be read as
evidence about a real strategy.

What makes it useful rather than decorative is that it injects *known ground
truth* and hands it back as :class:`SynthTruth`: which symbol was suspended on
which date, which days closed limit-up, and what the true IC of each planted
factor is. Tests assert against those facts instead of eyeballing plots, which
is the difference between a test that catches a masking regression and one that
merely runs.

Layout produced::

    <root>/daily/date=YYYYMMDD/part.parquet
    <root>/factor_frame/date=YYYYMMDD/part.parquet
    <root>/index/date=YYYYMMDD/part.parquet
    <root>/actions/date=YYYYMMDD/part.parquet
    <root>/membership/part.parquet
    <root>/listings/part.parquet
    <root>/bars/<freq>/date=YYYYMMDD/symbol=NNNNNN/part.parquet
    <root>/ticks/date=YYYYMMDD/symbol=NNNNNN/part.parquet

Daily-grain datasets partition by date alone; per-instrument datasets partition
by date and symbol. Both layouts are exercised on purpose, because
:meth:`quarry.db.DataRoot.hive_keys` has to handle either.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from crucible.determinism import DEFAULT_SEED, rng
from quarry.schema import FactorSpec, SchemaFingerprint

__all__ = ["SynthTruth", "make_store", "PLANTED_FACTORS"]

#: Factors planted in the dump, with the true correlation each carries against
#: the next-day return. ``alpha_strong`` through ``alpha_none`` form a
#: monotone ladder so an IC estimator can be checked for ordering, not just
#: sign. ``dupe_of_strong`` is a near-copy used to test pool rejection.
PLANTED_FACTORS: dict[str, float] = {
    "alpha_strong": 0.40,
    "alpha_mid": 0.15,
    "alpha_weak": 0.05,
    "alpha_none": 0.00,
    "dupe_of_strong": 0.40,
}

_INDUSTRIES = ("bank", "tech", "pharma", "energy", "consumer", "materials")


@dataclass(frozen=True)
class SynthTruth:
    """Ground truth about a generated store.

    Everything a test needs to assert correctness without re-deriving it from
    the very code under test.
    """

    root: Path
    symbols: tuple[str, ...]
    dates: tuple[dt.date, ...]
    freq: str
    suspensions: dict[str, tuple[dt.date, ...]]
    limit_ups: dict[str, tuple[dt.date, ...]]
    one_word_boards: dict[str, tuple[dt.date, ...]]
    st_symbols: tuple[str, ...]
    factor_ic: dict[str, float]
    fingerprint: SchemaFingerprint

    @property
    def n_suspensions(self) -> int:
        """Total injected suspension observations."""
        return sum(len(v) for v in self.suspensions.values())

    def suspended_on(self, symbol: str, day: dt.date) -> bool:
        """Whether ``symbol`` was injected as suspended on ``day``."""
        return day in self.suspensions.get(symbol, ())


def _make_symbols(n: int, gen: np.random.Generator) -> tuple[str, ...]:
    """A realistic board mix: SH main, SZ main, ChiNext, STAR.

    The mix matters because price-limit bands are board-dependent -- a fixture
    of only ``600xxx`` names would never exercise the 20% branch.
    """
    per = max(1, n // 4)
    out: list[str] = []
    out += [f"6000{i:02d}" for i in range(per)]
    out += [f"0000{i:02d}" for i in range(per)]
    out += [f"3000{i:02d}" for i in range(per)]
    out += [f"6880{i:02d}" for i in range(n - 3 * per)]
    return tuple(sorted(out[:n]))


def _sessions(start: dt.date, end: dt.date) -> tuple[dt.date, ...]:
    """Weekdays in range.

    Deliberately not the real CSRC calendar: this module must not depend on
    :mod:`almanac` (subpackages stay independently splittable). Choose a window
    without mainland holidays -- 2024-01-02..2024-01-31 is clean.
    """
    days = np.arange(start, end + dt.timedelta(days=1), dtype="datetime64[D]")
    return tuple(d.astype(dt.date) for d in days[np.is_busday(days)])


def make_store(
    root: str | Path,
    *,
    start: str | dt.date = "2024-01-02",
    end: str | dt.date = "2024-01-31",
    n_symbols: int = 20,
    bar_freq: str = "1min",
    bars_per_day: int = 240,
    with_ticks: bool = False,
    tick_symbols: int = 3,
    seed: int = DEFAULT_SEED,
) -> SynthTruth:
    """Generate a complete synthetic store and return its ground truth.

    Parameters
    ----------
    n_symbols:
        Instrument count. Kept small by default; the point is correctness
        coverage, not scale.
    with_ticks:
        Also emit 3-second snapshots for ``tick_symbols`` instruments. Off by
        default because tick data dominates fixture size and most tests do not
        need it.
    seed:
        Fixed seed. Two calls with the same arguments produce byte-identical
        files, which is what lets the determinism test compare digests.

    Returns
    -------
    SynthTruth
        Injected facts: suspensions, limit days, ST names, planted factor ICs
        and the dump's schema fingerprint.
    """
    root = Path(root)
    gen = rng(seed)
    d0 = dt.date.fromisoformat(str(start)) if not isinstance(start, dt.date) else start
    d1 = dt.date.fromisoformat(str(end)) if not isinstance(end, dt.date) else end
    dates = _sessions(d0, d1)
    symbols = _make_symbols(n_symbols, gen)
    n_d, n_s = len(dates), len(symbols)
    if n_d < 5:
        raise ValueError(f"date range {d0}..{d1} yields only {n_d} sessions; need at least 5")

    # --- latent return process -------------------------------------------
    # A market factor plus idiosyncratic noise, so cross-sectional dispersion
    # is realistic and a market-neutral hedge has something to remove.
    market = gen.normal(0.0, 0.010, size=n_d)
    beta = gen.uniform(0.6, 1.4, size=n_s)
    idio = gen.normal(0.0, 0.018, size=(n_d, n_s))
    rets = market[:, None] * beta[None, :] + idio

    st_flags = gen.random(n_s) < 0.10
    st_symbols = tuple(s for s, f in zip(symbols, st_flags) if f)
    industries = gen.integers(0, len(_INDUSTRIES), size=n_s)
    base_price = gen.uniform(5.0, 60.0, size=n_s)
    shares_out = gen.uniform(2e8, 5e9, size=n_s)

    # --- unadjusted daily price path --------------------------------------
    close = np.empty((n_d, n_s))
    prev = base_price.copy()
    for i in range(n_d):
        close[i] = np.round(prev * (1.0 + rets[i]), 2)
        prev = close[i]
    prev_close = np.vstack([base_price[None, :], close[:-1]])

    # --- injected impairments --------------------------------------------
    suspended = np.zeros((n_d, n_s), dtype=bool)
    for j in range(n_s):
        # Two suspensions per symbol, never on the first or last session so a
        # forward-return label always has somewhere to land.
        for i in gen.choice(np.arange(1, n_d - 1), size=2, replace=False):
            suspended[i, j] = True

    limit_pct = np.where(
        [s.startswith(("300", "688")) for s in symbols],
        0.20,
        np.where(st_flags, 0.05, 0.10),
    )
    limit_up_px = np.floor(prev_close * (1 + limit_pct[None, :]) * 100 + 0.5) / 100

    limit_up = np.zeros((n_d, n_s), dtype=bool)
    one_word = np.zeros((n_d, n_s), dtype=bool)
    for j in range(n_s):
        i = int(gen.integers(1, n_d - 1))
        if not suspended[i, j]:
            limit_up[i, j] = True
            close[i, j] = limit_up_px[i, j]
            # Half the limit days are one-word boards: opened locked, never traded.
            one_word[i, j] = bool(gen.random() < 0.5)

    high = np.round(close * (1 + np.abs(gen.normal(0, 0.006, (n_d, n_s)))), 2)
    low = np.round(close * (1 - np.abs(gen.normal(0, 0.006, (n_d, n_s)))), 2)
    open_ = np.round((high + low) / 2, 2)
    volume = np.round(gen.lognormal(15.5, 0.7, (n_d, n_s)), 0)

    # One-word boards trade at a single price all day.
    high = np.where(one_word, close, high)
    low = np.where(one_word, close, low)
    open_ = np.where(one_word, close, open_)

    # THE TRAP: a suspended name keeps printing its prior close and zero
    # volume, exactly as a vendor feed does. Nothing downstream may treat this
    # flat 0% as a real return.
    for arr in (open_, high, low, close):
        arr[suspended] = prev_close[suspended]
    volume[suspended] = 0.0

    vwap = np.round((high + low + close) / 3.0, 2)
    amount = np.round(vwap * volume, 2)

    # --- planted factors ---------------------------------------------------
    # Correlated with the NEXT session's return, so a same-day feature genuinely
    # predicts a forward label. The last row has no future and is left as noise.
    fwd = np.vstack([rets[1:], np.zeros((1, n_s))])
    fwd_z = (fwd - fwd.mean(axis=1, keepdims=True)) / (fwd.std(axis=1, keepdims=True) + 1e-12)
    factors: dict[str, np.ndarray] = {}
    for name, target_ic in PLANTED_FACTORS.items():
        noise = gen.normal(0, 1, (n_d, n_s))
        factors[name] = target_ic * fwd_z + np.sqrt(max(0.0, 1 - target_ic**2)) * noise
    # Near-duplicate: same signal, a whisper of independent noise.
    factors["dupe_of_strong"] = factors["alpha_strong"] + gen.normal(0, 0.02, (n_d, n_s))

    dr_root = root
    dr_root.mkdir(parents=True, exist_ok=True)

    sym_col = list(symbols)
    for i, day in enumerate(dates):
        ts = dt.datetime.combine(day, dt.time(15, 0))
        tag = day.strftime("%Y%m%d")

        daily = pl.DataFrame(
            {
                "ts": [ts] * n_s,
                "symbol": sym_col,
                "open": open_[i],
                "high": high[i],
                "low": low[i],
                "close": close[i],
                "prev_close": prev_close[i],
                "vwap": vwap[i],
                "volume": volume[i],
                "amount": amount[i],
                "is_st": st_flags,
                "industry": [_INDUSTRIES[k] for k in industries],
                "market_cap": close[i] * shares_out,
                "list_date": [dates[0] - dt.timedelta(days=400)] * n_s,
            }
        ).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
        _write(dr_root / "daily" / f"date={tag}", daily)

        ff = pl.DataFrame(
            {"ts": [ts] * n_s, "symbol": sym_col, **{k: v[i] for k, v in factors.items()}}
        ).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
        _write(dr_root / "factor_frame" / f"date={tag}", ff)

        idx_lvl = float(3000 * np.exp(np.sum(market[: i + 1])))
        index = pl.DataFrame(
            {
                "ts": [ts],
                "symbol": ["000300.SH"],
                "close": [round(idx_lvl, 2)],
                "ret": [float(market[i])],
            }
        ).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
        _write(dr_root / "index" / f"date={tag}", index)

        _write_bars(dr_root / "bars" / bar_freq / f"date={tag}", day, symbols, i, open_, high, low, close, volume, bars_per_day, gen)

        if with_ticks:
            _write_ticks(dr_root / "ticks" / f"date={tag}", day, symbols[:tick_symbols], i, close, gen)

    # --- dividends, membership, listings ----------------------------------
    ex_day = dates[len(dates) // 2]
    div_syms = list(symbols[:: max(1, n_s // 4)])
    actions = pl.DataFrame(
        {
            "ts": [dt.datetime.combine(ex_day, dt.time(15, 0))] * len(div_syms),
            "symbol": div_syms,
            "prev_close": [float(close[dates.index(ex_day) - 1, symbols.index(s)]) for s in div_syms],
            "cash_div": [0.25] * len(div_syms),
            "split_ratio": [0.0] * len(div_syms),
        }
    ).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
    _write(dr_root / "actions" / f"date={ex_day.strftime('%Y%m%d')}", actions)

    half = n_s // 2
    _write(
        dr_root / "membership",
        pl.DataFrame(
            {
                "index_code": ["000300.SH"] * half,
                "symbol": sym_col[:half],
                "start_date": [dates[0] - dt.timedelta(days=365)] * half,
                "end_date": [None] * half,
            }
        ).with_columns(pl.col("start_date").cast(pl.Date), pl.col("end_date").cast(pl.Date)),
        hive=False,
    )
    _write(
        dr_root / "listings",
        pl.DataFrame(
            {
                "symbol": sym_col,
                "list_date": [dates[0] - dt.timedelta(days=400)] * n_s,
                "delist_date": [None] * n_s,
            }
        ).with_columns(pl.col("list_date").cast(pl.Date), pl.col("delist_date").cast(pl.Date)),
        hive=False,
    )

    fingerprint = SchemaFingerprint(
        tuple(FactorSpec(n, {"synthetic": True}, "DOUBLE") for n in factors),
        producer=f"synth-v1(seed={seed})",
    )

    return SynthTruth(
        root=dr_root,
        symbols=symbols,
        dates=dates,
        freq=bar_freq,
        suspensions={
            s: tuple(dates[i] for i in np.flatnonzero(suspended[:, j])) for j, s in enumerate(symbols)
        },
        limit_ups={
            s: tuple(dates[i] for i in np.flatnonzero(limit_up[:, j])) for j, s in enumerate(symbols)
        },
        one_word_boards={
            s: tuple(dates[i] for i in np.flatnonzero(one_word[:, j])) for j, s in enumerate(symbols)
        },
        st_symbols=st_symbols,
        factor_ic=dict(PLANTED_FACTORS),
        fingerprint=fingerprint,
    )


def _write(directory: Path, df: pl.DataFrame, *, hive: bool = True) -> None:
    """Write one Parquet part, creating the partition directory."""
    directory.mkdir(parents=True, exist_ok=True)
    df.write_parquet(directory / "part.parquet", compression="zstd", statistics=True)


def _write_bars(
    directory: Path,
    day: dt.date,
    symbols: tuple[str, ...],
    i: int,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    n_bars: int,
    gen: np.random.Generator,
) -> None:
    """Emit right-labelled intraday bars consistent with the day's OHLC.

    The path is a Brownian bridge pinned to the true open and close, so
    intraday features are not independent of the daily series they must agree
    with. Bars are labelled at their closing edge (09:31 covers 09:30-09:31).
    """
    slots = _bar_labels(day, n_bars)
    for j, sym in enumerate(symbols):
        o, c = float(open_[i, j]), float(close[i, j])
        if volume[i, j] <= 0:  # suspended: no intraday trading at all
            continue
        steps = gen.normal(0, 0.0008, n_bars)
        path = np.cumsum(steps)
        path = path - np.linspace(0, path[-1], n_bars)  # bridge to zero drift
        px = np.round(o * np.exp(path) + np.linspace(0, c - o, n_bars), 2)
        px[-1] = c
        bar_hi = np.round(px * (1 + np.abs(gen.normal(0, 0.0004, n_bars))), 2)
        bar_lo = np.round(px * (1 - np.abs(gen.normal(0, 0.0004, n_bars))), 2)
        vol = np.round(volume[i, j] * _u_shape(n_bars, gen), 0)
        frame = pl.DataFrame(
            {
                "ts": slots,
                "symbol": [sym] * n_bars,
                "open": np.round(np.concatenate([[o], px[:-1]]), 2),
                "high": np.maximum(bar_hi, px),
                "low": np.minimum(bar_lo, px),
                "close": px,
                "volume": vol,
                "amount": np.round(px * vol, 2),
                "vwap": px,
            }
        ).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
        _write(directory / f"symbol={sym}", frame)


def _u_shape(n: int, gen: np.random.Generator) -> np.ndarray:
    """Intraday volume weights: heavy at the open and close, thin midday.

    Real A-share volume is strongly U-shaped. A flat profile would make any
    VWAP-based label unrealistically easy to achieve.
    """
    x = np.linspace(0, 1, n)
    shape = 0.6 + 2.2 * (x - 0.5) ** 2
    shape = shape * gen.uniform(0.85, 1.15, n)
    return shape / shape.sum()


def _bar_labels(day: dt.date, n_bars: int) -> list[dt.datetime]:
    """Right-labelled minute boundaries across both A-share sessions."""
    per = n_bars // 2
    out: list[dt.datetime] = []
    for base, count in ((dt.time(9, 30), per), (dt.time(13, 0), n_bars - per)):
        origin = dt.datetime.combine(day, base)
        out.extend(origin + dt.timedelta(minutes=k + 1) for k in range(count))
    return out


def _write_ticks(
    directory: Path,
    day: dt.date,
    symbols: tuple[str, ...],
    i: int,
    close: np.ndarray,
    gen: np.random.Generator,
) -> None:
    """Emit 3-second level-1 snapshots for a handful of symbols.

    Enough structure to exercise the streaming path and microstructure
    features: bid/ask around a mid, sizes, and a cumulative volume that only
    ever increases.
    """
    n = 4800  # 4 hours at 3s
    origin = dt.datetime.combine(day, dt.time(9, 30))
    ts = [origin + dt.timedelta(seconds=3 * (k + 1)) for k in range(n)]
    for j, sym in enumerate(symbols):
        mid = float(close[i, j]) * np.exp(np.cumsum(gen.normal(0, 0.00012, n)))
        spread = np.round(np.maximum(0.01, np.abs(gen.normal(0.012, 0.004, n))), 2)
        bid = np.round(mid - spread / 2, 2)
        ask = np.round(bid + spread, 2)
        frame = pl.DataFrame(
            {
                "ts": ts,
                "symbol": [sym] * n,
                "last": np.round(mid, 2),
                "bid1": bid,
                "ask1": ask,
                "bid_size1": gen.integers(100, 50_000, n).astype(np.int64),
                "ask_size1": gen.integers(100, 50_000, n).astype(np.int64),
                "cum_volume": np.cumsum(gen.integers(0, 4000, n)).astype(np.int64),
            }
        ).with_columns(pl.col("ts").cast(pl.Datetime("ns")))
        _write(directory / f"symbol={sym}", frame)
