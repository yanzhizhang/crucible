"""Market-data quality gate: the written standard that a trading day must pass before use.

Every check returns a :class:`CheckResult` graded ``PASS`` / ``WARN`` / ``FAIL`` against
thresholds loaded from ``cfg/md_quality.toml`` (the same file a C++ live checker reads). A day's
verdict is the worst grade among its checks; downstream builders refuse a ``FAIL`` day unless
explicitly overridden. Checks that are specified but not yet implemented report ``SKIP`` so the
report shows the gap instead of hiding it.

Inputs are **normalised** lazy frames (see :func:`quarry.feitu.normalize_columns`): timestamps
are int64 ns since epoch UTC, ``symbol`` is a 6-digit string, ``exchange`` is ``XSHG``/``XSHE``.
Everything is a polars aggregation or a per-channel numpy pass, so a full day (~7e8 records)
runs in bounded memory. The standard itself is documented in ``docs/MD_QUALITY.md``.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from quarry.raw import A_SHARE_PREFIXES


def _a_share(exchange: pl.Expr, symbol: pl.Expr) -> pl.Expr:
    """A-share common stock by code range (see :data:`quarry.raw.A_SHARE_PREFIXES`)."""
    out = pl.lit(False)
    for venue, prefixes in A_SHARE_PREFIXES.items():
        out = out | ((exchange == venue) & symbol.str.slice(0, 3).is_in(prefixes))
    return out


__all__ = [
    "CheckResult",
    "Level",
    "Thresholds",
    "check_book_sanity",
    "check_daily_totals",
    "check_latency",
    "check_minute_coverage",
    "check_sequence",
    "check_trade_prices",
    "check_trade_symbol_coverage",
    "check_volume_identity",
    "grade",
    "load_thresholds",
    "results_frame",
    "session_minutes",
    "skipped",
    "verdict",
]

_LOCAL_OFFSET_NS = 8 * 3600 * 1_000_000_000
_NS_PER_MS = 1_000_000


class Level(Enum):
    """Grade of one check; ordered by severity."""

    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"
    """Specified in the standard, not implemented yet (or its input is absent)."""

    @property
    def severity(self) -> int:
        """0 pass/skip, 1 warn, 2 fail."""
        return {"PASS": 0, "SKIP": 0, "WARN": 1, "FAIL": 2}[self.value]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of one check on one scope (e.g. ``transaction/XSHE``).

    ``value`` is the measured quantity, ``warn_at`` / ``fail_at`` the thresholds it was graded
    against, and ``n`` the number of observations behind the measurement.
    """

    check: str
    scope: str
    level: Level
    value: float
    warn_at: float | None
    fail_at: float | None
    n: int
    detail: str = ""

    def row(self) -> dict[str, Any]:
        """Flat dict for a frame / Parquet row."""
        d = asdict(self)
        d["level"] = self.level.value
        return d


Thresholds = Mapping[str, Mapping[str, Any]]


def load_thresholds(path: str | Path) -> Thresholds:
    """Read the TOML threshold file (one table per check)."""
    with Path(path).open("rb") as fh:
        return tomllib.load(fh)


def grade(value: float, warn_at: float | None, fail_at: float | None) -> Level:
    """Grade a higher-is-worse measurement: ``>= fail_at`` FAIL, ``>= warn_at`` WARN."""
    if fail_at is not None and value >= fail_at:
        return Level.FAIL
    if warn_at is not None and value >= warn_at:
        return Level.WARN
    return Level.PASS


def _result(
    th: Thresholds, check: str, scope: str, value: float, n: int, detail: str = ""
) -> CheckResult:
    cfg = th.get(check, {})
    warn_at, fail_at = cfg.get("warn_at"), cfg.get("fail_at")
    return CheckResult(
        check, scope, grade(value, warn_at, fail_at), float(value), warn_at, fail_at, int(n), detail
    )


def skipped(check: str, scope: str, why: str) -> CheckResult:
    """Record a check the standard requires but that could not run (graded ``SKIP``)."""
    return CheckResult(check, scope, Level.SKIP, float("nan"), None, None, 0, why)


def verdict(results: Iterable[CheckResult]) -> Level:
    """Worst grade among ``results`` (``SKIP`` never worsens the verdict)."""
    worst = Level.PASS
    for r in results:
        if r.level.severity > worst.severity:
            worst = r.level
    return worst


def results_frame(results: Iterable[CheckResult]) -> pl.DataFrame:
    """Collect results into one frame, worst first."""
    rows = [r.row() for r in results]
    order = {"FAIL": 0, "WARN": 1, "SKIP": 2, "PASS": 3}
    return pl.DataFrame(rows).sort(
        pl.col("level").replace_strict(order, return_dtype=pl.Int8), "check", "scope"
    )


_BINS_PER_E = 200.0


def _log_bin(ns: pl.Expr) -> pl.Expr:
    """Signed log bin of a nanosecond value: ``sign * round(ln(1 + |x|/1us) * 200)``."""
    mag = ((ns.abs().cast(pl.Float64) / 1_000.0).log1p() * _BINS_PER_E).round(0)
    return (ns.sign() * mag).cast(pl.Int32)


def _bin_center_ns(b: np.ndarray) -> np.ndarray:
    return np.sign(b) * np.expm1(np.abs(b) / _BINS_PER_E) * 1_000.0


def _hist_quantile(g: pl.DataFrame, q: float) -> float:
    """Quantile (ns) of a ``(bin, n)`` histogram."""
    s = g.select("bin", "n").group_by("bin").agg(pl.col("n").sum()).sort("bin")
    cum = np.cumsum(s["n"].to_numpy())
    idx = int(np.searchsorted(cum, q * cum[-1]))
    return float(_bin_center_ns(s["bin"].to_numpy()[idx : idx + 1])[0])


# ---------------------------------------------------------------------------
# 1. completeness
# ---------------------------------------------------------------------------


def session_minutes(windows: Iterable[tuple[str, str]]) -> list[str]:
    """Expand ``[("09:30", "11:30"), ...]`` half-open windows into ``HHMM`` labels."""
    out: list[str] = []
    for start, end in windows:
        t = int(start[:2]) * 60 + int(start[3:])
        stop = int(end[:2]) * 60 + int(end[3:])
        out.extend(f"{m // 60:02d}{m % 60:02d}" for m in range(t, stop))
    return out


def check_minute_coverage(manifest: pl.DataFrame, kind: str, th: Thresholds) -> list[CheckResult]:
    """Every continuous-session minute must have a non-empty dump file.

    ``manifest`` is the decoder's per-file table (``minute`` = local ``HHMM``, ``rows``). A
    missing minute means no data at all for every symbol in it -- the 20260921 quotation feed,
    which starts at 10:24, is the reference failure this check must catch.
    """
    cfg = th.get("completeness.minutes_missing", {})
    expected = session_minutes(
        tuple(w) for w in cfg.get("windows", [["09:30", "11:30"], ["13:00", "14:57"]])
    )
    have = set(manifest.filter(pl.col("rows") > 0)["minute"].to_list())
    missing = [m for m in expected if m not in have]
    detail = ""
    if missing:
        detail = f"first missing {missing[0]}, last {missing[-1]}"
    out = [_result(th, "completeness.minutes_missing", kind, len(missing), len(expected), detail)]
    if "n_truncated_books" in manifest.columns:
        trunc = int(manifest["n_truncated_books"].sum())
        out.append(
            _result(
                th, "completeness.book_levels_truncated", kind, trunc, int(manifest["rows"].sum())
            )
        )
    return out


def check_trade_symbol_coverage(
    quotes: pl.LazyFrame, trades: pl.LazyFrame, th: Thresholds
) -> list[CheckResult]:
    """Every stock that traded according to its snapshots must appear in the trade stream.

    The snapshot's cumulative ``total_volume`` is the exchange's own count, so a stock with a
    positive end-of-day volume but no trade records means the tick feed lost that symbol.
    """
    traded_by_quote = (
        quotes.group_by("exchange", "symbol")
        .agg(pl.col("total_volume").max().alias("vol"))
        .filter((pl.col("vol") > 0) & _a_share(pl.col("exchange"), pl.col("symbol")))
    )
    in_trades = (
        trades.filter(pl.col("record_type") == "trade").select("exchange", "symbol").unique()
    )
    have = traded_by_quote.join(in_trades, on=["exchange", "symbol"], how="semi")
    stats = (
        traded_by_quote.group_by("exchange")
        .agg(pl.len().alias("n"))
        .join(have.group_by("exchange").agg(pl.len().alias("hit")), on="exchange", how="left")
        .collect(engine="streaming")
    )
    out = []
    for r in stats.iter_rows(named=True):
        miss = 1.0 - (r["hit"] or 0) / r["n"]
        out.append(
            _result(
                th,
                "completeness.trade_symbols_missing_frac",
                f"trade/{r['exchange']}",
                miss,
                r["n"],
                f"{r['n'] - (r['hit'] or 0)} traded stocks absent",
            )
        )
    return out


# ---------------------------------------------------------------------------
# 2. sequence integrity (merged order + trade stream per channel)
# ---------------------------------------------------------------------------


def check_sequence(
    orders: pl.LazyFrame, trades: pl.LazyFrame, th: Thresholds, *, batch_rows: int = 5_000_000
) -> list[CheckResult]:
    """Contiguity of the per-channel sequence shared by the order and trade streams.

    Orders and trades share one sequence space per channel, so contiguity is only meaningful
    on the merged stream (checking either alone reads a clean capture as 50% loss). Which
    column carries that sequence is per exchange and set in the thresholds file
    (``sequence.seq_col``).

    Memory: two streaming passes and no sort. Pass one takes each channel's id range; pass
    two fills a saturating ``uint8`` presence counter per channel batch by batch, so the peak
    is one byte per id in the range (8x less than sorting the int64 ids), independent of how
    the day is split into files.
    """
    cfg = th.get("sequence", {})
    seq_col: Mapping[str, str] = cfg.get("seq_col", {"XSHG": "seq_id", "XSHE": "seq_id"})
    out: list[CheckResult] = []
    for exch, col in seq_col.items():
        merged = pl.concat(
            [
                f.filter(pl.col("exchange") == exch).select(
                    "channel_id", pl.col(col).cast(pl.Int64).alias("s")
                )
                for f in (orders, trades)
            ]
        )
        ranges = (
            merged.group_by("channel_id")
            .agg(pl.col("s").min().alias("lo"), pl.col("s").max().alias("hi"), pl.len().alias("n"))
            .collect(engine="streaming")
            .sort("channel_id")
        )
        lo = {r["channel_id"]: r["lo"] for r in ranges.iter_rows(named=True)}
        counts = {
            r["channel_id"]: np.zeros(r["hi"] - r["lo"] + 1, dtype=np.uint8)
            for r in ranges.iter_rows(named=True)
        }
        for batch in merged.collect_batches(chunk_size=batch_rows):
            ch_arr = batch["channel_id"].to_numpy()
            s_arr = batch["s"].to_numpy()
            for ch in np.unique(ch_arr):
                u, c = np.unique(s_arr[ch_arr == ch] - lo[ch], return_counts=True)
                cnt = counts[ch]
                cnt[u] = np.minimum(cnt[u].astype(np.int64) + c, 255).astype(np.uint8)
        total = missing = dups = gaps = 0
        worst_ch, worst_missing = None, -1
        for ch, cnt in counts.items():
            absent = cnt == 0
            ch_missing = int(absent.sum())
            total += int(cnt.astype(np.int64).sum())
            dups += int((cnt[cnt > 1].astype(np.int64) - 1).sum())
            missing += ch_missing
            gaps += int((absent[1:] & ~absent[:-1]).sum())
            if ch_missing > worst_missing:
                worst_ch, worst_missing = ch, ch_missing
        scope = f"{exch}/{col}"
        span = total + missing - dups
        out.append(
            _result(
                th,
                "sequence.missing_frac",
                scope,
                missing / span if span else 0.0,
                total,
                f"{missing} ids in {gaps} gaps over {len(counts)} channels; "
                f"worst channel {worst_ch} ({worst_missing})",
            )
        )
        out.append(_result(th, "sequence.duplicates", scope, dups, total))
    return out


# ---------------------------------------------------------------------------
# 3. timestamps
# ---------------------------------------------------------------------------


def _minute_of_day(hhmm: str) -> int:
    return int(hhmm[:2]) * 60 + int(hhmm[3:])


def check_latency(frame: pl.LazyFrame, kind: str, th: Thresholds) -> list[CheckResult]:
    """Presence, causality, steady-state spread, drift, burst backlog and resolution.

    ``latency = arrival_ts - exchange_ts``, measured on market events only (vendor status
    records, e.g. SSE product-status messages stamped 06:00 and captured at 09:14, are not
    latency). Two regimes are graded separately because they mean different things:

    * **steady state** -- 15-minute buckets inside ``steady_windows`` (continuous trading away
      from the opens). Its p99 and the range of its bucket medians (drift) describe the link
      and the capture clock.
    * **burst backlog** -- the day's worst latency, which on this capture comes from the
      open (09:30-09:45), the 13:00 reopen and the close, when the capture falls tens of
      seconds behind. That is a property a point-in-time backtest must replay (data arrives
      late exactly when volume peaks), not a reason to discard the day, so it grades WARN only.

    A negative latency beyond the tolerance means the capture clock is behind the exchange
    clock. Resolution reports the coarsest unit every exchange timestamp is a multiple of.
    """
    tol_ms = th.get("timestamps.negative_latency_frac", {}).get("tolerance_ms", 5.0)
    steady = th.get("timestamps.latency_p99_ms", {}).get(
        "steady_windows", [["09:45", "11:30"], ["13:15", "14:57"]]
    )
    names = frame.collect_schema().names()
    if "record_type" in names:
        frame = frame.filter(pl.col("record_type") != "status")
    lat = (pl.col("arrival_ts") - pl.col("exchange_ts")).alias("lat")
    tod_min = (((pl.col("exchange_ts") + _LOCAL_OFFSET_NS) // 60_000_000_000) % 1440).cast(pl.Int32)
    base = frame.select("exchange", "exchange_ts", "arrival_ts", lat, tod_min.alias("tod"))
    agg = (
        base.group_by("exchange")
        .agg(
            pl.len().alias("n"),
            ((pl.col("exchange_ts") <= 0) | (pl.col("arrival_ts") <= 0)).sum().alias("bad_ts"),
            (pl.col("lat") < -tol_ms * _NS_PER_MS).sum().alias("neg"),
            pl.col("lat").min().alias("lat_min"),
            pl.col("lat").max().alias("lat_max"),
            (pl.col("lat") > 1_000 * _NS_PER_MS).sum().alias("gt1s"),
            (pl.col("exchange_ts") % 1_000_000_000 == 0).mean().alias("res_1s"),
            (pl.col("exchange_ts") % 10_000_000 == 0).mean().alias("res_10ms"),
            (pl.col("exchange_ts") % 1_000_000 == 0).mean().alias("res_1ms"),
        )
        .collect(engine="streaming")
    )
    # Exact quantiles would hold the whole day's latency column in memory. A log-binned
    # histogram per 15-minute bucket streams in constant memory; bins are ~0.5% wide.
    hist = (
        base.with_columns(
            (pl.col("tod") // 15).alias("bucket"), _log_bin(pl.col("lat")).alias("bin")
        )
        .group_by("exchange", "bucket", "bin")
        .agg(pl.len().alias("n"), pl.col("lat").max().alias("bmax"))
        .collect(engine="streaming")
    )
    steady_buckets: set[int] = set()
    for lo, hi in steady:
        a, b = _minute_of_day(lo), _minute_of_day(hi)
        steady_buckets.update(range(-(-a // 15), b // 15))  # buckets fully inside [a, b)
    st = hist.filter(pl.col("bucket").is_in(sorted(steady_buckets)))

    def per_exchange(h: pl.DataFrame, q: float) -> dict[str, float]:
        return {e: _hist_quantile(g, q) for (e,), g in h.group_by("exchange")}

    p50, p99 = per_exchange(st, 0.5), per_exchange(st, 0.99)
    meds: dict[str, list[float]] = {}
    for (e, _b), g in st.group_by("exchange", "bucket"):
        if int(g["n"].sum()) >= 1000:
            meds.setdefault(str(e), []).append(_hist_quantile(g, 0.5))
    worst = (
        hist.group_by("exchange", "bucket")
        .agg(pl.col("bmax").max())
        .sort("bmax", descending=True)
        .group_by("exchange", maintain_order=True)
        .first()
    )
    worst_by = {r["exchange"]: r["bucket"] for r in worst.iter_rows(named=True)}

    out: list[CheckResult] = []
    for r in agg.iter_rows(named=True):
        e = r["exchange"]
        scope = f"{kind}/{e}"
        n = r["n"]
        out.append(_result(th, "timestamps.missing", scope, r["bad_ts"], n))
        out.append(
            _result(
                th,
                "timestamps.negative_latency_frac",
                scope,
                r["neg"] / n,
                n,
                f"{r['neg']} beyond -{tol_ms}ms; min {r['lat_min'] / _NS_PER_MS:.3f}ms",
            )
        )
        if e in p99:
            cfg = th.get(f"timestamps.latency_p99_ms.{kind}", {})
            v = p99[e] / _NS_PER_MS
            out.append(
                CheckResult(
                    "timestamps.latency_p99_ms",
                    scope,
                    grade(v, cfg.get("warn_at"), cfg.get("fail_at")),
                    v,
                    cfg.get("warn_at"),
                    cfg.get("fail_at"),
                    n,
                    f"steady state; p50 {p50[e] / _NS_PER_MS:.1f}ms",
                )
            )
        if len(meds.get(e, [])) >= 2:
            m = meds[e]
            out.append(
                _result(
                    th,
                    "timestamps.drift_ms",
                    scope,
                    (max(m) - min(m)) / _NS_PER_MS,
                    n,
                    f"range of {len(m)} steady-state 15-minute medians",
                )
            )
        wb = worst_by.get(e)
        when = f"{wb * 15 // 60:02d}:{wb * 15 % 60:02d}" if wb is not None else "?"
        out.append(
            _result(
                th,
                "timestamps.burst_backlog_s",
                scope,
                r["lat_max"] / 1e9,
                n,
                f"worst in the 15 min from {when}; {r['gt1s']} records > 1 s",
            )
        )
        res = (
            "1s"
            if r["res_1s"] > 0.999
            else "10ms"
            if r["res_10ms"] > 0.999
            else "1ms"
            if r["res_1ms"] > 0.999
            else "<1ms"
        )
        res_ms = {"1s": 1000.0, "10ms": 10.0, "1ms": 1.0, "<1ms": 0.0}[res]
        cfg_res = th.get(f"timestamps.resolution_ms.{kind}", {})
        out.append(
            CheckResult(
                "timestamps.resolution_ms",
                scope,
                grade(res_ms, cfg_res.get("warn_at"), cfg_res.get("fail_at")),
                res_ms,
                cfg_res.get("warn_at"),
                cfg_res.get("fail_at"),
                n,
                f"exchange_ts granularity {res}",
            )
        )
    return out


# ---------------------------------------------------------------------------
# 4. internal consistency
# ---------------------------------------------------------------------------


def check_volume_identity(
    quotes: pl.LazyFrame, trades: pl.LazyFrame, th: Thresholds
) -> list[CheckResult]:
    """End-of-day snapshot cumulative volume/amount must equal the sum of trade records.

    This is the exchange's own accounting against the tick stream: any trade lost, duplicated
    or mis-attributed shows up here per symbol.
    """
    q = (
        quotes.group_by("exchange", "symbol")
        .agg(
            pl.col("total_volume").max().alias("q_vol"), pl.col("total_amount").max().alias("q_amt")
        )
        .filter((pl.col("q_vol") > 0) & _a_share(pl.col("exchange"), pl.col("symbol")))
    )
    t = (
        trades.filter(pl.col("record_type") == "trade")
        .group_by("exchange", "symbol")
        .agg(
            pl.col("volume").sum().alias("t_vol"),
            (pl.col("price") * pl.col("volume")).sum().alias("t_amt"),
        )
    )
    amt_tol = th.get("consistency.amount_mismatch_frac", {}).get("rel_tolerance", 1e-6)
    j = (
        q.join(t, on=["exchange", "symbol"], how="inner")
        .group_by("exchange")
        .agg(
            pl.len().alias("n"),
            (pl.col("q_vol") != pl.col("t_vol")).sum().alias("vol_bad"),
            ((pl.col("q_amt") - pl.col("t_amt")).abs() > amt_tol * pl.col("q_amt").abs() + 1.0)
            .sum()
            .alias("amt_bad"),
        )
        .collect(engine="streaming")
    )
    out = []
    for r in j.iter_rows(named=True):
        scope = f"stock/{r['exchange']}"
        out.append(
            _result(
                th,
                "consistency.volume_mismatch_frac",
                scope,
                r["vol_bad"] / r["n"],
                r["n"],
                f"{r['vol_bad']} symbols",
            )
        )
        out.append(
            _result(
                th,
                "consistency.amount_mismatch_frac",
                scope,
                r["amt_bad"] / r["n"],
                r["n"],
                f"{r['amt_bad']} symbols",
            )
        )
    return out


def check_book_sanity(
    quotes: pl.LazyFrame | Iterable[pl.LazyFrame], th: Thresholds
) -> list[CheckResult]:
    """Crossed or locked top of book during continuous trading (``status == 3``).

    Accepts one lazy frame or an iterable of them (one per dump file). Pass the iterable for a
    full day: extracting level 1 from the fixed-size book arrays is not streamed by polars and
    materialises both 10-level books for everything in scope -- 6.1 GB for one day scanned as
    one frame, versus one minute file's worth when fed per file.
    """
    frames = [quotes] if isinstance(quotes, pl.LazyFrame) else quotes
    parts = []
    for lf in frames:
        parts.append(
            lf.filter(pl.col("is_a_share") & (pl.col("status") == 3))
            .select(
                "exchange",
                pl.col("bid_px").arr.get(0).alias("b1"),
                pl.col("ask_px").arr.get(0).alias("a1"),
            )
            .group_by("exchange")
            .agg(
                pl.len().alias("n"),
                ((pl.col("b1") > 0) & (pl.col("a1") > 0) & (pl.col("b1") >= pl.col("a1")))
                .sum()
                .alias("crossed"),
            )
            .collect()
        )
    agg = (
        pl.concat(parts).group_by("exchange").agg(pl.col("n").sum(), pl.col("crossed").sum())
        if parts
        else pl.DataFrame(schema={"exchange": pl.String, "n": pl.UInt32, "crossed": pl.UInt32})
    )
    return [
        _result(
            th,
            "consistency.crossed_book_frac",
            f"quotation/{r['exchange']}",
            r["crossed"] / r["n"] if r["n"] else 0.0,
            r["n"],
            f"{r['crossed']} snapshots",
        )
        for r in agg.iter_rows(named=True)
    ]


def check_trade_prices(
    quotes: pl.LazyFrame, trades: pl.LazyFrame, th: Thresholds
) -> list[CheckResult]:
    """Stock trade prices must sit inside the day's limits and on the 0.01 tick grid."""
    limits = (
        quotes.filter(pl.col("high_limited") > 0)
        .group_by("exchange", "symbol")
        .agg(pl.col("high_limited").max().alias("up"), pl.col("low_limited").min().alias("dn"))
        .filter(_a_share(pl.col("exchange"), pl.col("symbol")))
    )
    t = trades.filter(pl.col("record_type") == "trade").join(
        limits, on=["exchange", "symbol"], how="inner"
    )
    eps = 1e-6
    agg = (
        t.group_by("exchange")
        .agg(
            pl.len().alias("n"),
            ((pl.col("price") > pl.col("up") + eps) | (pl.col("price") < pl.col("dn") - eps))
            .sum()
            .alias("outside"),
            ((pl.col("price") * 100 - (pl.col("price") * 100).round(0)).abs() > 1e-4)
            .sum()
            .alias("off_tick"),
        )
        .collect(engine="streaming")
    )
    out = []
    for r in agg.iter_rows(named=True):
        scope = f"trade/{r['exchange']}"
        out.append(_result(th, "consistency.price_outside_limits", scope, r["outside"], r["n"]))
        out.append(_result(th, "consistency.off_tick_prices", scope, r["off_tick"], r["n"]))
    return out


# ---------------------------------------------------------------------------
# 5. multi-source consistency
# ---------------------------------------------------------------------------


def check_daily_totals(
    trades: pl.LazyFrame, reference: pl.DataFrame, source: str, th: Thresholds
) -> list[CheckResult]:
    """Per-stock daily volume/amount from our trade stream vs an independent daily source.

    ``reference`` needs ``exchange`` (``XSHG``/``XSHE``), ``symbol``, ``is_a_share``, ``volume``
    (shares) and ``amount`` (CNY) -- e.g. the TDX official end-of-day package. Volume must match exactly (a
    trade lost or duplicated anywhere in the day shows up); amount within a relative
    tolerance (the reference rounds to the fen). Stocks present on only one side are counted
    too: a stock the exchange reports as traded but absent from our capture is a coverage gap.
    """
    ours = (
        trades.filter(pl.col("record_type") == "trade")
        .group_by("exchange", "symbol")
        .agg(
            pl.col("volume").sum().alias("vol"),
            (pl.col("price") * pl.col("volume")).sum().alias("amt"),
        )
        .collect(engine="streaming")
    )
    # classify after aggregating: the per-row string test over ~2.5e8 trades cost 8.5 GB / 80 s,
    # on the ~5e3 aggregated symbols it is free
    ours = ours.filter(_a_share(pl.col("exchange"), pl.col("symbol")))
    ref = reference.filter((pl.col("volume") > 0) & pl.col("is_a_share")).select(
        "exchange", "symbol", pl.col("volume").alias("r_vol"), pl.col("amount").alias("r_amt")
    )
    tol = th.get("multisource.daily_amount_mismatch_frac", {}).get("rel_tolerance", 1e-4)
    j = ours.join(ref, on=["exchange", "symbol"], how="full", coalesce=True)
    out: list[CheckResult] = []
    for (exch,), g in j.group_by("exchange"):
        both = g.filter(pl.col("vol").is_not_null() & pl.col("r_vol").is_not_null())
        n = both.height
        only_ref = g.filter(pl.col("vol").is_null()).height
        only_ours = g.filter(pl.col("r_vol").is_null()).height
        vol_bad = both.filter(pl.col("vol") != pl.col("r_vol"))
        amt_bad = both.filter(
            (pl.col("amt") - pl.col("r_amt")).abs() > tol * pl.col("r_amt").abs() + 1.0
        )
        scope = f"{source}/{exch}"
        sample = ", ".join(vol_bad.head(3)["symbol"].to_list())
        out.append(
            _result(
                th,
                "multisource.daily_volume_mismatch_frac",
                scope,
                vol_bad.height / n if n else 0.0,
                n,
                f"{vol_bad.height} stocks differ" + (f", e.g. {sample}" if sample else ""),
            )
        )
        out.append(
            _result(
                th,
                "multisource.daily_amount_mismatch_frac",
                scope,
                amt_bad.height / n if n else 0.0,
                n,
                f"{amt_bad.height} stocks differ",
            )
        )
        out.append(
            _result(
                th,
                "multisource.coverage_missing_frac",
                scope,
                only_ref / (n + only_ref) if n + only_ref else 0.0,
                n + only_ref,
                f"{only_ref} traded per {source} but absent from capture; "
                f"{only_ours} only in capture",
            )
        )
    return out
