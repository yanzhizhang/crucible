"""Daily index constituent weights -> ``index_universe`` / ``index_universe_hist`` (port of shtcommon).

One row per symbol in any tracked index that day; one float column per index (name from
:mod:`ced.indices`, e.g. ``CSI 300 Index``), weight 0-1, null when the symbol is not in that
index. Weights are the **pre-open** view of day D (only data up to D-1's close):

    anchor (AIndexHS300FreeWeight, published after T's close):  w_close[T] = official
    pre-open D:   universe[D] = normalize(w_close[D-1] * overnight_adj[D])
    after close:  w_close[D]  = normalize(w_close[D-1] * close[D] / preclose[D])

Two independent paths:

``hist`` (backtest, authoritative after the close): one SQL on AShareEODPrices gives
    overnight_adj = adj[D] / adj[D-1] and ret = close / preclose; days whose EOD is not loaded
    are not written.
``live`` (pre-open): overnight_adj from AShareEXRightDividendRecord (previous close / ex-right
    reference price, rounded to the cent like daily SOD; total-share ``_DIF`` basis); the drift
    uses published EOD returns.

Anchors are fetched from 60 days before the range (non-HS300 indices publish monthly). Trading
days follow the SZSE calendar, as in CED.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ced import db, store
from ced.calendar import SZSE, calendar
from ced.exright import apply_total_share_ratios, round_cent
from ced.indices import NAMES, WIND_CODE_MAP

log = logging.getLogger(__name__)
WIND_SUPPORTED_INDICES = frozenset(WIND_CODE_MAP)


# ----------------------------------------------------------------------------- SQL
def get_wind_weights(index_names: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Anchor snapshots -> wide (publish day x (index_name, wind_code)), 0-1."""
    wind2name = {WIND_CODE_MAP[n]: n for n in index_names}
    codes_sql = ",".join(f"'{c}'" for c in wind2name)
    sql = f"""
        SELECT
            TRADE_DT AS trading_day,
            S_INFO_WINDCODE AS index_wind_code,
            S_CON_WINDCODE AS wind_code,
            I_WEIGHT AS weight
        FROM
            WindDB.dbo.AIndexHS300FreeWeight
        WHERE
            S_INFO_WINDCODE IN ({codes_sql})
            AND TRADE_DT BETWEEN '{_forward_fetch(start_date)}' AND '{end_date}'
    """
    df = db.read_sql(sql)
    df["index_name"] = df["index_wind_code"].map(wind2name)
    return _pivot_anchors(df)


def get_wind_eod(wind_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """EOD close / preclose / adj factor (+ LAG previous adj factor), long."""
    if not wind_codes:
        return pd.DataFrame()
    codes_sql = ",".join(f"'{c}'" for c in wind_codes)
    sql = f"""
        SELECT
            TRADE_DT AS trade_dt,
            S_INFO_WINDCODE AS wind_code,
            S_DQ_CLOSE AS close_price,
            S_DQ_PRECLOSE AS prev_close,
            S_DQ_ADJFACTOR AS adj_factor,
            LAG(S_DQ_ADJFACTOR) OVER (
                PARTITION BY S_INFO_WINDCODE ORDER BY TRADE_DT
            ) AS prev_adj_factor
        FROM
            WindDB.dbo.AShareEODPrices
        WHERE
            S_INFO_WINDCODE IN ({codes_sql})
            AND TRADE_DT BETWEEN '{start_date}' AND '{end_date}'
    """
    df = db.read_sql(sql)
    if not df.empty:
        df["trade_dt"] = df["trade_dt"].astype(str)
    return df


def get_ex_dividend(wind_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Ex-right records in [start, end] (available pre-open), on the total-share basis."""
    if not wind_codes:
        return pd.DataFrame()
    codes_sql = ",".join(f"'{c}'" for c in wind_codes)
    sql = f"""
        SELECT
            S_INFO_WINDCODE AS wind_code,
            EX_DATE AS ex_date,
            CASH_DIVIDEND_RATIO AS cash,
            BONUS_SHARE_RATIO AS bonus,
            CONVERSED_RATIO AS conversed,
            RIGHTSISSUE_RATIO AS rights_ratio,
            RIGHTSISSUE_PRICE AS rights_price
        FROM
            WindDB.dbo.AShareEXRightDividendRecord
        WHERE
            S_INFO_WINDCODE IN ({codes_sql})
            AND EX_DATE >= '{start_date}' AND EX_DATE <= '{end_date}'
    """
    df = db.read_sql(sql)
    if not df.empty:
        df["ex_date"] = df["ex_date"].astype(str)
    return apply_total_share_ratios(df, start_date, end_date, tag="index_universe", wind_codes=wind_codes)


def get_wind_overnight_hist(wind_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Hist: overnight_adj = adj[D] / adj[D-1], ret_intraday = close[D] / preclose[D]; loaded days only."""
    if not wind_codes:
        return pd.DataFrame()
    codes_sql = ",".join(f"'{c}'" for c in wind_codes)
    sql = f"""
        WITH eod AS (
            SELECT
                TRADE_DT AS trade_dt,
                S_INFO_WINDCODE AS wind_code,
                S_DQ_CLOSE AS close_price,
                S_DQ_PRECLOSE AS prev_close,
                S_DQ_ADJFACTOR AS adj_factor,
                LAG(S_DQ_ADJFACTOR) OVER (
                    PARTITION BY S_INFO_WINDCODE ORDER BY TRADE_DT
                ) AS prev_adj_factor
            FROM
                WindDB.dbo.AShareEODPrices
            WHERE
                S_INFO_WINDCODE IN ({codes_sql})
                AND TRADE_DT BETWEEN '{start_date}' AND '{end_date}'
        )
        SELECT
            trade_dt,
            wind_code,
            CASE WHEN prev_close > 0 THEN close_price / prev_close END AS ret_intraday,
            CASE WHEN prev_adj_factor > 0 THEN adj_factor / prev_adj_factor END AS overnight_adj
        FROM eod
        WHERE close_price IS NOT NULL AND prev_close IS NOT NULL
    """
    df = db.read_sql(sql)
    if not df.empty:
        df["trade_dt"] = df["trade_dt"].astype(str)
    return df


# ----------------------------------------------------------------------------- drift model
def _forward_fetch(start_date: str) -> str:
    return (pd.Timestamp(start_date) - pd.Timedelta(days=60)).strftime("%Y%m%d")


def _pivot_anchors(df: pd.DataFrame) -> pd.DataFrame:
    """Long -> wide (publish day x (index_name, wind_code)), percent -> 0-1."""
    if df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["index_name", "wind_code", "weight"])
    df = df.assign(trading_day=pd.to_datetime(df["trading_day"], errors="coerce").dt.strftime("%Y%m%d")
                   ).dropna(subset=["trading_day"])
    if df.empty:
        return pd.DataFrame()
    return df.pivot_table(index="trading_day", columns=["index_name", "wind_code"], values="weight",
                          aggfunc="mean").sort_index() / 100.0


def _nz(x) -> float:
    return 0.0 if x is None or pd.isna(x) else float(x)


def _eod_panels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """EOD long -> (ret_intraday, overnight_adj, close) wide (date x code)."""
    if df.empty:
        e = pd.DataFrame()
        return e, e, e
    df = df.copy()
    df["trade_dt"] = df["trade_dt"].astype(str)
    close = df["close_price"].astype("float64")
    preclose = df["prev_close"].astype("float64")
    adj = df["adj_factor"].astype("float64")
    prev_adj = df["prev_adj_factor"].astype("float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        df["ret"] = np.where(preclose > 0, close / preclose, np.nan)
        df["oadj"] = np.where(prev_adj > 0, adj / prev_adj, np.nan)
    kw = {"index": "trade_dt", "columns": "wind_code", "aggfunc": "mean"}
    return (df.pivot_table(values="ret", **kw).sort_index(), df.pivot_table(values="oadj", **kw).sort_index(),
            df.pivot_table(values="close_price", **kw).sort_index())


def _overnight_adj_exdiv(exdiv_df: pd.DataFrame, close_wide: pd.DataFrame, trading_days: list[str]) -> pd.DataFrame:
    """Ex-right records -> pre-open overnight factor = previous close / reference price (rounded)."""
    if exdiv_df.empty or close_wide.empty:
        return pd.DataFrame()
    day_set = set(trading_days)
    prev_cache: dict[str, str] = {}
    rows: dict[str, dict[str, float]] = {}
    for rec in exdiv_df.itertuples(index=False):
        ex_date = rec.ex_date
        if ex_date not in day_set:
            continue
        code = rec.wind_code
        prev_d = prev_cache.get(ex_date)
        if prev_d is None:
            try:
                prev_d = calendar.prev(ex_date, exchange=SZSE)
            except ValueError:
                continue
            prev_cache[ex_date] = prev_d
        if prev_d not in close_wide.index or code not in close_wide.columns:
            continue
        prev_close = float(close_wide.at[prev_d, code])
        if not np.isfinite(prev_close) or prev_close <= 0:
            continue
        denom = 1.0 + _nz(rec.bonus) + _nz(rec.conversed) + _nz(rec.rights_ratio)
        if denom <= 0:
            continue
        p_ex = float(round_cent((prev_close - _nz(rec.cash) + _nz(rec.rights_price) * _nz(rec.rights_ratio)) / denom))
        if not np.isfinite(p_ex) or p_ex <= 0:
            continue
        rows.setdefault(ex_date, {})[code] = prev_close / p_ex
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame.from_dict(rows, orient="index").sort_index()


def _carry_forward_morning(anchors: pd.DataFrame, ret_intraday: pd.DataFrame, overnight_adj: pd.DataFrame,
                           trading_days: list[str], seed_date: str) -> pd.DataFrame:
    """Walk every day: 1. emit pre-open weights, 2. drift with the day's return, 3. reset on an anchor."""
    anchor_dates = set(anchors.index)
    emit_set = set(trading_days)
    full_calendar = calendar.trade_days(seed_date, trading_days[-1], SZSE)
    walk_dates = sorted(set(full_calendar) | anchor_dates)
    index_names = list(anchors.columns.get_level_values("index_name").unique())
    w_close: dict[str, pd.Series] = {}
    out_rows: dict[str, dict[tuple[str, str], float]] = {}
    for d in walk_dates:
        if w_close and d in emit_set:
            adj_row = overnight_adj.loc[d] if d in overnight_adj.index else None
            row: dict[tuple[str, str], float] = {}
            for name, w in w_close.items():
                m = w * adj_row.reindex(w.index).fillna(1.0) if adj_row is not None else w
                total = float(m.sum())
                if total > 0:
                    m = m / total
                    for code, val in m.items():
                        row[(name, code)] = float(val)
            if row:
                out_rows[d] = row
        if w_close and d in ret_intraday.index:
            ret_row = ret_intraday.loc[d]
            for name, w in list(w_close.items()):
                raw = w * ret_row.reindex(w.index).fillna(1.0)
                total = float(raw.sum())
                if total > 0:
                    w_close[name] = raw / total
        if d in anchor_dates:
            day_row = anchors.loc[d]
            for name in index_names:
                try:
                    s = day_row.xs(name, level="index_name").dropna()
                except KeyError:
                    continue
                if not s.empty:
                    w_close[name] = s.astype("float64")
    if not out_rows:
        return pd.DataFrame(index=pd.Index(trading_days, name="trading_day"))
    result = pd.DataFrame.from_dict(out_rows, orient="index")
    result.columns = pd.MultiIndex.from_tuples(result.columns, names=["index_name", "wind_code"])
    result.index.name = "trading_day"
    return result.reindex(trading_days)


def _prelim(index_codes: list[str], start_date: str, end_date: str, trading_days: list[str]):
    try:
        anchors = get_wind_weights(index_codes, start_date, end_date)
    except Exception as e:
        log.error("index_universe: fetching weights [%s, %s] failed: %s", start_date, end_date, e, exc_info=True)
        return None
    if anchors.empty:
        log.warning("index_universe: no anchor rows in [%s, %s], every day will be empty", start_date, end_date)
        return None
    if not trading_days:
        return None
    seed_date = min(anchors.index.min(), trading_days[0])
    wind_codes = sorted(set(anchors.columns.get_level_values("wind_code")))
    return anchors, wind_codes, seed_date


def bulk_hist(index_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    trading_days = calendar.trade_days(start_date, end_date, SZSE)
    empty = pd.DataFrame(index=pd.Index([], name="trading_day"))
    prelim = _prelim(index_codes, start_date, end_date, trading_days)
    if prelim is None:
        return empty
    anchors, wind_codes, seed_date = prelim
    try:
        eod = get_wind_overnight_hist(wind_codes, seed_date, end_date)
    except Exception as e:
        log.error("index_universe: hist EOD fetch failed: %s", e, exc_info=True)
        return empty
    if eod.empty:
        log.warning("index_universe: hist [%s, %s] no EOD, nothing written", seed_date, end_date)
        return empty
    last_available = max(eod["trade_dt"].unique())
    trading_days = [d for d in trading_days if d <= last_available]
    if not trading_days:
        return empty
    kw = {"index": "trade_dt", "columns": "wind_code", "aggfunc": "mean"}
    ret_intraday = eod.pivot_table(values="ret_intraday", **kw).sort_index()
    overnight_adj = eod.pivot_table(values="overnight_adj", **kw).sort_index()
    return _carry_forward_morning(anchors, ret_intraday, overnight_adj, trading_days, seed_date)


def bulk_live(index_codes: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    trading_days = calendar.trade_days(start_date, end_date, SZSE)
    empty = pd.DataFrame(index=pd.Index(trading_days, name="trading_day"))
    prelim = _prelim(index_codes, start_date, end_date, trading_days)
    if prelim is None:
        return empty
    anchors, wind_codes, seed_date = prelim
    try:
        eod = get_wind_eod(wind_codes, seed_date, end_date)
    except Exception as e:
        log.error("index_universe: live EOD fetch failed, forward-filling the last anchor: %s", e, exc_info=True)
        all_dates = sorted(set(anchors.index) | set(trading_days))
        return anchors.reindex(all_dates).ffill().reindex(trading_days)
    ret_intraday, _, close_wide = _eod_panels(eod)
    try:
        exdiv = get_ex_dividend(wind_codes, trading_days[0], end_date)
    except Exception as e:
        log.error("index_universe: ex-right fetch failed, overnight factor = 1: %s", e, exc_info=True)
        exdiv = pd.DataFrame()
    overnight_adj = _overnight_adj_exdiv(exdiv, close_wide, trading_days)
    return _carry_forward_morning(anchors, ret_intraday, overnight_adj, trading_days, seed_date)


def _index_names() -> list[str]:
    return [c for c in NAMES if c in WIND_SUPPORTED_INDICES]


def day_frame(wide: pd.DataFrame, date: str, codes: list[str]) -> pd.DataFrame:
    """One day: symbol x index weight columns (symbols present in any index that day)."""
    day = wide.loc[date].dropna() if not wide.empty and date in wide.index else pd.Series(dtype="float64")
    if day.empty:
        panel = pd.DataFrame(columns=codes, dtype="float64")
    else:
        panel = day.unstack(level="index_name").reindex(columns=codes).sort_index()
    panel = panel.astype("float64")
    panel.index.name = "symbol"
    return panel.reset_index()


def convert_range(start: str, end: str, *, overwrite: bool = True) -> dict:
    """Live, pre-open weights for every trading day in [start, end]."""
    codes = _index_names()
    wide = bulk_live(codes, start, end)
    return {d: store.write(day_frame(wide, d, codes), "index_universe", d, overwrite=overwrite)
            for d in calendar.trade_days(start, end, SZSE)}


def convert_range_hist(start: str, end: str, *, overwrite: bool = True) -> dict:
    """Hist weights; days whose EOD is not loaded yet are not written."""
    codes = _index_names()
    wide = bulk_hist(codes, start, end)
    available = set(wide.index)
    return {d: store.write(day_frame(wide, d, codes), "index_universe_hist", d, overwrite=overwrite)
            for d in calendar.trade_days(start, end, SZSE) if d in available}
