"""Daily SOD / EOD panels from Wind (port of shtcommon ``ced.daily``; rules and SQL unchanged).

Datasets (one row per symbol per trading day):

``daily_sod``       live SOD, computable before the open from T-1 data + rules
``daily_sod_hist``  hist SOD: every field ``AShareEODPrices`` has is taken from its row T
                    after the close (listing = an EOD row exists, suspension = its status code,
                    limits, preclose, adj factors); only shares come from DerivativeIndicator.
                    The authoritative answer the live SOD tries to reproduce (in CED both were
                    written to the same file; here they are two datasets, so neither is lost)
``daily_eod``       EOD: raw OHLC, preclose, adj_factor, limits, vol (shares), tot (CNY), vwap,
                    ret_o_pc, ret_c_o

SOD columns: is_traded (int8), maxp_allowed, minp_allowed (0 = no price band, new listing;
NaN = no data), freeshare / circshare / totshare (shares, Int64; Wind 万股 x 1e4), ret_adj
(ex-right reference price / previous close; 1.0 without an ex-right event and on day one).

Live SOD(T) (``_iter_sod``):
    1. base = row T-1: DerivativeIndicator shares + EODPrices close / preclose / limits +
       Description board / list / delist dates                                 get_sod_base
    2. add T's first-day listings (no T-1 row)                           AShareIPO _with_ipo_rows
    3. shares switched to T's values when a change takes effect on T
                                         AShareCapitalization / AShareFreeFloat _with_shares_asof
    4. ex-right / suspension / listing / exchange rules -> columns              _build_sod_one

EOD units: S_DQ_VOLUME lots (x100 -> shares), S_DQ_AMOUNT thousand CNY (x1000 -> CNY),
S_DQ_PRECLOSE already carries the pre-suspension close through a suspension.

Range entry points take a closed interval; each source is fetched once for the whole range
and sliced per day, so a single day and a range give identical results. Days whose source rows
are missing are skipped with a WARNING (``strict=True`` raises).
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd

from ced import db, store
from ced.calendar import calendar
from ced.exright import apply_total_share_ratios

log = logging.getLogger(__name__)

SOD_FIELDS = ["is_traded", "maxp_allowed", "minp_allowed", "freeshare", "circshare", "totshare", "ret_adj"]
EOD_FIELDS = ["open", "high", "low", "close", "preclose", "adj_factor", "maxp_allowed", "minp_allowed",
              "vol", "tot", "vwap", "ret_o_pc", "ret_c_o"]
_SHARE_COLS = ("freeshare", "circshare", "totshare")


# ----------------------------------------------------------------------------- SQL
class DailySQL:
    """Wind daily queries, all on closed intervals."""

    @staticmethod
    def get_sod_base(start: str, end: str) -> pd.DataFrame:
        """Live base: rows with TRADE_DT in [start, end] (callers pass the day before the target)."""
        sql = f"""
            SELECT
                s.TRADE_DT AS trade_dt,
                s.S_INFO_WINDCODE AS wind_code,
                s.FREE_SHARES_TODAY AS freeshare,
                s.FLOAT_A_SHR_TODAY AS circshare,
                s.TOT_SHR_TODAY AS totshare,
                e.S_DQ_CLOSE AS prev_close,
                e.S_DQ_PRECLOSE AS prev_preclose,
                e.S_DQ_LIMIT AS prev_limit,
                e.S_DQ_STOPPING AS prev_stopping,
                d.S_INFO_LISTBOARDNAME AS list_board,
                d.S_INFO_LISTDATE AS list_date,
                d.S_INFO_DELISTDATE AS delist_date
            FROM dbo.AShareEODDerivativeIndicator s
                LEFT JOIN dbo.AShareEODPrices e
                    ON s.S_INFO_WINDCODE = e.S_INFO_WINDCODE
                    AND s.TRADE_DT = e.TRADE_DT
                LEFT JOIN dbo.AShareDescription d
                    ON s.S_INFO_WINDCODE = d.S_INFO_WINDCODE
            WHERE s.TRADE_DT >= '{start}' AND s.TRADE_DT <= '{end}'
        """
        return _clean_dates(db.read_sql(sql), ("trade_dt", "list_board", "list_date", "delist_date"))

    @staticmethod
    def get_sod_base_hist(start: str, end: str) -> pd.DataFrame:
        """Hist base: EODPrices rows in [start, end]; prev_adj_factor by LAG, so start one day early."""
        sql = f"""
            SELECT
                e.TRADE_DT AS trade_dt,
                e.S_INFO_WINDCODE AS wind_code,
                e.S_DQ_PRECLOSE AS prev_close,
                e.S_DQ_LIMIT AS maxp_allowed,
                e.S_DQ_STOPPING AS minp_allowed,
                e.S_DQ_ADJFACTOR AS adj_factor,
                LAG(e.S_DQ_ADJFACTOR) OVER (
                    PARTITION BY e.S_INFO_WINDCODE ORDER BY e.TRADE_DT
                ) AS prev_adj_factor,
                e.S_DQ_TRADESTATUSCODE AS trade_status_code,
                s.FREE_SHARES_TODAY AS freeshare,
                s.FLOAT_A_SHR_TODAY AS circshare,
                s.TOT_SHR_TODAY AS totshare,
                d.S_INFO_LISTBOARDNAME AS list_board,
                d.S_INFO_LISTDATE AS list_date,
                d.S_INFO_DELISTDATE AS delist_date
            FROM dbo.AShareEODPrices e
                LEFT JOIN dbo.AShareEODDerivativeIndicator s
                    ON e.S_INFO_WINDCODE = s.S_INFO_WINDCODE
                    AND e.TRADE_DT = s.TRADE_DT
                LEFT JOIN dbo.AShareDescription d
                    ON e.S_INFO_WINDCODE = d.S_INFO_WINDCODE
            WHERE e.TRADE_DT >= '{start}' AND e.TRADE_DT <= '{end}'
        """
        return _clean_dates(db.read_sql(sql), ("trade_dt", "list_board", "list_date", "delist_date"))

    @staticmethod
    def get_ex_dividend(start: str, end: str) -> pd.DataFrame:
        """Ex-right records with EX_DATE in [start, end], ratios switched to the total-share basis."""
        sql = f"""
            SELECT
                S_INFO_WINDCODE AS wind_code,
                EX_DATE AS ex_date,
                CASH_DIVIDEND_RATIO AS cash,
                BONUS_SHARE_RATIO AS bonus,
                CONVERSED_RATIO AS conversed,
                RIGHTSISSUE_RATIO AS rights_ratio,
                RIGHTSISSUE_PRICE AS rights_price
            FROM dbo.AShareEXRightDividendRecord
            WHERE EX_DATE >= '{start}' AND EX_DATE <= '{end}'
        """
        df = _clean_dates(db.read_sql(sql), ("ex_date",))
        return apply_total_share_ratios(df, start, end, tag="SOD")

    @staticmethod
    def _share_change_history(table: str, value_sql: dict, lo: str, end: str, *,
                              valid_only: bool = False) -> pd.DataFrame:
        """Share-change table: symbols whose max(CHANGE_DT, ANN_DT) falls in [lo, end], full history
        to ``end`` (the as-of value is the latest CHANGE_DT among effective records; OPDATE ignored)."""
        vals = ",\n                ".join(f"t.{src} AS {dst}" for dst, src in value_sql.items())
        valid = "AND IS_VALID = 1" if valid_only else ""
        sql = f"""
            SELECT
                t.S_INFO_WINDCODE AS wind_code,
                t.CHANGE_DT       AS change_dt,
                t.ANN_DT          AS ann_dt,
                {vals}
            FROM dbo.{table} t
            WHERE t.CHANGE_DT <= '{end}' AND t.ANN_DT <= '{end}' {valid.replace("IS_VALID", "t.IS_VALID")}
              AND t.S_INFO_WINDCODE IN (
                  SELECT S_INFO_WINDCODE FROM dbo.{table}
                  WHERE (CASE WHEN ANN_DT > CHANGE_DT THEN ANN_DT ELSE CHANGE_DT END)
                        BETWEEN '{lo}' AND '{end}' {valid})
        """
        df = db.read_sql(sql, what=table)
        cols = ["wind_code", "change_dt", "ann_dt", "eff_dt", *value_sql]
        if df.empty:
            return pd.DataFrame(columns=cols)
        for c in ("wind_code", "change_dt", "ann_dt"):
            df[c] = df[c].fillna("").astype(str).str.strip()
        for c in value_sql:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["eff_dt"] = np.where(df["ann_dt"] > df["change_dt"], df["ann_dt"], df["change_dt"])
        return df[cols].reset_index(drop=True)

    @staticmethod
    def get_freefloat_history(lo: str, end: str) -> pd.DataFrame:
        return DailySQL._share_change_history("AShareFreeFloat", {"freeshare": "S_SHARE_FREESHARES"}, lo, end)

    @staticmethod
    def get_capitalization_history(lo: str, end: str) -> pd.DataFrame:
        """IS_VALID = 1; CHANGE_DT is the listing date of the change (CHANGE_DT1, the record date, unused)."""
        return DailySQL._share_change_history(
            "AShareCapitalization", {"totshare": "TOT_SHR", "circshare": "FLOAT_A_SHR"}, lo, end, valid_only=True)

    @staticmethod
    def get_ipo_listings(start: str, end: str) -> pd.DataFrame:
        """AShareIPO listings in [start, end] that did not fail."""
        sql = f"""
            SELECT
                i.S_INFO_WINDCODE      AS wind_code,
                i.S_IPO_LISTDATE       AS list_date,
                i.S_IPO_PRICE          AS ipo_price,
                d.S_INFO_LISTBOARDNAME AS list_board,
                d.S_INFO_DELISTDATE    AS delist_date
            FROM dbo.AShareIPO i
                LEFT JOIN dbo.AShareDescription d
                    ON i.S_INFO_WINDCODE = d.S_INFO_WINDCODE
            WHERE i.S_IPO_LISTDATE >= '{start}' AND i.S_IPO_LISTDATE <= '{end}'
              AND ISNULL(i.IS_FAILURE, 0) = 0
        """
        df = db.read_sql(sql, what="AShareIPO")
        cols = ["wind_code", "list_date", "ipo_price", "list_board", "delist_date"]
        if df.empty:
            return pd.DataFrame(columns=cols)
        for c in ("wind_code", "list_date", "list_board", "delist_date"):
            df[c] = df[c].fillna("").astype(str).str.strip()
        df["ipo_price"] = pd.to_numeric(df["ipo_price"], errors="coerce")
        return df[cols].drop_duplicates("wind_code", keep="last").reset_index(drop=True)

    @staticmethod
    def get_suspension(start: str, end: str) -> pd.DataFrame:
        """Whole-day suspensions (444003000 / 444008000 / 444016000 / 444017000), one row per day."""
        sql = f"""
            SELECT
                S_INFO_WINDCODE AS wind_code,
                S_DQ_SUSPENDDATE AS suspend_date
            FROM dbo.AShareTradingSuspension
            WHERE S_DQ_SUSPENDTYPE IN (444003000, 444008000, 444016000, 444017000)
                AND S_DQ_SUSPENDDATE >= '{start}'
                AND S_DQ_SUSPENDDATE <= '{end}'
        """
        return _clean_dates(db.read_sql(sql), ("suspend_date",))

    @staticmethod
    def get_eod(start: str, end: str) -> pd.DataFrame:
        """EOD: AShareEODPrices row T (loaded 15:00-16:00)."""
        sql = f"""
            SELECT
                e.TRADE_DT AS trade_dt,
                e.S_INFO_WINDCODE AS wind_code,
                e.S_DQ_OPEN AS [open],
                e.S_DQ_HIGH AS high,
                e.S_DQ_LOW AS low,
                e.S_DQ_CLOSE AS [close],
                e.S_DQ_PRECLOSE AS preclose,
                e.S_DQ_ADJFACTOR AS adj_factor,
                e.S_DQ_LIMIT AS maxp_allowed,
                e.S_DQ_STOPPING AS minp_allowed,
                e.S_DQ_VOLUME AS volume_lot,
                e.S_DQ_AMOUNT AS amount_kyuan,
                e.S_DQ_AVGPRICE AS vwap,
                d.S_INFO_LISTDATE AS list_date
            FROM dbo.AShareEODPrices e
                LEFT JOIN dbo.AShareDescription d
                    ON e.S_INFO_WINDCODE = d.S_INFO_WINDCODE
            WHERE e.TRADE_DT >= '{start}' AND e.TRADE_DT <= '{end}'
        """
        return _clean_dates(db.read_sql(sql), ("trade_dt", "list_date"))


def _clean_dates(df: pd.DataFrame, cols: tuple) -> pd.DataFrame:
    """Date / text columns to str, NULL -> ''."""
    if df.empty:
        return df
    for c in cols:
        if c in df.columns:
            df[c] = df[c].fillna("").astype(str)
    return df


def _num(d: pd.DataFrame, col: str) -> np.ndarray:
    return pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=np.float64)


def _f64(d: pd.DataFrame, col: str) -> np.ndarray:
    return d[col].to_numpy(dtype=np.float64)


def _int(values: np.ndarray, scale: float = 1.0) -> pd.array:
    """Scale, round, nullable Int64 (NaN -> null; CED wrote the -1 sentinel here)."""
    v = np.asarray(values, dtype=np.float64) * scale
    return pd.array(np.where(np.isfinite(v), np.round(v), np.nan), dtype="Float64").astype("Int64")


# ----------------------------------------------------------------------------- price rules
def _round_half_up(x: np.ndarray) -> np.ndarray:
    """Round half up to the cent (not banker's: 3.75*1.1 = 4.125 -> 4.13); +1e-9 for binary error."""
    return np.floor(x * 100.0 + 0.5 + 1e-9) / 100.0


def _floor_cent(x: np.ndarray) -> np.ndarray:
    return np.floor(x * 100.0 + 1e-9) / 100.0


def _ceil_cent(x: np.ndarray) -> np.ndarray:
    return np.ceil(x * 100.0 - 1e-9) / 100.0


def _is_bj(symbols: list[str], boards: np.ndarray) -> np.ndarray:
    return np.array([s.endswith(".BJ") or "北交" in (b or "") or "北证" in (b or "")
                     for s, b in zip(symbols, boards, strict=True)], dtype=np.bool_)


def _limit_pct(symbols: list[str], boards: np.ndarray) -> np.ndarray:
    """Board rule: main 10 %, ChiNext / STAR 20 %, BSE 30 % (ST not narrowed; checked 2026-09-21)."""
    pct = np.full(len(symbols), 0.10)
    for i, code in enumerate(symbols):
        head, _, suffix = code.partition(".")
        bn = boards[i] or ""
        if suffix == "BJ" or "北交" in bn or "北证" in bn:
            pct[i] = 0.30
        elif head.startswith("688") or "科创" in bn:
            pct[i] = 0.20
        elif head.startswith(("300", "301")) or "创业" in bn:
            pct[i] = 0.20
    return pct


_VALID_PCTS = np.array([0.05, 0.10, 0.20, 0.30])


def _observed_pct(d: pd.DataFrame, symbols: list[str], boards: np.ndarray, date: str) -> np.ndarray:
    """Per symbol: the exchange's own T-1 S_DQ_LIMIT / S_DQ_PRECLOSE - 1 snapped to 5/10/20/30 %;
    the board rule only where T-1 has none. Disagreements logged (WARNING above 1 %)."""
    rule = _limit_pct(symbols, boards)
    lim = _num(d, "prev_limit")
    pre = _num(d, "prev_preclose")
    with np.errstate(divide="ignore", invalid="ignore"):
        raw = lim / pre - 1.0
    ok = np.isfinite(raw) & (lim > 0) & (pre > 0)
    snapped = np.full(len(raw), np.nan)
    if ok.any():
        near = _VALID_PCTS[np.abs(raw[ok, None] - _VALID_PCTS[None, :]).argmin(axis=1)]
        snapped[ok] = np.where(np.abs(raw[ok] - near) < 0.01, near, np.nan)
    use_obs = np.isfinite(snapped)
    differ = use_obs & (np.abs(snapped - rule) > 1e-9)
    n_diff = int(differ.sum())
    if n_diff:
        n_obs = int(use_obs.sum())
        sample = [f"{symbols[i]} obs={snapped[i]:.0%} rule={rule[i]:.0%}" for i in np.flatnonzero(differ)[:8]]
        lvl = logging.WARNING if n_diff > 0.01 * max(n_obs, 1) else logging.INFO
        log.log(lvl, "SOD(%s): %d/%d symbols' T-1 exchange limit differs from the board rule, using the "
                "exchange's: %s%s", date, n_diff, n_obs, sample,
                " ... update _limit_pct" if lvl == logging.WARNING else "")
    return np.where(use_obs, snapped, rule)


_IPO_NO_BAND_DAYS = 5
_IPO_NO_BAND_DAYS_BJ = 1


def _no_band(d: pd.DataFrame, symbols: list[str], list_arr: np.ndarray, boards: np.ndarray, date: str) -> np.ndarray:
    """No price band on T: n-th trading day since listing (day one = 1), SH/SZ n <= 5, BSE n <= 1.
    Only when the listing date is unknown: T-1 limit == 0 (0 = no band, NULL = no data)."""
    bj = _is_bj(symbols, boards)
    lo = (pd.Timestamp(date) - pd.Timedelta(days=30)).strftime("%Y%m%d")
    out = np.zeros(len(symbols), dtype=bool)
    for i, ld in enumerate(list_arr):
        ld = str(ld or "").split(".")[0].strip()
        if len(ld) != 8 or not ld.isdigit() or ld > date or ld < lo:
            continue
        out[i] = len(calendar.trade_days(ld, date)) <= (_IPO_NO_BAND_DAYS_BJ if bj[i] else _IPO_NO_BAND_DAYS)
    lim = _num(d, "prev_limit")
    return out | ((list_arr == "") & np.isfinite(lim) & (lim == 0.0))


# ----------------------------------------------------------------------------- live base
_SHARE_SOURCES = (
    ("get_freefloat_history", ("freeshare",), "AShareFreeFloat"),
    ("get_capitalization_history", ("totshare", "circshare"), "AShareCapitalization"),
)


def _load_share_sources(lo: str, end: str) -> list:
    """[(history, columns, table)]; a failed fetch keeps the T-1 values for those columns (ERROR)."""
    out = []
    for fetch, cols, what in _SHARE_SOURCES:
        try:
            df = getattr(DailySQL, fetch)(lo, end)
        except Exception:
            log.error("SOD: fetching %s failed, %s keep the T-1 derivative-indicator values", what,
                      "/".join(cols), exc_info=True)
            df = pd.DataFrame(columns=["wind_code", "change_dt", "ann_dt", "eff_dt", *cols])
        out.append((df, cols, what))
    return out


def _load_ipo(start: str, end: str) -> pd.DataFrame:
    try:
        return DailySQL.get_ipo_listings(start, end)
    except Exception:
        log.error("SOD: fetching AShareIPO failed, first-day listings are missing", exc_info=True)
        return pd.DataFrame(columns=["wind_code", "list_date", "ipo_price", "list_board", "delist_date"])


def _with_ipo_rows(base_df: pd.DataFrame, ipo: pd.DataFrame, date: str) -> pd.DataFrame:
    """Add T's first-day listings (board / list / delist only; the rest from the rules)."""
    if ipo.empty:
        return base_df
    new = ipo[(ipo["list_date"] == date) & ~ipo["wind_code"].isin(base_df.index)]
    if new.empty:
        return base_df
    rows = new.set_index("wind_code")[["list_board", "list_date", "delist_date"]].reindex(columns=base_df.columns)
    log.info("SOD(%s): %d first-day listings added (AShareIPO): %s", date, len(rows),
             [f"{s} ipo {p:g}" for s, p in zip(new["wind_code"], new["ipo_price"], strict=True)][:8])
    return pd.concat([base_df, rows])


def _with_shares_asof(base_df: pd.DataFrame, sources: list, prev_d: str, date: str) -> pd.DataFrame:
    """Switch T-1 shares to T's values where a change becomes effective in (T-1, T].

    The derivative table switches on max(CHANGE_DT, ANN_DT), rolled to the next trading day
    (verified 2026-09-23: 000600 / 000703 / 688347 / 688549). Value = latest CHANGE_DT among
    records effective by T. First-day listings take it regardless of the effective date.
    """
    out = base_df
    first_day = base_df.index[base_df["list_date"] == date]
    for src, cols, what in sources:
        if src.empty or out.empty:
            continue
        known = src[(src["change_dt"] <= date) & (src["ann_dt"] <= date)]
        asof = known.sort_values(["wind_code", "change_dt", "ann_dt"]).drop_duplicates("wind_code", keep="last")
        take = (asof["eff_dt"] > prev_d) | asof["wind_code"].isin(first_day)
        asof = asof[take & asof["wind_code"].isin(out.index)].set_index("wind_code")
        if asof.empty:
            continue
        if out is base_df:
            out = base_df.copy()
        for c in cols:
            new = asof[c].dropna()
            if new.empty:
                continue
            old = pd.to_numeric(out.loc[new.index, c], errors="coerce")
            out.loc[new.index, c] = new.to_numpy()
            changed = (old - new).abs() > 1e-6
            if changed.any():
                log.info("SOD(%s): %d %s switched to T's value from %s (10k shares): %s", date,
                         int(changed.sum()), c, what,
                         [f"{s} {old[s]:g}->{new[s]:g}" for s in new.index[changed][:8]])
    return out


def _exdiv_on(exdiv: pd.DataFrame, date: str) -> pd.DataFrame:
    if exdiv.empty:
        return exdiv.iloc[0:0].set_index("wind_code")
    return exdiv[exdiv["ex_date"] == date].drop_duplicates("wind_code").set_index("wind_code")


def _suspended_set(susp: pd.DataFrame, date: str) -> set[str]:
    if susp.empty:
        return set()
    return set(susp.loc[susp["suspend_date"] == date, "wind_code"])


# ----------------------------------------------------------------------------- one day
def _sod_frame(symbols: list[str], d: pd.DataFrame, is_traded, maxp, minp, ret_adj) -> pd.DataFrame:
    """SOD panel; shares 10k -> shares, nullable Int64."""
    out = pd.DataFrame({"symbol": symbols, "is_traded": np.asarray(is_traded, dtype=np.int8),
                        "maxp_allowed": np.asarray(maxp, dtype=np.float64),
                        "minp_allowed": np.asarray(minp, dtype=np.float64)})
    for c in _SHARE_COLS:
        out[c] = _int(_f64(d, c), 1e4)
    out["ret_adj"] = np.asarray(ret_adj, dtype=np.float64)
    return out[["symbol", *SOD_FIELDS]]


def _build_sod_one(date: str, base_df: pd.DataFrame, exdiv_day: pd.DataFrame, susp_set: set[str],
                   symbols: list[str] | None) -> pd.DataFrame:
    """Live SOD(T) from the prepared base (T-1 rows + first-day listings + T shares)."""
    if symbols is None:
        symbols = sorted(base_df.index.unique().tolist())
    d = base_df.reindex(symbols)

    prev_close = _f64(d, "prev_close")
    boards = d["list_board"].fillna("").to_numpy(dtype=object)
    list_arr = d["list_date"].fillna("").to_numpy(dtype=object)
    delist_arr = d["delist_date"].fillna("").to_numpy(dtype=object)
    first_day = list_arr == date
    valid_pc = np.isfinite(prev_close)

    # ex-right reference price rounded to the cent (the exchange uses the rounded price)
    ex = exdiv_day.reindex(symbols)
    has_ex = ex["ex_date"].notna().to_numpy()
    cash, bonus, conversed, rratio, rprice = (
        ex[c].fillna(0.0).to_numpy(dtype=np.float64)
        for c in ("cash", "bonus", "conversed", "rights_ratio", "rights_price"))
    with np.errstate(divide="ignore", invalid="ignore"):
        p_ex = _round_half_up((prev_close - cash + rprice * rratio) / (1.0 + bonus + conversed + rratio))
        ratio = p_ex / prev_close
    has_valid_ex = has_ex & valid_pc & np.isfinite(ratio) & (ratio > 0)
    ret_adj = np.where(has_valid_ex & ~first_day, ratio, 1.0)

    # limits: BSE up floors / down ceils, the rest round half up; no band (new listing) -> 0
    limit_base = np.where(has_valid_ex, p_ex, prev_close)
    pct = _observed_pct(d, symbols, boards, date)
    bj = _is_bj(symbols, boards)
    with np.errstate(invalid="ignore"):
        raw_max = limit_base * (1.0 + pct)
        raw_min = limit_base * (1.0 - pct)
        maxp = np.where(valid_pc, np.where(bj, _floor_cent(raw_max), _round_half_up(raw_max)), np.nan)
        minp = np.where(valid_pc, np.where(bj, _ceil_cent(raw_min), _round_half_up(raw_min)), np.nan)
    no_band = _no_band(d, symbols, list_arr, boards, date)
    if no_band.any():
        maxp = np.where(no_band, 0.0, maxp)
        minp = np.where(no_band, 0.0, minp)
        log.info("SOD(%s): %d new listings inside the no-band window, limits written as 0: %s", date,
                 int(no_band.sum()), [symbols[i] for i in np.flatnonzero(no_band)[:8]])

    # delist_date is the first non-trading day: strict >
    listed = (valid_pc | first_day) & ((delist_arr == "") | (delist_arr > date))
    suspended = np.array([s in susp_set for s in symbols], dtype=np.bool_)
    is_traded = (listed & ~suspended).astype(np.int8)
    return _sod_frame(symbols, d, is_traded, maxp, minp, ret_adj)


_TRADE_STATUS_SUSPENDED = 0
_TRADE_STATUS_UNVERIFIED = -2


def _build_sod_hist_one(date: str, base_df: pd.DataFrame, symbols: list[str] | None) -> pd.DataFrame:
    """Hist SOD(T): every field AShareEODPrices has comes from its row T -- nothing from other tables.

    Listed = the symbol has an EOD row on T (not AShareDescription's list / delist dates);
    is_traded = listed and S_DQ_TRADESTATUSCODE != 0 (-2 unverified / NULL count as trading,
    listed per day as a WARNING). ret_adj = adj[T-1] / adj[T]; a first trading day has no
    previous factor (LAG is NULL) and gets 1.0. Limits and preclose are the EOD fields.
    Only the three share columns come from elsewhere (DerivativeIndicator): EOD has none.
    """
    if symbols is None:
        symbols = sorted(base_df.index.unique().tolist())
    d = base_df.reindex(symbols)

    listed = np.isin(np.asarray(symbols, dtype=object), base_df.index.to_numpy(dtype=object))
    first_day = np.zeros(len(symbols), dtype=bool)  # handled by the NULL LAG below
    status = _num(d, "trade_status_code")
    unsure = listed & (~np.isfinite(status) | (status == _TRADE_STATUS_UNVERIFIED))
    if unsure.any():
        log.warning("SOD_hist(%s): %d symbols with S_DQ_TRADESTATUSCODE -2/NULL counted as trading: %s",
                    date, int(unsure.sum()), [symbols[i] for i in np.flatnonzero(unsure)[:10]])
    is_traded = (listed & (status != _TRADE_STATUS_SUSPENDED)).astype(np.int8)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = _f64(d, "prev_adj_factor") / _f64(d, "adj_factor")
    ret_adj = np.where(np.isfinite(ratio) & (ratio > 0) & ~first_day, ratio, 1.0)
    return _sod_frame(symbols, d, is_traded, _f64(d, "maxp_allowed"), _f64(d, "minp_allowed"), ret_adj)


def _build_eod_one(date: str, day_df: pd.DataFrame, symbols: list[str] | None) -> pd.DataFrame:
    if symbols is None:
        symbols = sorted(day_df.index.unique().tolist())
    d = day_df.reindex(symbols)
    open_ = _f64(d, "open")
    close = _f64(d, "close")
    first_day = d["list_date"].fillna("").to_numpy(dtype=object) == date
    with np.errstate(divide="ignore", invalid="ignore"):
        ret_o_pc = np.where(first_day, 1.0, open_ / _f64(d, "preclose"))
        ret_c_o = np.where(open_ > 0, close / open_, np.nan)
    return pd.DataFrame({
        "symbol": symbols,
        "open": open_,
        "high": _f64(d, "high"),
        "low": _f64(d, "low"),
        "close": close,
        "preclose": _f64(d, "preclose"),
        "adj_factor": _f64(d, "adj_factor"),
        "maxp_allowed": _f64(d, "maxp_allowed"),
        "minp_allowed": _f64(d, "minp_allowed"),
        "vol": _int(_f64(d, "volume_lot"), 100.0),
        "tot": _f64(d, "amount_kyuan") * 1000.0,
        "vwap": _f64(d, "vwap"),
        "ret_o_pc": np.where(np.isfinite(ret_o_pc), ret_o_pc, np.nan),
        "ret_c_o": np.where(np.isfinite(ret_c_o), ret_c_o, np.nan),
    })


# ----------------------------------------------------------------------------- ranges
def _slice_by_trade_date(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    return {dt: grp.drop(columns="trade_dt").set_index("wind_code") for dt, grp in df.groupby("trade_dt", sort=False)}


def _trade_days_with_prev(start: str, end: str) -> tuple[list[str], dict[str, str]]:
    days = calendar.trade_days(start, end)
    if not days:
        return [], {}
    return days, dict(zip(days, [calendar.prev(days[0]), *days[:-1]], strict=True))


def _missing(by_date: dict, key: str, msg: str, strict: bool) -> bool:
    df = by_date.get(key)
    if df is not None and not df.empty:
        return False
    if strict:
        raise RuntimeError(msg)
    log.warning("%s, skipped", msg)
    return True


def iter_sod(start: str, end: str, symbols=None, strict: bool = False) -> Iterator[tuple[str, pd.DataFrame]]:
    """Live SOD per day; days whose base (T-1 rows) is missing are skipped."""
    days, prev_of = _trade_days_with_prev(start, end)
    if not days:
        return
    src_lo, src_hi = prev_of[days[0]], prev_of[days[-1]]
    base_by_date = _slice_by_trade_date(DailySQL.get_sod_base(src_lo, src_hi))
    exdiv = DailySQL.get_ex_dividend(start, end)
    susp = DailySQL.get_suspension(start, end)
    shares = _load_share_sources(src_lo, end)
    ipo = _load_ipo(start, end)
    for d in days:
        prev_d = prev_of[d]
        if _missing(base_by_date, prev_d, f"SOD({d}): base (TRADE_DT={prev_d}) has no rows", strict):
            continue
        base = _with_shares_asof(_with_ipo_rows(base_by_date[prev_d], ipo, d), shares, prev_d, d)
        yield d, _build_sod_one(d, base, _exdiv_on(exdiv, d), _suspended_set(susp, d), symbols)


def iter_sod_hist(start: str, end: str, symbols=None, strict: bool = False) -> Iterator[tuple[str, pd.DataFrame]]:
    """Hist SOD per day; fetched from the day before the first target so LAG has its value."""
    days, prev_of = _trade_days_with_prev(start, end)
    if not days:
        return
    base_by_date = _slice_by_trade_date(DailySQL.get_sod_base_hist(prev_of[days[0]], end))
    for d in days:
        if _missing(base_by_date, d, f"SOD_hist({d}): AShareEODPrices has no rows (before the close?)", strict):
            continue
        yield d, _build_sod_hist_one(d, base_by_date[d], symbols)


def iter_eod(start: str, end: str, symbols=None, strict: bool = False) -> Iterator[tuple[str, pd.DataFrame]]:
    days, _ = _trade_days_with_prev(start, end)
    if not days:
        return
    by_date = _slice_by_trade_date(DailySQL.get_eod(start, end))
    for d in days:
        if _missing(by_date, d, f"EOD({d}): EODPrices has no rows (before the close?)", strict):
            continue
        yield d, _build_eod_one(d, by_date[d], symbols)


def _save(it: Iterator[tuple[str, pd.DataFrame]], dataset: str, overwrite: bool) -> dict[str, object]:
    return {d: store.write(df, dataset, d, overwrite=overwrite) for d, df in it}


def convert_range_sod(start: str, end: str, *, overwrite: bool = True, strict: bool = False) -> dict:
    return _save(iter_sod(start, end, None, strict), "daily_sod", overwrite)


def convert_range_sod_hist(start: str, end: str, *, overwrite: bool = True, strict: bool = False) -> dict:
    return _save(iter_sod_hist(start, end, None, strict), "daily_sod_hist", overwrite)


def convert_range_eod(start: str, end: str, *, overwrite: bool = True, strict: bool = False) -> dict:
    return _save(iter_eod(start, end, None, strict), "daily_eod", overwrite)
