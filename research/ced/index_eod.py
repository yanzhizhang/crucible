"""Index daily bars -> dataset ``index_eod`` (port of shtcommon ``ced.index_eod``), after the close.

One row per code in :data:`INDEX_EOD_CODES`, in that order, every day; an index without a bar
that day is a row of nulls (not dropped). A few CSI indices live under ``.CSI`` in Wind: queried
as ``.CSI``, written as ``.SH`` (:data:`WIND_SOURCE_CODE`).

Columns: open / high / low / close (points; ~31 % of indices publish only a close), vol
(S_DQ_VOLUME lots x 100 -> shares), tot (S_DQ_AMOUNT, already CNY in AIndexEODPrices -- unlike
the stock table's thousand CNY).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ced import db, store
from ced.calendar import calendar
from ced.daily import _f64, _int

log = logging.getLogger(__name__)

INDEX_EOD_CODES: tuple[str, ...] = (
    # SSE
    "000016.SH", "000300.SH", "000905.SH", "000852.SH", "000510.SH", "000001.SH", "000010.SH",
    "000009.SH", "000903.SH", "000904.SH", "000906.SH", "000985.SH", "000688.SH", "000698.SH",
    "000015.SH", "000922.SH", "000934.SH", "000928.SH", "000932.SH", "000933.SH", "000935.SH",
    # SZSE
    "399300.SZ", "399905.SZ", "399852.SZ", "399001.SZ", "399004.SZ", "399005.SZ", "399006.SZ",
    "399673.SZ", "399311.SZ", "399303.SZ", "399324.SZ", "399986.SZ", "399975.SZ", "399809.SZ",
    "399997.SZ",
)
WIND_SOURCE_CODE: dict[str, str] = {"000904.SH": "000904.CSI", "000922.SH": "000922.CSI"}
IDX_EOD_FIELDS = ["open", "high", "low", "close", "vol", "tot"]


def get_index_eod(start: str, end: str, codes: tuple[str, ...]) -> pd.DataFrame:
    """AIndexEODPrices rows in [start, end] for ``codes`` (our codes in, our codes out)."""
    to_ours = {WIND_SOURCE_CODE.get(c, c): c for c in codes}
    in_list = ",".join(f"'{c}'" for c in to_ours)
    sql = f"""
        SELECT
            TRADE_DT        AS trade_dt,
            S_INFO_WINDCODE AS wind_code,
            S_DQ_OPEN       AS [open],
            S_DQ_HIGH       AS high,
            S_DQ_LOW        AS low,
            S_DQ_CLOSE      AS [close],
            S_DQ_VOLUME     AS volume_lot,
            S_DQ_AMOUNT     AS amount
        FROM dbo.AIndexEODPrices
        WHERE TRADE_DT >= '{start}' AND TRADE_DT <= '{end}'
          AND S_INFO_WINDCODE IN ({in_list})
    """
    df = db.read_sql(sql, what="AIndexEODPrices")
    if not df.empty:
        df["trade_dt"] = df["trade_dt"].astype(str)
        df["wind_code"] = df["wind_code"].astype(str).str.strip().map(lambda c: to_ours.get(c, c))
        for c in ("open", "high", "low", "close", "volume_lot", "amount"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _build_one(date: str, day_df: pd.DataFrame, codes: tuple[str, ...]) -> pd.DataFrame:
    d = day_df.drop_duplicates("wind_code", keep="last").set_index("wind_code").reindex(list(codes))
    missing = d["close"].isna()
    if missing.any():
        log.warning("idx_EOD(%s): %d/%d indices without a close (not loaded / discontinued), null row: %s",
                    date, int(missing.sum()), len(codes), list(d.index[missing])[:10])
    return pd.DataFrame({
        "symbol": list(codes),
        "open": _f64(d, "open"), "high": _f64(d, "high"), "low": _f64(d, "low"), "close": _f64(d, "close"),
        "vol": _int(_f64(d, "volume_lot"), 100.0),
        "tot": _f64(d, "amount").astype(np.float64),
    })


def build_range(start: str, end: str, codes: tuple[str, ...] = INDEX_EOD_CODES, *,
                strict: bool = False) -> dict[str, pd.DataFrame]:
    days = calendar.trade_days(start, end)
    if not days:
        return {}
    df = get_index_eod(days[0], days[-1], codes)
    by_date = {dt: g for dt, g in df.groupby("trade_dt", sort=False)} if not df.empty else {}
    out: dict[str, pd.DataFrame] = {}
    for d in days:
        if d not in by_date:
            msg = f"idx_EOD({d}): AIndexEODPrices has no rows (before the close?)"
            if strict:
                raise RuntimeError(msg)
            log.warning("%s, skipped", msg)
            continue
        out[d] = _build_one(d, by_date[d], codes)
    return out


def convert_range(start: str, end: str, *, overwrite: bool = True, strict: bool = False) -> dict:
    return {d: store.write(df, "index_eod", d, overwrite=overwrite)
            for d, df in build_range(start, end, strict=strict).items()}
