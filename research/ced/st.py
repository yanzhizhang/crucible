"""ST / delisting state -> dataset ``st`` (port of shtcommon ``ced.st``; Wind AShareST).

One row per symbol with any active ST event that day; columns ``st_<type>`` (int8 0/1) for the
Wind types in :data:`ST_TYPES` (``st_S`` = ST, ``st_Y`` = *ST, ...). Active on d:
``ENTRY_DT <= d AND (REMOVE_DT IS NULL OR REMOVE_DT > d)``; a range is one query sliced per day.
Trading days follow the SZSE calendar, as in CED.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ced import db, store
from ced.calendar import SZSE, calendar

ST_TYPES: tuple[str, ...] = ("L", "P", "Q", "R", "S", "T", "V", "X", "Y", "Z")
ST_TYPE_LABELS: dict[str, str] = {
    "L": "退市整理", "P": "PT", "Q": "强制终止挂牌风险警示", "R": "恢复上市", "S": "ST", "T": "退市",
    "V": "退市覆核", "X": "创业板暂停上市风险警示", "Y": "*ST", "Z": "暂停上市",
}


def get_events(start: str, end: str) -> pd.DataFrame:
    """All AShareST events overlapping [start, end]."""
    sql = f"""
        SELECT
            S_INFO_WINDCODE AS wind_code,
            S_TYPE_ST       AS st_type,
            ENTRY_DT        AS entry_dt,
            REMOVE_DT       AS remove_dt
        FROM dbo.AShareST
        WHERE ENTRY_DT <= '{end}'
          AND (REMOVE_DT IS NULL OR REMOVE_DT > '{start}')
    """
    df = db.read_sql(sql)
    if df.empty:
        return pd.DataFrame(columns=["wind_code", "st_type", "entry_dt", "remove_dt"])
    df["wind_code"] = df["wind_code"].astype(str)
    df["st_type"] = df["st_type"].astype(str).str.strip().str.upper()
    df["entry_dt"] = df["entry_dt"].astype(str)
    df["remove_dt"] = df["remove_dt"].fillna("").astype(str)
    return df.reset_index(drop=True)


def _slice_active(events: pd.DataFrame, date: str) -> pd.DataFrame:
    if events.empty:
        return events
    entry = events["entry_dt"].to_numpy()
    remove = events["remove_dt"].to_numpy()
    return events.loc[(entry <= date) & ((remove == "") | (remove > date))].reset_index(drop=True)


def _frame(active: pd.DataFrame) -> pd.DataFrame:
    active = active.loc[active["st_type"].isin(ST_TYPES)] if not active.empty else active
    symbols = sorted(active["wind_code"].unique().tolist()) if not active.empty else []
    pos = {s: i for i, s in enumerate(symbols)}
    arr = np.zeros((len(symbols), len(ST_TYPES)), dtype=np.int8)
    tpos = {t: j for j, t in enumerate(ST_TYPES)}
    for sym, st in zip(active.get("wind_code", []), active.get("st_type", []), strict=True):
        arr[pos[sym], tpos[st]] = 1
    out = pd.DataFrame(arr, columns=[f"st_{t}" for t in ST_TYPES])
    out.insert(0, "symbol", symbols)
    return out


def build_range(start: str, end: str) -> dict[str, pd.DataFrame]:
    days = calendar.trade_days(start, end, SZSE)
    if not days:
        return {}
    events = get_events(start, end)
    return {d: _frame(_slice_active(events, d)) for d in days}


def convert_range(start: str, end: str, *, overwrite: bool = True) -> dict:
    return {d: store.write(df, "st", d, overwrite=overwrite) for d, df in build_range(start, end).items()}
