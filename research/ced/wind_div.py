"""Cash dividends taking effect today -> dataset ``wind_div`` (port of shtcommon ``ced.wind_div``).

Day ``d`` holds dividends whose ex-date ``EX_DT == d`` (not announcements), implemented plans
only (``S_DIV_PROGRESS = '3'``). Columns ``cash_dvd_per_sh_pre_tax`` / ``_after_tax`` (CNY per
share). Pure bonus / conversion events stay as 0 rows ("an event without cash" differs from
"no event"); symbols without an event are absent (read as 0). Several plans on one day are
added; differentiated rows (object not all holders) are dropped -- :func:`aggregate_per_share`,
shared with ``zy_div``. Trading days follow the SZSE calendar, as in CED.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ced import db, store
from ced.calendar import SZSE, calendar

log = logging.getLogger(__name__)

DIV_FIELDS: tuple[str, ...] = ("cash_dvd_per_sh_pre_tax", "cash_dvd_per_sh_after_tax")
DIV_FIELD_LABELS = {"cash_dvd_per_sh_pre_tax": "每股派息(税前)(元)", "cash_dvd_per_sh_after_tax": "每股派息(税后)(元)"}
_PROGRESS_IMPLEMENTED = "3"
PUBLIC_OBJECT_RE = r"普通|全体|流通|无限售|公众|A股|所有"


def get_ex_dividend(start: str, end: str) -> pd.DataFrame:
    """Implemented AShareDividend records with EX_DT in [start, end], one row per Wind record."""
    sql = f"""
        SELECT
            S_INFO_WINDCODE            AS wind_code,
            EX_DT                      AS ex_dt,
            CASH_DVD_PER_SH_PRE_TAX    AS cash_dvd_per_sh_pre_tax,
            CASH_DVD_PER_SH_AFTER_TAX  AS cash_dvd_per_sh_after_tax,
            S_DIV_OBJECT               AS div_object
        FROM dbo.AShareDividend
        WHERE S_DIV_PROGRESS = '{_PROGRESS_IMPLEMENTED}'
          AND EX_DT >= '{start}'
          AND EX_DT <= '{end}'
    """
    df = db.read_sql(sql)
    cols = ["wind_code", "ex_dt", *DIV_FIELDS, "div_object", "is_public"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["wind_code"] = df["wind_code"].astype(str).str.strip()
    df["ex_dt"] = df["ex_dt"].astype(str).str.strip()
    for f in DIV_FIELDS:
        df[f] = pd.to_numeric(df[f], errors="coerce").astype(np.float64)
    obj = df["div_object"].fillna("").astype(str).str.strip()
    df["div_object"] = obj
    df["is_public"] = (obj == "") | obj.str.contains(PUBLIC_OBJECT_RE, regex=True)
    return df[cols].reset_index(drop=True)


def slice_date(records: pd.DataFrame, date: str) -> pd.DataFrame:
    if records.empty:
        return records
    return records.loc[records["ex_dt"].to_numpy() == date].reset_index(drop=True)


def aggregate_per_share(records: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Per symbol: add plans implemented the same day; drop differentiated (non-public) rows;
    if a symbol has only non-public rows, add them all (WARNING)."""
    if records.empty:
        return pd.DataFrame({"symbol": pd.Series(dtype=str), **{f: pd.Series(dtype=np.float64) for f in DIV_FIELDS}})
    grp_any_public = records.groupby("wind_code")["is_public"].transform("any")
    grp_multi = records["wind_code"].duplicated(keep=False)
    keep = records["is_public"] | ~grp_any_public
    dropped = records.loc[~keep]
    if len(dropped):
        log.info("%s: %d differentiated rows (object not all holders) dropped: %s", tag, len(dropped),
                 (dropped["wind_code"] + " " + dropped["div_object"]).tolist()[:10])
    no_public = records.loc[~grp_any_public & grp_multi, "wind_code"].unique().tolist()
    if no_public:
        log.warning("%s: %d symbols with several same-day rows and none for all holders, added up -- "
                    "check: %s", tag, len(no_public), no_public[:10])
    records = records.loc[keep]
    dup = records["wind_code"].duplicated(keep=False)
    if dup.any():
        log.info("%s: %d symbols with several plans on one day, added: %s", tag,
                 records.loc[dup, "wind_code"].nunique(), sorted(records.loc[dup, "wind_code"].unique().tolist())[:10])
    agg = records.groupby("wind_code", sort=True)[list(DIV_FIELDS)].sum(min_count=1)
    return agg.reset_index().rename(columns={"wind_code": "symbol"}).astype({f: np.float64 for f in DIV_FIELDS})


def build_range(start: str, end: str) -> dict[str, pd.DataFrame]:
    days = calendar.trade_days(start, end, SZSE)
    if not days:
        return {}
    records = get_ex_dividend(start, end)
    return {d: aggregate_per_share(slice_date(records, d), "wind_div") for d in days}


def convert_range(start: str, end: str, *, overwrite: bool = True) -> dict:
    return {d: store.write(df, "wind_div", d, overwrite=overwrite, empty_ok=True)
            for d, df in build_range(start, end).items()}
