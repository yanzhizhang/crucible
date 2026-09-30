"""Wind concept boards -> dataset ``concept`` (port of shtcommon ``ced.concept``).

Long table, one row per (symbol, concept_code) membership that day (CED: one-hot matrix);
names in ``_static/catalog.parquet``. Membership on d: ``entry_dt < d < remove_dt`` (strict on
both ends -- no same-day look-ahead), latest entry per pair, then CED's filters: index code
647090000, name exclusions, board active and listed by d, no BSE symbols, and per symbol more
than 1 and fewer than 200 concepts.

Listing date: a board counts from the day after it lists (``list_dt < d``), like members
(``entry_dt < d``). CED's code filtered ``list_dt <= d`` after an SQL bound of ``< end``, so a
board listed on d counted inside a range but not on a single day; its production files are the
single-day case (the daily ``[T-1, T]`` job never rewrites T-1), so strict ``<`` is both
consistent and what production actually wrote.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

from ced import db, store
from ced.calendar import SSE, calendar

_DEFAULT_INDEX_CODE = 647090000
_DEFAULT_CT_MIN = 1
_DEFAULT_CT_MAX = 200
_DEFAULT_NAME_LIKE_EXCLUDES = ("HK", "US", "SS", "金股", "精选", "国资", "陆股通", "沪伦通", "龙头", "综合")
_DEFAULT_NAME_IN_EXCLUDES = ("A50指数", "股权转让指数")
_FAR_FUTURE = "299991231"


def get_concept_range(start: str, end: str) -> pd.DataFrame:
    """Memberships active on any day of [start, end]."""
    sql = f"""
        SELECT A.S_CON_WINDCODE   AS instrument,
               B.S_INFO_NAME      AS concept_name,
               B.S_INFO_WINDCODE  AS concept_code,
               B.S_INFO_INDEXCODE AS index_code,
               A.S_CON_INDATE     AS entry_dt,
               A.S_CON_OUTDATE    AS remove_dt,
               B.S_INFO_LISTDATE  AS list_dt,
               B.EXPIRE_DATE      AS expire_dt
        FROM dbo.AIndexMembersWind A WITH (NOLOCK)
        INNER JOIN dbo.AIndexDescription B WITH (NOLOCK)
          ON A.F_INFO_WINDCODE = B.S_INFO_WINDCODE
        WHERE B.S_INFO_LISTDATE < '{end}'
          AND A.S_CON_INDATE   < '{end}'
          AND (A.S_CON_OUTDATE > '{start}' OR A.S_CON_OUTDATE IS NULL OR A.S_CON_OUTDATE = '')
          AND (B.EXPIRE_DATE   > '{start}' OR B.EXPIRE_DATE   IS NULL OR B.EXPIRE_DATE   = '')
    """
    df = db.read_sql(sql)
    str_cols = ["instrument", "concept_name", "concept_code", "entry_dt", "remove_dt", "list_dt", "expire_dt"]
    if df.empty:
        return pd.DataFrame(columns=[*str_cols, "index_code"])
    for c in str_cols:
        df[c] = df[c].astype("string").fillna("").astype(str).str.strip()
    df.loc[df["remove_dt"] == "", "remove_dt"] = _FAR_FUTURE
    df.loc[df["expire_dt"] == "", "expire_dt"] = _FAR_FUTURE
    df["index_code"] = pd.to_numeric(df["index_code"], errors="coerce").astype("Int32")
    return df.reset_index(drop=True)


def get_concept_catalog() -> pd.DataFrame:
    sql = ("SELECT DISTINCT S_INFO_WINDCODE AS concept_code, S_INFO_NAME AS concept_name "
           "FROM dbo.AIndexDescription WITH (NOLOCK) ORDER BY S_INFO_WINDCODE")
    df = db.read_sql(sql)
    if df.empty:
        return pd.DataFrame(columns=["concept_code", "concept_name"])
    for c in ("concept_code", "concept_name"):
        df[c] = df[c].astype("string").fillna("").astype(str).str.strip()
    return df.reset_index(drop=True)


def slice_asof(range_df: pd.DataFrame, date: str) -> pd.DataFrame:
    if range_df.empty:
        return range_df.iloc[0:0].copy()
    sub = range_df.loc[(range_df["entry_dt"] < date) & (range_df["remove_dt"] > date)]
    if sub.empty:
        return sub.reset_index(drop=True)
    return (sub.sort_values(["instrument", "concept_code", "entry_dt"])
            .drop_duplicates(subset=["instrument", "concept_code"], keep="last").reset_index(drop=True))


def apply_filters(snapshot: pd.DataFrame, date: str, *, index_code: int | None = _DEFAULT_INDEX_CODE,
                  ct_min: int | None = _DEFAULT_CT_MIN, ct_max: int | None = _DEFAULT_CT_MAX,
                  name_like_excludes: Iterable[str] = _DEFAULT_NAME_LIKE_EXCLUDES,
                  name_in_excludes: Iterable[str] = _DEFAULT_NAME_IN_EXCLUDES, require_listed: bool = True,
                  require_active: bool = True, exclude_bj: bool = True) -> pd.DataFrame:
    if snapshot.empty:
        return snapshot
    df = snapshot
    if index_code is not None:
        df = df.loc[df["index_code"] == index_code]
    if name_in_excludes:
        df = df.loc[~df["concept_name"].isin(list(name_in_excludes))]
    if name_like_excludes:
        pattern = "|".join(re.escape(k) for k in name_like_excludes)
        df = df.loc[~df["concept_name"].str.contains(pattern, na=False, regex=True)]
    if require_active:
        df = df.loc[df["expire_dt"] == _FAR_FUTURE]
    if require_listed:
        df = df.loc[(df["list_dt"] != "") & (df["list_dt"] < date)]
    if exclude_bj:
        df = df.loc[~df["instrument"].str.endswith("BJ")]
    if (ct_min is not None or ct_max is not None) and not df.empty:
        ct = df.groupby("instrument")["concept_code"].transform("count")
        if ct_min is not None:
            df = df.loc[ct > ct_min]
            ct = ct.loc[df.index]
        if ct_max is not None:
            df = df.loc[ct < ct_max]
    return df.reset_index(drop=True)


def day_frame(snap: pd.DataFrame) -> pd.DataFrame:
    out = snap[["instrument", "concept_code"]].drop_duplicates() if not snap.empty else \
        pd.DataFrame(columns=["instrument", "concept_code"])
    return (out.rename(columns={"instrument": "symbol"}).astype(str)
            .sort_values(["symbol", "concept_code"]).reset_index(drop=True))


def convert_range(start: str, end: str, *, overwrite: bool = True) -> dict:
    store.write_static(get_concept_catalog(), "concept", "catalog")
    rng = get_concept_range(start, end)
    return {d: store.write(day_frame(apply_filters(slice_asof(rng, d), d)), "concept", d, overwrite=overwrite)
            for d in calendar.trade_days(start, end, SSE)}
