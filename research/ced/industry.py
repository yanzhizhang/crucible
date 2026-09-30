"""Industry membership -> ``sw_industry`` (JYDB, 申万) and ``wind_industry`` (Wind), ports of shtcommon.

CED wrote one one-hot ``(symbol, industry_code)`` matrix per level; each symbol belongs to one
industry per level, so here one row per symbol with one code column per level (lossless):

``sw_industry``    symbol, level1, level2, level3             (Standard 38 = 申万行业分类(新))
``wind_industry``  symbol, level1 .. level4 (code prefixes of 4 / 6 / 8 / 10 digits of the
                   10-digit leaf; a symbol whose leaf is shorter has null at the deeper levels)

Names: ``_static/catalog.parquet`` (industry_code -> industry_name) per dataset.

SW: every record is fetched once (main board LC_ExgIndustry + STAR LC_STIBExgIndustry) and each
day keeps records with ``pub_date <= d < cancel_date`` (latest pub_date per symbol).
Wind: CED queried each day; here one query for the range, sliced per day with the same rule
(``ENTRY_DT <= d AND (REMOVE_DT empty OR > d)``, latest ENTRY_DT per symbol) -- identical rows.
Trading days follow the SZSE calendar, as in CED.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ced import db, store
from ced.calendar import SZSE, calendar

SW_STANDARD_DEFAULT = 38
SW_LEVELS: tuple[int, ...] = (1, 2, 3)
WIND_LEVEL_LEN: dict[int, int] = {1: 4, 2: 6, 3: 8, 4: 10}
WIND_LEVELS: tuple[int, ...] = (1, 2, 3, 4)
_WIND_CODE_PREFIX = "62"  # 万得行业分类 inside the mixed code dictionary


# ----------------------------------------------------------------------------- SW (JYDB)
def get_all_sw(standard: int = SW_STANDARD_DEFAULT) -> pd.DataFrame:
    """Every SW record (main board + STAR), no date filter; all columns as str (NULL -> 'None')."""
    cols = (
        "CONVERT(VARCHAR(8), A.InfoPublDate, 112) AS pub_date, "
        "CONVERT(VARCHAR(8), A.CancelDate,    112) AS cancel_date, "
        "A.FirstIndustryCode  AS level1_code, A.FirstIndustryName  AS level1_name, "
        "A.SecondIndustryCode AS level2_code, A.SecondIndustryName AS level2_name, "
        "A.ThirdIndustryCode  AS level3_code, A.ThirdIndustryName  AS level3_name, "
        "A.FourthIndustryCode  AS level4_code, A.FourthIndustryName  AS level4_name, "
        "B.SecuCode AS secu_code"
    )
    sql_main = f"""
        SELECT {cols} FROM JYDB.dbo.LC_ExgIndustry AS A
        INNER JOIN JYDB.dbo.SecuMain AS B ON A.CompanyCode = B.CompanyCode
        WHERE A.Standard = {int(standard)} AND A.InfoSource = '申万研究所' AND B.SecuCategory = 1
    """
    sql_stib = f"""
        SELECT {cols} FROM JYDB.dbo.LC_STIBExgIndustry AS A
        INNER JOIN JYDB.dbo.SecuMain AS B ON A.CompanyCode = B.CompanyCode
        WHERE A.Standard = {int(standard)} AND B.SecuCategory = 1
    """
    df = pd.concat([db.read_sql(sql_main, db.JY), db.read_sql(sql_stib, db.JY)], ignore_index=True)
    secu = df["secu_code"].astype(str)
    df["wind_code"] = secu + np.where(secu.str.startswith("6"), ".SH",
                                      np.where(secu.str[0].isin(["4", "8"]), ".BJ", ".SZ"))
    return df.astype(str)


def sw_snapshot(df: pd.DataFrame, date: str) -> pd.DataFrame:
    """Records valid on ``date``, latest pub_date per symbol."""
    mask = (df["pub_date"] <= date) & ((df["cancel_date"] > date) | (df["cancel_date"] == "None"))
    return df[mask].sort_values("pub_date").drop_duplicates("wind_code", keep="last").reset_index(drop=True)


def _sw_day(snap: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"symbol": snap["wind_code"]})
    for lvl in SW_LEVELS:  # CED dropped NULL codes per level; astype(str) had made them 'None'
        out[f"level{lvl}"] = snap[f"level{lvl}_code"].where(snap[f"level{lvl}_code"] != "None")
    return out.sort_values("symbol").reset_index(drop=True)


def sw_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """code -> name over levels 1-3 (first name seen wins, as in CED)."""
    pairs: dict[str, str] = {}
    for lvl in SW_LEVELS:
        sub = df[df[f"level{lvl}_code"] != "None"]
        for code, name in zip(sub[f"level{lvl}_code"], sub[f"level{lvl}_name"], strict=True):
            pairs.setdefault(code, name)
    return pd.DataFrame({"industry_code": sorted(pairs), "industry_name": [pairs[c] for c in sorted(pairs)]})


def convert_range_sw(start: str, end: str, *, overwrite: bool = True, standard: int = SW_STANDARD_DEFAULT) -> dict:
    allrec = get_all_sw(standard)
    store.write_static(sw_catalog(allrec), "sw_industry", "catalog")
    return {d: store.write(_sw_day(sw_snapshot(allrec, d)), "sw_industry", d, overwrite=overwrite)
            for d in calendar.trade_days(start, end, SZSE)}


# ----------------------------------------------------------------------------- Wind
def get_wind_range(start: str, end: str) -> pd.DataFrame:
    """AShareIndustriesClass rows active on any day of [start, end]."""
    sql = f"""
        SELECT S_INFO_WINDCODE AS wind_code, WIND_IND_CODE AS wind_ind_code,
               ENTRY_DT AS entry_dt, REMOVE_DT AS remove_dt
        FROM dbo.AShareIndustriesClass
        WHERE ENTRY_DT <= '{end}'
          AND (REMOVE_DT IS NULL OR REMOVE_DT = '' OR REMOVE_DT > '{start}')
    """
    df = db.read_sql(sql)
    if df.empty:
        return pd.DataFrame(columns=["wind_code", "wind_ind_code", "entry_dt", "remove_dt"])
    df["wind_ind_code"] = df["wind_ind_code"].astype(str)
    df["entry_dt"] = df["entry_dt"].astype(str)
    df["remove_dt"] = df["remove_dt"].fillna("").astype(str)
    return df


def wind_snapshot(rng: pd.DataFrame, date: str) -> pd.DataFrame:
    """CED's single-day query on ``date``: active, latest ENTRY_DT per symbol."""
    if rng.empty:
        return rng
    act = rng[(rng["entry_dt"] <= date) & ((rng["remove_dt"] == "") | (rng["remove_dt"] > date))]
    return act.sort_values(["wind_code", "entry_dt"]).drop_duplicates("wind_code", keep="last")


def _wind_day(snap: pd.DataFrame) -> pd.DataFrame:
    sub = snap.dropna(subset=["wind_ind_code"])
    out = pd.DataFrame({"symbol": sub["wind_code"].to_numpy()})
    code = sub["wind_ind_code"].to_numpy(dtype=object)
    for lvl, n in WIND_LEVEL_LEN.items():
        out[f"level{lvl}"] = [c[:n] if len(c) >= n else None for c in code]
    out = out[out[[f"level{lvl}" for lvl in WIND_LEVELS]].notna().any(axis=1)]
    return out.sort_values("symbol").reset_index(drop=True)


def get_wind_codes() -> pd.DataFrame:
    """ASHAREINDUSTRIESCODE, USED = 1, Wind industry codes only ('62...')."""
    sql = f"""
        SELECT INDUSTRIESCODE AS industry_code, INDUSTRIESNAME AS industry_name,
               LEVELNUM AS level_num FROM dbo.ASHAREINDUSTRIESCODE
        WHERE USED = '1' AND INDUSTRIESCODE LIKE '{_WIND_CODE_PREFIX}%'
    """
    df = db.read_sql(sql)
    if df.empty:
        return pd.DataFrame(columns=["industry_code", "industry_name", "level_num"])
    return df.astype({"industry_code": str, "industry_name": str, "level_num": int}).reset_index(drop=True)


def convert_range_wind(start: str, end: str, *, overwrite: bool = True) -> dict:
    store.write_static(get_wind_codes(), "wind_industry", "catalog")
    rng = get_wind_range(start, end)
    return {d: store.write(_wind_day(wind_snapshot(rng, d)), "wind_industry", d, overwrite=overwrite)
            for d in calendar.trade_days(start, end, SZSE)}
