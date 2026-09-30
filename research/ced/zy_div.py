"""Same dividends from 朝阳永续 ``bas_stk_hisdistribution`` -> dataset ``zy_div`` (port of shtcommon).

Same shape and meaning as ``wind_div`` so the two sources compare cell by cell. Source values
are per 10 shares (/10 here). Rules (dictionary table_cd=129, checked 2026-09-22): distri_type
分红 1002 / 送股 1004 / 转增 1005 only, cash only on 1002 rows (bonus / conversion rows give 0
so the symbol appears, like Wind); final plans only (scheme_type 1001); is_newest = 1 and
is_valid = 1 (NULL counts as 1); public objects 1001 all / 1002 circulating, 9999/empty by the
description; when a group has a 1002 row the 1001 row is not counted (differentiated dividend:
the all-holders row is an average that no actual holder receives). NULL cash -> 0.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ced import db, store
from ced.calendar import SZSE, calendar
from ced.wind_div import DIV_FIELDS, PUBLIC_OBJECT_RE, aggregate_per_share, slice_date

log = logging.getLogger(__name__)

_PER_10_SHARES = 10.0
_DIV_TYPES = ("1002", "1004", "1005")
_CASH_TYPE = "1002"
_SCHEME_FINAL = "1001"


def _to_wind_code(stock_code: pd.Series) -> pd.Series:
    """6-digit code -> Wind code: 6xx .SH, 4xx/8xx/92x .BJ, else .SZ."""
    code = stock_code.astype(str).str.strip().str.zfill(6)
    suffix = np.where(code.str.startswith("6"), ".SH",
                      np.where(code.str[0].isin(["4", "8"]) | code.str.startswith("92"), ".BJ", ".SZ"))
    return code + suffix


def get_ex_dividend(start: str, end: str) -> pd.DataFrame:
    """Records with xr_xd_date in [start, end], per share, one row per source row (not aggregated)."""
    div_types = ", ".join(f"'{t}'" for t in _DIV_TYPES)
    sql = f"""
        SELECT
            stock_code,
            CONVERT(VARCHAR(8), xr_xd_date, 112) AS ex_dt,
            beftax_maxcashdiv AS pre_tax_per_10,
            beftax_mincashdiv AS pre_tax_min_per_10,
            aftax_cashdiv     AS after_tax_per_10,
            distri_type,
            scheme_type,
            object_type,
            object_desc,
            is_newest,
            is_valid
        FROM dbo.bas_stk_hisdistribution
        WHERE xr_xd_date >= '{start}'
          AND xr_xd_date <= '{end}'
          AND distri_type IN ({div_types})
    """
    df = db.read_sql(sql, db.ZY)
    cols = ["wind_code", "ex_dt", *DIV_FIELDS, "div_object", "is_public"]
    if df.empty:
        return pd.DataFrame(columns=cols)

    def flag(c: str) -> pd.Series:
        return pd.to_numeric(df[c], errors="coerce").fillna(1) == 1

    live = flag("is_newest") & flag("is_valid")
    if (~live).any():
        stale = df.loc[~live]
        log.info("zy_div: %d rows with is_newest=0 or is_valid=0 dropped: %s", len(stale),
                 (stale["stock_code"].astype(str) + " " + stale["ex_dt"].astype(str) + " newest="
                  + stale["is_newest"].astype(str) + " valid=" + stale["is_valid"].astype(str)).tolist()[:10])
        df = df.loc[live].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=cols)

    def code(c: str) -> pd.Series:  # numeric code columns may come back as 1001.0
        return df[c].fillna("").astype(str).str.strip().str.replace(r"\.0$", "", regex=True)

    scheme = code("scheme_type")
    not_final = scheme != _SCHEME_FINAL
    if not_final.any():
        log.info("zy_div: %d draft rows (scheme_type != 1001) dropped: %s", int(not_final.sum()),
                 (df.loc[not_final, "stock_code"].astype(str) + " " + df.loc[not_final, "ex_dt"].astype(str)).tolist()[:10])
        df = df.loc[~not_final].reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(columns=cols)
    otype = code("object_type")
    odesc = df["object_desc"].fillna("").astype(str).str.strip()
    is_cash = code("distri_type") == _CASH_TYPE
    pre = pd.to_numeric(df["pre_tax_per_10"], errors="coerce").fillna(
        pd.to_numeric(df["pre_tax_min_per_10"], errors="coerce"))
    aft = pd.to_numeric(df["after_tax_per_10"], errors="coerce")
    out = pd.DataFrame({
        "wind_code": _to_wind_code(df["stock_code"]),
        "ex_dt": df["ex_dt"].astype(str).str.strip(),
        DIV_FIELDS[0]: pre.where(is_cash, 0.0).astype(np.float64) / _PER_10_SHARES,
        DIV_FIELDS[1]: aft.where(is_cash, 0.0).astype(np.float64) / _PER_10_SHARES,
        "div_object": otype + ":" + odesc,
        "is_public": otype.isin(("1001", "1002"))
        | (otype.isin(("9999", "")) & odesc.str.contains(PUBLIC_OBJECT_RE, regex=True))
        | (otype.isin(("9999", "")) & (odesc == "")),
    })
    for f in DIV_FIELDS:
        out[f] = out[f].fillna(0.0)
    out["_otype"] = otype.to_numpy()
    has_circ = out.groupby(["wind_code", "ex_dt"])["_otype"].transform(lambda s: (s == "1002").any())
    demote = has_circ & (out["_otype"] != "1002") & out["is_public"]
    if demote.any():
        log.info("zy_div: %d all-holder rows superseded by a circulating-holder row in their group: %s",
                 int(demote.sum()), (out.loc[demote, "wind_code"] + " " + out.loc[demote, "ex_dt"]).unique().tolist()[:10])
        out.loc[demote, "is_public"] = False
    return out[cols].reset_index(drop=True)


def build_range(start: str, end: str) -> dict[str, pd.DataFrame]:
    days = calendar.trade_days(start, end, SZSE)
    if not days:
        return {}
    records = get_ex_dividend(start, end)
    return {d: aggregate_per_share(slice_date(records, d), "zy_div") for d in days}


def convert_range(start: str, end: str, *, overwrite: bool = True) -> dict:
    return {d: store.write(df, "zy_div", d, overwrite=overwrite, empty_ok=True)
            for d, df in build_range(start, end).items()}
