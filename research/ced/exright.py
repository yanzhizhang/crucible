"""Pre-open ex-right reference-price inputs on the total-share basis (``_DIF``), verbatim port.

AShareEXRightDividendRecord gives the announced per-share ratios. With a buyback account or a
differentiated dividend the exchange prices the ex-right on the *total-share* ratios, which
AShareDividend publishes as ``CASH_DVD_PER_SH_PRE_TAX_DIF`` / ``DIV_BONUSRATE_DIF`` /
``DIV_CONVERSEDRATE_DIF``. Checked 2026-09-23 on 301237 / 002322 / 600211 / 000703.
Differentiated rows of one plan carry the same ``_DIF`` (take the all-holders row, do not
add); separate plans on the same day are added. **Price adjustment only** -- holders still
receive the announced cash (wind_div). Rights issues are not in AShareDividend.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from ced import db

log = logging.getLogger(__name__)
_PROGRESS_IMPLEMENTED = "3"


def round_cent(x):
    """Round half up to the cent (exchange rule); +1e-9 absorbs binary error."""
    return np.floor(np.asarray(x, dtype=np.float64) * 100.0 + 0.5 + 1e-9) / 100.0


def _get_dif(start: str, end: str, wind_codes: list | None = None) -> pd.DataFrame:
    """AShareDividend on the total-share basis, one row per (wind_code, ex_date)."""
    from ced.wind_div import PUBLIC_OBJECT_RE

    code_filter = ""
    if wind_codes:
        code_filter = "AND S_INFO_WINDCODE IN (" + ",".join(f"'{c}'" for c in wind_codes) + ")"
    sql = f"""
        SELECT
            S_INFO_WINDCODE              AS wind_code,
            EX_DT                        AS ex_date,
            S_DIV_OBJECT                 AS div_object,
            CASH_DVD_PER_SH_PRE_TAX_DIF  AS cash_dif,
            DIV_BONUSRATE_DIF            AS bonus_dif,
            DIV_CONVERSEDRATE_DIF        AS conversed_dif
        FROM dbo.AShareDividend
        WHERE S_DIV_PROGRESS = '{_PROGRESS_IMPLEMENTED}'
          AND EX_DT >= '{start}' AND EX_DT <= '{end}'
          {code_filter}
    """
    df = db.read_sql(sql, what="AShareDividend(_DIF)")
    cols = ["wind_code", "ex_date", "cash_dif", "bonus_dif", "conversed_dif"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    df["wind_code"] = df["wind_code"].astype(str).str.strip()
    df["ex_date"] = df["ex_date"].astype(str).str.strip()
    for c in cols[2:]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    obj = df["div_object"].fillna("").astype(str).str.strip()
    public = (obj == "") | obj.str.contains(PUBLIC_OBJECT_RE, regex=True)
    grp_any_public = public.groupby([df["wind_code"], df["ex_date"]]).transform("any")
    df = df.loc[public | ~grp_any_public]
    return df.groupby(["wind_code", "ex_date"], as_index=False)[cols[2:]].sum()[cols]


def apply_total_share_ratios(exdiv: pd.DataFrame, start: str, end: str, *, tag: str,
                             wind_codes: list | None = None) -> pd.DataFrame:
    """Replace cash / bonus / conversed with the total-share ``_DIF`` values where usable.

    Field by field, only downwards: ``_DIF`` must be > 0 and within [50 %, 100 %] of the
    announced value (differentiated dividends measured 89-99 %); otherwise the announced value
    stays. A fetch failure keeps the announced basis (logged) rather than failing the SOD.
    """
    if exdiv.empty:
        return exdiv
    try:
        dif = _get_dif(start, end, wind_codes)
    except Exception:
        log.error("%s: AShareDividend _DIF fetch failed, ex-rights on the announced basis", tag, exc_info=True)
        return exdiv
    if dif.empty:
        return exdiv
    m = exdiv.merge(dif, on=["wind_code", "ex_date"], how="left")
    before = m[["cash", "bonus", "conversed"]].fillna(0.0).to_numpy(dtype=np.float64)
    dif_v = m[["cash_dif", "bonus_dif", "conversed_dif"]].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        usable = (np.isfinite(dif_v) & (dif_v > 0) & (before > 0)
                  & (dif_v <= before * (1 + 1e-6)) & (dif_v >= before * 0.5))
    after = np.where(usable, dif_v, before)
    rejected = np.isfinite(dif_v) & (dif_v > 0) & (before > 0) & ~usable
    changed = (np.abs(after - before) > 1e-9).any(axis=1)
    names = ("cash", "bonus", "conversed")

    def _desc(i: int, mask: np.ndarray) -> str:
        parts = [f"{names[j]} {before[i, j]:g}->{dif_v[i, j]:g}" for j in range(3) if mask[i, j]]
        return f"{m.at[i, 'wind_code']} {m.at[i, 'ex_date']} " + " ".join(parts)

    if changed.any():
        sample = [_desc(i, usable & (np.abs(after - before) > 1e-9)) for i in np.flatnonzero(changed)[:8]]
        log.info("%s: %d ex-right records lowered to the total-share basis (_DIF): %s",
                 tag, int(changed.sum()), sample)
    if rejected.any():
        rows = np.flatnonzero(rejected.any(axis=1))
        log.info("%s: %d _DIF values outside 50-100%% of the announced value, kept announced: %s",
                 tag, len(rows), [_desc(i, rejected) for i in rows[:8]])
    out = exdiv.copy()
    out[["cash", "bonus", "conversed"]] = after
    return out
