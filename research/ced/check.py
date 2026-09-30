"""CED checks (port of shtcommon ``ced.check``): two panels compared cell by cell.

``check-div``    wind_div vs zy_div (two dividend sources)
``check-hist``   pre-open live vs after-close hist, for ``daily_sod`` and ``index_universe``
                 (hist taken from its stored dataset, else rebuilt in memory; never overwritten)
``check-index``  our pre-open index weights vs Wind: (1) sum(w * stock return) must equal the
                 index's own return (every index, every day), (2) drifted weights must equal the
                 official close weights where published (HS300 daily, others monthly)

Tolerances and return codes are CED's: 0 identical / 1 scattered (share of rows below
``warn_frac``) / 2 above threshold / 3 file or hist missing. Limits must match to the cent
(``warn_frac`` 0); share columns are compared but never alarm (live is T-1, hist T); index
weights allow 1e-5. Results go to the log as in CED -- and every mismatching cell also to
``store/ced/_checks/<check>/date=<D>/part.parquet`` (symbol, column, left, right), so a month
of checks is one DuckDB query.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch

import numpy as np
import pandas as pd

from ced import db, store
from ced.calendar import SZSE, calendar
from ced.indices import WIND_CODE_MAP

log = logging.getLogger(__name__)

RC_OK, RC_MINOR, RC_WARN, RC_MISSING = 0, 1, 2, 3
DIV_ATOL = 1e-6
HIST_RTOL = 1e-6
HIST_ATOL = 1e-9
HIST_WARN_FRAC = 0.01
HIST_COL_RULES = {
    "daily_sod": (("maxp_allowed", 0.0, 1e-9, 0.0), ("minp_allowed", 0.0, 1e-9, 0.0), ("*share", 0.0, 0.0, 1.0)),
    "index_universe": (("*", 0.0, 1e-5, 0.01),),
}
HIST_DATASETS: tuple[str, ...] = ("daily_sod", "index_universe")
INDEX_W_ATOL = 1e-4
INDEX_RET_WARN_BPS = 3.0


def _board(symbol: str) -> str:
    head, _, suffix = symbol.partition(".")
    if suffix == "BJ":
        return "BSE"
    if head.startswith("688"):
        return "STAR"
    if head.startswith(("300", "301")):
        return "ChiNext"
    return "SH main" if suffix == "SH" else "SZ main"


def _col_rule(col: str, rules, default: tuple[float, float, float]) -> tuple[float, float, float]:
    for pat, rtol, atol, wf in rules or ():
        if fnmatch(col, pat):
            return rtol, atol, wf
    return default


def frame(df: pd.DataFrame) -> pd.DataFrame:
    """Stored day -> symbol-indexed numeric frame (non-numeric columns dropped)."""
    out = df.set_index("symbol")
    out = out[[c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]]
    return out.astype("float64").sort_index()


@dataclass
class CheckResult:
    rc: int = RC_OK
    notes: list[str] = field(default_factory=list)
    n_both: int = 0
    rows: dict[str, list[str]] = field(default_factory=dict)
    cells: list[tuple[str, str, float, float]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return "; ".join(self.notes)

    def mismatches(self) -> pd.DataFrame:
        return pd.DataFrame(self.cells, columns=["symbol", "column", "left", "right"])


def compare_frames(left: pd.DataFrame, right: pd.DataFrame, *, names: tuple[str, str], rtol: float, atol: float,
                   warn_frac: float, col_rules=None) -> CheckResult:
    """Cell by cell; both NaN = same, one NaN = different, else |a-b| > atol + rtol*|b|."""
    ln, rn = names
    r = CheckResult()
    n_union = max(len(left.index.union(right.index)), 1)

    def bump(n_bad: int, n_total: int) -> None:
        r.rc = max(r.rc, RC_WARN if n_bad > warn_frac * max(n_total, 1) else RC_MINOR)

    only_l = left.index.difference(right.index)
    only_r = right.index.difference(left.index)
    both = left.index.intersection(right.index)
    r.n_both = len(both)
    for only, df, nm in ((only_l, left, ln), (only_r, right, rn)):
        if len(only):
            bump(len(only), n_union)
            r.notes.append(f"only in {nm}: {len(only)}")
            for s in only:
                r.rows.setdefault(s, []).append(f"only in {nm}")
                r.cells.append((s, f"<only in {nm}>", np.nan, np.nan))
    cols_l, cols_r = set(left.columns), set(right.columns)
    if cols_l ^ cols_r:
        r.notes.append(f"columns differ, comparing the common ones; difference {sorted(cols_l ^ cols_r)}")
    cols = [c for c in left.columns if c in cols_r]
    a, b = left.loc[both, cols], right.loc[both, cols]
    boards = np.array([_board(s) for s in both], dtype=object)
    board_total = {g: int((boards == g).sum()) for g in dict.fromkeys(boards)}
    for c in cols:
        c_rtol, c_atol, c_wf = _col_rule(c, col_rules, (rtol, atol, warn_frac))
        x = a[c].to_numpy(dtype=np.float64)
        y = b[c].to_numpy(dtype=np.float64)
        nan_mis = np.isnan(x) ^ np.isnan(y)
        both_ok = ~np.isnan(x) & ~np.isnan(y)
        d = np.abs(x - y)
        bad = nan_mis | (both_ok & (d > c_atol + c_rtol * np.abs(y)))
        n = int(bad.sum())
        if not n:
            continue
        r.rc = max(r.rc, RC_WARN if n > c_wf * max(len(both), 1) else RC_MINOR)
        by_board = " ".join(f"{g} {int((bad & (boards == g)).sum())}/{board_total[g]}"
                            for g in board_total if (bad & (boards == g)).any())
        hit = both_ok & bad
        mx = float(np.nanmax(np.where(hit, d, np.nan))) if hit.any() else float("nan")
        r.notes.append(f"{c}: {n}/{len(both)} rows differ [{by_board}], one-sided NaN {int(nan_mis.sum())}, "
                       f"max|{ln}-{rn}|={mx:.6g}")
        for s, xv, yv in zip(both[bad], x[bad], y[bad], strict=True):
            r.rows.setdefault(s, []).append(f"{c} {ln}={xv:.6g} {rn}={yv:.6g}")
            r.cells.append((s, c, float(xv), float(yv)))
    if r.rc == RC_OK:
        r.notes.append(f"{len(both)} common symbols x {len(cols)} columns identical (rtol={rtol}, atol={atol})")
    return r


def report(check: str, what: str, date: str, r: CheckResult, warn_frac: float) -> None:
    """Log like CED (summary + one line per differing symbol) and store the mismatching cells."""
    if r.cells:
        store.write(r.mismatches(), f"_checks/{check}", date)
    if r.rc == RC_OK:
        log.info("%s %s: identical. %s", what, date, r.notes[-1])
        return
    lvl = logging.INFO if r.rc == RC_MINOR else logging.WARNING
    log.log(lvl, "%s %s: %s %g: %s", what, date, "scattered differences below" if lvl == logging.INFO
            else "differences above", warn_frac, r.summary)
    for s in sorted(r.rows):
        log.log(lvl, "%s %s   %s: %s", what, date, s, " | ".join(r.rows[s]))


def _load(dataset: str, date: str) -> pd.DataFrame | None:
    return store.read_day(dataset, date) if store.exists(dataset, date) else None


# ----------------------------------------------------------------------------- check-div
def _drop_one_sided_zero(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    def zero_rows(df: pd.DataFrame, only: pd.Index) -> pd.Index:
        if not len(only):
            return only
        return only[(df.loc[only].fillna(0.0).abs().to_numpy() < 1e-12).all(axis=1)]

    za = zero_rows(a, a.index.difference(b.index))
    zb = zero_rows(b, b.index.difference(a.index))
    return a.drop(za), b.drop(zb), len(za) + len(zb)


def check_div_date(date: str, *, atol: float = DIV_ATOL) -> int:
    w, z = _load("wind_div", date), _load("zy_div", date)
    if w is None or z is None:
        log.error("check-div %s: dividend file missing wind_div=%s zy_div=%s", date, w is not None, z is not None)
        return RC_MISSING
    wf, zf, n_zero = _drop_one_sided_zero(frame(w), frame(z))
    if n_zero:
        log.info("check-div %s: %d symbols only on one side with zero cash (bonus/conversion only), same", date, n_zero)
    r = compare_frames(wf, zf, names=("wind", "zy"), rtol=0.0, atol=atol, warn_frac=0.0)
    report("check-div", "wind_div vs zy_div", date, r, 0.0)
    return r.rc


def check_div_range(start: str, end: str, **_) -> dict[str, int]:
    return {d: check_div_date(d) for d in calendar.trade_days(start, end, SZSE)}


# ----------------------------------------------------------------------------- check-hist
def _hist(name: str, date: str) -> pd.DataFrame | None:
    stored = _load(f"{name}_hist", date)
    if stored is not None:
        return stored
    if name == "daily_sod":
        from ced import daily

        return dict(daily.iter_sod_hist(date, date)).get(date)
    from ced import index_universe as iu

    codes = [c for c in iu.NAMES if c in iu.WIND_SUPPORTED_INDICES]
    wide = iu.bulk_hist(codes, date, date)
    day = iu.day_frame(wide, date, codes)
    return day if len(day) else None


def check_hist_date(name: str, date: str, *, rtol: float = HIST_RTOL, atol: float = HIST_ATOL,
                    warn_frac: float = HIST_WARN_FRAC) -> int:
    live = _load(name, date)
    if live is None:
        log.error("check-hist %s %s: live day missing", name, date)
        return RC_MISSING
    try:
        hist = _hist(name, date)
    except Exception:
        log.exception("check-hist %s %s: building hist failed", name, date)
        return RC_MISSING
    if hist is None:
        log.error("check-hist %s %s: hist unavailable (EOD not loaded? run after the close)", name, date)
        return RC_MISSING
    r = compare_frames(frame(live), frame(hist), names=("live", "hist"), rtol=rtol, atol=atol,
                       warn_frac=warn_frac, col_rules=HIST_COL_RULES[name])
    report(f"check-hist-{name}", f"{name} live vs hist", date, r, warn_frac)
    return r.rc


def check_hist_range(start: str, end: str, **_) -> dict[str, int]:
    return {d: max(check_hist_date(n, d) for n in HIST_DATASETS) for d in calendar.trade_days(start, end, SZSE)}


# ----------------------------------------------------------------------------- check-index
def _index_official_return(date: str) -> pd.Series:
    name_by_code = {WIND_CODE_MAP[n]: n for n in WIND_CODE_MAP}
    codes = ",".join(f"'{c}'" for c in name_by_code)
    df = db.read_sql(f"""
        SELECT S_INFO_WINDCODE AS index_code, S_DQ_CLOSE AS close_px, S_DQ_PRECLOSE AS pre_px
        FROM dbo.AIndexEODPrices
        WHERE TRADE_DT = '{date}' AND S_INFO_WINDCODE IN ({codes})
    """, what="AIndexEODPrices")
    if df.empty:
        return pd.Series(dtype=np.float64)
    ret = df["close_px"].astype(float) / df["pre_px"].astype(float) - 1.0
    return pd.Series(ret.to_numpy(), index=df["index_code"].map(name_by_code).to_numpy()).dropna()


def _stock_intraday_return(date: str, symbols: list) -> pd.Series:
    codes = ",".join(f"'{s}'" for s in symbols)
    df = db.read_sql(f"""
        SELECT S_INFO_WINDCODE AS wind_code, S_DQ_CLOSE AS close_px, S_DQ_PRECLOSE AS pre_px
        FROM dbo.AShareEODPrices
        WHERE TRADE_DT = '{date}' AND S_INFO_WINDCODE IN ({codes})
    """, what="AShareEODPrices")
    if df.empty:
        return pd.Series(dtype=np.float64)
    ret = df["close_px"].astype(float) / df["pre_px"].astype(float) - 1.0
    return pd.Series(ret.to_numpy(), index=df["wind_code"].to_numpy()).dropna()


def _official_close_weights(date: str) -> pd.DataFrame:
    from ced import index_universe as iu

    wide = iu.get_wind_weights(sorted(iu.WIND_SUPPORTED_INDICES), date, date)
    if wide.empty or date not in wide.index:
        return pd.DataFrame()
    row = wide.loc[date].dropna()
    return pd.DataFrame() if row.empty else row.unstack(level="index_name").sort_index()


def check_index_date(date: str, *, w_atol: float = INDEX_W_ATOL, ret_warn_bps: float = INDEX_RET_WARN_BPS) -> int:
    live = _load("index_universe", date)
    if live is None:
        log.error("check-index %s: live index_universe missing", date)
        return RC_MISSING
    w = frame(live).fillna(0.0)
    if w.empty:
        log.error("check-index %s: live file has no constituents", date)
        return RC_MISSING
    ret_off = _index_official_return(date)
    if ret_off.empty:
        log.error("check-index %s: AIndexEODPrices has no row for the day (run after the close)", date)
        return RC_MISSING
    ret_stk = _stock_intraday_return(date, w.index.tolist()).reindex(w.index)
    n_missing = int(ret_stk.isna().sum())
    ret_stk = ret_stk.fillna(0.0)
    rc = RC_OK
    for name in w.columns:
        if name not in ret_off.index or w[name].sum() <= 0:
            continue
        wv = w[name]
        r_model = float((wv * ret_stk).sum() / wv.sum())
        diff_bps = (r_model - float(ret_off[name])) * 1e4
        if abs(diff_bps) > ret_warn_bps:
            lvl, rc = logging.WARNING, max(rc, RC_WARN)
        else:
            lvl = logging.INFO
            if abs(diff_bps) > 0.5:
                rc = max(rc, RC_MINOR)
        log.log(lvl, "check-index %s %s: return model=%.4f%% official=%.4f%% diff=%+.2f bp (members %d, no EOD %d)",
                date, name, r_model * 100, float(ret_off[name]) * 100, diff_bps, int((wv > 0).sum()), n_missing)
    off = _official_close_weights(date)
    if off.empty:
        log.info("check-index %s: no official close weights today (non-HS300 monthly), return check only", date)
        return rc
    pred = w * (1.0 + ret_stk.to_numpy()[:, None])
    pred = pred / pred.sum(axis=0).replace(0, np.nan)
    for name in [c for c in off.columns if c in pred.columns]:
        a = pred[[name]].rename(columns={name: "w"})
        b = off[[name]].rename(columns={name: "w"})
        r = compare_frames(a[a["w"] > 0], b[b["w"] > 0], names=("model", "official"), rtol=0.0, atol=w_atol,
                           warn_frac=0.01)
        report(f"check-index-{name.replace(' ', '_')}", f"{name} close weights model vs official", date, r, 0.01)
        rc = max(rc, r.rc)
    return rc


def check_index_range(start: str, end: str, **_) -> dict[str, int]:
    return {d: check_index_date(d) for d in calendar.trade_days(start, end, SZSE)}
