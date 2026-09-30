"""Barra CNE6 delivery files -> five datasets (port of shtcommon ``ced.barra``; parsing unchanged).

Reads ``{repo}/SMD_CNTRD_100_D_YYYY/<pattern>.YYYYMMDD`` ('|'-separated, '!' metadata lines,
the first '!' line with a separator is the header). Repo: ``$CRUCIBLE_BARRA_REPO``, default
``/mnt/BarraDataRepo/BarraCNE6``. Each day depends only on that day's files.

``barra_exp``    symbol x factor exposure (one column per factor, ``CNTRD_`` prefix stripped)
``barra_stats``  symbol x SRet / Yield / TRisk / SRisk / HBeta / PBeta / Close / Capt / Ret / EstU
``barra_rate``   symbol, USDCNY (the day's CNY rate on every symbol of the price file)
``barra_fret``   factor, factor_return
``barra_cov``    factor x factor covariance (symmetrised: the files fill one triangle)

Symbols are Wind codes via ``BarraID/CHN_LOCALID_Asset_ID`` (latest file on or before the
day; rows with StartDate <= d <= EndDate; CN600612 -> 600612.SH, 0/3 .SZ, 4/8/920 .BJ);
unmapped Barrids are dropped (INFO, WARNING above 5 %), duplicates keep the first.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ced import store

log = logging.getLogger(__name__)

_PATTERNS: dict[str, str] = {
    "exposure": "CNTRD_100_Asset_Exposure",
    "factor_ret": "CNTRD_100_DlyFacRet",
    "covariance": "CNTRD_100_Covariance",
    "asset_data": "CNTRD_100_Asset_Data",
    "spec_ret": "CNTR_100_Asset_DlySpecRet",
    "price": "CNTR_Daily_Asset_Price",
    "estu": "CNTR_ESTU_POR",
    "rates": "CNTR_Rates",
}
_STAT_SOURCES: dict[str, tuple[str, str]] = {
    "SRet": ("spec_ret", "SpecificReturn"),
    "Yield": ("asset_data", "Yield%"),
    "TRisk": ("asset_data", "TotalRisk%"),
    "SRisk": ("asset_data", "SpecRisk%"),
    "HBeta": ("asset_data", "HistBeta"),
    "PBeta": ("asset_data", "PredBeta"),
    "Close": ("price", "Price"),
    "Capt": ("price", "Capt"),
    "Ret": ("price", "DlyReturn%"),
    "EstU": ("estu", "Shares"),
}
_YEAR_DIR_RE = re.compile(r"SMD_CNTRD_?\d*_D_(\d{4})$")
_FACTOR_PREFIX = "CNTRD_"
BARRA_KINDS: tuple[str, ...] = ("exp", "fret", "cov", "stats", "rate")
_BARRA_ID_SUBDIR = "BarraID"
_LOCALID_PATTERN = "CHN_LOCALID_Asset_ID"
_IDENTITY_PATTERN = "CHN_Asset_Identity"
_LOCALID_PREFIX = "CN"
_DATE_SUFFIX_RE = re.compile(r"\.(\d{8})$")


def repo_root() -> Path:
    return Path(os.environ.get("CRUCIBLE_BARRA_REPO", "/mnt/BarraDataRepo/BarraCNE6"))


def _year_dirs(repo: Path, year: int | None = None) -> list[Path]:
    if not repo.exists():
        return []
    out = []
    for p in sorted(repo.iterdir()):
        m = _YEAR_DIR_RE.search(p.name) if p.is_dir() else None
        if m and (year is None or int(m.group(1)) == year):
            out.append(p)
    return out


def find_source(kind: str, date: str, repo: Path | None = None) -> Path | None:
    """The day's file of ``kind`` (its year's directory first, then any year)."""
    repo = repo or repo_root()
    pattern = _PATTERNS[kind]
    for d in [*_year_dirs(repo, year=int(date[:4])), *_year_dirs(repo)]:
        cand = d / f"{pattern}.{date}"
        if cand.exists():
            return cand
    return None


def parse_barra_file(filepath: str | Path, *, separator: str = "|", skip_prefix: str = "!") -> pd.DataFrame:
    header = None
    rows: list[list[str]] = []
    with open(filepath, encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(skip_prefix):
                if header is None and separator in line:
                    header = line.lstrip(skip_prefix).strip()
                continue
            rows.append(line.split(separator))
    if header is None:
        raise ValueError(f"no header row in {filepath}")
    columns = header.split(separator)
    return pd.DataFrame([r[: len(columns)] for r in rows if len(r) >= len(columns)], columns=columns)


def _to_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _strip(names: Iterable[str]) -> list[str]:
    return [str(s).removeprefix(_FACTOR_PREFIX) for s in names]


def _symbol_frame(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.astype("float64")
    out.columns = [str(c) for c in out.columns]
    out.index.name = "symbol"
    return out.reset_index()


def build_exposure(date: str, repo: Path, symbol_map: dict[str, str]) -> pd.DataFrame:
    src = find_source("exposure", date, repo)
    if src is None:
        raise FileNotFoundError(f"barra exposure missing for {date}")
    df = _to_numeric(parse_barra_file(src), ["Exposure"])
    df["Factor"] = _strip(df["Factor"])
    pv = df.pivot_table(index="Barrid", columns="Factor", values="Exposure", aggfunc="first").sort_index().sort_index(axis=1)
    return _symbol_frame(reindex_barrid_to_windcode(pv, symbol_map))


def build_stats(date: str, repo: Path, symbol_map: dict[str, str]) -> pd.DataFrame:
    series: dict[str, pd.Series] = {}
    for stat, (kind, col) in _STAT_SOURCES.items():
        src = find_source(kind, date, repo)
        if src is None:
            continue
        df = _to_numeric(parse_barra_file(src), [col])
        if "Barrid" not in df.columns or col not in df.columns:
            continue
        series[stat] = df.dropna(subset=["Barrid"]).set_index("Barrid")[col]
    if not series:
        raise FileNotFoundError(f"no barra stats sources for {date} under {repo}")
    panel = pd.DataFrame(series).sort_index()
    panel = panel.reindex(columns=[c for c in _STAT_SOURCES if c in panel.columns])
    return _symbol_frame(reindex_barrid_to_windcode(panel, symbol_map))


def build_rate(date: str, repo: Path, symbol_map: dict[str, str]) -> pd.DataFrame:
    rates_src = find_source("rates", date, repo)
    if rates_src is None:
        raise FileNotFoundError(f"barra rates missing for {date}")
    rates = _to_numeric(parse_barra_file(rates_src), ["USDxrate"])
    if "Currency" not in rates.columns or "USDxrate" not in rates.columns:
        raise ValueError(f"unexpected rates schema in {rates_src}: {list(rates.columns)}")
    cny = rates.loc[rates["Currency"].astype(str).str.upper() == "CNY"]
    if cny.empty:
        raise ValueError(f"CNY row not found in {rates_src}")
    usd_cny = float(cny["USDxrate"].iloc[0])
    sym_src = find_source("price", date, repo) or find_source("asset_data", date, repo)
    if sym_src is None:
        raise FileNotFoundError(f"no price/asset_data for symbol universe @ {date}")
    sym_df = parse_barra_file(sym_src)
    if "Barrid" not in sym_df.columns:
        raise ValueError(f"unexpected schema in {sym_src}: {list(sym_df.columns)}")
    symbols = np.sort(sym_df["Barrid"].dropna().astype(str).map(symbol_map).dropna().drop_duplicates().to_numpy())
    return pd.DataFrame({"symbol": symbols.astype(str), "USDCNY": np.full(symbols.size, usd_cny)})


def build_factor_return(date: str, repo: Path) -> pd.DataFrame:
    src = find_source("factor_ret", date, repo)
    if src is None:
        raise FileNotFoundError(f"barra factor_ret missing for {date}")
    df = _to_numeric(parse_barra_file(src), ["DlyReturn"])
    df["Factor"] = _strip(df["Factor"])
    if "DataDate" in df.columns:
        df = df[df["DataDate"].astype(str) == date]
    df = df.dropna(subset=["Factor"]).set_index("Factor").sort_index()
    return pd.DataFrame({"factor": df.index.astype(str), "factor_return": df["DlyReturn"].to_numpy(dtype="float64")})


def build_covariance(date: str, repo: Path) -> pd.DataFrame:
    src = find_source("covariance", date, repo)
    if src is None:
        raise FileNotFoundError(f"barra covariance missing for {date}")
    df = _to_numeric(parse_barra_file(src), ["VarCovar"])
    df["Factor1"] = _strip(df["Factor1"])
    df["Factor2"] = _strip(df["Factor2"])
    pv = df.pivot_table(index="Factor1", columns="Factor2", values="VarCovar", aggfunc="first")
    factors = sorted(set(pv.index.astype(str)).union(pv.columns.astype(str)))
    arr = pv.reindex(index=factors, columns=factors).to_numpy(dtype="float64").copy()
    mask = np.isnan(arr)
    arr[mask] = arr.T[mask]
    out = pd.DataFrame(arr, columns=factors)
    out.insert(0, "factor", factors)
    return out


_SYMBOL_BUILDERS = {"exp": build_exposure, "stats": build_stats, "rate": build_rate}
_FACTOR_BUILDERS = {"fret": build_factor_return, "cov": build_covariance}


def convert_date(date: str, *, kinds: Iterable[str] = BARRA_KINDS, repo: Path | None = None,
                 overwrite: bool = True, skip_missing: bool = True) -> dict[str, Path]:
    repo = repo or repo_root()
    symbol_map: dict[str, str] | None = None
    out: dict[str, Path] = {}
    for kind in kinds:
        try:
            if kind in _SYMBOL_BUILDERS:
                if symbol_map is None:
                    symbol_map = build_barrid_to_windcode(date, repo=repo)
                df = _SYMBOL_BUILDERS[kind](date, repo, symbol_map)
            else:
                df = _FACTOR_BUILDERS[kind](date, repo)
        except FileNotFoundError as e:
            if skip_missing:
                log.warning("barra %s %s: source missing, skipped: %s", kind, date, e)
                continue
            raise
        out[kind] = store.write(df, f"barra_{kind}", date, overwrite=overwrite)
    return out


def discover_dates(repo: Path, start_date: str, end_date: str) -> list[str]:
    """Days with an exposure file in [start, end] (non-trading days never have one)."""
    dates: set[str] = set()
    for y in range(int(start_date[:4]), int(end_date[:4]) + 1):
        for d in _year_dirs(repo, year=y):
            for f in d.glob(f"{_PATTERNS['exposure']}.*"):
                ds = f.name[-8:]
                if not f.is_dir() and ds.isdigit() and start_date <= ds <= end_date:
                    dates.add(ds)
    return sorted(dates)


def convert_range(start: str, end: str, *, overwrite: bool = True) -> dict:
    repo = repo_root()
    found = discover_dates(repo, start, end)
    if not found:
        log.warning("barra [%s, %s]: no source files under %s, nothing written", start, end, repo)
    return {d: convert_date(d, repo=repo, overwrite=overwrite) for d in found}


# ----------------------------------------------------------------------------- Barrid -> Wind code
def _assetid_to_windcode(asset_id: str) -> str | None:
    s = str(asset_id).strip().upper().removeprefix(_LOCALID_PREFIX)
    if len(s) != 6 or not s.isdigit():
        return None
    if s[0] == "6":
        return f"{s}.SH"
    if s[0] in ("0", "3"):
        return f"{s}.SZ"
    if s[0] in ("4", "8") or s.startswith("920"):
        return f"{s}.BJ"
    return None


def _find_latest_id_file(pattern: str, date: str, repo: Path) -> Path | None:
    """Latest ``BarraID/<pattern>.YYYYMMDD`` on or before ``date`` (else the earliest, WARNING)."""
    id_dir = repo / _BARRA_ID_SUBDIR
    if not id_dir.exists():
        return None
    candidates = sorted((m.group(1), f) for f in id_dir.glob(f"{pattern}.*")
                        if not f.is_dir() and (m := _DATE_SUFFIX_RE.search(f.name)))
    if not candidates:
        return None
    le = [c for c in candidates if c[0] <= date]
    if le:
        return le[-1][1]
    log.warning("barra id: no %s file on or before %s, fallback to earliest %s", pattern, date, candidates[0][1].name)
    return candidates[0][1]


def build_barrid_to_windcode(date: str, repo: Path | None = None) -> dict[str, str]:
    repo = repo or repo_root()
    src = _find_latest_id_file(_LOCALID_PATTERN, date, repo)
    if src is None:
        raise FileNotFoundError(f"barra LOCALID file missing under {repo / _BARRA_ID_SUBDIR} for {date}")
    df = parse_barra_file(src)
    missing = {"Barrid", "AssetIDType", "AssetID", "StartDate", "EndDate"} - set(df.columns)
    if missing:
        raise ValueError(f"unexpected LOCALID schema in {src}: missing {sorted(missing)}")
    df = df[df["AssetIDType"].astype(str).str.upper() == "LOCALID"]
    df = df[(df["StartDate"].astype(str) <= date) & (df["EndDate"].astype(str) >= date)]
    df = df.sort_values("StartDate").drop_duplicates("Barrid", keep="last")
    mapping = {}
    for barrid, asset_id in zip(df["Barrid"].astype(str), df["AssetID"].astype(str), strict=True):
        wc = _assetid_to_windcode(asset_id)
        if wc is not None:
            mapping[barrid] = wc
    log.info("barra id %s: %s -> %d Barrids mapped", date, src.name, len(mapping))
    return mapping


def reindex_barrid_to_windcode(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    win = df.index.to_series().astype(str).map(mapping)
    keep = win.notna().to_numpy()
    n_drop = int((~keep).sum())
    if n_drop:
        lvl = logging.WARNING if n_drop > 0.05 * len(keep) else logging.INFO
        log.log(lvl, "barra id: %d/%d Barrids without a Wind code dropped (e.g. %s)", n_drop, len(keep),
                df.index[~keep][:5].tolist())
    out = df.loc[keep].copy()
    out.index = win[keep].to_numpy()
    dup = out.index.duplicated(keep="first")
    if dup.any():
        log.warning("barra id: %d duplicate Wind codes after remap, keeping first", int(dup.sum()))
        out = out.loc[~dup]
    out.index.name = df.index.name
    return out.sort_index()
