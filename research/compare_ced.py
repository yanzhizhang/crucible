"""Compare our CED Parquet (``research/ced``) with the production CED ``.xr`` files, day by day.

Intranet only (reads the netCDF files with xarray -- one of the files allowed to). Every CED
file is turned into the shape we store and compared:

* numeric panels (SOD, EOD, index EOD, index weights, ST, dividends, Barra): cell by cell,
  ``|ours - ced| <= atol + rtol * |ced|`` (default 1e-9 / 1e-9), both-NaN equal, CED's ``-1``
  integer sentinels read as null;
* industry: the one-hot ``(symbol, industry_code)`` file of each level -> one code per symbol;
* concept: the set of ``(symbol, concept_code)`` pairs.

Our live and hist versions of SOD and index weights are each compared with the one CED file,
which also shows which basis the production files are on.

    python research/compare_ced.py --ced /data/share/CED --start 20260401 --end 20260430
    python research/compare_ced.py --ced /data/prod/CEDxr4d --ext .nc      # the PM's research CED
    python research/compare_ced.py --ced /data/prod/CEDxr4d/sod --ext .nc  # its pre-open copy

A CED root may hold only some datasets (CEDxr4d: daily, index, barra): datasets whose directory
is absent are skipped once, not reported missing every day. Without ``--start`` / ``--end`` the
range is the days present in the root (``daily/*_SOD``, else ``index/*``, else ``barra/*_exp``).

Output (``--out``, default ``data/reports/ced_vs_share/<ts>``): ``summary.parquet`` /
``summary.csv`` (one row per dataset x day: rows on each side, symbols on one side only,
cells that differ, worst columns, rc 0 identical / 1 differences in <= 1 % of rows / 2 more /
3 a file missing) and ``cells.parquet`` (every differing cell, capped per day); a pivot of
the summary is printed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ced import store  # noqa: E402
from ced.calendar import SZSE, calendar  # noqa: E402
from ced.check import compare_frames  # noqa: E402

MAX_CELLS_PER_DAY = 2000
# per-dataset (rtol, atol) where the default is too strict: index weights come out of a drift /
# renormalisation whose float order differs (~1e-7 relative); CED's own check used atol 1e-5
TOLERANCE = {"index_universe": (1e-6, 1e-9), "index_universe_hist": (1e-6, 1e-9)}

# ours -> (CED path pattern relative to the CED root, kind)
NUMERIC = {
    "daily_sod": "daily/{d}_SOD{ext}",
    "daily_sod_hist": "daily/{d}_SOD{ext}",
    "daily_eod": "daily/{d}_EOD{ext}",
    "index_eod": "daily/{d}_idx_EOD{ext}",
    "index_universe": "index/{d}{ext}",
    "index_universe_hist": "index/{d}{ext}",
    "st": "st/{d}{ext}",
    "wind_div": "wind_div/{d}{ext}",
    "zy_div": "zy_div/{d}{ext}",
    "barra_exp": "barra/{d}_exp{ext}",
    "barra_stats": "barra/{d}_stats{ext}",
    "barra_rate": "barra/{d}_rate{ext}",
    "barra_fret": "barra/{d}_fret{ext}",
    "barra_cov": "barra/{d}_cov{ext}",
}
LEVELS = {"sw_industry": (1, 2, 3), "wind_industry": (1, 2, 3, 4)}


def _labels(values) -> list[str]:
    out = []
    for v in np.asarray(values).ravel():
        s = v.decode("ascii", "replace") if isinstance(v, bytes | np.bytes_) else str(v)
        out.append(s.rstrip("\x00").strip())
    return out


def load(path: Path) -> xr.Dataset:
    for engine in ("netcdf4", "h5netcdf"):
        try:
            with xr.open_dataset(path, engine=engine) as src:
                return src.load()
        except (ValueError, ImportError, ModuleNotFoundError):
            continue
    raise RuntimeError(f"cannot open {path} with netcdf4 or h5netcdf")


def xr4d_frame(ds: xr.Dataset) -> tuple[pd.DataFrame, str | None]:
    """CEDxr4d layout: one variable ``x`` over (S, D, I, V) -> S x V frame, and the D it holds.

    A ``sod/`` file named T holds D = T-1 (the start-of-day view), so callers compare with our
    dataset at D, not at the file's date.
    """
    da = ds[next(iter(ds.data_vars))]
    day = _labels(ds["D"].values)[0] if "D" in ds.coords else None
    sel = {d: 0 for d in da.dims if d not in ("S", "V")}
    two = da.isel(sel).transpose("S", "V")
    df = pd.DataFrame(two.values.astype("float64"), index=pd.Index(_labels(ds["S"].values), name="S"),
                      columns=_labels(ds["V"].values))
    return df.sort_index(), day


def _numeric_ids(idx: pd.Index) -> bool:
    """CEDxr4d stock keys are bare numbers ('1' = 000001) where CED used Wind codes."""
    return len(idx) > 0 and all(str(x).isdigit() for x in idx[:50])


def to_numeric_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Our Wind codes -> CEDxr4d keys: SH / SZ as bare numbers (000001.SZ -> '1'), BSE kept as is
    (CEDxr4d writes 920000.BJ with its suffix)."""

    def key(x: str) -> str:
        code, _, suffix = str(x).partition(".")
        return str(int(code)) if suffix in ("SH", "SZ") and code.isdigit() else str(x)

    out = df.copy()
    out.index = [key(x) for x in out.index]
    return out[~out.index.duplicated(keep="first")]


def xr_frame(ds: xr.Dataset) -> pd.DataFrame:
    """Row dim (symbol_id or factor) x columns: 1-D vars by name, 2-D vars by their column label."""
    row_dim = next(iter(ds[next(iter(ds.data_vars))].dims))
    cols: dict[str, np.ndarray] = {}
    for name, da in ds.data_vars.items():
        if da.dims == (row_dim,):
            cols[str(name)] = da.values
        elif len(da.dims) == 2 and da.dims[0] == row_dim:
            for j, lab in enumerate(_labels(ds[da.dims[1]].values)):
                cols[lab] = da.values[:, j]
    df = pd.DataFrame(cols, index=pd.Index(_labels(ds[row_dim].values), name=row_dim))
    for c in df.columns:
        if df[c].dtype.kind in "iu":
            v = df[c].to_numpy(dtype=np.float64)
            v[v == -1.0] = np.nan  # CED's INT64_NULL / INT8_NULL
            df[c] = v
    return df.astype("float64").sort_index()


def ours_frame(dataset: str, date: str) -> pd.DataFrame:
    df = store.read_day(dataset, date)
    key = "factor" if dataset in ("barra_fret", "barra_cov") else "symbol"
    df = df.set_index(key)
    if dataset == "st":
        df.columns = [c.removeprefix("st_") for c in df.columns]
    num = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[num].astype("float64").sort_index()


def _row(dataset: str, date: str, path: Path, **kw) -> dict:
    return {"dataset": dataset, "date": date, "ced_file": str(path), **kw}


EXT = ".xr"


def present(root: Path) -> set[str]:
    """Datasets whose directory exists under this CED root."""
    have = {ds for ds, pat in NUMERIC.items() if (root / pat.split("/")[0]).is_dir()}
    have |= {ds for ds in LEVELS if (root / ds).is_dir()}
    return have | ({"concept"} if (root / "concept").is_dir() else set())


def root_days(root: Path) -> list[str]:
    for pat in ("daily/*_SOD", "index/*", "barra/*_exp"):
        days = sorted({f.name[:8] for f in root.glob(pat + EXT) if f.name[:8].isdigit()})
        if days:
            return days
    return []


def compare_numeric(dataset: str, date: str, root: Path, rtol: float, atol: float) -> tuple[dict, pd.DataFrame]:
    path = root / NUMERIC[dataset].format(d=date, ext=EXT)
    if not path.exists():
        return _row(dataset, date, path, rc=3, note="missing: ced file"), pd.DataFrame()
    ds = load(path)
    if {"S", "V"} <= set(ds.dims):  # CEDxr4d
        ced, held = xr4d_frame(ds)
        day = held or date
    else:
        ced, day = xr_frame(ds), date
    if not store.exists(dataset, day):
        return _row(dataset, date, path, rc=3, note=f"missing: ours {dataset} {day}"), pd.DataFrame()
    ours = ours_frame(dataset, day)
    if dataset == "barra_fret" and len(ced) == 1 and "factor_return" in ours.columns:
        # CEDxr4d stores factor returns as one row x factors; ours is factors x one column
        ours = ours[["factor_return"]].T.set_axis(ced.index, axis=0)
    if _numeric_ids(ced.index):
        ours = to_numeric_ids(ours)
    rtol, atol = TOLERANCE.get(dataset, (rtol, atol))
    r = compare_frames(ours, ced, names=("ours", "ced"), rtol=rtol, atol=atol, warn_frac=0.01)
    cells = r.mismatches()
    worst = cells[~cells["column"].str.startswith("<")]["column"].value_counts().head(5)
    return _row(dataset, date, path, rc=r.rc, data_day=day, ours_rows=len(ours), ced_rows=len(ced),
                only_ours=int(ours.index.difference(ced.index).size), only_ced=int(ced.index.difference(ours.index).size),
                cells_diff=int((~cells["column"].str.startswith("<")).sum()),
                worst=", ".join(f"{c} {n}" for c, n in worst.items()), note=r.summary[:500]), cells


def _onehot_codes(ds: xr.Dataset) -> pd.Series:
    """One-hot (symbol_id, code) -> the code of each symbol ('None' / 'nan' codes -> null)."""
    var = ds[next(iter(ds.data_vars))]
    codes = np.array(_labels(ds[var.dims[1]].values), dtype=object)
    syms = _labels(ds["symbol_id"].values)
    arr = var.values
    out = {}
    for i, s in enumerate(syms):
        hit = np.flatnonzero(arr[i] == 1)
        out[s] = codes[hit[0]] if len(hit) else None
    ser = pd.Series(out, dtype=object)
    return ser.where(~ser.isin(["None", "nan", ""]))


def compare_industry(dataset: str, date: str, root: Path) -> list[tuple[dict, pd.DataFrame]]:
    res = []
    have = store.exists(dataset, date)
    ours = store.read_day(dataset, date).set_index("symbol") if have else None
    for lvl in LEVELS[dataset]:
        path = root / dataset / f"{date}_ind{lvl}{EXT}"
        name = f"{dataset}_ind{lvl}"
        if not path.exists() or ours is None:
            res.append((_row(name, date, path, rc=3, note=f"missing: ced={path.exists()} ours={have}"), pd.DataFrame()))
            continue
        ced = _onehot_codes(load(path)).dropna()
        mine = ours[f"level{lvl}"].dropna()
        both = mine.index.intersection(ced.index)
        diff = both[mine[both].to_numpy() != ced[both].to_numpy()]
        only_o, only_c = mine.index.difference(ced.index), ced.index.difference(mine.index)
        n_bad = len(diff) + len(only_o) + len(only_c)
        rc = 0 if n_bad == 0 else (1 if n_bad <= 0.01 * max(len(both), 1) else 2)
        cells = pd.DataFrame({"symbol": list(diff) + list(only_o) + list(only_c),
                              "column": [f"level{lvl}"] * len(diff) + ["<only in ours>"] * len(only_o)
                              + ["<only in ced>"] * len(only_c),
                              "left": [mine[s] for s in diff] + [mine[s] for s in only_o] + [None] * len(only_c),
                              "right": [ced[s] for s in diff] + [None] * len(only_o) + [ced[s] for s in only_c]})
        res.append((_row(name, date, path, rc=rc, ours_rows=len(mine), ced_rows=len(ced), only_ours=len(only_o),
                         only_ced=len(only_c), cells_diff=len(diff)), cells.astype(str)))
    return res


def compare_concept(date: str, root: Path) -> tuple[dict, pd.DataFrame]:
    path = root / "concept" / f"{date}{EXT}"
    have = store.exists("concept", date)
    if not path.exists() or not have:
        return _row("concept", date, path, rc=3, note=f"missing: ced={path.exists()} ours={have}"), pd.DataFrame()
    ds = load(path)
    var = ds[next(iter(ds.data_vars))]
    syms, codes = _labels(ds["symbol_id"].values), _labels(ds[var.dims[1]].values)
    ii, jj = np.nonzero(var.values == 1)
    ced = {(syms[i], codes[j]) for i, j in zip(ii, jj, strict=True)}
    ours_df = store.read_day("concept", date)
    ours = set(zip(ours_df["symbol"], ours_df["concept_code"], strict=True))
    only_o, only_c = sorted(ours - ced), sorted(ced - ours)
    n_bad = len(only_o) + len(only_c)
    rc = 0 if n_bad == 0 else (1 if n_bad <= 0.01 * max(len(ced), 1) else 2)
    cells = pd.DataFrame({"symbol": [s for s, _ in only_o + only_c],
                          "column": ["<only in ours>"] * len(only_o) + ["<only in ced>"] * len(only_c),
                          "left": [c for _, c in only_o] + [None] * len(only_c),
                          "right": [None] * len(only_o) + [c for _, c in only_c]})
    return _row("concept", date, path, rc=rc, ours_rows=len(ours), ced_rows=len(ced), only_ours=len(only_o),
                only_ced=len(only_c), cells_diff=0), cells.astype(str)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ced", type=Path, default=Path("/data/share/CED"))
    ap.add_argument("--start", help="default: first day present in the CED root")
    ap.add_argument("--end", help="default: last day present in the CED root")
    ap.add_argument("--ext", default=".xr", help="file extension of the CED files (.xr, or .nc for CEDxr4d)")
    ap.add_argument("--rtol", type=float, default=1e-9)
    ap.add_argument("--atol", type=float, default=1e-9)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    global EXT
    EXT = a.ext
    stamp = dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).strftime("%Y%m%d_%H%M%S")
    out = a.out or Path("data/reports/ced_vs_share") / stamp
    out.mkdir(parents=True, exist_ok=True)

    rows, cells = [], []

    def add(row: dict, c: pd.DataFrame) -> None:
        rows.append(row)
        if len(c):
            cells.append(c.head(MAX_CELLS_PER_DAY).astype(str).assign(dataset=row["dataset"], date=row["date"]))

    have = present(a.ced)
    skipped = sorted((set(NUMERIC) | set(LEVELS) | {"concept"}) - have)
    if a.start and a.end:
        days = calendar.trade_days(a.start, a.end, SZSE)
    else:
        days = [d for d in root_days(a.ced) if (not a.start or d >= a.start) and (not a.end or d <= a.end)]
    if not days:
        raise SystemExit(f"no days to compare under {a.ced} (ext {EXT})")
    print(f"{len(days)} days {days[0]}..{days[-1]}, CED root {a.ced} ({EXT}), ours {store.STORE}")
    print(f"datasets compared: {sorted(have)}; not in this root, skipped: {skipped}")
    for d in days:
        for ds in [x for x in NUMERIC if x in have]:
            try:
                add(*compare_numeric(ds, d, a.ced, a.rtol, a.atol))
            except Exception as e:  # one broken file must not stop the month
                add(_row(ds, d, a.ced, rc=3, note=f"error: {type(e).__name__}: {e}"), pd.DataFrame())
        for ds in [x for x in LEVELS if x in have]:
            try:
                for row, c in compare_industry(ds, d, a.ced):
                    add(row, c)
            except Exception as e:
                add(_row(ds, d, a.ced, rc=3, note=f"error: {type(e).__name__}: {e}"), pd.DataFrame())
        if "concept" in have:
            try:
                add(*compare_concept(d, a.ced))
            except Exception as e:
                add(_row("concept", d, a.ced, rc=3, note=f"error: {type(e).__name__}: {e}"), pd.DataFrame())
        print(f"  {d} done")

    summary = pd.DataFrame(rows)
    summary.to_parquet(out / "summary.parquet", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    if cells:
        pd.concat(cells, ignore_index=True).to_parquet(out / "cells.parquet", index=False)
    rc_names = {0: "identical", 1: "minor", 2: "differs", 3: "missing"}
    piv = (summary.assign(result=summary["rc"].map(rc_names)).pivot_table(
        index="dataset", columns="result", values="date", aggfunc="count", fill_value=0))
    agg = summary.groupby("dataset").agg(cells_diff=("cells_diff", "sum"), only_ours=("only_ours", "sum"),
                                         only_ced=("only_ced", "sum"))
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print("\ndays per result:")
        print(piv.join(agg))
        worst = summary[summary["rc"].isin([1, 2])].groupby("dataset")["worst"].first()
        if len(worst):
            print("\nworst columns (first differing day):")
            print(worst.to_string())
        miss = summary[summary["rc"] == 3].groupby("dataset")["note"].first()
        if len(miss):
            print("\nmissing / errors (first):")
            print(miss.to_string())
    print(f"\nreports: {out}/summary.csv, cells.parquet")


if __name__ == "__main__":
    main()
