"""Request-level Parquet cache for WindPy. A request that has been answered is never resent."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import polars as pl

from crucible.determinism import stable_hash

__all__ = ["CACHE_ROOT", "cached_request", "request_log"]

CACHE_ROOT = Path(__file__).resolve().parents[2] / "data" / "wind_cache"
_LOG = CACHE_ROOT / "_requests.parquet"


def _key(func: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    return stable_hash(func, args, sorted(kwargs.items()))[:24]


def cached_request(func: str, *args: Any, refresh: bool = False, **kwargs: Any) -> pl.DataFrame:
    """Return the cached answer to ``w.<func>(*args, **kwargs)``, fetching it only once.

    Parameters
    ----------
    func:
        WindPy function name (``wsd``, ``wss``, ``wset``, ``wsi``, ``wst``, ``tdays`` ...).
    refresh:
        Refetch even if cached. Only for data that is known to have been restated; it costs
        quota, so it is never the default.

    Notes
    -----
    The request itself is stored next to the answer (``<hash>.json``), so the cache is
    self-describing and an entry can be audited or deleted by hand.
    """
    key = _key(func, args, kwargs)
    d = CACHE_ROOT / func
    data, meta = d / f"{key}.parquet", d / f"{key}.json"
    if data.exists() and not refresh:
        return pl.read_parquet(data)

    from wind.client import call  # imported lazily: reading the cache must work without WindPy

    df = call(func, *args, **kwargs)
    d.mkdir(parents=True, exist_ok=True)
    tmp = data.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(data)
    meta.write_text(
        json.dumps(
            {"func": func, "args": list(args), "kwargs": kwargs},
            ensure_ascii=False,
            default=str,
            indent=1,
        ),
        encoding="utf-8",
    )
    _append_log(func, key, df.height, args)
    return df


def _append_log(func: str, key: str, rows: int, args: tuple[Any, ...]) -> None:
    row = pl.DataFrame(
        {
            "at": [dt.datetime.now(dt.UTC).isoformat(timespec="seconds")],
            "func": [func],
            "key": [key],
            "rows": [rows],
            "args": [repr(args)[:500]],
        }
    )
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    if _LOG.exists():
        row = pl.concat([pl.read_parquet(_LOG), row])
    row.write_parquet(_LOG)


def request_log() -> pl.DataFrame:
    """Every request that actually went to Wind (quota accounting)."""
    return pl.read_parquet(_LOG) if _LOG.exists() else pl.DataFrame()
