"""Thin, strict WindPy session wrapper: connect once, raise on any non-zero ErrorCode."""

from __future__ import annotations

from typing import Any

import polars as pl

__all__ = ["WindError", "call", "connect"]

_session: Any = None


class WindError(RuntimeError):
    """A WindPy call returned a non-zero ErrorCode. Never swallowed: bad data is worse."""


def connect(wait_s: int = 60) -> Any:
    """Start (or reuse) the WindPy session. Requires the Wind terminal running and logged in."""
    global _session
    if _session is not None and _session.isconnected():
        return _session
    try:
        from WindPy import w  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - Windows + terminal only
        raise WindError(
            "WindPy not importable. It exists only on Windows with the Wind terminal installed; "
            "the venv needs a WindPy.pth pointing at C:\\Wind\\Wind.NET.Client\\WindNET\\x64"
        ) from exc
    r = w.start(waitTime=wait_s)
    if r.ErrorCode != 0 or not w.isconnected():
        raise WindError(
            f"w.start failed (ErrorCode {r.ErrorCode}: {r.Data}). Is the Wind terminal open and "
            "logged in? WindPy cannot log in by itself."
        )
    _session = w
    return w


def call(func: str, *args: Any, **kwargs: Any) -> pl.DataFrame:
    """Invoke ``w.<func>(*args, usedf=True, **kwargs)`` and return a polars frame.

    The index WindPy puts on the pandas frame (dates for wsd/wsi, codes for wss) becomes a
    regular column named ``index`` so nothing is lost in the conversion.
    """
    w = connect()
    out = getattr(w, func)(*args, usedf=True, **kwargs)
    err, df = out
    if err != 0:
        raise WindError(f"w.{func}{args} -> ErrorCode {err}: {df}")
    return pl.from_pandas(df.reset_index())
