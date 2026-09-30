"""Intranet database access for CED: Wind, JY (聚源), ZY (朝阳永续) over SQLAlchemy.

Connection URLs come from the environment only -- nothing is written into the repo:

* ``CRUCIBLE_WIND_URL`` / ``CRUCIBLE_JY_URL`` / ``CRUCIBLE_ZY_URL``, e.g.
  ``mssql+pymssql://user:pwd@host:1433/WindDB?charset=utf8&tds_version=7.0``
* or ``CRUCIBLE_SHTCOMMON=<path to shtcommon/py>``: reuse the URLs an existing shtcommon
  install is configured with (its ``configs/configs.py`` is loaded by file path, so its own
  ``ced`` / ``db`` packages never shadow ours).

``tds_version=7.0`` is required by the Wind SQL Server (without it FreeTDS negotiates 7.3/7.4
and fails with ``DB-Lib 20002``, although TCP connects).

Every query goes through :func:`read_sql`, which logs caller, table, rows and seconds -- the
first place to look when a dataset comes out empty.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

import pandas as pd

WIND, JY, ZY = "wind", "jydb", "zyyx"
_ENV = {WIND: "CRUCIBLE_WIND_URL", JY: "CRUCIBLE_JY_URL", ZY: "CRUCIBLE_ZY_URL"}
_SHT_ATTR = {WIND: "wind_url", JY: "jy_url", ZY: "zy_url"}
_FROM_RE = re.compile(r"\bFROM\s+([\w.\[\]]+)", re.IGNORECASE)
_engines: dict[str, object] = {}
_lock = threading.Lock()


def _shtcommon_url(db: str) -> str | None:
    root = os.environ.get("CRUCIBLE_SHTCOMMON")
    if not root:
        return None
    path = Path(root) / "configs" / "configs.py"
    spec = importlib.util.spec_from_file_location("_crucible_shtcommon_configs", path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(f"CRUCIBLE_SHTCOMMON: no configs/configs.py under {root}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod.db_config, _SHT_ATTR[db])


def url_for(db: str) -> str:
    """Connection URL for ``db`` (``WIND`` / ``JY`` / ``ZY``) from the environment."""
    url = os.environ.get(_ENV[db]) or _shtcommon_url(db)
    if not url:
        raise RuntimeError(f"no URL for {db}: set {_ENV[db]} or CRUCIBLE_SHTCOMMON")
    return url


def masked(url: str) -> str:
    """URL with the password hidden, for logs."""
    return re.sub(r"(://[^:/@]+:)[^@]*@", r"\1***@", url)


def engine(db: str):
    """One pooled engine per database, created on first use."""
    if db not in _engines:
        url = url_for(db)  # raises RuntimeError when unconfigured (callers fall back, e.g. the calendar cache)
        with _lock:
            if db not in _engines:
                import sqlalchemy

                _engines[db] = sqlalchemy.create_engine(url, echo=False, pool_recycle=300)
    return _engines[db]


def read_sql(sql: str, db: str = WIND, *, what: str | None = None) -> pd.DataFrame:
    """Run one query; log ``caller <- table: rows, seconds`` under the caller's logger."""
    frame = sys._getframe(1)
    log = logging.getLogger(frame.f_globals.get("__name__", __name__))
    if what is None:
        m = _FROM_RE.search(sql)
        what = m.group(1) if m else "?"
    t0 = time.perf_counter()
    with engine(db).begin() as conn:
        df = pd.read_sql(sql, conn)
    log.info("%s <- %s: %d rows, %.1fs", frame.f_code.co_name, what, len(df), time.perf_counter() - t0)
    return df
