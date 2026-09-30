"""Intranet database access for CED: Wind, JY (聚源), ZY (朝阳永续) over SQLAlchemy.

The server is fixed here (``10.14.3.20:1433``, databases ``WindDB`` / ``JYDB`` /
``Zyyx2.0``, pymssql, ``tds_version=7.0``). The account: ``DB_USER`` / ``DB_PASSWORD`` below,
filled in on the intranet copy only (never committed with values), or the environment
``CRUCIBLE_DB_USER`` / ``CRUCIBLE_DB_PASSWORD``, which wins when set.
Overrides: a full ``CRUCIBLE_WIND_URL`` / ``CRUCIBLE_JY_URL`` / ``CRUCIBLE_ZY_URL``, or
``CRUCIBLE_SHTCOMMON=<path to shtcommon/py>`` to reuse an existing shtcommon install's URLs
(its ``configs/configs.py`` is loaded by file path, so its ``ced`` / ``db`` never shadow ours).

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
# intranet SQL Server (fixed): host, port, database per source; tds_version=7.0 is mandatory
HOST, PORT = "10.14.3.20", 1433
DATABASES = {WIND: "WindDB", JY: "JYDB", ZY: "Zyyx2.0"}
_QUERY = "charset=utf8&tds_version=7.0"
# account: fill in on the intranet machine only -- never commit these with values
DB_USER = ""
DB_PASSWORD = ""
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
    """Connection URL for ``db`` (``WIND`` / ``JY`` / ``ZY``).

    A full URL in the environment wins; otherwise the fixed intranet server above with the
    account from ``CRUCIBLE_DB_USER`` / ``CRUCIBLE_DB_PASSWORD`` (one account for all three).
    """
    url = os.environ.get(_ENV[db]) or _shtcommon_url(db)
    if url:
        return url
    user = os.environ.get("CRUCIBLE_DB_USER") or DB_USER
    pwd = os.environ.get("CRUCIBLE_DB_PASSWORD") or DB_PASSWORD
    if not user or not pwd:
        raise RuntimeError(f"no database account for {db}: fill DB_USER / DB_PASSWORD in research/ced/db.py, "
                           f"or set CRUCIBLE_DB_USER / CRUCIBLE_DB_PASSWORD (or a full {_ENV[db]})")
    from urllib.parse import quote

    return f"mssql+pymssql://{quote(user, safe='')}:{quote(pwd, safe='')}@{HOST}:{PORT}/{DATABASES[db]}?{_QUERY}"


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
