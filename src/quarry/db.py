"""DuckDB connection and views over the Parquet partition tree.

The storage layout is ``<dataset>/date=YYYYMMDD/symbol=NNNNNN/*.parquet``.
DuckDB registers *views* over that tree rather than loading frames, so a query
touching three symbols on one date reads three files, not a year of the panel.
Partition pruning is the entire performance story at tick resolution: the
difference between a 40ms read and a 40GB one is whether the predicate lands on
``date``/``symbol`` before the scan.

Hive partition columns are pinned to VARCHAR via ``hive_types``. Left to
inference, ``symbol=000001`` becomes the integer 1, which then fails to join
against the string ``"000001"`` everywhere else in the stack -- and fails by
silently matching nothing rather than by erroring.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

if TYPE_CHECKING:
    import duckdb

__all__ = [
    "DataRoot",
    "DATASETS",
    "open_db",
    "register_view",
    "stream",
    "hive_date",
]

DATASETS: Mapping[str, str] = {
    "factor_frame": "factor_frame",
    "daily": "daily",
    "index": "index",
    "ticks": "ticks",
    "actions": "actions",
    "membership": "membership",
    "listings": "listings",
}
"""Dataset name -> subdirectory under the data root.

Bar datasets are frequency-scoped (``bars/1min``) and registered separately by
:meth:`DataRoot.bar_datasets`, since each frequency is its own partition tree.
"""


@dataclass(frozen=True)
class DataRoot:
    """Filesystem layout of a crucible data store.

    Parameters
    ----------
    root:
        Directory containing the dataset subdirectories.
    """

    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", Path(self.root))

    def path(self, dataset: str) -> Path:
        """Directory for ``dataset``, whether or not it exists."""
        return self.root / DATASETS.get(dataset, dataset)

    def glob(self, dataset: str) -> str:
        """Recursive Parquet glob for ``dataset``, POSIX-separated.

        DuckDB wants forward slashes even on Windows; a backslash glob matches
        nothing and returns an empty result instead of an error.
        """
        return (self.path(dataset) / "**" / "*.parquet").as_posix()

    def exists(self, dataset: str) -> bool:
        """Whether ``dataset`` has at least one Parquet file."""
        p = self.path(dataset)
        return p.is_dir() and any(p.rglob("*.parquet"))

    def hive_keys(self, dataset: str) -> tuple[str, ...]:
        """Which hive keys actually appear in ``dataset``'s paths.

        Detected rather than assumed: pinning ``hive_types`` for a key that is
        not in the path is an error, and a store may partition by ``date``
        alone or by ``date``/``symbol``. Both are valid layouts and both must
        open without the caller knowing which they have.
        """
        p = self.path(dataset)
        if not p.is_dir():
            return ()
        first = next(iter(p.rglob("*.parquet")), None)
        if first is None:
            return ()
        rel = first.relative_to(p).parts
        return tuple(part.split("=", 1)[0] for part in rel if "=" in part)

    def bar_datasets(self) -> dict[str, str]:
        """Discovered ``bars_<freq>`` datasets keyed by view name."""
        bars = self.root / "bars"
        if not bars.is_dir():
            return {}
        return {
            f"bars_{d.name}": f"bars/{d.name}"
            for d in sorted(bars.iterdir())
            if d.is_dir() and any(d.rglob("*.parquet"))
        }

    def available(self) -> dict[str, str]:
        """Every dataset present under this root, keyed by view name."""
        found = {name: rel for name, rel in DATASETS.items() if self.exists(name)}
        found.update(self.bar_datasets())
        return found


def register_view(
    conn: duckdb.DuckDBPyConnection,
    name: str,
    glob: str,
    *,
    hive_keys: Sequence[str] = ("date", "symbol"),
) -> None:
    """Register ``name`` as a view over a Parquet glob.

    ``union_by_name`` is on so a dataset that gained a column mid-history still
    reads as one view, with nulls before the column existed -- the alternative
    is a hard failure on the first schema evolution.

    ``hive_keys`` are pinned to VARCHAR. Pass only the keys genuinely present in
    the paths; naming an absent key is an error in DuckDB.
    """
    hive_clause = ""
    if hive_keys:
        types = ", ".join(f"'{k}': 'VARCHAR'" for k in hive_keys)
        hive_clause = f", hive_partitioning = 1, hive_types = {{{types}}}"
    conn.execute(
        f"CREATE OR REPLACE VIEW {name} AS "  # noqa: S608 -- name/glob are caller-controlled paths
        f"SELECT * FROM read_parquet('{glob}', union_by_name = 1{hive_clause})"
    )


def open_db(
    root: str | Path,
    *,
    read_only: bool = True,
    threads: int | None = None,
    memory_limit: str = "4GB",
    database: str | Path = ":memory:",
    extra_views: Mapping[str, str] | None = None,
) -> duckdb.DuckDBPyConnection:
    """Open a configured DuckDB connection with views over ``root``.

    Parameters
    ----------
    root:
        Data root containing dataset subdirectories.
    read_only:
        Advisory for research use. Views are created regardless; this sets the
        connection's access mode when ``database`` is a real file.
    threads, memory_limit:
        DuckDB resource caps. The default 4GB limit is deliberately modest:
        DuckDB spills to disk when it hits the cap, so a tick query that would
        have OOM-killed the process instead runs slowly and finishes. Raising
        this trades that safety for speed.
    database:
        ``":memory:"`` by default. Point at a file to persist views and
        DuckDB's statistics across sessions.
    extra_views:
        Additional ``view_name -> glob`` pairs registered after the standard
        datasets.

    Returns
    -------
    A connection with one view per discovered dataset. Datasets that are absent
    are skipped rather than erroring, so a store holding only daily data opens
    fine.

    Notes
    -----
    Nothing is loaded into memory here. The returned connection is a query
    planner over files; ``SELECT *`` without a date predicate will still try to
    read the whole tree.
    """
    import duckdb

    dr = DataRoot(Path(root))
    config: dict[str, Any] = {"memory_limit": memory_limit}
    if threads is not None:
        config["threads"] = threads
    if str(database) != ":memory:" and read_only:
        config["access_mode"] = "READ_ONLY"

    conn = duckdb.connect(str(database), config=config)
    # Deterministic ordering of otherwise-unordered scans; without this, two
    # runs can emit identical rows in different order and break byte-identity.
    conn.execute("SET preserve_insertion_order = true")

    for name in dr.available():
        dataset = name.replace("bars_", "bars/")
        register_view(conn, name, dr.glob(dataset), hive_keys=dr.hive_keys(dataset))

    for name, glob in (extra_views or {}).items():
        register_view(conn, name, glob)

    return conn


def stream(
    conn: duckdb.DuckDBPyConnection,
    sql: str,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
    *,
    batch_rows: int = 1_000_000,
) -> Iterator[pl.DataFrame]:
    """Execute ``sql`` and yield polars frames in batches.

    The streaming counterpart to a plain ``.pl()`` fetch. Use it whenever the
    result could exceed memory -- a day of all-symbol ticks is tens of millions
    of rows, and materialising it is the specific failure this codebase is
    built to avoid.

    Yields
    ------
    polars DataFrames of at most ``batch_rows`` rows, in query order.
    """
    cur = conn.execute(sql, params) if params is not None else conn.execute(sql)
    reader = cur.fetch_record_batch(batch_rows)
    for batch in reader:
        yield pl.from_arrow(batch)  # type: ignore[misc]


def hive_date(col: str = "date") -> pl.Expr:
    """Parse a ``date=YYYYMMDD`` hive column into a polars Date.

    Hive partition values arrive as strings by construction (see module
    docstring); this is the one place that string becomes a real date.
    """
    return pl.col(col).cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d")
