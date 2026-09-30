"""Inventory PM's result tree: the schema baseline every reproduction is compared against.

Intranet only (reads netCDF with xarray, one of two files allowed to). Walks ``--root``
(default ``/work/prod``) and records, per file: path, size, format, and for each format what
can be read without loading data -- netCDF dims/coords/vars/dtypes (+ first/last coordinate
values), Parquet schema and row count, CSV header and row count, LightGBM text model header,
and for unknown formats the first 16 bytes (magic). Writes ``manifest.parquet`` and a
readable ``manifest.md`` next to it.

Usage::

    python research/pm_inventory.py --root /work/prod --out data/pm_manifest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import pyarrow.parquet as pq

_NC = {".nc", ".xr", ".nc4"}


def _coord_preview(values: Any, k: int = 3) -> str:
    v = list(values)
    if len(v) <= 2 * k:
        return ", ".join(map(str, v))
    return ", ".join(map(str, v[:k])) + " ... " + ", ".join(map(str, v[-k:]))


def _describe_nc(p: Path) -> dict[str, Any]:
    import xarray as xr

    with xr.open_dataset(p) as ds:
        return {
            "dims": json.dumps({k: int(v) for k, v in ds.sizes.items()}),
            "coords": json.dumps(
                {
                    c: {"dtype": str(ds[c].dtype), "values": _coord_preview(ds[c].values)}
                    for c in ds.coords
                },
                ensure_ascii=False,
            ),
            "vars": json.dumps(
                {v: {"dims": list(ds[v].dims), "dtype": str(ds[v].dtype)} for v in ds.data_vars}
            ),
            "attrs": json.dumps({k: str(v) for k, v in ds.attrs.items()}, ensure_ascii=False),
        }


def _describe_parquet(p: Path) -> dict[str, Any]:
    meta = pq.read_metadata(p)
    return {
        "rows": meta.num_rows,
        "vars": json.dumps({f.name: str(f.type) for f in meta.schema.to_arrow_schema()}),
    }


def _describe_text(p: Path, n: int = 5) -> dict[str, Any]:
    with p.open("r", encoding="utf-8", errors="replace") as fh:
        head = [next(fh, "") for _ in range(n)]
    rows = sum(1 for _ in p.open("rb"))
    return {"rows": rows, "head": "".join(head)[:2000]}


def describe(p: Path) -> dict[str, Any]:
    """One manifest row for ``p``; never raises -- a read error is recorded, not hidden."""
    row: dict[str, Any] = {"path": str(p), "suffix": p.suffix.lower(), "bytes": p.stat().st_size}
    with p.open("rb") as fh:
        row["magic"] = fh.read(16).hex()
    try:
        if row["suffix"] in _NC:
            row.update(_describe_nc(p))
        elif row["suffix"] == ".parquet":
            row.update(_describe_parquet(p))
        elif row["suffix"] in {".csv", ".txt"}:
            row.update(_describe_text(p))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=Path("/work/prod"))
    ap.add_argument("--out", type=Path, default=Path("data/pm_manifest"))
    a = ap.parse_args()
    rows = [describe(p) for p in sorted(a.root.rglob("*")) if p.is_file()]
    df = pl.DataFrame(rows, infer_schema_length=None)
    a.out.mkdir(parents=True, exist_ok=True)
    df.write_parquet(a.out / "manifest.parquet")
    lines = [f"# PM manifest of `{a.root}`", "", f"{df.height} files", ""]
    for r in df.iter_rows(named=True):
        lines.append(f"## {r['path']}")
        lines.append(f"- size {r['bytes']:,} B, magic `{r['magic']}`")
        for k in ("dims", "coords", "vars", "rows", "attrs", "head", "error"):
            if r.get(k) not in (None, "", "{}"):
                lines.append(f"- {k}: `{r[k]}`" if k != "head" else f"- head:\n```\n{r[k]}\n```")
        lines.append("")
    (a.out / "manifest.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{df.height} files -> {a.out}")


if __name__ == "__main__":
    main()
