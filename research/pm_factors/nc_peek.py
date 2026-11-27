r"""Structure of PM's netCDF data files: dimensions, variables, coordinate labels and one stock's slice. Read-only.

    python research/pm_factors/nc_peek.py FILE [FILE ...] [--sid 1]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def labels(v) -> list[str]:
    a = np.asarray(v[...]).ravel().tolist()
    return [x.decode() if isinstance(x, bytes) else str(x) for x in a]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--sid", default="1", help="stock label to slice (matches the S / symbol coordinate as text)")
    a = ap.parse_args()
    import h5netcdf
    for p in a.files:
        print(f"== {p} ({p.stat().st_size:,} B)")
        f = h5netcdf.File(p, "r")
        groups = [f] + [f.groups[g] for g in f.groups]
        for g in groups:
            if g is not f:
                print(f"  group {g.name}")
            print("  dims", {k: (len(v) if hasattr(v, "__len__") else v) for k, v in g.dimensions.items()})
            coords = {}
            for k, v in g.variables.items():
                print(f"  var {k} {v.dimensions} {v.shape} {v.dtype}")
                if v.ndim == 1 and k in g.dimensions:
                    coords[k] = labels(v)
            for k, lab in coords.items():
                print(f"   {k}: n {len(lab)} first {lab[:12]}{' ...' if len(lab) > 12 else ''} last {lab[-3:]}")
            sdim = next((d for d in coords if a.sid in coords[d]), None)
            for k, v in g.variables.items():
                if k in coords or sdim is None or sdim not in v.dimensions:
                    continue
                i = coords[sdim].index(a.sid)
                idx = tuple(i if d == sdim else slice(None) for d in v.dimensions)
                x = np.asarray(v[idx])
                flat = x.ravel()
                print(f"   {k}[{sdim}={a.sid}] shape {x.shape}: {np.round(flat[:16].astype(float), 6).tolist() if flat.dtype.kind in 'fiu' else flat[:8].tolist()}"
                      f"{' ...' if flat.size > 16 else ''}")
        f.close()
    print("[done]")


if __name__ == "__main__":
    main()
