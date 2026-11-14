r"""Input / output of the C++ base-quantity program (``cpp/featgen.cpp``) and its Python reference.

``write_inputs`` packs each stock's snapshot ("fast"), trade ("slow") and order frames -- the frames ``ic_score`` hands to
``generic`` / ``lit_factors`` / ``tick_flow`` -- into one binary file; ``run`` builds and starts the program; ``read_output``
returns ``{sid: {name: values at T}}``. ``reference`` computes the same quantities with the Python definitions, so the two
can be compared value by value (``selftest.py`` on synthetic ticks, ``compare`` on real days).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pm_factors"))
sys.path.insert(0, str(HERE.parent))

CPP = HERE / "cpp"
# column order = the enums of featgen.cpp
FAST = (["t_local", "s", "price", "op", "pc", "b1", "a1", "bv1", "av1", "cv", "ca", "bv5", "av5", "bwp5", "awp5", "ofi2_5", "ul", "ll"]
        + [f"ofil{i}" for i in range(2, 7)] + [f"bq{i}" for i in range(2, 7)] + [f"aq{i}" for i in range(2, 7)])
SLOW = ["t_local", "s", "price", "vol", "amt", "side", "bsq", "ssq"]
ORD = ["t_local", "s", "kind", "side", "amt", "px", "seq"]
SEED = ["zzpk.low", "zzeg.c.high", "zzeg.c.low"]
NULL_AS = {"bsq": -1.0, "ssq": -1.0}  # lit_factors._size_tiers fills these nulls with -1


def _cols(g: pl.DataFrame | None, names: list[str]) -> np.ndarray:
    """The frame's columns as float64 [len(names), rows]; a missing column or a null is NaN (-1 for bsq / ssq)."""
    if g is None or g.height == 0:
        return np.zeros((len(names), 0))
    out = np.full((len(names), g.height), np.nan)
    for i, c in enumerate(names):
        if c in g.columns:
            col = g[c].cast(pl.Float64)
            out[i] = (col.fill_null(NULL_AS[c]) if c in NULL_AS else col).to_numpy()
    return out


def write_inputs(dst: Path, T: np.ndarray, stocks: list[dict]) -> None:
    """``stocks``: dicts with sid, gf, gs, go (frames or None) and seed ({state key: value}, yesterday's state)."""
    with dst.open("wb") as f:
        f.write(b"CRUFG001")
        f.write(np.array([len(stocks), T.size], "<u8").tobytes())
        f.write(np.asarray(T, "<f8").tobytes())
        for st in stocks:
            a, b, c = _cols(st.get("gf"), FAST), _cols(st.get("gs"), SLOW), _cols(st.get("go"), ORD)
            sd = st.get("seed") or {}
            f.write(np.array([st["sid"]], "<i8").tobytes())
            f.write(np.array([a.shape[1], b.shape[1], c.shape[1]], "<u8").tobytes())
            f.write(np.array([sd.get(k, np.nan) for k in SEED], "<f8").tobytes())
            for m in (a, b, c):
                f.write(np.ascontiguousarray(m, "<f8").tobytes())


def ensure_binary(build: Path | None = None) -> Path:
    """Build featgen when it is missing or older than its source (same flags as the state engine: no fast-math)."""
    build = build or CPP / "build"
    exe, src = build / "featgen", CPP / "featgen.cpp"
    if exe.exists() and exe.stat().st_mtime >= src.stat().st_mtime:
        return exe
    build.mkdir(parents=True, exist_ok=True)
    cxx = shutil.which("g++") or "g++"
    subprocess.run([cxx, "-std=c++20", "-O2", "-DNDEBUG", "-ffp-contract=off", "-pthread", str(src), "-o", str(exe)], check=True)
    return exe


def run(inp: Path, out: Path, names: Path, threads: int = 8, exe: Path | None = None) -> None:
    subprocess.run([str(exe or ensure_binary()), str(inp), str(out), "--names", str(names), "--threads", str(threads)], check=True)


def read_output(out: Path, names: Path) -> tuple[list[str], dict[int, np.ndarray]]:
    """(base names, {sid: array [n_base, n_T]})."""
    nm = names.read_text().split()
    b = out.read_bytes()
    assert b[:8] == b"CRUFO001", "not a featgen output file"
    ns, nt, nb = (int(x) for x in np.frombuffer(b, "<u8", 3, 8))
    assert nb == len(nm), (nb, len(nm))
    rec = np.frombuffer(b, np.dtype([("sid", "<i8"), ("v", "<f8", (nb, nt))]), ns, 32)
    return nm, {int(r["sid"]): r["v"] for r in rec}


def reference(gf: pl.DataFrame | None, gs: pl.DataFrame | None, go: pl.DataFrame | None, seed: dict, T: np.ndarray) -> dict[str, np.ndarray]:
    """The Python definitions at T: what ``ic_score`` puts under ``gen.`` (order flow on), ``seed_feats`` and ``px``."""
    import dcp_ema as E
    import feature_intraday as FI
    import ic_score as IC
    import tick_flow as TF
    from lit_factors import order_stream

    gen = IC.generic(gf, gs, T)
    gen.update({f"flow_{k}": v for k, v in TF.flow(go, gs, gf, T).items()})
    gen.update(order_stream(go, gf, gs, T))
    if gf is not None and gf.height > 2:
        a = E.arrays(gf, "fast")
        old, IC.MARKS = IC.MARKS, T  # seed_feats reads the module's grid
        try:
            gen.update(IC.seed_feats(seed or {}, a))
        finally:
            IC.MARKS = old
        gen["px"] = FI.at_marks(a["t_local"], a["price"], T)
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in ((k, np.broadcast_to(np.asarray(v, float), T.shape)) for k, v in gen.items())}


def compare(names: list[str], got: dict[int, np.ndarray], ref: dict[int, dict[str, np.ndarray]], tol: float = 1e-9,
            share: float = 0.005) -> tuple[int, list[str]]:
    """Per base quantity over all stocks: cells, cells where one side is NaN and the other not, cells differing by more
    than ``tol`` (relative), the largest difference. A quantity is off on any NaN mismatch, or when more than ``share`` of
    its cells differ: the snapshot-signed flows (simb*) sign a volume by exp(log(mid)) against a price, and a tie there
    falls either way with the last bit of exp / log (numpy's and the C library's differ), a handful of cells a day.
    Returns (number of quantities off, report lines)."""
    lines, bad = [], 0
    for i, nm in enumerate(names):
        cells = nan_off = n_ref = n_diff = 0
        worst = 0.0
        for sid, v in got.items():
            r = ref[sid].get(nm)
            if r is None:
                r = np.full(v.shape[1], np.nan)
            c = v[i]
            fa, fb = np.isfinite(c), np.isfinite(r)
            cells += c.size
            n_ref += int(fb.sum())
            nan_off += int((fa != fb).sum())
            both = fa & fb
            if both.any():
                d = np.abs(c[both] - r[both]) / np.maximum(np.abs(r[both]), 1e-12)
                n_diff += int((d > tol).sum())
                worst = max(worst, float(d.max()))
        off = nan_off > 0 or n_diff > share * max(n_ref, 1) or n_ref == 0
        bad += off
        lines.append(f"  {'OFF' if off else 'ok '} {nm:<52} finite in ref {n_ref:>8} / {cells:<8} NaN mismatches {nan_off:>6}  "
                     f"cells differing {n_diff:>6}  max rel diff {worst:.3g}")
    return bad, lines
