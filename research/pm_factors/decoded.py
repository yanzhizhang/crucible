r"""Features decoded from the runtime's value slots (11-26/27), as extra ic_score candidates.

From the engine's state at the marks (gen.dz_*):
  dz_eg = channel mid / its 300 s average  (ZZEG slot 7: [+0x138] = (hi + lo) / 2 over the EMA's mean)
  dz_ba = mean of zzba.r / mean of zzba.p  (ZZBA slot 7: range average over price average, 0 when the price mean <= 0)
  dz_mh = signed sqrt of the mean of zzmh.r  (ZZMH slot 7, up to its constant divisor, which a cal: transform absorbs)
Per-stock constants read at start-up (ced.dz.<FAM>.<V>): the previous trading day's samplerS.nc of ZZSQ (zzop's three
order-age priors ab / a / b) and ZZDS (zzel.b's file), and the day's barra stats.nc (zzel.a's file; sod naming = the
trading day, holding the previous day's values). Read-only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def mean(e: dict[str, np.ndarray], stem: str) -> np.ndarray | None:
    s, w = e.get(f"{stem}.sum"), e.get(f"{stem}.wgt")
    if s is None or w is None:
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(w != 0, s / w, np.nan)


def state_feats(e: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {}
    hi, lo, em = e.get("zzeg.c.high"), e.get("zzeg.c.low"), mean(e, "zzeg.c.rEma")
    with np.errstate(divide="ignore", invalid="ignore"):
        if hi is not None and lo is not None and em is not None:
            out["dz_eg"] = (hi + lo) / 2 / em
        p, r = mean(e, "zzba.p"), mean(e, "zzba.r")
        if p is not None and r is not None:
            out["dz_ba"] = np.where(p > 0, r / p, 0.0)
        m = mean(e, "zzmh.r")
        if m is not None:
            out["dz_mh"] = np.sign(m) * np.sqrt(np.abs(m))
    return out


def _nc_table(path: Path, i_slot: int = 0) -> dict[int, dict[str, float]]:
    """sid -> V label -> value of x[S, 0, i_slot, V] (the samplerS / stats layout)."""
    import h5netcdf
    out: dict[int, dict[str, float]] = {}
    with h5netcdf.File(path, "r") as f:
        lab = lambda k: [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(f.variables[k][...]).ravel().tolist()]  # noqa: E731
        S, V = lab("S"), lab("V")
        x = np.asarray(f.variables["x"][...])[:, 0, min(i_slot, f.variables["x"].shape[2] - 1), :]
    for i, s in enumerate(S):
        code = s.split(".")[0]  # "600000" or a wind code "600000.SH"
        if code.isdigit():
            out[int(code)] = {v: float(x[i, j]) for j, v in enumerate(V)}
    return out


def daily_consts(d: str, hstats: Path, barra: Path, on_day: bool = False, i_slot: int = 0) -> tuple[dict[int, dict[str, float]], list[str]]:
    """sid -> {'<FAM>.<V>': value} for the day's start-up constants, and a note of which files were read.
    ``on_day``: the samplerS folder named D (sod naming: it holds D-1's data) instead of the latest before D; ``i_slot``: the I index."""
    out: dict[int, dict[str, float]] = {}
    note = []
    for fam in ("ZZSQ", "ZZDS"):
        days = sorted(p.name for p in (hstats / fam).glob("2*") if (p.name <= d if on_day else p.name < d) and (p / "samplerS.nc").exists())
        if days:
            for sid, kv in _nc_table(hstats / fam / days[-1] / "samplerS.nc", i_slot).items():
                out.setdefault(sid, {}).update({f"{fam}.{k}": v for k, v in kv.items()})
            note.append(f"{fam} {days[-1]} I={i_slot}")
    prev = sorted(p for p in barra.glob("2*_stats.nc") if p.name[:8] < d)  # on day T barra has only T-1's file (the user, 2026-11-27)
    bf = prev[-1] if prev else barra / f"{d}_stats.nc"
    if bf.exists():
        for sid, kv in _nc_table(bf).items():
            out.setdefault(sid, {}).update({f"barra.{k}": v for k, v in kv.items()})
        note.append(bf.name)
    return out, note
