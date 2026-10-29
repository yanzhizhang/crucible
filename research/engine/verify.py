r"""Check an engine day-end state against PM's ``.jz`` and / or against the vectorized research rebuild.

Per key the share of stocks where the two agree (|diff| <= max(1e-6, 1e-9 |ref|)); averaged values (``<p>.sum / .wgt``)
are compared as ``sum / wgt``; grid families (``<p>.c.<n>``) as the share of equal bins; clocks (``.t``) within 10 ms.

* ``--pm <jz>``: the real test (intranet) -- solved keys must read 1.0 on SZSE;
* ``--reference``: the vectorized rebuild (``feature_intraday.rebuild`` at the stop, ``dcp_build.build`` scalars,
  the ``diag_slots`` grid helpers) -- the engine must reproduce it exactly, anywhere our ticks are.

    python research/engine/verify.py --date 20261009 --engine data/engine/engine_20261009.json.gz --pm /data/prod/h/HS300/p1_intrading_dcp/20261009.jz
    python research/engine/verify.py --date 20261009 --engine data/engine/engine_20261009.json.gz --reference --stop 54150
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import sys
from pathlib import Path

import duckdb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))

import dcp_build as B  # scalars, state loader
import dcp_ema as E  # tick loading
import diag_slots as DS  # grid helpers
import feature_intraday as FI  # EMAs at a mark
from state import GH_EDGES, GH_EMPTY, HJ_SAMPLE, JB_DROP

BIN = re.compile(r"^(.+\.[cr])\.(\d+)$")  # <p>.c.<n> bins and <p>.r.<k> grid points fold into one family


def load_engine(path: Path) -> dict[int, dict[str, float]]:
    """Engine output as sid -> key -> value."""
    with gzip.open(path) as f:
        d = json.load(f)
    return {int(s): {k: (math.nan if v is None else float(v)) for k, v in kv.items()} for s, kv in d.items()}


def values(kv: dict[str, float]) -> dict[str, float]:
    """Keys with <p>.sum / .wgt folded into <p> = sum / wgt."""
    out = {}
    for k, v in kv.items():
        if k.endswith(".sum"):
            w = kv.get(k[:-4] + ".wgt")
            out[k[:-4]] = v / w if w else math.nan
        elif not k.endswith(".wgt"):
            out[k] = v
    return out


def reference(con: duckdb.DuckDBPyConnection, date: int, sids: list[int], stop: float) -> dict[int, dict[str, float]]:
    """Rebuild every solved key at the stop the vectorized way, per stock."""
    tc = stop - 34200 if stop <= 41400 else stop - 46800 + 7200
    marks = np.array([tc])
    up = E.load(con, date, sids, stop)
    fast = {int(g["sid"][0]): g for g in up["fast"].partition_by("sid")}
    slow = {int(g["sid"][0]): g for g in up["slow"].partition_by("sid")}
    scal = B.build(con, date, sids, stop)
    scal = {int(r["sid"]): r for r in scal.iter_rows(named=True)}
    out = {}
    for sid in sids:
        gf, gs = fast.get(sid), slow.get(sid)
        if gf is None and gs is None:
            continue
        kv = {k: float(v[0]) for k, v in FI.rebuild(gf, gs, marks).items() if not k.endswith(".wgt")}
        kv = {(k[:-4] if k.endswith(".sum") else k): v for k, v in kv.items()}
        if sid in scal:
            for k, v in scal[sid].items():
                if k not in ("sid", "mkt") and not k.startswith("clock."):
                    kv[k] = float(v) if v is not None else math.nan
        if gf is not None:
            ft = gf["t_local"].to_numpy().astype(float)
            px = gf["price"].to_numpy().astype(float)
            pxn = np.where(px > 0, px, np.nan)
            b1, a1 = gf["b1"].to_numpy().astype(float), gf["a1"].to_numpy().astype(float)
            mid = np.where((b1 > 0) & (a1 > 0), (b1 + a1) / 2, pxn)
            for k, v in enumerate(DS.last_per_slot(ft, gf["ca"].to_numpy().astype(float), 711)):
                kv[f"zzfb.c.{k}"] = float(v) if np.isfinite(v) else 0.0
            pc = gf["pc"].to_numpy().astype(float)
            pcv = pc[pc > 0][0] if (pc > 0).any() else np.nan
            for k, v in enumerate(DS.last_per_slot(ft, pxn, 948)):
                kv[f"zzjb.c.{k}"] = 0.0 if np.isfinite(v) and np.isfinite(pcv) and v <= pcv * (1 - JB_DROP) else 4.0
            for k, v in enumerate(FI.at_marks(ft, mid, np.array(HJ_SAMPLE))):
                kv[f"zzhj.c.{k}"] = float(v)
        if gs is not None:
            tl = gs["t_local"].to_numpy().astype(float)
            s = gs["s"].to_numpy().astype(float)
            amt, side = gs["amt"].to_numpy().astype(float), gs["side"].to_numpy().astype(float)
            cont = (s >= 34200) & (s < 53820)
            b = DS.per_slot(tl, amt * cont * (side > 0), 948)
            sl = DS.per_slot(tl, amt * cont * (side < 0), 948)
            with np.errstate(divide="ignore", invalid="ignore"):
                x = (b - sl) / (b + sl)
            code = np.where(b + sl > 0, np.searchsorted(np.array(GH_EDGES), x, side="right"), GH_EMPTY)
            for k, v in enumerate(code):
                kv[f"zzgh.c.{k}"] = float(v)
        out[sid] = kv
    return out


def compare(ours: dict[int, dict[str, float]], ref: dict[int, dict[str, float]], mkt: dict[int, int], label: str) -> None:
    """Print agreement per key / grid family, overall and on SZSE."""
    rows = []
    for sid, kv in ours.items():
        rv = ref.get(sid)
        if rv is None:
            continue
        ov = values(kv)
        for k, v in ov.items():
            r = rv.get(k)
            if r is None:
                continue
            fam = BIN.sub(r"\1.<n>", k)
            tol = 0.01 if k.endswith(".t") else max(1e-6, 1e-9 * abs(r))
            ok = (math.isnan(v) and math.isnan(r)) or abs(v - r) <= tol
            rows.append({"key": fam, "sid": sid, "szse": mkt.get(sid) == 3554, "ok": ok, "ours": v, "ref": r})
    if not rows:
        print(f"{label}: nothing to compare")
        return
    t = pl.DataFrame(rows)
    agg = (t.group_by("key").agg(pl.len().alias("n"), pl.col("ok").mean().alias("all"), pl.col("ok").filter(pl.col("szse")).mean().alias("SZSE"),
                                 ((pl.col("ours") - pl.col("ref")).abs() / pl.col("ref").abs()).median().alias("mre"))
           .sort("all", "key"))
    with pl.Config(tbl_rows=200, tbl_width_chars=160, float_precision=4):
        print(f"\n== engine vs {label}: share of stocks (bins) equal, worst first")
        print(agg)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--pm", type=Path, help="PM's day-end state file")
    ap.add_argument("--reference", action="store_true", help="compare with the vectorized research rebuild")
    ap.add_argument("--stop", type=float, default=54150.005, help="local second of day (reference mode; pm mode takes it from the file)")
    ap.add_argument("--catalog", type=Path, default=Path("/work/crucible_data/catalog.duckdb"))
    a = ap.parse_args()
    ours = load_engine(a.engine)
    sids = sorted(ours)
    con = duckdb.connect(str(a.catalog), read_only=True)
    mkt = {int(r[0]): int(r[1]) for r in con.sql(f"SELECT DISTINCT symbol_id, market_id FROM raw_quotation WHERE date = {int(a.date)}").fetchall()}
    if a.pm:
        pm = B.load_state(a.pm)
        tz = pm.filter(pl.col("key") == "zzba.p.t")["pm"].max()
        print(f"PM {a.pm.name}: {pm['sid'].n_unique()} stocks, stop (zzba.p.t) {tz:.3f} on the trading clock")
        ref = {}
        for sid, g in pm.group_by("sid"):
            kv = dict(zip(g["key"].to_list(), g["pm"].to_list(), strict=True))
            ref[int(sid[0])] = values(kv)
        compare(ours, ref, mkt, "PM")
    if a.reference:
        compare(ours, reference(con, int(a.date), sids, a.stop), mkt, "vectorized reference")


if __name__ == "__main__":
    main()
