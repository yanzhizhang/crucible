r"""Self-test of the C++ engine against the Python one on synthetic events (no catalog needed).

    python research/engine/selftest.py [--stocks 20] [--events 4000] [--seed 1]

Random snapshots and trades (zero prices, missing ranges, auction / lunch / after-close stamps, stamp ties, carry seeds
from a random previous state) go through ``rules.on_snap / on_trade`` and through ``cpp/build/engine``; every key of the
day-end state must match exactly (NaN == NaN).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from export import EVENT_DTYPE, write_events, write_prev, write_rules
from rules import EMAS, on_cancel, on_order, on_snap, on_trade
from run_cpp import ensure_binary
from state import Stock


def tc(s: float) -> float:
    """Trading clock of a second of day (dcp_build.tc)."""
    return s - 34200 if s <= 41400 else (7200.0 if s < 46800 else s - 46800 + 7200)


def synth(rng: np.random.Generator, sid: int, n: int) -> np.ndarray:
    """One stock's random day, sorted like the export."""
    ls = np.sort(rng.integers(33000, 54200, n).astype(float) + rng.integers(0, 3, n) * 0.0)  # whole seconds -> ties
    kind = rng.integers(0, 4, n).astype(np.int16)  # 0 snapshot, 1 trade, 2 order, 3 cancel
    out = np.zeros(n, dtype=EVENT_DTYPE)
    out["sid"], out["mkt"], out["kind"] = sid, 1, kind
    out["lsod"] = ls
    out["s"] = ls - rng.choice([0.0, 1.0, 3.0], n)
    out["t"] = np.array([tc(x) for x in out["lsod"]])
    px = 10.0 + np.cumsum(rng.normal(0, 0.01, n)).round(2)
    px[rng.random(n) < 0.02] = 0.0
    out["price"] = px
    snap = kind == 0
    b1, a1 = px - 0.01, px + 0.01
    b1[rng.random(n) < 0.05] = 0.0
    hi, lo = px + 0.05, px - 0.05
    hi[rng.random(n) < 0.1] = 0.0
    op = np.where(rng.random(n) < 0.9, 10.0, 0.0)
    pc = np.where(rng.random(n) < 0.9, 9.9, 0.0)
    amt = np.cumsum(rng.integers(0, 100000, n).astype(float))
    vol = rng.integers(1, 50, n) * 100.0
    side = rng.choice([1.0, -1.0, 0.0], n)
    trade = kind == 1
    out["f"][:, 0] = np.where(snap, b1, np.where(kind == 3, 0.0, vol))
    out["f"][:, 1] = np.where(snap, a1, np.where(kind == 3, 0.0, side))
    out["price"] = np.where(kind == 3, 0.0, px)
    out["f"][:, 2] = np.where(snap, hi, 0.0)
    out["f"][:, 3] = np.where(snap, lo, 0.0)
    out["f"][:, 4] = np.where(snap, op, 0.0)
    out["f"][:, 5] = np.where(snap, pc, 0.0)
    out["f"][:, 6] = np.where(snap, amt, 0.0)
    order = np.lexsort((kind, ls))
    return out[order]


def py_run(ev: np.ndarray, prev: dict[int, dict[str, float]]) -> dict[int, dict[str, float]]:
    """The Python engine over the same records."""
    out = {}
    for sid in np.unique(ev["sid"]):
        st = Stock(EMAS, prev.get(int(sid)))
        for e in ev[ev["sid"] == sid]:
            f = [float(x) for x in e["f"]]
            if e["kind"] == 0:
                on_snap(st, float(e["t"]), float(e["s"]), float(e["lsod"]), float(e["price"]), *f)
            elif e["kind"] == 1:
                on_trade(st, float(e["t"]), float(e["s"]), float(e["price"]), f[0], f[1])
            elif e["kind"] == 3:
                on_cancel(st, float(e["t"]), float(e["s"]))
            else:
                on_order(st, float(e["t"]), float(e["s"]), float(e["price"]), f[0], f[1])
        out[int(sid)] = st.dump()
    return out


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stocks", type=int, default=20)
    ap.add_argument("--events", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    sids = [600000 + i for i in range(a.stocks)]
    ev = np.concatenate([synth(rng, s, a.events) for s in sids])
    bases = {p.partition("|")[0] for p in EMAS}
    prev = {s: {f"{b}{x}": v for b in bases for x, v in ((".sum", float(rng.normal(1000, 10))), (".wgt", float(rng.integers(0, 3) * 50)))}
            for s in sids[::2]}
    exe = ensure_binary()
    with tempfile.TemporaryDirectory() as d:
        w = Path(d)
        names = write_rules(EMAS, w / "rules.tsv")
        write_prev(prev, names, w / "prev.bin")
        write_events(ev, w / "events.bin")
        subprocess.run([str(exe), str(w / "events.bin"), str(w / "rules.tsv"), str(w / "state.json"), "--prev", str(w / "prev.bin"), "--threads", "4"], check=True)
        cpp = {int(s): kv for s, kv in json.loads((w / "state.json").read_text()).items()}
    py = py_run(ev, prev)
    bad, n = [], 0
    for sid, kv in py.items():
        ckv = cpp[sid]
        if set(kv) != set(ckv):
            bad.append((sid, "keys", sorted(set(kv) ^ set(ckv))[:5]))
            continue
        for k, v in kv.items():
            n += 1
            c = ckv[k]
            c = math.nan if c is None else float(c)
            if not ((math.isnan(v) and math.isnan(c)) or v == c):
                bad.append((sid, k, v, c))
    print(f"{len(py)} stocks, {n} values compared, {len(bad)} mismatches")
    for b in bad[:20]:
        print("  ", b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
