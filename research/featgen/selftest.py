r"""featgen against the Python definitions on synthetic ticks: every base quantity, every stock, every time of the grid.

    python research/featgen/selftest.py                         # build, run, compare (Linux)
    python research/featgen/selftest.py --stage make --work W   # only write W/in.bin and W/ref.npz
    python research/featgen/selftest.py --stage check --work W  # only compare W/out.bin (+ W/names.txt) with W/ref.npz

The two-stage form is for a Windows checkout whose compiler lives in WSL: make on Windows, featgen in WSL, check on Windows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import featio as IO  # noqa: E402


def synth(sid: int, rng: np.random.Generator) -> dict:
    """One stock's day: snapshots about every 3 s from 09:15, trades and orders at random times, a few empty quotes."""
    clock = lambda t: np.where(t <= 7200, 34200 + t, 46800 + t - 7200)  # noqa: E731 -- trading clock -> exchange second
    n = int(rng.integers(3000, 5200))
    t = np.sort(rng.uniform(-900, 14220, n))
    pc = float(np.round(rng.uniform(5, 80), 2))
    mid = np.round(pc * np.exp(np.cumsum(rng.normal(0, 4e-4, n))), 2)
    half = rng.integers(1, 4, n) * 0.01 / 2
    b1, a1 = np.round(mid - half, 2), np.round(mid + half + 0.01, 2)
    dead = rng.random(n) < 0.01  # a snapshot with one side empty
    b1 = np.where(dead, 0.0, b1)
    lv = lambda: rng.integers(1, 400, n) * 100.0  # noqa: E731 -- a level's volume
    bq, aq = [lv() for _ in range(6)], [lv() for _ in range(6)]
    bv5, av5 = sum(bq[:5]), sum(aq[:5])
    dvol = np.where(rng.random(n) < 0.7, rng.integers(0, 50, n) * 100.0, 0.0)
    price = np.where(rng.random(n) < 0.5, b1, a1)
    price = np.where(price > 0, price, a1)
    price[:3] = 0.0  # before the first trade
    cv, ca = np.cumsum(dvol), np.cumsum(dvol * np.where(price > 0, price, pc))
    gf = pl.DataFrame({
        "t_local": t, "t_exch": t - rng.uniform(0, 0.5, n), "s": clock(t), "price": price, "op": np.where(t > -300, np.round(pc * 1.003, 2), 0.0),
        "pc": np.full(n, pc), "hi": np.maximum.accumulate(np.where(price > 0, price, 0.0)), "lo": np.full(n, pc * 0.97),
        "b1": b1, "a1": a1, "bv1": bq[0], "av1": aq[0], "tbv": bv5 * 3, "tav": av5 * 3, "cv": cv, "ca": ca,
        "bv5": bv5, "av5": av5, "bwp5": np.round(b1 - 0.02, 4), "awp5": np.round(a1 + 0.02, 4),
        "ofi2_5": np.where(rng.random(n) < 0.05, np.nan, rng.normal(0, 3e4, n)),
        "ul": np.full(n, np.round(pc * 1.1, 2)), "ll": np.full(n, np.round(pc * 0.9, 2)),
        **{f"ofil{i}": rng.normal(0, 1e4, n) for i in range(2, 7)},
        **{f"bq{i}": bq[i - 1] for i in range(2, 7)}, **{f"aq{i}": aq[i - 1] for i in range(2, 7)},
        # columns the Python definitions read for quantities the baseline does not use (levels 5 and 7..10)
        "b5px": np.round(b1 - 0.04, 2), "a5px": np.round(a1 + 0.04, 2),
        **{f"ofil{i}": rng.normal(0, 1e4, n) for i in range(7, 11)},
        **{f"bq{i}": lv() for i in range(7, 11)}, **{f"aq{i}": lv() for i in range(7, 11)},
    })
    m = int(rng.integers(2000, 9000))
    ts = np.sort(rng.uniform(-300, 14220, m))
    k = np.clip(np.searchsorted(t, ts, side="right") - 1, 0, n - 1)
    side = rng.choice([1.0, -1.0, 0.0], m, p=[0.48, 0.48, 0.04])
    tp = np.where(side > 0, a1[k], np.where(b1[k] > 0, b1[k], a1[k]))
    vol = rng.integers(1, 60, m) * 100.0
    ids = rng.integers(1, m // 3 + 2, m).astype(float)  # an aggressive order fills in several trades
    gs = pl.DataFrame({"t_local": ts, "t_exch": ts - 0.01, "s": clock(ts), "ls": np.floor(ts * 4), "price": tp, "op": np.full(m, np.round(pc * 1.003, 2)),
                       "pc": np.full(m, pc), "vol": vol, "amt": vol * tp, "side": side,
                       "bsq": pl.Series(np.where(rng.random(m) < 0.02, np.nan, ids)).fill_nan(None),
                       "ssq": pl.Series(np.where(rng.random(m) < 0.02, np.nan, ids + 1e6)).fill_nan(None)})
    q = int(rng.integers(4000, 12000))
    to = np.sort(rng.uniform(-900, 14220, q))
    k = np.clip(np.searchsorted(t, to, side="right") - 1, 0, n - 1)
    oside = rng.choice([1.0, -1.0, 0.0], q, p=[0.49, 0.49, 0.02])
    opx = np.where(rng.random(q) < 0.03, 0.0, np.round(mid[k] + rng.integers(-5, 6, q) * 0.01, 2))
    ovol = rng.integers(1, 80, q) * 100.0
    seq = np.cumsum(rng.integers(1, 40, q)).astype(float)
    go = pl.DataFrame({"t_local": to, "s": clock(to), "kind": rng.choice([1, 2], q, p=[0.7, 0.3]), "side": oside, "vol": ovol,
                       "amt": ovol * opx, "px": opx, "seq": np.where(rng.random(q) < 0.01, np.nan, seq)})
    seed = {"zzpk.low": pc * 0.98, "zzeg.c.high": pc * 1.04, "zzeg.c.low": pc * 0.95}
    if sid % 5 == 0:  # a stock PM's state has nothing for
        seed = {}
    return {"sid": sid, "gf": gf, "gs": gs, "go": go, "seed": seed}


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["all", "make", "check"], default="all")
    ap.add_argument("--work", type=Path, default=Path("data/featgen_selftest"))
    ap.add_argument("--stocks", type=int, default=12)
    ap.add_argument("--times", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    a.work.mkdir(parents=True, exist_ok=True)
    inp, out, names, refp = a.work / "in.bin", a.work / "out.bin", a.work / "names.txt", a.work / "ref.npz"
    if a.stage in ("all", "make"):
        rng = np.random.default_rng(a.seed)
        T = np.sort(rng.uniform(1.0, 13800.0, a.times))
        stocks = [synth(600000 + i if i % 2 else 1 + i, rng) for i in range(a.stocks)]
        stocks.append({"sid": 999999, "gf": stocks[0]["gf"].head(2), "gs": None, "go": None, "seed": {}})  # a stock with almost nothing
        IO.write_inputs(inp, T, stocks)
        ref = {f"{st['sid']}|{k}": v for st in stocks for k, v in IO.reference(st["gf"], st["gs"], st["go"], st["seed"], T).items()}
        np.savez_compressed(refp, T=T, **ref)
        print(f"{len(stocks)} stocks x {T.size} times -> {inp}, {refp} ({len(ref)} reference series)")
    if a.stage == "all":
        IO.run(inp, out, names)
    if a.stage in ("all", "check"):
        z = np.load(refp)
        ref: dict[int, dict[str, np.ndarray]] = {}
        for key in z.files:
            if "|" in key:
                sid, _, k = key.partition("|")
                ref.setdefault(int(sid), {})[k] = z[key]
        nm, got = IO.read_output(out, names)
        for sid in got:
            ref.setdefault(sid, {})
        bad, lines = IO.compare(nm, got, ref)
        print("\n".join(lines))
        print(f"{len(nm)} base quantities, {len(got)} stocks: {bad} off")
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
