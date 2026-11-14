r"""The whole-book quantity (gen.best_level_order_count_imbalance) from the C++ replay (cpp/bookgen.cpp).

Loads a day's order / trade events exactly as ``full_book.py`` does (``bt.events`` cache, exchange sequence order, the
tick band from the day's limits), packs them for ``bookgen`` and writes ``<out>/date=<D>.parquet`` (sid, mark,
best_level_order_count_imbalance) -- the layout ``ic_score --full-book`` and ``vec_day.py --full-book`` read.

    CRUCIBLE_SOURCE=dysl2 python research/featgen/book_day.py --date 20261112 --state $SEED/20261112.jz --marks rand:240 \
        --check $DATA/store/pm_factors/full_book_dysl2/date=20261112.parquet
    python research/featgen/book_day.py --selftest            # synthetic events against full_book.book_at (numba)
    python research/featgen/book_day.py --selftest --stage make|check --work W     # split, for a Windows checkout + WSL
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pm_factors"))
sys.path.insert(0, str(HERE.parent))

CPP = HERE / "cpp"
COLS = ("kind", "ats", "side", "tick", "qty", "id_a", "id_b", "otype")
NAME = "best_level_order_count_imbalance"


def pack(dst: Path, n_t: int, stocks: list[dict]) -> None:
    """``stocks``: dicts with sid, venue (0 SSE / 1 SZSE), lo_tick, n_ticks, cut (int64 ns per time) and the COLS arrays."""
    with dst.open("wb") as f:
        f.write(b"CRUBK001")
        f.write(np.array([len(stocks), n_t], "<u8").tobytes())
        for st in stocks:
            n = st["kind"].shape[0]
            f.write(np.array([st["sid"], st["venue"], st["lo_tick"], st["n_ticks"]], "<i8").tobytes())
            f.write(np.array([n], "<u8").tobytes())
            f.write(np.asarray(st["cut"], "<i8").tobytes())
            for c in COLS:
                f.write(np.ascontiguousarray(st[c], "<i8").tobytes())


def ensure_binary(build: Path | None = None) -> Path:
    build = build or CPP / "build"
    exe, src = build / "bookgen", CPP / "bookgen.cpp"
    if exe.exists() and exe.stat().st_mtime >= src.stat().st_mtime:
        return exe
    build.mkdir(parents=True, exist_ok=True)
    subprocess.run([shutil.which("g++") or "g++", "-std=c++20", "-O2", "-DNDEBUG", "-pthread", str(src), "-o", str(exe)], check=True)
    return exe


def read(out: Path) -> dict[int, dict[str, np.ndarray]]:
    """{sid: imb / bid / ask (per time), stats}."""
    b = out.read_bytes()
    assert b[:8] == b"CRUBO001", "not a bookgen output file"
    ns, nt = (int(x) for x in np.frombuffer(b, "<u8", 2, 8))
    rec = np.frombuffer(b, np.dtype([("sid", "<i8"), ("imb", "<f8", (nt,)), ("bid", "<f8", (nt,)), ("ask", "<f8", (nt,)), ("stats", "<i8", (3,))]), ns, 24)
    return {int(r["sid"]): {"imb": r["imb"], "bid": r["bid"], "ask": r["ask"], "stats": r["stats"]} for r in rec}


def synth(sid: int, venue: int, rng: np.random.Generator, n: int) -> dict:
    """A made-up event stream that walks every branch: priced and market adds, own-best, FAK tags, cancels, trades,
    unknown ids, out-of-band prices, arrival stamps that step back."""
    lo, nt = 1000, 400
    kind, side, tick, qty, ida, idb, ot = (np.zeros(n, np.int64) for _ in range(7))
    live = {1: [], -1: []}
    nxt, mid = 1, lo + nt // 2
    for i in range(n):
        u = rng.random()
        if u < 0.55 or not (live[1] and live[-1]):
            sd = int(rng.choice([1, -1])) if rng.random() > 0.01 else 0
            kind[i], side[i], ida[i], qty[i] = 0, sd, nxt, int(rng.integers(0, 50)) * 100
            off = int(rng.integers(-3, 25))
            tick[i] = (mid - off if sd > 0 else mid + off) if rng.random() > 0.005 else lo - 50
            ot[i] = int(rng.choice([2, 2, 2, 2, 1, 3, 11, 12, 15])) if venue == 1 else 2
            if sd:
                live[sd].append(nxt)
            nxt += 1
        elif u < 0.78:
            sd = int(rng.choice([1, -1]))
            k = live[sd].pop(int(rng.integers(len(live[sd])))) if rng.random() > 0.03 else 10**9 + i
            kind[i], side[i], ida[i], qty[i] = 1, sd, k, int(rng.choice([0, 0, 100]))
        else:
            b_, a_ = live[1][int(rng.integers(len(live[1])))], live[-1][int(rng.integers(len(live[-1])))]
            if rng.random() < 0.3:  # the newest order is the aggressor (SZSE: its own burst)
                if rng.random() < 0.5:
                    b_ = live[1][-1]
                else:
                    a_ = live[-1][-1]
            kind[i], side[i], ida[i], idb[i] = 2, int(rng.choice([1, -1, 0])), b_, a_
            qty[i], tick[i] = int(rng.integers(1, 30)) * 100, mid + int(rng.integers(-2, 3))
            if rng.random() < 0.02:
                idb[i] = 10**9 + i
        if rng.random() < 0.02:
            mid = int(np.clip(mid + rng.integers(-2, 3), lo + 40, lo + nt - 40))
    ats = np.cumsum(rng.integers(0, 3_000_000, n)).astype(np.int64) + rng.integers(-2_000_000, 2_000_000, n)
    return {"sid": sid, "venue": venue, "lo_tick": lo, "n_ticks": nt, "kind": kind, "ats": ats, "side": side, "tick": tick,
            "qty": qty, "id_a": ida, "id_b": idb, "otype": ot}


def reference(st: dict) -> np.ndarray:
    """full_book.book_at + factors for one packed stock: the imbalance at each time."""
    import full_book as FB
    from bt import engine as E

    rec = np.searchsorted(np.maximum.accumulate(st["ats"]), st["cut"], side="right") - 1
    raw, _ = FB.book_at(st["kind"], st["ats"], st["side"], st["tick"], st["qty"], st["id_a"], st["id_b"], st["otype"], st["lo_tick"],
                        st["n_ticks"], E.VENUE_SSE if st["venue"] == 0 else E.VENUE_SZSE, rec, st["cut"])
    return FB.factors(raw)[NAME]


def selftest(a: argparse.Namespace) -> None:
    a.work.mkdir(parents=True, exist_ok=True)
    inp, out = a.work / "book_in.bin", a.work / "book_out.bin"
    if a.stage in ("all", "make"):
        rng = np.random.default_rng(11)
        stocks = [synth(i + 1, i % 2, rng, int(rng.integers(20_000, 60_000))) for i in range(10)]
        for st in stocks:
            st["cut"] = np.sort(rng.integers(int(st["ats"].min()) - 10, int(st["ats"].max()) + 10, 300)).astype(np.int64)
        pack(inp, 300, stocks)
        np.savez_compressed(a.work / "book_ref.npz", **{str(st["sid"]): reference(st) for st in stocks})
        print(f"{len(stocks)} synthetic stocks -> {inp}")
    if a.stage == "all":
        subprocess.run([str(ensure_binary()), str(inp), str(out)], check=True)
    if a.stage in ("all", "check"):
        got, ref, bad = read(out), np.load(a.work / "book_ref.npz"), 0
        for sid, g in got.items():
            r = ref[str(sid)]
            fa, fb = np.isfinite(g["imb"]), np.isfinite(r)
            n = int((fa != fb).sum()) + int((np.abs(g["imb"][fa & fb] - r[fa & fb]) > 1e-12).sum())
            bad += n > 0
            print(f"  {'OFF' if n else 'ok '} sid {sid}: finite {int(fb.sum())} of {r.size}, cells differing {n}, stats {g['stats'].tolist()}")
        print(f"{len(got)} stocks: {bad} off")
        sys.exit(1 if bad else 0)


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--stage", choices=["all", "make", "check"], default="all")
    ap.add_argument("--work", type=Path, default=Path("data/bookgen_work"))
    ap.add_argument("--date")
    ap.add_argument("--state", type=Path, help="PM's state file (.jz): the stocks (an earlier day's file when the day has none)")
    ap.add_argument("--marks", help="ic_score's grid spec: five | rand:N[:seed] | a comma list")
    ap.add_argument("--grid", type=float, help="instead of --marks: every GRID seconds from GRID to 14220")
    ap.add_argument("--out", type=Path, default=Path("data/featgen_book"))
    ap.add_argument("--threads", type=int, default=32)
    ap.add_argument("--check", type=Path, help="full_book.py's parquet of the day: compare the column with it")
    a = ap.parse_args()
    if a.selftest:
        return selftest(a)
    import dcp_build as B
    import full_book as FB
    from bt import engine as E
    from bt import events as EV
    from ic_score import mark_grid

    t0 = time.perf_counter()
    T = np.arange(a.grid, 14220.0 + 1e-9, a.grid) if a.grid else mark_grid(a.marks or "five")
    if not a.state.exists():
        a.state = sorted(p for p in a.state.parent.glob("2*.jz") if p.stem < a.date)[-1]
    sids = sorted(B.load_state(a.state)["sid"].unique().to_list())
    uni = pl.DataFrame({"symbol_id": sids, "market_id": [FB.SSE if c >= 600000 else FB.SZSE for c in sids]}, schema={"symbol_id": pl.Int32, "market_id": pl.Int16})
    ev, lim = EV.load(a.date, uni)
    parts = ev.partition_by("symbol_id", "market_id", as_dict=True)
    stocks = []
    for li in lim.iter_rows(named=True):
        e = parts.get((li["symbol_id"], li["market_id"]))
        if e is None or e.height == 0:
            continue
        lo = int(li["dn_tick"]) - 2
        st = {c: e[c].to_numpy().astype(np.int64) for c in COLS}
        st.update(sid=int(li["symbol_id"]), venue=E.VENUE_SSE if li["market_id"] == FB.SSE else E.VENUE_SZSE, lo_tick=lo,
                  n_ticks=int(li["up_tick"]) - lo + 3, cut=FB.mark_ns(int(st["ats"][0]), T))
        stocks.append(st)
    t1 = time.perf_counter()
    a.work.mkdir(parents=True, exist_ok=True)
    inp, outb = a.work / f"book_in_{a.date}.bin", a.work / f"book_out_{a.date}.bin"
    pack(inp, T.size, stocks)
    t2 = time.perf_counter()
    subprocess.run([str(ensure_binary()), str(inp), str(outb), "--threads", str(a.threads)], check=True)
    t3 = time.perf_counter()
    got = read(outb)
    df = pl.DataFrame({"sid": np.repeat(np.array(list(got), np.int64), T.size), "mark": np.tile(T, len(got)),
                       NAME: np.concatenate([g["imb"] for g in got.values()])})
    a.out.mkdir(parents=True, exist_ok=True)
    dst = a.out / f"date={a.date}.parquet"
    df.write_parquet(dst)
    inp.unlink()
    outb.unlink()
    stats = np.sum([g["stats"] for g in got.values()], axis=0)
    print(f"{a.date}: {len(stocks)} stocks, {sum(s['kind'].shape[0] for s in stocks):,} events, {T.size} times; load {t1 - t0:.0f}s, pack {t2 - t1:.1f}s, "
          f"bookgen {t3 - t2:.1f}s; finite {df[NAME].is_finite().mean():.4f}; unresolved cancels {stats[0]:,}, unresolved passive fills {stats[1]:,}, "
          f"out-of-band adds {stats[2]:,} -> {dst}")
    if a.check and a.check.exists():
        r = pl.read_parquet(a.check, columns=["sid", "mark", NAME]).with_columns(pl.col("sid").cast(pl.Int64))
        j = df.join(r, on=["sid", "mark"], how="inner", suffix="_py")
        x, y = j[NAME].to_numpy(), j[NAME + "_py"].to_numpy().astype(float)
        fa, fb = np.isfinite(x), np.isfinite(y)
        n = int((fa != fb).sum()) + int((np.abs(x[fa & fb] - y[fa & fb]) > 1e-12).sum())
        print(f"{a.date}: against {a.check}: {j.height:,} cells matched of {df.height:,}, finite in Python {int(fb.sum()):,}, cells differing {n:,}")


if __name__ == "__main__":
    main()
