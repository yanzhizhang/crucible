"""Stdlib-only factor preview on the fetched A-share panel.

**This is a preview, not the pipeline.** It exists because the project venv was
still installing and real numbers were wanted today. It re-implements a small
slice of :mod:`assay` in pure Python, which means it is a *second*
implementation and therefore a divergence risk -- exactly the thing this repo
argues against.

It earns its place in one specific way: once the real pipeline runs, its output
is an **independent cross-check** of ``assay.ic``. Two implementations agreeing
is evidence; this one alone is not authoritative. Where they disagree, the
crucible pipeline wins and this file is the suspect.

Factors are the cheap subset of ``research/factors.py`` -- the ones expressible
without numpy. Labels are next-day close-to-close on adjusted prices, with the
entry lag omitted, so these are *diagnostic* ICs and not tradable ones.
"""

from __future__ import annotations

import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001, S110
        pass

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
MIN_NAMES = 20


def load() -> dict[str, list[dict[str, float | str]]]:
    """Per-symbol chronological bar lists with adjusted prices."""
    by_sym: dict[str, list[dict[str, float | str]]] = defaultdict(list)
    with (RAW / "bars.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            adj = float(r["adj_factor"])
            by_sym[r["symbol"]].append(
                {
                    "date": r["date"],
                    "close": float(r["close"]),
                    "adj_close": float(r["close"]) * adj,
                    "high": float(r["high"]) * adj,
                    "low": float(r["low"]) * adj,
                    "volume": float(r["volume"]),
                    "amount": float(r["amount"]),
                    "turnover": float(r["turnover"]),
                }
            )
    for rows in by_sym.values():
        rows.sort(key=lambda x: x["date"])
    return by_sym


def _ret(rows: list[dict], i: int, n: int) -> float | None:
    if i - n < 0:
        return None
    a, b = rows[i - n]["adj_close"], rows[i]["adj_close"]
    return None if a <= 0 else b / a - 1.0


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def build(by_sym: dict[str, list[dict]]) -> dict[str, dict[str, dict[str, float]]]:
    """``{date: {symbol: {factor: value, 'y1': label}}}``.

    Every factor uses only bars at or before ``i``; the label uses ``i+1``.
    """
    out: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for sym, rows in by_sym.items():
        for i, row in enumerate(rows):
            if i < 60 or i + 1 >= len(rows):
                continue
            if row["volume"] <= 0:  # suspended: stale price, no real observation
                continue
            r1 = [_ret(rows, j, 1) for j in range(i - 19, i + 1)]
            r1 = [x for x in r1 if x is not None]
            if len(r1) < 20:
                continue
            r60 = [_ret(rows, j, 1) for j in range(i - 59, i + 1)]
            r60 = [x for x in r60 if x is not None]

            turn20 = sum(rows[j]["turnover"] for j in range(i - 19, i + 1)) / 20
            turn60 = sum(rows[j]["turnover"] for j in range(i - 59, i + 1)) / 60
            illiq = sum(
                abs(x) / (rows[j]["amount"] + 1.0)
                for j, x in zip(range(i - 19, i + 1), r1)
            ) / 20 * 1e9
            ma20 = sum(rows[j]["adj_close"] for j in range(i - 19, i + 1)) / 20

            lab = _ret(rows, i + 1, 1)
            if lab is None:
                continue

            out[str(row["date"])][sym] = {
                "rev_5": -(_ret(rows, i, 5) or 0.0),
                "rev_20": -(_ret(rows, i, 20) or 0.0),
                "mom_60_20": (rows[i - 20]["adj_close"] / rows[i - 60]["adj_close"] - 1.0),
                "vol_20": _std(r1),
                "vol_60": _std(r60),
                "turn_20": turn20,
                "turn_bias": turn20 / turn60 - 1.0 if turn60 else 0.0,
                "illiq": illiq,
                "max_5": max(r1[-5:]),
                "bias_20": row["adj_close"] / ma20 - 1.0,
                "hl_range_20": sum(
                    (rows[j]["high"] - rows[j]["low"]) / rows[j]["adj_close"]
                    for j in range(i - 19, i + 1)
                ) / 20,
                "y1": lab,
            }
    return out


def _rank(vals: list[float]) -> list[float]:
    """Average ranks, ties shared."""
    order = sorted(range(len(vals)), key=lambda k: vals[k])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float | None:
    if len(a) < 2:
        return None
    ra, rb = _rank(a), _rank(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    return None if da == 0 or db == 0 else num / (da * db)


def main() -> None:
    panel = build(load())
    dates = sorted(panel)
    factors = [
        "rev_5", "rev_20", "mom_60_20", "vol_20", "vol_60",
        "turn_20", "turn_bias", "illiq", "max_5", "bias_20", "hl_range_20",
    ]
    print(f"panel: {len(dates)} cross-sections, {dates[0]} .. {dates[-1]}")
    print(f"mean breadth: {sum(len(panel[d]) for d in dates) / len(dates):.0f} names\n")

    rows = []
    for f in factors:
        ics, q_spread = [], []
        for d in dates:
            cs = panel[d]
            if len(cs) < MIN_NAMES:
                continue
            xs = [cs[s][f] for s in cs]
            ys = [cs[s]["y1"] for s in cs]
            c = spearman(xs, ys)
            if c is None:
                continue
            ics.append(c)
            paired = sorted(zip(xs, ys))
            k = max(1, len(paired) // 5)
            top = sum(y for _, y in paired[-k:]) / k
            bot = sum(y for _, y in paired[:k]) / k
            q_spread.append(top - bot)

        n = len(ics)
        if n < 2:
            continue
        mean = sum(ics) / n
        sd = _std(ics)
        icir = mean / sd if sd else 0.0
        rows.append(
            {
                "factor": f,
                "ic": mean,
                "icir": icir,
                "t": icir * math.sqrt(n),
                "pos": sum(1 for x in ics if x > 0) / n,
                "q_spread": sum(q_spread) / len(q_spread),
                "n": n,
            }
        )

    rows.sort(key=lambda r: -abs(r["icir"]))
    print(f"{'factor':<14}{'IC':>9}{'ICIR':>8}{'t':>8}{'pos%':>8}{'Q5-Q1':>10}{'n':>6}")
    print("-" * 63)
    for r in rows:
        print(
            f"{r['factor']:<14}{r['ic']:>+9.4f}{r['icir']:>+8.3f}{r['t']:>+8.2f}"
            f"{r['pos'] * 100:>7.1f}%{r['q_spread']:>+10.4%}{r['n']:>6}"
        )

    print()
    print("Diagnostic ICs: next-day close-to-close, no entry lag, so NOT tradable.")
    print("Universe is today's top-80 by market cap -- selection biased.")
    print("Preview only; crucible's assay is the authority once deps land.")


if __name__ == "__main__":
    main()
