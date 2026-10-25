r"""Stage 4f: the per-minute profiles in PM's day-end state (``<p>.c.<bin>``), from the state files alone.

Every profile has a multiple of 237 bins (the continuous-session minutes), i.e. k blocks of 237.
Needs no ticks; per profile it prints:

* **values**: share zero, integer share, the most common values;
* **shape** per block: median, share zero, mean over the first / middle / last 30 minutes, and the
  share of stocks whose block never decreases along the minutes (1 = a cumulative curve);
* **day to day** (consecutive state files): the share of cells unchanged in the same block
  (``same``), equal to the previous day's block j-1 / j+1 (``from_prev_blk`` / ``from_next_blk``: a
  sliding window of days), or to the previous minute (``from_prev_min``);
* **across blocks**: median ratio of each block to block 0.

Both bin layouts are reported: block-major (bin = j * 237 + m) and minute-major (bin = m * k + j).

    python research/pm_factors/diag_profile.py --dates 20260923,20260924,20260929,20260930,20261009
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl

DCP = Path("/data/prod/h/HS300/p1_intrading_dcp")
HIST = re.compile(r"^(.+)\.c\.(-?\d+)$")
NMIN = 237


def load(path: Path) -> dict[str, dict[int, np.ndarray]]:
    """Map profile -> sid -> its bin values (NaN where a bin is missing)."""
    with gzip.open(path) as fh:
        raw = json.loads(fh.read())
    cells: dict[str, dict[int, dict[int, float]]] = {}
    for s, kv in raw.items():
        for k, v in kv.items():
            if v is not None and (m := HIST.match(k)):
                cells.setdefault(m.group(1), {}).setdefault(int(s), {})[int(m.group(2))] = float(v)
    out = {}
    for p, by in cells.items():
        nb = max(max(c) for c in by.values()) + 1
        out[p] = {}
        for sid, c in by.items():
            a = np.full(nb, np.nan)
            a[list(c)] = list(c.values())
            out[p][sid] = a
    return out


def blocks(a: np.ndarray, k: int, layout: str) -> np.ndarray:
    """(k, 237) view of one stock's bins."""
    return a.reshape(k, NMIN) if layout == "block" else a.reshape(NMIN, k).T


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--dcp-dir", type=Path, default=DCP)
    ap.add_argument("--out", type=Path, default=Path("data/reports/stage4"))
    args = ap.parse_args()
    files = sorted(p.stem for p in args.dcp_dir.glob("*.jz"))
    days = []
    for d in [x.strip() for x in args.dates.split(",") if x.strip()]:
        prev = [f for f in files if f < d]
        if d in files and prev:
            days.append((d, load(args.dcp_dir / f"{d}.jz"), load(args.dcp_dir / f"{prev[-1]}.jz")))
            print(f"loaded {d} and {prev[-1]}")
    if not days:
        return
    vals, shape, move, ratio = [], [], [], []
    for p in sorted(days[0][1]):
        allv = np.concatenate([a for _, cur, _ in days for a in cur.get(p, {}).values()])
        allv = allv[np.isfinite(allv)]
        nb = max(len(a) for a in days[0][1][p].values())
        common = Counter(np.round(allv, 6).tolist()).most_common(6)
        vals.append({"p": p, "bins": nb, "k": nb / NMIN, "zero": float(np.mean(allv == 0)),
                     "integer": float(np.mean(allv == np.round(allv))), "p10": float(np.quantile(allv, 0.1)),
                     "med": float(np.median(allv)), "p90": float(np.quantile(allv, 0.9)),
                     "common": ", ".join(f"{v:g}:{c / allv.size:.2f}" for v, c in common)})
        if nb % NMIN:
            continue
        k = nb // NMIN
        for layout in ("block", "inter"):
            B = np.stack([blocks(a, k, layout) for _, cur, _ in days for a in cur[p].values() if len(a) == nb])  # (n, k, 237)
            for j in range(k):
                x = B[:, j, :]
                with np.errstate(invalid="ignore"):
                    mono = np.mean([np.all(np.diff(r[np.isfinite(r)]) >= 0) for r in x])
                shape.append({"p": p, "layout": layout, "block": j, "med": float(np.nanmedian(x)), "zero": float(np.nanmean(x == 0)),
                              "first30": float(np.nanmean(x[:, :30])), "mid": float(np.nanmean(x[:, 100:137])),
                              "last30": float(np.nanmean(x[:, -30:])), "nondecr": float(mono)})
                if j:
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio.append({"p": p, "layout": layout, "block": j, "ratio_to_b0": float(np.nanmedian(x / B[:, 0, :]))})
            eq = {"same": [], "from_prev_blk": [], "from_next_blk": [], "from_prev_min": []}
            for _, cur, pre_all in days:
                pre = pre_all.get(p, {})
                for sid, a in cur[p].items():
                    if sid not in pre or len(a) != nb or len(pre[sid]) != nb:
                        continue
                    c, z = blocks(a, k, layout), blocks(pre[sid], k, layout)
                    tol = lambda u, v: np.abs(u - v) <= 1e-9 * np.maximum(1.0, np.abs(v))  # noqa: E731 -- equal to float noise
                    eq["same"].append(np.nanmean(tol(c, z)))
                    if k > 1:
                        eq["from_prev_blk"].append(np.nanmean(tol(c[1:], z[:-1])))
                        eq["from_next_blk"].append(np.nanmean(tol(c[:-1], z[1:])))
                    eq["from_prev_min"].append(np.nanmean(tol(c[:, 1:], z[:, :-1])))
            move.append({"p": p, "layout": layout, **{n: float(np.mean(v)) if v else None for n, v in eq.items()}})
    args.out.mkdir(parents=True, exist_ok=True)
    with pl.Config(tbl_rows=80, tbl_cols=14, tbl_width_chars=230, fmt_str_lengths=70, float_precision=4):
        print("\n== values per profile (common = value:share)")
        print(pl.DataFrame(vals))
        print("\n== day to day: share of cells equal to the previous day's same block / block j-1 / j+1 / previous minute")
        print(pl.DataFrame(move))
        print("\n== shape per block (nondecr = share of stocks whose block never decreases along the minutes)")
        sh = pl.DataFrame(shape)
        print(sh)
        if ratio:
            print("\n== median ratio of each block to block 0")
            print(pl.DataFrame(ratio))
    sh.write_csv(args.out / "profile_shape.csv")


if __name__ == "__main__":
    main()
