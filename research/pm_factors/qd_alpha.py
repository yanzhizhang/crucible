r"""QD's per-day alpha from 224 (``/tmp/o/sig/YYYYMMDD/alpha.parquet``, pulled to ``$DATA/qd_alpha/<D>/``): what it holds, and
how it lines up with the signals 224's chrono log says runS received (``pm_signals``, ``trk SIG``).

The user, 2026-11-25: QD's 224 alphas for the recent days reach 0.99999 correlation and can be used. This reads them
read-only, prints the schema and coverage, and joins them to pm_signals on (sid, time) -- exact on any int64 ns column
that matches ``md_ts`` / ``md_recv_ts`` at 10 us, else nearest-before within 3 s -- then prints the correlation and
the share of exactly equal values per day. ``--write`` also stores the joined frame as ``$DATA/qd_alpha/joined/date=D.parquet``.

    python research/pm_factors/qd_alpha.py --dates 20261123,20261124
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import polars as pl

DATA = Path("/work/crucible_data")
NS_LO, NS_HI = 1_600_000_000 * 10**9, 2_100_000_000 * 10**9  # plausible epoch-ns range


def alpha_file(d: str, root: Path) -> Path | None:
    for p in (root / d / "alpha.parquet", root / f"date={d}" / "alpha.parquet"):
        if p.exists():
            return p
    return None


def guess(df: pl.DataFrame) -> tuple[str | None, list[str], list[str]]:
    """(symbol column, epoch-ns time columns, float value columns)."""
    sym = next((c for c in df.columns if c.lower() in ("sid", "symbol", "code", "wind_code", "windcode", "ticker", "securityid", "symbol_id")), None)
    times = [c for c in df.columns if df[c].dtype in (pl.Int64, pl.UInt64) and df.height
             and NS_LO < int(df[c].drop_nulls().median() or 0) < NS_HI]
    times += [c for c in df.columns if df[c].dtype == pl.Datetime("ns") or str(df[c].dtype).startswith("Datetime")]
    vals = [c for c in df.columns if df[c].dtype in (pl.Float64, pl.Float32) and c != sym]
    return sym, times, vals


def sid_of(df: pl.DataFrame, sym: str) -> pl.Expr:
    return pl.col(sym).cast(pl.String).str.extract(r"(\d{6})", 1).cast(pl.Int64).alias("sid") if df[sym].dtype == pl.String \
        else pl.col(sym).cast(pl.Int64).alias("sid")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--root", type=Path, default=DATA / "qd_alpha")
    ap.add_argument("--signals", type=Path, default=DATA / "pm_signals")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    for d in a.dates.split(","):
        f = alpha_file(d, a.root)
        if f is None:
            print(f"== {d}: no alpha.parquet under {a.root}/{d}")
            continue
        q = pl.read_parquet(f)
        sym, times, vals = guess(q)
        print(f"== {d}: {f} {q.height:,} rows x {q.width} cols")
        print("   schema:", dict(zip(q.columns[:40], [str(t) for t in q.dtypes[:40]])), "..." if q.width > 40 else "")
        print(f"   symbol column {sym}, epoch-ns time columns {times}, float columns {vals[:12]}{' ...' if len(vals) > 12 else ''}")
        if sym is None or not times or not vals:
            print("   cannot join to pm_signals automatically: page the schema above")
            continue
        q = q.with_columns(sid_of(q, sym))
        print(f"   stocks {q['sid'].n_unique()}, rows/stock median {q.group_by('sid').len()['len'].median():.0f}")
        s_path = a.signals / f"date={d}.parquet"
        if not s_path.exists():
            print(f"   no {s_path}: run pm_signals for {d} first")
            continue
        s = pl.read_parquet(s_path).filter(pl.col("prediction").is_not_null())
        s = s.filter(pl.col("src") == "sig") if "src" in s.columns else s
        best = None
        for tq in times:
            qt = q.with_columns(pl.col(tq).cast(pl.Int64).alias("_t"))
            for ts in [c for c in ("md_ts", "md_recv_ts") if c in s.columns]:
                k = lambda c: ((pl.col(c) + 5_000) // 10_000).alias("_k")  # noqa: E731 -- 10 us buckets
                j = s.with_columns(k(ts)).join(qt.with_columns(k("_t")), on=["sid", "_k"], how="inner")
                if best is None or j.height > best[0]:
                    best = (j.height, tq, ts, j)
        if best is None or best[0] == 0:
            print("   no exact (sid, time) match on any time column pair; nearest-before join within 3 s on the first pair")
            tq, ts = times[0], "md_recv_ts" if "md_recv_ts" in s.columns else "md_ts"
            j = s.sort(ts).join_asof(q.with_columns(pl.col(tq).cast(pl.Int64).alias(ts)).sort(ts), on=ts, by="sid",
                                     strategy="backward", tolerance=3 * 10**9).drop_nulls(vals[0])
        else:
            _, tq, ts, j = best
        print(f"   join {tq} ~ pm_signals.{ts}: {j.height:,} of {s.height:,} signals matched ({j.height / max(s.height, 1):.4f})")
        for v in vals[:12]:
            x, y = j["prediction"].to_numpy(), j[v].to_numpy()
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 10:
                continue
            r = float(np.corrcoef(x[ok], y[ok])[0, 1])
            eq = float(np.mean(x[ok] == y[ok]))
            print(f"   {v:>24s}: corr {r:.6f}, exactly equal {eq:.4f}, max |diff| {np.abs(x[ok] - y[ok]).max():.3g} (n {ok.sum():,})")
        if a.write:
            dst = a.root / "joined" / f"date={d}.parquet"
            dst.parent.mkdir(parents=True, exist_ok=True)
            j.write_parquet(dst)
            print(f"   -> {dst}")


if __name__ == "__main__":
    main()
