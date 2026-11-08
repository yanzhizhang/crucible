r"""zhangyiqing's momentum_trade output on 224 (``/tmp/zyq/stocks``, rsynced to 97): what its per-stock fit means.

Each day at 20:01 momentum_trade writes ``logs/YYYY-MM-DD.txt`` -- three config lines, ``Fit:``, one line per stock
``600000.SH, (k, bp, p, q)``, then ``Failed:`` and the codes it could not fit -- and overwrites ``stocklist.txt`` (codes
only). Read on 97 so far: q = p x (1 + bp / 1e4) (bp a price offset in basis points), many bp pinned at -1000 / +2000
(the daily limits), stocklist = the k == 3 group.

This script parses every day and checks those readings against CED's end-of-day prices: is p the day's close (or what),
is q the next day's limit price, where bp sits against the stock's own limit, which k the stocks of each kind get, the
day's amplitude per k (config minAmplitude / maxAmplitude 100 / 500), and whether the next day reached q.

    python research/pm_factors/zyq_stocks.py --root /work/crucible_data/zyq_stocks --dates 20261104,20261105,20261106
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

ROW = re.compile(r"^(\d{6}\.(?:SH|SZ|BJ)),\s*\(\s*(-?\d+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\)\s*$")
CODE = re.compile(r"\d{6}\.(?:SH|SZ|BJ)")


def parse(f: Path, d: str) -> tuple[pl.DataFrame, list[str], list[str]]:
    """(per-stock fits, failed codes, the header lines)."""
    rows, failed, head, part = [], [], [], "head"
    for ln in f.read_text(errors="replace").splitlines():
        s = ln.strip()
        if s.startswith("Fit"):
            part = "fit"
            continue
        if s.startswith("Failed"):
            part = "failed"
            failed += CODE.findall(s)
            continue
        if part == "head":
            head.append(s)
        elif part == "fit" and (m := ROW.match(s)):
            rows.append((m[1], int(m[2]), float(m[3]), float(m[4]), float(m[5])))
        elif part == "failed":
            failed += CODE.findall(s)
        elif s:
            head.append(f"(unparsed) {s[:120]}")
    df = pl.DataFrame(rows, schema={"code": pl.String, "k": pl.Int64, "bp": pl.Float64, "p": pl.Float64, "q": pl.Float64}, orient="row")
    return df.with_columns(pl.lit(d).alias("date"), pl.col("code").str.slice(0, 6).cast(pl.Int64).alias("sid")), failed, head


def board_limit(code: pl.Expr) -> pl.Expr:
    """The usual daily limit in bp by board (ST and new listings differ: CED's limit prices decide below)."""
    c = code.str.slice(0, 3)
    return (pl.when(c.is_in(["300", "301", "688", "689"])).then(2000.0).when(code.str.ends_with(".BJ")).then(3000.0).otherwise(1000.0))


def q(x: pl.Series | np.ndarray, qs=(1, 10, 50, 90, 99)) -> str:
    v = np.asarray(x, float)
    v = v[np.isfinite(v)]
    return "/".join(f"{np.percentile(v, p):.4g}" for p in qs) if v.size else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/work/crucible_data/zyq_stocks"))
    ap.add_argument("--dates", required=True, help="YYYYMMDD,... (the log of each day: logs/YYYY-MM-DD.txt)")
    ap.add_argument("--ced", action=argparse.BooleanOptionalAction, default=True, help="check against CED end-of-day prices")
    ap.add_argument("--out", type=Path, default=Path("data/reports/zyq_stocks"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    S = None
    if a.ced:
        import ced_match as CM

        S = CM.S
        eod_days = S.dates("daily_eod")
    days = [x.strip() for x in a.dates.split(",") if x.strip()]
    allf = []
    for d in days:
        f = a.root / "logs" / f"{d[:4]}-{d[4:6]}-{d[6:]}.txt"
        if not f.exists():
            print(f"skip {d}: no {f}")
            continue
        df, failed, head = parse(f, d)
        allf.append(df)
        print(f"\n== {d}: {df.height} fits, {len(failed)} failed; header: " + " | ".join(h[:100] for h in head))
        err = (df["q"] - df["p"] * (1 + df["bp"] / 1e4)).abs() / df["p"]
        print(f"  q = p (1 + bp/1e4): max rel error {err.max():.2e}")
        for k, g in df.group_by("k", maintain_order=True):
            print(f"  k={k[0]}: {g.height} stocks; bp p1/p10/p50/p90/p99 {q(g['bp'])}; share bp > 0 {(g['bp'] > 0).mean():.2f}; "
                  f"p {q(g['p'], (10, 50, 90))}")
        lim = df.with_columns(board_limit(pl.col("code")).alias("lim"))
        at = lim.with_columns(((pl.col("bp").abs() - pl.col("lim")).abs() < 0.01).alias("at"))
        print(f"  |bp| at the board limit (1000 main / 2000 ChiNext-STAR / 3000 BJ): {at['at'].mean():.3f}; "
              f"by k: " + ", ".join(f"k={k[0]} {g['at'].mean():.2f}" for k, g in at.group_by("k", maintain_order=True)))
        top = df.with_columns(pl.col("bp").round(2)).group_by("bp").len().sort("len", descending=True).head(8)
        print("  most common bp: " + ", ".join(f"{b:g} x{n}" for b, n in top.iter_rows()))
        if S is not None and d in eod_days:
            e = S.read("daily_eod", d, d).collect().with_columns(CM.sid_of(pl.col("symbol")).alias("sid"))
            j = df.join(e.select("sid", "open", "high", "low", "close", "preclose", *[c for c in ("maxp_allowed", "minp_allowed") if c in e.columns]),
                        on="sid", how="left")
            for c in ("close", "preclose", "open", "high", "low"):
                r = (j["p"] / j[c]).to_numpy()
                print(f"  p / {c}: equal (1e-6) {np.mean(np.abs(r - 1) < 1e-6):.3f}; p10/p50/p90 {q(r, (10, 50, 90))}")
            amp = ((j["high"] - j["low"]) / j["preclose"] * 1e4).to_numpy()
            print("  day's amplitude (high - low) / preclose in bp by k: "
                  + ", ".join(f"k={k} {q(amp[(j['k'] == k).to_numpy()], (10, 50, 90))}" for k in sorted(j["k"].unique().to_list())))
            ret = ((j["close"] / j["preclose"] - 1) * 1e4).to_numpy()
            print("  day's return in bp by k: " + ", ".join(f"k={k} {q(ret[(j['k'] == k).to_numpy()], (10, 50, 90))}" for k in sorted(j["k"].unique().to_list())))
            print(f"  corr(bp, day's return) {np.corrcoef(np.nan_to_num(ret), df['bp'].to_numpy())[0, 1]:+.3f}")
            nxt = [x for x in eod_days if x > d]
            if nxt:
                n = S.read("daily_eod", nxt[0], nxt[0]).collect().with_columns(CM.sid_of(pl.col("symbol")).alias("sid"))
                cols = [c for c in ("maxp_allowed", "minp_allowed") if c in n.columns]
                jn = df.join(n.select("sid", "high", "low", "close", *cols).rename({"high": "h1", "low": "l1", "close": "c1"}), on="sid", how="left")
                for c in cols:
                    print(f"  q = next day's {c} ({nxt[0]}): {np.mean(np.abs(jn['q'].to_numpy() / jn[c].to_numpy() - 1) < 1e-4):.3f}")
                up, dn = (jn["bp"] > 0).to_numpy(), (jn["bp"] < 0).to_numpy()
                print(f"  next day reached q: bp > 0 high >= q {np.nanmean((jn['h1'] >= jn['q']).to_numpy()[up]):.3f} ({up.sum()}), "
                      f"bp < 0 low <= q {np.nanmean((jn['l1'] <= jn['q']).to_numpy()[dn]):.3f} ({dn.sum()})")
                r1 = ((jn["c1"] / jn["p"] - 1) * 1e4).to_numpy()
                print("  next day's close / p - 1 in bp by k: "
                      + ", ".join(f"k={k} {q(r1[(jn['k'] == k).to_numpy()], (10, 50, 90))}" for k in sorted(jn["k"].unique().to_list())))
            else:
                print("  next day: no daily_eod yet")
    if not allf:
        return
    full = pl.concat(allf)
    full.write_parquet(a.out / "fits.parquet")
    sl = a.root / "stocklist.txt"
    if sl.exists():
        codes = set(CODE.findall(sl.read_text(errors="replace")))
        last = full.filter(pl.col("date") == full["date"].max())
        for k in sorted(last["k"].unique().to_list()):
            ks = set(last.filter(pl.col("k") == k)["code"].to_list())
            print(f"stocklist.txt ({len(codes)} codes) vs {full['date'].max()} k={k} ({len(ks)}): common {len(codes & ks)}")
    if full["date"].n_unique() > 1:  # how a stock's k moves from day to day
        w = full.pivot(on="date", index="code", values="k", aggregate_function="first")
        ds = [c for c in w.columns if c != "code"]
        for x, y in zip(ds, ds[1:]):
            t = w.group_by(x, y).len().sort(x, y)
            print(f"k {x} -> {y}: " + ", ".join(f"{r[0]}->{r[1]} {r[2]}" for r in t.iter_rows()))
    print(f"-> {a.out / 'fits.parquet'}")


if __name__ == "__main__":
    main()
