r"""Two open state keys against PM's day-end file: which of our values PM's equals, and a brute-force search for zzgh.nb.

``who``: for each target key, on the stocks where our value differs from PM's: the sign of ours - PM, by exchange, and every
OTHER engine key whose value equals PM's on those stocks (count per key) -- e.g. zzmh.hm equal to the trades' high on the
stocks we miss says which high PM tracks.

``nb``: PM's zzgh.nb against (buy - sell) / (buy + sell) built straight from the trade stream under every combination of
quantity (amount, volume, count), window (exchange clock 09:30-14:57 / 09:30-15:00 / 09:25-15:00 / whole day; local clock
09:30-14:57), side source (the trade's own flag; the larger of buy / sell order id where the stream has both) and
denominator (buy + sell; all trades incl. unflagged), plus buy / (buy + sell): the share of stocks exact at 1e-6 and 1e-4
and the correlation with PM, best first.

    python research/engine/gap_probe.py --date 20261120 --engine data/engine_r187/dysl2/engine_20261120.json.gz --pm $SP/20261120.jz \
        --catalog /work/crucible_data/catalog_dysl2.duckdb --who zzmh.hm,zzmh.r,zzgh.nb
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pm_factors"))

SOD = "((time / 1e9 + 28800) % 86400)"
LSOD = "((spider_ts / 1e9 + 28800) % 86400)"
WINDOWS = {"x0930_1457": f"{SOD} >= 34200 AND {SOD} < 53820", "x0930_1500": f"{SOD} >= 34200 AND {SOD} <= 54000",
           "x0925_1500": f"{SOD} >= 33900 AND {SOD} <= 54000", "day": "TRUE", "l0930_1457": f"{LSOD} >= 34200 AND {LSOD} < 53820"}
QTY = {"amt": "volume * price", "vol": "volume", "cnt": "1"}


def load(path: Path) -> dict[int, dict[str, float]]:
    """A state file with <p>.sum / <p>.wgt folded into <p>: sid -> key -> value."""
    with gzip.open(path) as f:
        d = json.load(f)
    out = {}
    for s, kv in d.items():
        o = {}
        for k, v in kv.items():
            if not isinstance(v, (int, float)):
                continue
            if k.endswith(".sum"):
                w = kv.get(k[:-4] + ".wgt")
                o[k[:-4]] = v / w if w else math.nan
            elif not k.endswith(".wgt"):
                o[k] = float(v)
        out[int(s)] = o
    return out


def same(a: float, b: float, tol: float = 1e-6) -> bool:
    """Equal within a relative tolerance."""
    return a == a and b == b and abs(a - b) <= tol * max(1.0, abs(b))


def who(ours: dict[int, dict[str, float]], pm: dict[int, dict[str, float]], keys: list[str]) -> None:
    """Per target key: the mismatches and which other engine keys carry PM's value there."""
    for key in keys:
        sids = [s for s in pm if key in pm[s] and s in ours and key in ours[s]]
        bad = [s for s in sids if not same(ours[s][key], pm[s][key])]
        print(f"\n{key}: {len(sids)} stocks, {len(bad)} differ at 1e-6")
        if not bad:
            continue
        sign = Counter(("ours > pm" if ours[s][key] > pm[s][key] else "ours < pm") + (" SH" if s >= 600000 else " SZ") for s in bad)
        rel = np.array([abs(ours[s][key] - pm[s][key]) / max(abs(pm[s][key]), 1e-12) for s in bad])
        print(f"   {dict(sign)}; relative gap median {np.median(rel):.2e}, 90% {np.quantile(rel, 0.9):.2e}, max {rel.max():.2e}")
        eq = Counter(k for s in bad for k, v in ours[s].items() if k != key and same(v, pm[s][key]))
        print("   other engine keys equal to PM's value on those stocks: " + (", ".join(f"{k} {v}" for k, v in eq.most_common(12)) or "none"))
        eqp = Counter(k for s in bad for k, v in pm[s].items() if k != key and same(v, ours[s][key]))
        print("   PM keys equal to OUR value on those stocks: " + (", ".join(f"{k} {v}" for k, v in eqp.most_common(8)) or "none"))
        for s in bad[:5]:
            print(f"   e.g. {s}: ours {ours[s][key]:.8g}  pm {pm[s][key]:.8g}")


def nb(con, date: int, pm: dict[int, dict[str, float]]) -> None:
    """Brute-force (buy - sell) / (buy + sell) from the trade stream against PM's zzgh.nb."""
    import polars as pl

    tgt = {s: kv["zzgh.nb"] for s, kv in pm.items() if "zzgh.nb" in kv and kv["zzgh.nb"] == kv["zzgh.nb"]}
    v = np.array(list(tgt.values()))
    print(f"\nzzgh.nb in PM's file: {len(v)} stocks; min {v.min():.4f}, quartiles {np.quantile(v, .25):.4f} / {np.median(v):.4f} / {np.quantile(v, .75):.4f}, max {v.max():.4f}; share exactly 0: {(v == 0).mean():.3f}")
    cols = {r[0] for r in con.sql("DESCRIBE raw_transaction").fetchall()}
    sides = {"flag": "CASE dir WHEN 1063 THEN 1 WHEN 1550 THEN -1 ELSE 0 END"}
    if {"buy_seq_id", "sell_seq_id"} <= cols:
        sides["seq"] = "CASE WHEN buy_seq_id > sell_seq_id THEN 1 WHEN sell_seq_id > buy_seq_id THEN -1 ELSE 0 END"
    con.register("want_nb", pl.DataFrame({"sid": list(tgt)}))
    sel = []
    for sn, sx in sides.items():
        for wn, wx in WINDOWS.items():
            for qn, qx in QTY.items():
                sel += [f"sum(CASE WHEN ({wx}) AND ({sx}) = 1 THEN {qx} ELSE 0 END)::DOUBLE AS b_{sn}_{wn}_{qn}",
                        f"sum(CASE WHEN ({wx}) AND ({sx}) = -1 THEN {qx} ELSE 0 END)::DOUBLE AS s_{sn}_{wn}_{qn}"]
    for wn, wx in WINDOWS.items():
        for qn, qx in QTY.items():
            sel.append(f"sum(CASE WHEN ({wx}) THEN {qx} ELSE 0 END)::DOUBLE AS a_{wn}_{qn}")
    df = con.sql(f"SELECT symbol_id AS sid, {', '.join(sel)} FROM raw_transaction r JOIN want_nb w ON w.sid = r.symbol_id WHERE r.date = {date} AND r.trade_type = 1 GROUP BY 1").pl()
    p = np.array([tgt[int(s)] for s in df["sid"].to_list()])
    res = []
    for sn in sides:
        for wn in WINDOWS:
            for qn in QTY:
                b, s, al = df[f"b_{sn}_{wn}_{qn}"].to_numpy(), df[f"s_{sn}_{wn}_{qn}"].to_numpy(), df[f"a_{wn}_{qn}"].to_numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    for dn, x in (("(b-s)/(b+s)", (b - s) / (b + s)), ("(b-s)/all", (b - s) / al), ("b/(b+s)", b / (b + s)), ("(s-b)/(b+s)", (s - b) / (b + s)),
                                  ("b/all", b / al), ("(b-s)/(b+s) x2 clipped", np.clip(2 * (b - s) / (b + s), -1, 1))):
                        ok = np.isfinite(x)
                        c = float(np.corrcoef(x[ok], p[ok])[0, 1]) if ok.sum() > 3 and np.std(x[ok]) > 0 else np.nan
                        res.append((float((np.abs(x - p) <= 1e-6).mean()), float((np.abs(x - p) <= 1e-4).mean()), c, f"{dn:<24} side {sn:<4} window {wn:<11} of {qn}"))
    res.sort(key=lambda r: (-r[0], -r[1], -abs(r[2]) if r[2] == r[2] else 0))
    print(f"{len(res)} candidates on {len(p)} stocks; best 15 by exact share, then the 8 most correlated")
    for e6, e4, c, name in res[:15]:
        print(f"   exact 1e-6 {e6:.3f}  1e-4 {e4:.3f}  corr {c:+.4f}  {name}")
    for e6, e4, c, name in sorted(res, key=lambda r: -abs(r[2]) if r[2] == r[2] else 0)[:8]:
        print(f"   corr {c:+.4f}  exact 1e-6 {e6:.3f}  {name}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--date", type=int, required=True)
    ap.add_argument("--engine", type=Path, required=True)
    ap.add_argument("--pm", type=Path, required=True)
    ap.add_argument("--catalog", type=Path, help="with it: the zzgh.nb search over the trade stream")
    ap.add_argument("--who", default="zzmh.hm,zzmh.r,zzgh.nb,zzba.p")
    a = ap.parse_args()
    ours, pm = load(a.engine), load(a.pm)
    print(f"{a.date}: engine {len(ours)} stocks, PM {len(pm)} stocks")
    who(ours, pm, [k for k in a.who.split(",") if k])
    if a.catalog:
        import duckdb

        nb(duckdb.connect(str(a.catalog), read_only=True), a.date, pm)


if __name__ == "__main__":
    main()
