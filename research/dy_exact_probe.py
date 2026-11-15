"""Read-only: how big DataYes numbers get (index volume vs int64) and whether dysl2_raw kept them exactly.

Per day and kind, every numeric text column of the package is scanned (index: every index entry; stocks: the largest
entries): exact max |value| (Decimal on the text), rows above 2^53 (float64 stops being exact) and 2^63 (int64 overflows),
rows with a non-zero fraction, rows in scientific notation. Then, for the indices dysl2_raw keeps, the last cumulative
volume / amount in the text vs the value dysl2_raw holds.

    python research/dy_exact_probe.py --dates 20261112,20261113
"""

from __future__ import annotations

import argparse
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decode_datayes_day as DY  # noqa: E402
from decode_tdf_day import INDEXES, STOCKS  # noqa: E402

I53, I63 = 2**53, 2**63 - 1
NUM = r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$"


def dec(s: str) -> Decimal | None:
    """Exact value of one numeric string."""
    try:
        return abs(Decimal(s))
    except InvalidOperation:
        return None


def scan(frames: list[pl.DataFrame]) -> dict[str, dict[str, object]]:
    """Per numeric column over all frames: n, exact max |v|, n > 2^53, n > 2^63, n with a fraction, n in sci notation."""
    out: dict[str, dict[str, object]] = {}
    for df in frames:
        for c in df.columns:
            s = df[c].drop_nulls().str.strip_chars()
            s = s.filter(s != "")
            if s.is_empty():
                continue
            ok = s.str.contains(NUM)
            if ok.mean() < 0.99:
                continue  # a text column
            s = s.filter(ok)
            r = out.setdefault(c, {"n": 0, "max": Decimal(0), "gt53": 0, "gt63": 0, "frac": 0, "sci": 0})
            r["n"] += s.len()
            r["sci"] += int(s.str.contains(r"[eE]").sum())
            r["frac"] += int(s.str.contains(r"\.\d*[1-9]").sum())
            ip = s.str.replace(r"^-", "").str.replace(r"[.eE].*$", "").str.len_chars()
            wide = s.filter(ip >= 15).to_list()  # only these can pass 2^53 (16 digits), cheap to check exactly
            wide += s.filter(s.str.contains(r"[eE]")).to_list()
            for v in (dec(x) for x in wide):
                if v is None:
                    continue
                r["gt53"] += v > I53
                r["gt63"] += v > I63
            cand = s.gather(ip.arg_sort(descending=True).head(64)).to_list() + wide[:64]
            m = max((v for v in (dec(x) for x in cand) if v is not None), default=Decimal(0))
            r["max"] = max(r["max"], m)
    return out


def show(title: str, res: dict[str, dict[str, object]]) -> None:
    """One line per column, widest first."""
    print(f"  {title}")
    for c, r in sorted(res.items(), key=lambda kv: -kv[1]["max"]):
        print(f"    {c:<24} n={r['n']:>11,}  max={r['max']:.6E}  >2^53={r['gt53']:<6} >2^63={r['gt63']:<6} "
              f"frac={r['frac']:<9} sci={r['sci']}")


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dates", required=True)
    ap.add_argument("--root", type=Path, default=Path("/mnt/TickDataRepo/DataYesSharesL2"))
    ap.add_argument("--raw", type=Path, default=Path("/work/crucible_data/dysl2_raw"))
    ap.add_argument("--top", type=int, default=12, help="largest stock entries scanned per kind")
    a = ap.parse_args()
    for d in a.dates.split(","):
        pkg = DY.package(a.root, d)
        if pkg is None:
            print(f"{d}: no package under {a.root}")
            continue
        idx = DY.index(pkg)
        pre = f"{pkg}!" if pkg.is_file() else ""
        print(f"== {d}  {pkg}  {len(idx)} entries")
        for sub, kind in DY.SUBS.items():
            ents = [(sym, ref, b) for (sym, s), (ref, b) in idx.items() if s == sub]
            if kind != "index":
                ents = sorted((e for e in ents if STOCKS.match(e[0])), key=lambda e: -e[2])[: a.top]
            frames = {sym: DY.read(pre + ref) for sym, ref, _ in ents}
            show(f"{kind}: {len(frames)} entries ({'all' if kind == 'index' else 'largest stocks'})", scan(list(frames.values())))
            if kind != "index":
                continue
            got = list(a.raw.glob(f"kind=index/date={d}/*.parquet"))
            raw = pl.read_parquet(got, columns=["symbol_id", "time", "total_volume", "total_amount"]) if got else None
            for sym, df in sorted(frames.items()):
                if not INDEXES.match(sym) or df.is_empty():
                    continue
                df = DY.from_datayes(df, kind)
                m, _ = DY.quote_mapping(df.columns)
                last = df.tail(1).to_dicts()[0]
                tv, ta = last.get(m.get("total_volume", ""), ""), last.get(m.get("total_amount", ""), "")
                line = f"    {sym}: text volume={tv} amount={ta}"
                if raw is not None:
                    r = raw.filter(pl.col("symbol_id") == int(sym[:6])).sort("time").tail(1)
                    if r.height:
                        rv, ra = r["total_volume"][0], r["total_amount"][0]
                        dv = (dec(str(rv)) or 0) - (dec(tv or "0") or 0)
                        line += f" | dysl2_raw volume={rv} amount={ra!r} (volume diff {dv})"
                print(line)


if __name__ == "__main__":
    main()
