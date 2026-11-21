r"""runS's silent runs: do the orders with no logged decision come from an unlogged evaluation on the stock's next snapshot?

R194: the logged blocks (catchRun "alpha.1" + alg.cpp lines) are the alpha-arrival runs; half of the SO lines are printed
with no block of their own, ~one snapshot after the stock's last logged block, and finish what that block wanted. Test:
for EVERY SO line (field 2 = E, taken as the receive epoch of the snapshot that triggered the run)
  state  = the stock's last logged block before E (fa, vpen, vr, poscst, cpos, asq, hs, lthr, winsdQty), with the orders
           sent since that block added to the position as a pending leg and taken off the sellable quantity;
  book   = the first snapshot of the stock in the tick store received from E - 0.05 s on (bid / ask / level-1 sizes, mid);
  replay = GTp on the new mid -> executp; cOrd (threshold, room, caps, level-1 size in shares, lots) -> side, price, qty.
An SO line whose own block is the logged one (E == that block's book epoch) is replayed on the block itself (control).
Printed: per class (own block / silent, by seconds since the last block) the share with an order replayed and side /
price / quantity exact; the same by half hour; with the logged hs against the level-1 spread / mid as the cost; the
stale-alpha rule (adur 12 s); first mismatches. Read-only.

    python research/pm_factors/runs_silent.py --logs /work/crucible_data/pm_224/20261118/chrono_logs --date 20261118 --catalog /work/crucible_data/catalog_dysl2.duckdb
"""

from __future__ import annotations

import argparse
import bisect
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))

ADUR = 12.0


def parse(files: list[str]) -> tuple[dict[int, list[dict]], list[dict]]:
    """(per stock: its logged blocks in order, every SO line) of the day's log."""
    from pm_orders import GLUE, PLUGIN, SRC
    from pm_signals import NUM

    blocks: dict[int, list[dict]] = defaultdict(list)
    sos: list[dict] = []
    for f in files:
        cur: dict | None = None
        last_so: dict | None = None
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in (part for raw in fh for part in GLUE.split(raw) if part.strip()):
                if " trk " in line:
                    continue
                s = SRC.search(line)
                if not s or (line.startswith("[20") and not PLUGIN.search(line)):
                    continue
                where, rest = s.group(1), s.group(2)
                fname = where.rsplit(":", 1)[0].rsplit("/", 1)[-1]
                kv = {k.replace("this->", "").replace("book->", "book_").rstrip("_"): float(v) for k, v in NUM.findall(rest)}
                if fname == "catchRun.cpp" and "srcEpoch" in kv:
                    cur = {"sid": int(kv.get("sid", -1)), "rec": kv.get("recEpoch", kv["srcEpoch"]), "ev": False, "dec": False, "order": None}
                    blocks[cur["sid"]].append(cur)
                elif fname == "alg.cpp" and "emsMktVol" in kv and cur is not None:
                    cur.update(fa0=kv.get("fa"), bk=kv.get("book_epoch"))
                elif fname == "alg.cpp" and "vpen" in kv and "executp" not in kv and cur is not None:
                    cur.update(ev=True, **{k: kv.get(k, np.nan) for k in ("mp", "vpen", "vr", "poscst", "cpos", "asq", "hs")})
                    cur["fa"] = kv.get("fa", cur.get("fa0", np.nan))
                elif fname == "alg.cpp" and "executp" in kv and cur is not None:
                    cur.update(dec=True, executp=kv["executp"], lthr=kv.get("lthr", 0.0), wq=kv.get("winsdQty", np.nan), pp=kv.get("pp", 0.0))
                    if cur.get("fa") is None or cur.get("fa") != cur.get("fa"):
                        cur["fa"] = kv.get("fa", cur.get("fa0", np.nan))
                elif fname == "alg.cpp" and "nullptr" in rest and cur is not None:
                    m = re.search(r"nullptr\s*:\s*(\d+|true|false)", rest)
                    cur["order"] = bool(m) and m.group(1) in ("0", "false")
                elif fname == "es.cpp" and "nowTs:" in rest:
                    fld = rest.split("nowTs:", 1)[1].split(",")
                    try:
                        last_so = {"e": float(fld[1]), "sid": int(float(fld[5])), "px": float(fld[6]), "qty": float(fld[7].split()[0]), "side": None,
                                   "in_block": cur is not None and int(float(fld[5])) == cur["sid"]}
                        sos.append(last_so)
                    except (IndexError, ValueError):
                        last_so = None
                elif fname == "catchRun.cpp" and "place_order" in rest and last_so is not None:
                    m = re.search(r"symbol = (\d{6})\S*, volume = ([\d.]+), price = ([\d.]+), side = (\w+)", rest)
                    if m and int(m.group(1)) == last_so["sid"]:
                        last_so["side"] = m.group(4).lower()
                    last_so = None
    return blocks, sos


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--catalog", type=Path, required=True)
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()
    from pm_signals import log_files
    from runs_ftp import gtp
    from runs_ord import cord

    blocks, sos = parse(log_files(a.logs, a.date))
    for s in blocks:
        blocks[s] = [b for b in blocks[s] if b.get("ev")]
    recs = {s: [b["rec"] for b in v] for s, v in blocks.items()}
    print(f"== {a.date}: {sum(len(v) for v in blocks.values()):,} logged blocks with an evaluation line on {len(blocks)} stocks; {len(sos):,} SO lines, with a side {sum(1 for s in sos if s['side'])}, printed in their own stock's block {sum(s['in_block'] for s in sos)}")
    if not sos:
        return
    import duckdb

    con = duckdb.connect(str(a.catalog), read_only=True)
    so = pl.DataFrame({"i": list(range(len(sos))), "sid": [s["sid"] for s in sos], "e": [round(s["e"] * 1e9) for s in sos]})
    con.register("so", so.to_arrow())
    bk = con.sql(f"""SELECT s.i, (q.spider_ts - s.e) / 1e6 AS lag, q.bid_px[1]::DOUBLE AS b1, q.ask_px[1]::DOUBLE AS a1, q.bid_vol[1]::DOUBLE AS bv1, q.ask_vol[1]::DOUBLE AS av1, q.price::DOUBLE AS last
        FROM so s JOIN raw_quotation q ON q.symbol_id = s.sid AND q.date = {int(a.date)} AND q.spider_ts BETWEEN s.e - 50000000 AND s.e + 2600000000
        QUALIFY row_number() OVER (PARTITION BY s.i ORDER BY q.spider_ts) = 1""").pl()
    book = {r["i"]: r for r in bk.iter_rows(named=True)}
    lag = bk["lag"].to_numpy()
    print(f"   snapshot found for {len(book):,} SO lines; tick-store receive - SO field 2, ms, q1/10/50/90/99: {' / '.join(f'{v:.0f}' for v in np.percentile(lag, [1, 10, 50, 90, 99])) if lag.size else '-'}")

    sent: dict[int, list[tuple[float, float]]] = defaultdict(list)  # stock -> (E, signed qty) of the SO lines so far
    res = {hsk: Counter() for hsk in ("logged hs", "book spread / mid")}
    mism: list[str] = []
    why_no = Counter()
    for i, s in enumerate(sorted(range(len(sos)), key=lambda k: sos[k]["e"])):
        o = sos[s]
        sid, e = o["sid"], o["e"]
        rs = recs.get(sid, [])
        j = bisect.bisect_right(rs, e + 1e-4) - 1
        b = book.get(s)
        sign = 1.0 if o["side"] == "buy" else -1.0 if o["side"] == "sell" else (1.0 if b and b["a1"] and abs(o["px"] - b["a1"]) < 1e-6 else -1.0)
        if j < 0 or b is None or not (b["b1"] > 0 and b["a1"] > 0):
            why_no["no earlier block" if j < 0 else "no snapshot / one-sided book"] += 1
            sent[sid].append((e, sign * o["qty"]))
            continue
        blk = blocks[sid][j]
        age = e - blk["rec"]
        own = age < 0.02
        since = [q for (t, q) in sent[sid] if t > blk["rec"] + 1e-4]
        pend = sum(since)
        sells = -sum(q for q in since if q < 0)
        cls = "own block" if own else "silent <1.5 s" if age < 1.5 else "silent 1.5-3.5 s" if age < 3.5 else "silent 3.5-12 s" if age < ADUR else "silent, alpha older than 12 s"
        hh = int(((e + 28800) % 86400) // 1800)
        mid = (b["b1"] + b["a1"]) / 2
        asq = blk["asq"] - sells
        cur = blk["cpos"] + pend
        for hsk in res:
            hs = blk["hs"] if hsk == "logged hs" else (b["a1"] - b["b1"]) / mid
            if own and blk.get("dec"):
                ep = blk["executp"]
            else:
                ep = gtp(mid, blk["fa"], blk["vpen"], blk["vr"], blk["poscst"], cur, asq, hs)
            r = {"sid": sid, "fa": blk["fa"], "pp": (blk.get("pp") or 0.0) if own else pend, "hs": hs, "lthr": blk.get("lthr") or 0.0, "ev_cpos": cur, "executp": ep,
                 "winsdQty": blk.get("wq") if blk.get("wq") == blk.get("wq") and blk.get("wq") is not None else 1e12, "asq": asq, "a1": b["a1"], "av1": b["av1"], "b1": b["b1"], "bv1": b["bv1"]}
            side, px, q, why = cord(r)
            c = res[hsk]
            for key in (cls, f"hh {hh // 2:02d}:{'30' if hh % 2 else '00'}", "all"):
                c[(key, "n")] += 1
                c[(key, "order")] += bool(side)
                c[(key, "side")] += bool(side) and (o["side"] is None or side == o["side"])
                c[(key, "px")] += bool(side) and abs(px - o["px"]) < 1e-6
                c[(key, "qty")] += bool(side) and abs(q - o["qty"]) < 1e-6
                c[(key, "all3")] += bool(side) and (o["side"] is None or side == o["side"]) and abs(px - o["px"]) < 1e-6 and abs(q - o["qty"]) < 1e-6
            if hsk == "logged hs":
                if not side:
                    why_no[f"{'own' if own else 'silent'}: {why}"] += 1
                if (not side or abs(q - o["qty"]) > 1e-6 or abs(px - o["px"]) > 1e-6) and not own and len(mism) < a.show:
                    mism.append(f"   {sid:06d} SO {o['side']} {o['qty']:.0f} @ {o['px']} | replay {side or 'none'} {q:.0f} @ {px} ({why}) | age {age:.2f} s fa {blk['fa']:.5g} hs {blk['hs']:.5g} book hs {(b['a1'] - b['b1']) / mid:.5g} lthr {blk.get('lthr') or 0:.4g}"
                                f" cur {cur:.0f} (block {blk['cpos']:.0f} + sent {pend:.0f}) asq {asq:.0f} ep {ep:.1f} room {r['winsdQty']:.0f} | bid {b['b1']} x {b['bv1']:.0f} ask {b['a1']} x {b['av1']:.0f} | block mp {blk['mp']}")
        sent[sid].append((e, sign * o["qty"]))
    for hsk, c in res.items():
        print(f"-- cost x = {hsk}: class: SO lines | replay has an order | side | price | quantity | all three")
        keys = [k for k in ("all", "own block", "silent <1.5 s", "silent 1.5-3.5 s", "silent 3.5-12 s", "silent, alpha older than 12 s") if c[(k, "n")]] + sorted(k for k in {k for k, _ in c} if k.startswith("hh "))
        for k in keys:
            n = c[(k, "n")]
            print(f"   {k:<30} {n:>6,} | " + " | ".join(f"{c[(k, m)] / n:.4f}" for m in ("order", "side", "px", "qty", "all3")))
    print(f"-- no replay / no order, by reason (logged hs): {dict(why_no.most_common(12))}")
    print(f"   first {len(mism)} silent-run mismatches:")
    print("\n".join(mism))
    print("[done]")


if __name__ == "__main__":
    main()
