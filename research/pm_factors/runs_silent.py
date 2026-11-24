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


def parse(files: list[str]) -> tuple[dict[int, list[dict]], list[dict], dict[int, tuple[list[float], list[float]]]]:
    """(per stock: its logged blocks in order; every SO line; per stock: chrono's trk SIG handle times (epoch s) and predictions)."""
    from pm_orders import GLUE, KV, PLUGIN, SRC, TRK
    from pm_signals import NUM, repair

    sig: dict[int, tuple[list[float], list[float]]] = defaultdict(lambda: ([], []))

    blocks: dict[int, list[dict]] = defaultdict(list)
    sos: list[dict] = []
    for f in files:
        cur: dict | None = None
        last_so: dict | None = None
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in (part for raw in fh for part in GLUE.split(raw) if part.strip()):
                if " trk " in line:
                    m = TRK.search(line)
                    if m and m.group(1) == "SIG":
                        kv = repair({k.group(1): k.group(3) if k.group(3) is not None else k.group(4) for k in KV.finditer(m.group(2))})
                        try:
                            t = sig[int(str(kv.get("symbol"))[:6])]
                            t[0].append(float(kv["handle_ts"]) / 1e9)
                            t[1].append(float(kv["prediction"]))
                        except (KeyError, ValueError, TypeError):
                            pass
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
    for sid, (ts, pv) in sig.items():
        o = np.argsort(ts, kind="stable")
        sig[sid] = ([ts[i] for i in o], [pv[i] for i in o])
    return blocks, sos, dict(sig)


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

    blocks, sos, sigs = parse(log_files(a.logs, a.date))
    for s in blocks:
        blocks[s] = [b for b in blocks[s] if b.get("ev")]
    recs = {s: [b["rec"] for b in v] for s, v in blocks.items()}
    print(f"== {a.date}: {sum(len(v) for v in blocks.values()):,} logged blocks with an evaluation line on {len(blocks)} stocks; {len(sos):,} SO lines, with a side {sum(1 for s in sos if s['side'])}, printed in their own stock's block {sum(s['in_block'] for s in sos)}")
    cal = []  # the logged fa against the last signal handled before the block (is fa the signal's prediction?)
    for sid, v in blocks.items():
        sg = sigs.get(sid)
        if not sg:
            continue
        for blk in v[:: max(len(v) // 50, 1)]:
            k = bisect.bisect_right(sg[0], blk["rec"]) - 1
            if k >= 0 and blk.get("fa") == blk.get("fa") and blk.get("fa") is not None:
                cal.append((blk["fa"], sg[1][k], blk["rec"] - sg[0][k]))
    if cal:
        c_ = np.array(cal)
        print(f"   signals: {sum(len(v[0]) for v in sigs.values()):,} trk SIG on {len(sigs)} stocks; block fa == the last signal's prediction before the block (1e-9) on {float((np.abs(c_[:, 0] - c_[:, 1]) < 1e-9).mean()):.4f} of {len(c_):,} sampled blocks;"
              f" that signal's age at the block, s, q10/50/90: {' / '.join(f'{x:.3f}' for x in np.percentile(c_[:, 2], [10, 50, 90]))}")
    else:
        print("   no trk SIG lines parsed")
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
    res = {hsk: Counter() for hsk in ("alpha run (block as logged)", "snapshot run (previous alpha, state before the order; pending = logged - order)",
                                      "snapshot run (previous alpha, state before the order; pending = logged)", "snapshot run (previous alpha, state before the order; pending = 0)")}
    flag, which = Counter(), Counter()
    mism: list[str] = []
    qo = Counter()
    why_no = Counter()
    sil, sil_hh, cross = Counter(), Counter(), Counter()  # R206: no-block rows under 12 input choices; own rows whose replay crosses zero
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
        fa_prev, prev_age = None, float("nan")  # the last signal handled before the snapshot: the alpha a snapshot-driven run still holds
        fa_sig, sig_dt = None, float("nan")  # the first signal of the stock handled from the snapshot's receive time on (within 1.5 s)
        sg = sigs.get(sid)
        if sg:
            k = bisect.bisect_left(sg[0], e - 0.002)
            if k < len(sg[0]) and sg[0][k] <= e + 1.5:
                fa_sig, sig_dt = sg[1][k], sg[0][k] - e
            if k > 0:
                fa_prev, prev_age = sg[1][k - 1], e - sg[0][k - 1]
        if own:
            flag[("flagged" if blk.get("order") else "not flagged" if blk.get("dec") else "no decision line", "signal after" if fa_sig is not None else "no signal after")] += 1
        outcome = {}
        state = {}
        sq = sign * o["qty"]
        wq0 = blk.get("wq") if blk.get("wq") == blk.get("wq") and blk.get("wq") is not None else 1e12
        for hsk in res:
            hs = blk["hs"]
            if hsk == "alpha run (block as logged)":
                fa, cur_, asq_, wq_, pp_ = blk["fa"], cur, asq, wq0, ((blk.get("pp") or 0.0) if own else pend)
                ep = blk["executp"] if own and blk.get("dec") else gtp(mid, fa, blk["vpen"], blk["vr"], blk["poscst"], cur_, asq_, hs)
            else:  # the snapshot-driven run just before the logged one: the previous alpha (0 when older than adur), the state before this order
                fa = fa_prev if fa_prev is not None and prev_age < ADUR else 0.0
                if own:  # the block was logged after this order: take the order back out of its state
                    cur_, asq_, wq_ = cur - sq, asq + (o["qty"] if sign < 0 else 0.0), wq0 + o["qty"] / 0.85
                else:  # an older block never held this order (R198)
                    cur_, asq_, wq_ = cur, asq, wq0
                base = (blk.get("pp") or 0.0) if own else pend  # the pending leg the snapshot run saw: the logged pp_ less this order, as logged, or none
                pp_ = base - sq if hsk.endswith("pending = logged - order)") else base if hsk.endswith("pending = logged)") else 0.0
                ep = gtp(mid, fa, blk["vpen"], blk["vr"], blk["poscst"], cur_, asq_, hs)
            r = {"sid": sid, "fa": fa, "pp": pp_, "hs": hs, "lthr": blk.get("lthr") or 0.0, "ev_cpos": cur_, "executp": ep,
                 "winsdQty": wq_, "asq": asq_, "a1": b["a1"], "av1": b["av1"], "b1": b["b1"], "bv1": b["bv1"]}
            side, px, q, why = cord(r)
            state[hsk] = cur_
            good = bool(side) and (o["side"] is None or side == o["side"]) and abs(px - o["px"]) < 1e-6 and abs(q - o["qty"]) < 1e-6
            outcome[hsk] = (side, px, q, why, good, fa, ep)
        ka = "alpha run (block as logged)"
        for tag in ("pending = logged)", "pending = 0)"):
            kb = next(k for k in res if k.endswith(tag))
            outcome[f"alpha run if exact, else snapshot run with {tag[:-1]}"] = outcome[ka] if outcome[ka][4] else outcome[kb]
        outcome["alpha run if exact, else snapshot run"] = outcome["alpha run if exact, else snapshot run with pending = logged"]
        for hsk, (side, px, q, why, good, fa, ep) in outcome.items():
            if hsk == "alpha run if exact, else snapshot run":
                continue
            c = res.setdefault(hsk, Counter())
            for key in (cls, f"hh {hh // 2:02d}:{'30' if hh % 2 else '00'}", "all"):
                c[(key, "n")] += 1
                c[(key, "order")] += bool(side)
                c[(key, "side")] += bool(side) and (o["side"] is None or side == o["side"])
                c[(key, "px")] += bool(side) and abs(px - o["px"]) < 1e-6
                c[(key, "qty")] += bool(side) and abs(q - o["qty"]) < 1e-6
                c[(key, "all3")] += good
        side, px, q, why, good, fa, ep = outcome["alpha run if exact, else snapshot run"]
        sod = (e + 28800) % 86400
        if not own:  # R206: which alpha / spread / threshold reproduces an order whose snapshot has no logged block
            sil_hh[f"{int(sod // 3600):02d}h"] += 1
            hs_new = (b["a1"] - b["b1"]) / 2 / mid
            for fn, fv in (("old block's alpha", blk["fa"]), ("signal before", fa_prev), ("signal after", fa_sig)):
                for hn, hv in (("old spread", blk["hs"]), ("new spread", hs_new)):
                    for tn, tv in (("threshold", blk.get("lthr") or 0.0), ("no threshold", -1.0)):
                        sil[(fn, hn, tn, "n")] += 1
                        if fv is None:
                            continue
                        r2 = {"sid": sid, "fa": fv, "pp": pend, "hs": hv, "lthr": tv, "ev_cpos": cur, "executp": gtp(mid, fv, blk["vpen"], blk["vr"], blk["poscst"], cur, asq, hv),
                              "winsdQty": wq0, "asq": asq, "a1": b["a1"], "av1": b["av1"], "b1": b["b1"], "bv1": b["bv1"]}
                        s2, p2, q2, _ = cord(r2)
                        sil[(fn, hn, tn, "order")] += bool(s2)
                        sil[(fn, hn, tn, "sidepx")] += bool(s2) and abs(p2 - o["px"]) < 1e-6
                        sil[(fn, hn, tn, "all3")] += bool(s2) and abs(p2 - o["px"]) < 1e-6 and abs(q2 - o["qty"]) < 1e-6
        elif side and abs(px - o["px"]) < 1e-6:  # R206: own rows where the replay's order would take the position through zero
            c0 = state["alpha run (block as logged)" if outcome[ka][4] else next(k for k in res if k.endswith("pending = logged)"))]
            sgn = 1.0 if side == "buy" else -1.0
            if c0 * sgn < 0 and q > abs(c0) + 1e-6:
                tb = f"{int(sod // 3600):02d}:{int(sod % 3600 // 600) * 10:02d}"
                cross[(tb, "n")] += 1
                cross[(tb, "SO == replay")] += abs(q - o["qty"]) < 1e-6
                cross[(tb, "SO == |cur|")] += abs(o["qty"] - abs(c0)) < 1e-6
        which[("alpha run exact" if outcome[ka][4] else "snapshot run exact" if good else "neither", "flagged" if blk.get("order") else "not flagged")] += 1
        if not side:
            why_no[f"{'own' if own else 'silent'}: {why}"] += 1
        qty_only = bool(side) and abs(px - o["px"]) < 1e-6 and not good
        if qty_only:
            qo[(("last 60 s" if (e + 28800) % 86400 >= 53880 else "after 14:30" if (e + 28800) % 86400 >= 52200 else "earlier"), why,
                "SO > replay" if o["qty"] > q else "SO < replay", "SO == level-1 size" if abs(o["qty"] - (b["av1"] if sign > 0 else b["bv1"])) < 1e-6 else
                "SO == |cur|" if abs(o["qty"] - abs(cur)) < 1e-6 else "SO == asq" if abs(o["qty"] - asq) < 1e-6 else "other")] += 1
        if not good and (qty_only or not own) and len(mism) < a.show:
            sb = outcome[ka]
            mism.append(f"   {sid:06d} SO {o['side']} {o['qty']:.0f} @ {o['px']} | replay {side or 'none'} {q:.0f} @ {px} ({why}) | block age {age:.3f} s flagged {blk.get('order')} fa {blk['fa']:.5g} -> ep {sb[6]:.1f} ({sb[3]})"
                        f" | previous signal fa {fa_prev if fa_prev is None else round(fa_prev, 7)} ({prev_age:.2f} s old) -> ep {ep:.1f}, state before: cur {cur - sq:.0f} | next signal after {sig_dt * 1e3:.1f} ms | hs {blk['hs']:.5g} lthr {blk.get('lthr') or 0:.4g}"
                        f" cur {cur:.0f} (block {blk['cpos']:.0f} + sent {pend:.0f}) asq {asq:.0f} room {r['winsdQty']:.0f} | bid {b['b1']} x {b['bv1']:.0f} ask {b['a1']} x {b['av1']:.0f}")
        sent[sid].append((e, sign * o["qty"]))
    print(f"   SO lines on their own block, by the block's logged flag and whether a signal of the stock was handled within 1.5 s after the snapshot: {dict(flag.most_common())}")
    print(f"   which run explains the SO line exactly, by the block's logged flag: {dict(which.most_common())}")
    for hsk, c in res.items():
        print(f"-- {hsk}: class: SO lines | replay has an order | side | price | quantity | all three")
        keys = [k for k in ("all", "own block", "silent <1.5 s", "silent 1.5-3.5 s", "silent 3.5-12 s", "silent, alpha older than 12 s") if c[(k, "n")]] + sorted(k for k in {k for k, _ in c} if k.startswith("hh "))
        for k in keys:
            n = c[(k, "n")]
            print(f"   {k:<30} {n:>6,} | " + " | ".join(f"{c[(k, m)] / n:.4f}" for m in ("order", "side", "px", "qty", "all3")))
    print(f"-- no replay / no order, by reason (alpha run if exact, else snapshot run): {dict(why_no.most_common(12))}")
    print(f"-- right side and price, other quantity ({sum(qo.values())}): (when, what set the replay's quantity, direction, what the SO quantity equals) -> count")
    for k, v in qo.most_common(12):
        print(f"   {v:>4}  " + " | ".join(k))
    print(f"-- R206 orders with no logged block at their snapshot ({sum(sil_hh.values())}), by hour: {dict(sorted(sil_hh.items()))}")
    print("   replay on the old block's position with: alpha | spread | threshold -> has an order | side and price | all three")
    for key in sorted({k[:3] for k in sil}, key=lambda k: -sil[(*k, "all3")]):
        n = max(sil[(*key, "n")], 1)
        print(f"   {' | '.join(key):<52} " + " | ".join(f"{sil[(*key, m)] / n:.4f}" for m in ("order", "sidepx", "all3")))
    print("-- R206 own-block orders whose replay goes through zero (replay quantity > |position|, opposite sign), per 10 minutes: n | SO == replay | SO == |position|")
    for tb in sorted({k[0] for k in cross}):
        print(f"   {tb}  {cross[(tb, 'n')]:>5} | {cross[(tb, 'SO == replay')]:>5} | {cross[(tb, 'SO == |cur|')]:>5}")
    print(f"   first {len(mism)} mismatches of the combined replay (quantity-only and no-block rows):")
    print("\n".join(mism))
    print("[done]")


if __name__ == "__main__":
    main()
