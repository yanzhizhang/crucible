r"""runS's execution target, exact: GTp with its caps, then cFtp with the state shared by all stocks (decoded alg.cpp / pos.cpp).

The band target alone gives the logged executp on ~93% of the decisions (R180b / R182); the rest are sells capped at the
sellable quantity and buys cut or scaled. Decoded:
  GTp(mid, alpha, vpen, vr, poscst, pos, asq, x): band step r (lo / hi = (alpha -+ (x + cr) / 1.5) / (2 vpen (vr / 3) mid) - pos),
      tp = clamp(pos + r, +-poscst), step = tp - pos capped to [-cap, cap] with cap = asq for a step <= 0 and asq - pos for a step > 0;
  cFtp(tp, pos, nmv, nmvcst): a step > 0 becomes 0 when nmv > nmvcst, else it is scaled by (nmvcst - nmv - sp) / bp when
      bp + sp > nmvcst - nmv; nmv = sum over stocks of pos x price, bp / sp = sums of the positive / negative (target - pos) x price,
      all three carried across every stock of the alg and updated at each stock's own snapshot (SyncP, then Calc).
The log is re-read in order (every alg.cpp evaluation line: mp fa vpen vr poscst cpos asq hs, and the decision that follows
it when one does) and the three sums are carried exactly as the code does. Printed:
  1  evaluation lines with / without a decision line (is the state visible on every snapshot or only on decisions);
  2  lthr == -w (cr + msr / 2), w = ((vpen - cfg) / (0.001 - cfg))^2: exact share after 13:00;
  3  executp against GTp alone (band, band + caps), by the sign of the step;
  4  executp against GTp + cFtp with the carried state; the confusion table (kept / scaled / cut: logged vs replayed);
  5  the nmvcst each logged scaled buy implies (nmv + sp + s bp, s = logged step / GTp step): one constant if the state is right;
  6  nmv, bp, sp through the day; the first --show mismatches with their inputs.
Read-only.

    python research/pm_factors/runs_ftp.py --logs /work/crucible_data/pm_224/20261118/chrono_logs --date 20261118 --sts /work/crucible_data/sts_224
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

CR, CFG_VPEN, NMVCST = 0.0004, 1e-7, 1e7


def gtp(mid: float, alpha: float, vpen: float, vr: float, poscst: float, pos: float, asq: float, x: float, caps: bool = True) -> float:
    """Alg::GTp as decoded (alg.cpp @76-@122); caps=False stops after the +-poscst clamp."""
    r = 0.0
    if vr > 0.0:
        den = 2 * vpen * (vr / 3.0) * mid
        lo, hi = (alpha - x / 1.5 - CR / 1.5) / den - pos, (alpha + x / 1.5 + CR / 1.5) / den - pos
        r = lo if lo > 0.0 else hi if hi < 0.0 else 0.0
    tp = min(max(pos + r, -poscst), poscst)
    if not caps:
        return tp
    dlt = tp - pos
    cap = asq if dlt <= 0.0 else asq - pos
    if dlt > cap:
        dlt = cap
    elif dlt < -cap:
        dlt = -cap
    return pos + dlt


class Book:
    """MktVs + Estitp of one alg: positions, prices, last targets and the three sums."""

    def __init__(self) -> None:
        self.pos, self.px, self.ep = {}, {}, {}
        self.nmv = self.bp = self.sp = 0.0

    def _bs(self, old: float, new: float) -> None:
        if old > 0.0:
            self.bp -= old
        else:
            self.sp -= old
        if new > 0.0:
            self.bp += new
        else:
            self.sp += new

    def price(self, s: int, npx: float) -> None:
        """HandleSpf: a new price for the stock (nmv and the stock's pending notional move with it)."""
        opx = self.px.get(s, 0.0)
        if npx == opx:
            return
        p, e = self.pos.get(s, 0.0), self.ep.get(s, 0.0)
        self.nmv += p * (npx - opx)
        self._bs((e - p) * opx, (e - p) * npx)
        self.px[s] = npx

    def sync(self, s: int, np_: float) -> None:
        """Estitp::SyncP: a new position."""
        op = self.pos.get(s, 0.0)
        if np_ == op:
            return
        px, e = self.px.get(s, 0.0), self.ep.get(s, 0.0)
        self.nmv += px * (np_ - op)
        self.pos[s] = np_
        self._bs((e - op) * px, (e - np_) * px)

    def calc(self, s: int, tp: float) -> None:
        """Estitp::Calc: a new target."""
        p, px = self.pos.get(s, 0.0), self.px.get(s, 0.0)
        self._bs((self.ep.get(s, 0.0) - p) * px, (tp - p) * px)
        self.ep[s] = tp


def cftp(tp: float, pos: float, nmv: float, bp: float, sp: float, nmvcst: float = NMVCST) -> tuple[float, str]:
    """Alg::cFtp: (executp, kept | cut | scaled)."""
    dp, how = tp - pos, "kept"
    if dp > 0.0:
        if nmv > nmvcst:
            dp, how = 0.0, "cut"
        elif bp + sp > nmvcst - nmv:
            dp, how = dp * (nmvcst - nmv - sp) / bp, "scaled"
    return dp + pos, how


def parse(files: list[str]) -> list[dict]:
    """Every evaluation line in log order, with the decision that follows it (same snapshot) folded in."""
    from pm_orders import GLUE, PLUGIN, SRC
    from pm_signals import NUM

    rows: list[dict] = []
    for f in files:
        last: dict[str, float] = {}
        cur: dict | None = None
        pend: dict | None = None
        with open(f, encoding="utf-8", errors="replace") as fh:
            for line in (part for raw in fh for part in GLUE.split(raw) if part.strip()):
                if " trk " in line:
                    continue
                s = SRC.search(line)
                if not s or (line.startswith("[20") and not PLUGIN.search(line)):
                    continue
                where, rest = s.group(1), s.group(2)
                fname = where.rsplit(":", 1)[0].rsplit("/", 1)[-1]
                if fname not in ("catchRun.cpp", "alg.cpp"):
                    continue
                kv = {k.replace("this->", "").replace("book->", "book_").rstrip("_"): float(v) for k, v in NUM.findall(rest)}
                if fname == "catchRun.cpp" and "srcEpoch" in kv:
                    last, cur, pend = kv, None, None
                elif fname == "alg.cpp" and "vpen" in kv and "executp" not in kv and last:
                    cur = {"sid": int(last.get("sid", -1)), "epoch": last["srcEpoch"], **{k: kv.get(k, np.nan) for k in ("mp", "fa", "vpen", "vr", "poscst", "cpos", "asq", "hs")}, "dec": False}
                    rows.append(cur)
                elif fname == "alg.cpp" and "executp" in kv and cur is not None:
                    cur.update(dec=True, executp=kv["executp"], lthr=kv.get("lthr", np.nan), wq=kv.get("winsdQty", np.nan), pos=kv.get("pos", np.nan), pp=kv.get("pp", np.nan), order=False)
                    pend = cur
                elif fname == "alg.cpp" and "nullptr" in rest and pend is not None:
                    m = re.search(r"nullptr\s*:\s*(\d+|true|false)", rest)
                    pend["order"] = bool(m) and m.group(1) in ("0", "false")
                    pend = None
    return rows


def same(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol * max(1.0, abs(b))


def hm(sod: float) -> str:
    return f"{int(sod // 3600):02d}:{int(sod % 3600 // 60):02d}:{sod % 60:04.1f}"


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--sts", type=Path, default=Path("/work/crucible_data/sts_224"))
    ap.add_argument("--show", type=int, default=20)
    a = ap.parse_args()
    from pm_signals import log_files

    rows = parse(log_files(a.logs, a.date))
    n, nd = len(rows), sum(r["dec"] for r in rows)
    print(f"== {a.date}: 1  {n:,} evaluation lines; with a decision line {nd:,}, without {n - nd:,}; stocks {len({r['sid'] for r in rows})}")
    if not rows:
        return
    for r in rows:
        r["sod"] = (r["epoch"] + 28800) % 86400
    nod = [r for r in rows if not r["dec"]]
    if nod:
        print(f"   lines without a decision: before 13:00 {sum(r['sod'] < 46800 for r in nod):,}, after {sum(r['sod'] >= 46800 for r in nod):,}")

    try:  # 2 lthr
        import polars as pl
        from ic_score import pm_daily

        msr = dict(pm_daily(a.sts, a.date, sorted({r["sid"] for r in rows}), "on").select("sid", "pm.msr").iter_rows())
        aft = [r for r in rows if r["dec"] and r["sod"] >= 46800 and r["lthr"] != 0 and msr.get(r["sid"]) is not None]
        ok = sum(same(r["lthr"], -(((r["vpen"] - CFG_VPEN) / (0.001 - CFG_VPEN)) ** 2) * (CR + msr[r["sid"]] / 2), 1e-6) for r in aft)
        ok3 = sum(same(r["lthr"], -(((r["vpen"] - CFG_VPEN) / (0.001 - CFG_VPEN)) ** 2) * (CR + msr[r["sid"]] / 2), 1e-3) for r in aft)
        print(f"-- 2  lthr == -w (cr + msr / 2) on {ok / max(len(aft), 1):.4f} at 1e-6, {ok3 / max(len(aft), 1):.4f} at 1e-3, of {len(aft):,} decisions after 13:00 (tolerance relative to max(1, |lthr|): see 2b)")
        rel = np.array([abs(r["lthr"] / (-(((r["vpen"] - CFG_VPEN) / (0.001 - CFG_VPEN)) ** 2) * (CR + msr[r["sid"]] / 2)) - 1) for r in aft]) if aft else np.array([np.nan])
        print(f"   2b ratio lthr / formula - 1: |.| q50/90/99/max {' / '.join(f'{x:.2e}' for x in np.nanpercentile(rel, [50, 90, 99, 100]))}")
    except Exception as e:  # noqa: BLE001
        print(f"-- 2  start-of-day table not read ({e!r})")

    bk = Book()
    st = Counter()
    conf = Counter()
    impl, mism, track = [], [], []
    for i, r in enumerate(rows):
        s, pos = r["sid"], r["cpos"]
        bk.price(s, r["mp"])
        bk.sync(s, pos)
        late = r["sod"] >= 53880  # last 60 s before 14:59: the code targets flat
        t0 = gtp(r["mp"], r["fa"], r["vpen"], r["vr"], r["poscst"], pos, r["asq"], r["hs"], caps=False)
        tp = 0.0 if late else gtp(r["mp"], r["fa"], r["vpen"], r["vr"], r["poscst"], pos, r["asq"], r["hs"])
        bk.calc(s, tp)
        nmv, bp, sp = bk.nmv, bk.bp, bk.sp
        pred, how = cftp(tp, pos, nmv, bp, sp) if abs(tp - pos) > 1e-6 else (pos, "kept")
        if i % max(n // 12, 1) == 0:
            track.append(f"{hm(r['sod'])} nmv {nmv:,.0f} bp {bp:,.0f} sp {sp:,.0f}")
        if not r["dec"] or late:
            continue
        ex = r["executp"]
        side = "buy step" if tp > pos else "sell step" if tp < pos else "no step"
        st[("n", side)] += 1
        st[("band", side)] += same(ex, t0)
        st[("gtp", side)] += same(ex, tp)
        st[("ftp", side)] += same(ex, pred)
        if tp > pos:
            logged = "kept" if same(ex, tp) else "cut" if same(ex, pos) else "scaled" if pos < ex < tp else "other"
            conf[(logged, how)] += 1
            if logged == "scaled" and bp > 0:
                impl.append(nmv + sp + (ex - pos) / (tp - pos) * bp)
            if logged == "cut":
                st[("cutn",)] += 1
                st[("cut_nmv_sum",)] += nmv
        if not same(ex, pred) and len(mism) < a.show:
            mism.append(f"   {hm(r['sod'])} {s:06d} executp {ex:.4f} replay {pred:.4f} ({how}) gtp {tp:.4f} band {t0:.4f} | cpos {pos:.1f} asq {r['asq']:.0f} poscst {r['poscst']:.0f} fa {r['fa']:.6g} hs {r['hs']:.6g} mp {r['mp']:.4f}"
                        f" vpen {r['vpen']:.4g} vr {r['vr']:.4g} | nmv {nmv:,.0f} bp {bp:,.0f} sp {sp:,.0f}")
        if r.get("order"):  # after Po the position includes the new pending leg (estimate: the lot-rounded step, capped by the room)
            d = ex - pos
            q = np.floor(min(abs(d), r["wq"] if r["wq"] == r["wq"] else abs(d)) / 100) * 100
            bk.sync(s, pos + np.sign(d) * q)
    print("-- 3 / 4  executp exact (1e-6) by the sign of the GTp step: band only | band + caps (GTp) | GTp + cFtp replay")
    tot = Counter()
    for side in ("buy step", "sell step", "no step"):
        k = st[("n", side)]
        if k:
            print(f"   {side:<9} {k:>9,}: {st[('band', side)] / k:.4f} | {st[('gtp', side)] / k:.4f} | {st[('ftp', side)] / k:.4f}")
        for m in ("n", "band", "gtp", "ftp"):
            tot[m] += st[(m, side)]
    if tot["n"]:
        print(f"   {'all':<9} {tot['n']:>9,}: {tot['band'] / tot['n']:.4f} | {tot['gtp'] / tot['n']:.4f} | {tot['ftp'] / tot['n']:.4f}")
    print("   buy steps, logged (rows) vs replayed (columns kept / scaled / cut):")
    for lg in ("kept", "scaled", "cut", "other"):
        print(f"     logged {lg:<7} " + "  ".join(f"{hw} {conf[(lg, hw)]:>8,}" for hw in ("kept", "scaled", "cut")))
    if impl:
        v = np.array(impl)
        print(f"-- 5  nmvcst implied by {len(v):,} logged scaled buys: q1/10/50/90/99 {' / '.join(f'{x:,.0f}' for x in np.percentile(v, [1, 10, 50, 90, 99]))} (config {NMVCST:,.0f})")
    else:
        print("-- 5  no logged scaled buy")
    if st[("cutn",)]:
        print(f"   logged cut-to-zero buys {st[('cutn',)]:,}: mean replayed nmv there {st[('cut_nmv_sum',)] / st[('cutn',)]:,.0f}")
    print("-- 6  state through the day: " + "; ".join(track))
    print(f"   first {len(mism)} mismatches of the full replay:")
    print("\n".join(mism))
    print("[done]")


if __name__ == "__main__":
    main()
