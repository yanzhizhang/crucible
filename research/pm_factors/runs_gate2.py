r"""runS order gate, second pass: what starts an order CHAIN?

R178: only ~1% of lot-sized, past-threshold decisions send, and orders cluster within seconds of the stock's previous
order (re-places of one chain: order rate 0.47 at 3-5.3 s after an order, 0.004 after 120 s). So the question is what
STARTS a chain. A chain start = an order with no order of that stock in the previous --quiet seconds; the decisions under
study are the "quiet" ones (no order of the stock in the previous --quiet s), chain start yes / no.
  - the wanted trade measured three ways: executp - cpos, executp - pos, executp - pp (which of the logged positions
    is the one the order is cut from), raw, in lots, and in value;
  - the signal measured in threshold units: (|fa| - |lthr|) / hs, |fa| / hs, and the sign agreement of the trade with fa;
  - for every feature the best single cut (the threshold maximising F1 as a classifier of "chain start") and its
    precision / recall; then the best pairs of cuts among the top features;
  - feature quantiles on chain starts against the quiet decisions without an order;
  - what changes at a chain start against the stock's previous decision (did executp jump, did the sign flip, did pos
    or pp move), since a gate on a CHANGE would show there.
Read-only.

    python research/pm_factors/runs_gate2.py --orders /work/crucible_data/pm_orders --date 20261118
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
import polars as pl


def q(v: np.ndarray, qs=(10, 50, 90)) -> str:
    v = v[np.isfinite(v)]
    return " / ".join(f"{x:.4g}" for x in np.percentile(v, qs)) if v.size else "-"


def best_cut(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, str]:
    """(F1, threshold, precision, recall, direction) of the best x >= t or x <= t rule for y."""
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if y.sum() < 5:
        return 0.0, np.nan, 0.0, 0.0, ">="
    best = (0.0, np.nan, 0.0, 0.0, ">=")
    for direction, xs in ((">=", x), ("<=", -x)):
        o = np.argsort(-xs, kind="stable")
        tp = np.cumsum(y[o])
        k = np.arange(1, x.size + 1)
        last = np.r_[xs[o][1:] != xs[o][:-1], True]  # cut only between distinct values
        prec, rec = tp / k, tp / y.sum()
        f1 = np.where(last, 2 * prec * rec / np.maximum(prec + rec, 1e-12), 0.0)
        i = int(np.argmax(f1))
        if f1[i] > best[0]:
            best = (float(f1[i]), float(xs[o][i] if direction == ">=" else -xs[o][i]), float(prec[i]), float(rec[i]), direction)
    return best


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--quiet", type=float, default=30.0)
    a = ap.parse_args()
    c = pl.col
    d = pl.read_parquet(a.orders / f"date={a.date}" / "decisions.parquet").with_columns(c("sid").cast(pl.Int64), c("src_epoch").cast(pl.Int64)).drop_nulls(["src_epoch", "executp"]).sort("src_epoch")
    num = [k for k in ("fa", "pos", "pp", "executp", "asq", "lthr", "hs", "winsdQty", "nullptr", "liid", "ev_mp", "ev_cpos", "ev_poscst", "ev_vpen", "ev_vr") if k in d.columns]
    d = d.with_columns(c(k).cast(pl.Float64) for k in num).with_columns((c("nullptr") == 0).alias("order"), ((c("src_epoch") / 1e9 + 28800) % 86400).alias("sod"))
    o = d.filter(c("order")).select("sid", c("src_epoch").alias("prev_order"))
    d = d.join_asof(o, left_on="src_epoch", right_on="prev_order", by="sid", strategy="backward", allow_exact_matches=False)
    d = d.with_columns(((c("src_epoch") - c("prev_order")) / 1e9).fill_null(1e9).alias("since"))
    for k in ("executp", "pos", "pp", "ev_cpos", "fa", "lthr"):
        if k in d.columns:
            d = d.with_columns(c(k).shift(1).over("sid").alias(f"{k}_prev"))
    quiet = d.filter(c("since") >= a.quiet)
    y = quiet["order"].to_numpy()
    print(f"== {a.date}: {d.height:,} decisions, {int(d['order'].sum()):,} orders; quiet decisions (no order of the stock in the last {a.quiet:g} s): {quiet.height:,}, chain starts {int(y.sum()):,} ({y.mean():.5f})")
    print(f"   orders that are re-places (within {a.quiet:g} s of the stock's previous order): {int(d.filter(c('order') & (c('since') < a.quiet)).height):,}")
    print("   chain starts per half hour: " + ", ".join(f"{int(b) // 2:02d}:{'30' if int(b) % 2 else '00'} {n}" for b, n in sorted(quiet.filter(c('order')).group_by((c('sod') // 1800).cast(pl.Int64)).len().iter_rows())))
    g = lambda k: quiet[k].to_numpy().astype(float) if k in quiet.columns else np.full(quiet.height, np.nan)  # noqa: E731
    ex, pos, pp, cp, fa, lt, hs, mp, w, asq = (g(k) for k in ("executp", "pos", "pp", "ev_cpos", "fa", "lthr", "hs", "ev_mp", "winsdQty", "asq"))
    with np.errstate(divide="ignore", invalid="ignore"):
        F = {"|executp - cpos|": np.abs(ex - cp), "|executp - pos|": np.abs(ex - pos), "|executp - pp|": np.abs(ex - pp), "|executp|": np.abs(ex), "|pos - cpos|": np.abs(pos - cp), "|pp - cpos|": np.abs(pp - cp),
             "|executp - cpos| x mp": np.abs(ex - cp) * mp, "|executp - pos| x mp": np.abs(ex - pos) * mp, "|executp - cpos| / winsdQty": np.abs(ex - cp) / w, "|executp - pos| / winsdQty": np.abs(ex - pos) / w,
             "(|fa| - |lthr|) / hs": (np.abs(fa) - np.abs(lt)) / hs, "|fa| / hs": np.abs(fa) / hs, "|fa|": np.abs(fa), "|fa| - |lthr|": np.abs(fa) - np.abs(lt), "|fa| - hs": np.abs(fa) - hs, "|fa| - |lthr| - hs": np.abs(fa) - np.abs(lt) - hs,
             "sign(executp - cpos) == sign(fa)": (np.sign(ex - cp) == np.sign(fa)).astype(float), "sign(executp - pos) == sign(fa)": (np.sign(ex - pos) == np.sign(fa)).astype(float),
             "asq": asq, "winsdQty": w, "hs": hs, "|lthr|": np.abs(lt), "|executp - executp_prev|": np.abs(ex - g("executp_prev")), "|pos - pos_prev|": np.abs(pos - g("pos_prev")), "|pp - pp_prev|": np.abs(pp - g("pp_prev")),
             "|cpos - cpos_prev|": np.abs(cp - g("ev_cpos_prev")), "sign(executp) != sign(executp_prev)": (np.sign(ex) != np.sign(g("executp_prev"))).astype(float),
             "executp_prev == 0 and executp != 0": ((g("executp_prev") == 0) & (ex != 0)).astype(float), "lthr_prev == 0 and lthr != 0": ((g("lthr_prev") == 0) & (lt != 0)).astype(float)}
    cuts = sorted(((*best_cut(v, y), k) for k, v in F.items()), reverse=True)
    print("-- best single cut per feature (F1, rule, precision, recall):")
    for f1, t, p, r, dr, k in cuts:
        print(f"   F1 {f1:.4f}  {k} {dr} {t:.6g}   precision {p:.4f} recall {r:.4f}")
    top = [k for *_, k in cuts[:7]]
    best = []
    for k1, k2 in itertools.combinations(top, 2):
        f1a, t1, *_r, d1, _ = next(x for x in cuts if x[-1] == k1)
        m1 = F[k1] >= t1 if d1 == ">=" else F[k1] <= t1
        f2 = best_cut(np.where(m1, F[k2], np.nan), y)
        best.append((f2[0], f"{k1} {d1} {t1:.6g}  AND  {k2} {f2[4]} {f2[1]:.6g}", f2[2], f2[3] * (y[m1 & np.isfinite(F[k2])].sum() / max(y.sum(), 1))))
    print("-- best pairs (second cut chosen inside the first; F1 within the first cut, precision, recall of all chain starts):")
    for f1, name, p, r in sorted(best, reverse=True)[:6]:
        print(f"   F1 {f1:.4f}  {name}   precision {p:.4f} recall {r:.4f}")
    print("-- feature quantiles q10 / q50 / q90: chain starts | quiet decisions without an order")
    for k in ("|executp - cpos|", "|executp - pos|", "|executp - pp|", "|executp - cpos| x mp", "(|fa| - |lthr|) / hs", "|fa| / hs", "|fa| - |lthr| - hs", "|executp - executp_prev|", "|pos - pos_prev|", "asq", "winsdQty"):
        print(f"   {k:<28} {q(F[k][y])}  |  {q(F[k][~y])}")
    s = quiet.filter(c("order"))
    if s.height:
        tr = (s["executp"] - s["ev_cpos"]).to_numpy() if "ev_cpos" in s.columns else (s["executp"] - s["pos"]).to_numpy()
        print(f"-- on chain starts: executp == 0 on {float((s['executp'] == 0).mean()):.3f}; trade sign + {float((tr > 0).mean()):.3f}; sign(trade) == sign(fa) {float((np.sign(tr) == np.sign(s['fa'].to_numpy())).mean()):.3f};"
              f" lthr == 0 on {float((s['lthr'] == 0).mean()):.3f}; pos == cpos on {float((s['pos'] == s['ev_cpos']).mean()) if 'ev_cpos' in s.columns else float('nan'):.3f}; pp == pos on {float((s['pp'] == s['pos']).mean()):.3f}")
        print(f"   quiet decisions without an order: executp == 0 on {float((quiet.filter(~c('order'))['executp'] == 0).mean()):.3f}; pp == pos on {float((quiet.filter(~c('order'))['pp'] == quiet.filter(~c('order'))['pos']).mean()):.3f}")
    print("[done]")


if __name__ == "__main__":
    main()
