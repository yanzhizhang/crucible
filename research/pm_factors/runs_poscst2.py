r"""poscst = min(floor(5 x sod.mb / 100) x 100, 1e7) holds on ~95% of the stocks (R152); the rest sit far below. Which cap?

Per stock on the day (same-day stats/sod file): candidate caps next to the lot-rounded 5 x mb -- the net-market-value cap
over the price (nmvcst / mp, lot-rounded; first price of the day, and 10% up / down from it as stand-ins for the limit
prices), the quantity cap tqub, multiples of msr and vr, and the STAR 200-share minimum. For every formula the share of
stocks it reproduces exactly; then the stocks the best one misses, with their numbers. Read-only.

    python research/pm_factors/runs_poscst2.py --orders /work/crucible_data/pm_orders --date 20261113 --sts /work/crucible_data/pm_stsfile
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--orders", type=Path, default=Path("/work/crucible_data/pm_orders"))
    ap.add_argument("--date", required=True)
    ap.add_argument("--sts", type=Path, default=Path("/work/crucible_data/pm_stsfile"))
    a = ap.parse_args()
    from ic_score import pm_daily
    c = pl.col
    t = pl.read_parquet(a.orders / f"date={a.date}" / "decisions.parquet").with_columns(
        c("sid").cast(pl.Int64), *[c(k).cast(pl.Float64) for k in ("ev_poscst", "ev_mp", "ev_cpos", "ev_asq")]).sort("src_epoch")
    per = t.group_by("sid").agg(c("ev_poscst").first().alias("poscst"), c("ev_mp").first().alias("mp0"), c("ev_mp").min().alias("mp_min"), c("ev_mp").max().alias("mp_max"),
                                c("ev_cpos").first().alias("cpos0"), c("ev_asq").max().alias("asq_max"), c("ev_cpos").abs().max().alias("cpos_max")).sort("sid")
    j = per.join(pm_daily(a.sts, a.date, per["sid"].to_list(), "on"), on="sid", how="left")
    g = lambda k: j[k].to_numpy().astype(float)  # noqa: E731
    pc, mb, msr, vr, mp = g("poscst"), g("pm.mb"), g("pm.msr"), g("pm.vr"), g("mp0")
    lot = lambda x: np.floor(x / 100) * 100  # noqa: E731
    base = lot(5 * mb)
    forms = {"lot(5 mb)": base, "min(lot(5 mb), 1e7)": np.minimum(base, 1e7)}
    for nm, cap in (("lot(1e7 / mp0)", lot(1e7 / mp)), ("lot(1e7 / (1.1 mp0))", lot(1e7 / (1.1 * mp))), ("lot(1e7 / mp_max)", lot(1e7 / g("mp_max"))),
                    ("lot(1e7 / mp_min)", lot(1e7 / g("mp_min"))), ("lot(5 msr)", lot(5 * msr)), ("lot(msr)", lot(msr)), ("lot(10 msr)", lot(10 * msr)),
                    ("lot(mb)", lot(mb)), ("lot(5 vr)", lot(5 * vr)), ("lot(0.09 x 1e7 / mp0)", lot(0.09 * 1e7 / mp)), ("lot(1e6 / mp0)", lot(1e6 / mp)),
                    ("lot(9e6 / mp0)", lot(9e6 / mp)), ("lot(5e6 / mp0)", lot(5e6 / mp)), ("lot(2e6 / mp0)", lot(2e6 / mp))):
        forms[f"min(lot(5 mb), 1e7, {nm})"] = np.minimum(np.minimum(base, 1e7), cap)
    ok = np.isfinite(pc) & np.isfinite(mb)
    print(f"== {a.date}: {int(ok.sum())} stocks with poscst and mb; table columns {[k for k in j.columns if k.startswith('pm')][:14]}")
    res = sorted(((float((f[ok] == pc[ok]).mean()), nm) for nm, f in forms.items()), reverse=True)
    for sh, nm in res:
        print(f"   exact on {sh:.3f}  {nm}")
    best = forms[res[0][1]]
    miss = ok & (best != pc)
    print(f"-- the {int(miss.sum())} stocks '{res[0][1]}' misses (sid, poscst, formula, poscst/formula, mb, msr, vr, mp0, poscst x mp0, cpos0, asq_max, cpos_max):")
    sid = j["sid"].to_numpy()
    for i in np.flatnonzero(miss)[:40]:
        print(f"   {sid[i]}  {pc[i]:.0f}  {best[i]:.0f}  {pc[i] / best[i]:.5f}  {mb[i]:.6g}  {msr[i]:.6g}  {vr[i]:.6g}  {mp[i]:.4g}  {pc[i] * mp[i]:.6g}  {g('cpos0')[i]:.0f}  {g('asq_max')[i]:.0f}  {g('cpos_max')[i]:.0f}")
    if miss.sum() >= 5:
        r = pc[miss] / best[miss]
        print(f"   ratio quantiles 0/25/50/75/100: {' / '.join(f'{v:.5f}' for v in np.percentile(r, [0, 25, 50, 75, 100]))};"
              f" poscst x mp0 quantiles: {' / '.join(f'{v:.5g}' for v in np.percentile(pc[miss] * mp[miss], [0, 25, 50, 75, 100]))};"
              f" share of misses on 688 / 689 codes: {float(np.mean([str(s).startswith(('688', '689')) for s in sid[miss]])):.2f}, on 30xxxx: {float(np.mean([str(s).startswith('30') for s in sid[miss]])):.2f}")
    print("[done]")


if __name__ == "__main__":
    main()
