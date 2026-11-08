r"""Textbook high-frequency factors, with names that say what they are.

Naming rule for every factor we write ourselves (``docs`` / ``NAMING.md``, "Our own names"): ``<quantity>_<detail>_<window>``,
whole words from a short fixed vocabulary, the window in seconds with an ``s`` (``_300s``), no letter codes. A reader must
know what the column is from its name alone; ``DESCRIBE`` holds one plain sentence per name and is printed next to it.

All at times ``T`` (trading-clock seconds, local arrival), from the stock's own snapshots (``gf``) and trades (``gs``), as
``ic_score.generic`` gets them; everything is a running sum read off with ``searchsorted``, O(events + times) per stock.

* order-flow imbalance at level 1 (Cont, Kukanov & Stoikov 2014), summed over a window and scaled by the mean depth;
* microprice minus mid (Stoikov 2018);
* realized skewness / kurtosis and the downside share of the realized variance (Amaya et al. 2015);
* price-volume correlation over the window's snapshots (a common A-share broker factor);
* lag-1 autocorrelation of trade signs (order splitting / herding);
* Amihud illiquidity: |return| per 100 million CNY traded;
* toxicity: the mean over 1-minute buckets of |buy - sell| / (buy + sell) amount (a time-bar VPIN, Easley, Lopez de
  Prado & O'Hara 2012);
* price level: log mid and tick size over mid (one 0.01 CNY tick is a large step on a cheap stock).
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl

import dcp_ema as E  # per-update arrays
import feature_intraday as FI  # at_marks

TICK = 0.01  # CNY, every A-share board

DESCRIBE = {
    "ofi_level1_norm_{w}s": "level-1 order-flow imbalance (bid queue added - ask queue added) over the last {w} s, divided by the mean level-1 depth",
    "microprice_minus_mid_rel": "(depth-weighted microprice - mid) / mid at the latest snapshot",
    "realized_skew_{w}s": "realized skewness of snapshot mid log returns over the last {w} s",
    "realized_kurt_{w}s": "realized kurtosis of snapshot mid log returns over the last {w} s",
    "downside_var_share_{w}s": "share of the realized variance from falling snapshots over the last {w} s",
    "price_volume_corr_{w}s": "correlation of log mid and volume traded per snapshot interval over the last {w} s",
    "trade_sign_autocorr_{w}s": "lag-1 autocorrelation of trade signs (buy +1, sell -1) over the last {w} s",
    "amihud_illiq_{w}s": "|mid log return| over the last {w} s per 100 million CNY traded in it",
    "toxicity_abs_imb_1min_mean_{w}s": "mean over 1-minute buckets in the last {w} s of |buy - sell| / (buy + sell) traded amount",
    "log_mid_price": "log of the mid price",
    "tick_size_over_mid": "one price tick (0.01 CNY) over the mid price",
}


def describe(name: str) -> str:
    """Plain description of one of our names (``gen.`` prefix allowed), '' when unknown."""
    base = name.removeprefix("gen.")
    for pat, text in DESCRIBE.items():
        if "{w}" not in pat:
            if base == pat:
                return text
            continue
        head, tail = pat.split("{w}")
        if base.startswith(head) and base.endswith(tail) and base[len(head):-len(tail)].isdigit():
            return text.format(w=base[len(head):-len(tail)])
    return ""


def _win(cs: np.ndarray, t: np.ndarray, T: np.ndarray, w: float) -> np.ndarray:
    """Sum over (T - w, T] from a running sum ``cs`` (len(t) + 1) of per-event values at times ``t``."""
    return cs[np.searchsorted(t, T, side="right")] - cs[np.searchsorted(t, T - w, side="right")]


def _cs(v: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(np.nan_to_num(v))])


def literature(gf: pl.DataFrame | None, gs: pl.DataFrame | None, T: np.ndarray) -> dict[str, np.ndarray]:
    """Every factor of the module docstring at times T."""
    out: dict[str, np.ndarray] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if gf is not None and gf.height > 3:
            a = E.arrays(gf, "fast")
            ok = np.isfinite(a["mid"]) & (a["mid"] > 0)
            t, mid = a["t_local"][ok], a["mid"][ok]
            lm = np.log(mid)
            b1, a1 = gf["b1"].to_numpy().astype(float)[ok], gf["a1"].to_numpy().astype(float)[ok]
            bv, av = gf["bv1"].to_numpy().astype(float)[ok], gf["av1"].to_numpy().astype(float)[ok]
            m0 = FI.at_marks(t, lm, T)
            out["log_mid_price"] = m0
            out["tick_size_over_mid"] = TICK / np.exp(m0)
            micro = np.where((bv + av > 0) & (b1 > 0) & (a1 > 0), (b1 * av + a1 * bv) / (bv + av), mid)
            out["microprice_minus_mid_rel"] = FI.at_marks(t, (micro - mid) / mid, T)
            # Cont-Kukanov-Stoikov level-1 order-flow event per snapshot
            e = np.zeros(t.size)
            e[1:] = (bv[1:] * (b1[1:] >= b1[:-1]) - bv[:-1] * (b1[1:] <= b1[:-1])
                     - av[1:] * (a1[1:] <= a1[:-1]) + av[:-1] * (a1[1:] >= a1[:-1]))
            cs_e, cs_d, cs_n = _cs(e), _cs((bv + av) / 2), _cs(np.ones(t.size))
            for w in (10, 30, 60, 300):
                n = _win(cs_n, t, T, w)
                out[f"ofi_level1_norm_{w}s"] = _win(cs_e, t, T, w) / (_win(cs_d, t, T, w) / n)
            r = np.concatenate([[0.0], np.diff(lm)])
            cs1, cs2, cs3, cs4 = _cs(np.ones(t.size)), _cs(r ** 2), _cs(r ** 3), _cs(r ** 4)
            cs_dn = _cs(np.where(r < 0, r ** 2, 0.0))
            dv = np.clip(np.nan_to_num(a["dvol"][ok]), 0, None)
            cx, cy, cxx, cyy, cxy = _cs(lm), _cs(dv), _cs(lm * lm), _cs(dv * dv), _cs(lm * dv)
            for w in (300, 900, 1800):
                n, s2 = _win(cs1, t, T, w), _win(cs2, t, T, w)
                out[f"realized_skew_{w}s"] = np.sqrt(n) * _win(cs3, t, T, w) / s2 ** 1.5
                out[f"realized_kurt_{w}s"] = n * _win(cs4, t, T, w) / s2 ** 2
                out[f"downside_var_share_{w}s"] = _win(cs_dn, t, T, w) / s2
                sx, sy = _win(cx, t, T, w), _win(cy, t, T, w)
                vx, vy = _win(cxx, t, T, w) - sx * sx / n, _win(cyy, t, T, w) - sy * sy / n
                out[f"price_volume_corr_{w}s"] = (_win(cxy, t, T, w) - sx * sy / n) / np.sqrt(vx * vy)
        if gs is not None and gs.height > 3:
            b = E.arrays(gs, "slow")
            cont = gs["s"].to_numpy() >= 34200
            ts, side, amt = b["t_local"], np.where(cont, b["side"], 0.0), np.where(cont, b["amt"], 0.0)
            prod = np.concatenate([[0.0], side[1:] * side[:-1]])
            cs_p, cs_n = _cs(prod), _cs(np.abs(side))
            for w in (60, 300):
                out[f"trade_sign_autocorr_{w}s"] = _win(cs_p, ts, T, w) / _win(cs_n, ts, T, w)
            cs_a, cs_na = _cs(amt), _cs(amt * side)
            if "log_mid_price" in out:
                for w in (300, 900):
                    ret = np.abs(out["log_mid_price"] - FI.at_marks(t, lm, T - w))
                    out[f"amihud_illiq_{w}s"] = ret / (_win(cs_a, ts, T, w) / 1e8)
            for w in (900, 1800):
                ratios = []
                for j in range(w // 60):  # 1-minute buckets ending j minutes before T
                    tot = _win(cs_a, ts, T - 60 * j, 60)
                    ratios.append(np.where(tot > 0, np.abs(_win(cs_na, ts, T - 60 * j, 60)) / tot, np.nan))
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # a window with no trade at all
                    out[f"toxicity_abs_imb_1min_mean_{w}s"] = np.nanmean(np.stack(ratios), axis=0)
    return out
