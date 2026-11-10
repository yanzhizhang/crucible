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
* price level: log mid and tick size over mid (one 0.01 CNY tick is a large step on a cheap stock);
* (2026-11-08, round 11) aimed at PM families still unsolved: aggressor-order size tiers (qf), deeper-book shape and
  multi-level OFI (Xu, Gould & Howison 2019; Cao, Hansch & Wang 2009; Naes & Skjeltorp 2006 -- wj / dw), the open
  auction and the first half hour (Lou, Polk & Skouras 2019; Gao, Han, Li & Zhou 2018 -- op / ri / kp), the distance to
  the daily price limits, the trading-clock time itself, and the order stream (``order_stream``: order aggressiveness
  against the quote, Biais, Hillion & Spatt 1995; order rate, order-to-trade ratio, mean order size, channel sequence
  rate). Inputs need ``dcp_ema.load(..., extra=True)`` and ``tick_flow.load_orders``; a missing column skips its group.
"""

from __future__ import annotations

import re
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
    # aggressor-order size tiers: a trade's tier is the total amount its aggressive order traded (all its fills)
    "net_buy_amount_share_orders_{tier}_{win}": "(buy - sell aggressor amount) of orders in the {tier} tier over all traded amount, {win}",
    "net_buy_amount_share_orders_over_200k_cny_{win}": "(buy - sell aggressor amount) of orders trading over 200k CNY over all traded amount, {win}",
    "amount_share_orders_over_200k_cny_{win}": "share of the traded amount from aggressor orders trading over 200k CNY, {win}",
    # deeper book
    "ofi_levels1to5_norm_{w}s": "order-flow imbalance summed over book levels 1-5 over the last {w} s, divided by the mean 5-level depth",
    "ofi_levels2to5_norm_{w}s": "order-flow imbalance summed over book levels 2-5 over the last {w} s, divided by the mean level 2-5 depth",
    "depth_weighted_price_5lvl_minus_mid_rel": "(price averaged over both sides' 5 levels weighted by volume - mid) / mid",
    "bid_depth_weighted_distance_5lvl_rel": "(mid - volume-weighted bid price of levels 1-5) / mid",
    "ask_depth_weighted_distance_5lvl_rel": "(volume-weighted ask price of levels 1-5 - mid) / mid",
    "log_bid_depth_5lvl_per_rel_distance_to_level5": "log(bid volume of levels 1-5 / ((mid - bid level-5 price) / mid)): bid book slope",
    "log_ask_depth_5lvl_per_rel_distance_to_level5": "log(ask volume of levels 1-5 / ((ask level-5 price - mid) / mid)): ask book slope",
    "book_slope_log_ratio_bid_over_ask_5lvl": "bid book slope minus ask book slope (both logs)",
    "ofi_levels1to{k}_norm_{w}s": "order-flow imbalance summed over book levels 1-{k} over the last {w} s, divided by the mean 1-{k}-level depth",
    "ofi_level{k}_norm_{w}s": "order-flow imbalance at book level {k} alone over the last {w} s, divided by its mean depth",
    "depth_imbalance_levels1to{k}": "(bid - ask volume of levels 1-{k}) / their sum",
    "depth_imbalance_level{k}": "(bid - ask volume at level {k}) / their sum",
    "ofi_levels1to10_norm_{w}s": "order-flow imbalance summed over book levels 1-10 over the last {w} s, divided by the mean 10-level depth",
    "ofi_levels6to10_norm_{w}s": "order-flow imbalance summed over book levels 6-10 over the last {w} s, divided by the mean level 6-10 depth",
    "depth_weighted_price_10lvl_minus_mid_rel": "(price averaged over both sides' 10 levels weighted by volume - mid) / mid",
    "bid_depth_weighted_distance_10lvl_rel": "(mid - volume-weighted bid price of levels 1-10) / mid",
    "ask_depth_weighted_distance_10lvl_rel": "(volume-weighted ask price of levels 1-10 - mid) / mid",
    "depth_imbalance_10lvl": "(bid - ask volume of levels 1-10) / their sum",
    "depth_imbalance_levels6to10": "(bid - ask volume of levels 6-10) / their sum",
    "bid_depth_share_beyond_level5": "bid volume of levels 6-10 / bid volume of levels 1-10",
    "ask_depth_share_beyond_level5": "ask volume of levels 6-10 / ask volume of levels 1-10",
    "book_slope_log_ratio_bid_over_ask_10lvl": "bid minus ask log(10-level volume / relative distance to level 10)",
    # open auction, first half hour, limits, time
    "overnight_gap_log": "log(open / previous close)",
    "return_open_to_1000_log": "log(mid at min(now, 10:00) / open)",
    "log_open_auction_amount_cny": "log(1 + amount traded in the 09:25 open auction, CNY)",
    "open_auction_amount_share_of_day_so_far": "open-auction amount over the amount traded since the open auction",
    "open_auction_imbalance_level1_last_before_0925": "(bid1 - ask1 volume) / (bid1 + ask1 volume) on the last snapshot before 09:25",
    "open_auction_indicative_change_0920_to_open_log": "log(open / indicative bid1 on the last snapshot before 09:20)",
    "log_distance_to_up_limit": "log(up limit price / mid)",
    "log_distance_to_down_limit": "log(mid / down limit price)",
    "position_in_limit_band": "(mid - down limit) / (up limit - down limit)",
    "trading_seconds_since_0930": "seconds since 09:30 on the trading clock (lunch removed), the same for every stock",
    # order stream
    "marketable_order_amount_share_{w}s": "share of the new-order amount over the last {w} s priced to cross the opposite best quote (or market)",
    "marketable_order_net_amount_share_{w}s": "(buy - sell) / (buy + sell) amount of marketable new orders over the last {w} s",
    "passive_order_net_amount_share_at_or_inside_best_{w}s": "(buy - sell) / (buy + sell) amount of non-marketable new orders priced at or better than their own best quote, last {w} s",
    "order_price_aggressiveness_mean_{w}s": "amount-weighted mean of (buy: price - mid, sell: mid - price) / mid over new limit orders in the last {w} s",
    "log_new_order_count_per_s_{w}s": "log(1 + new orders per second over the last {w} s)",
    "log_new_order_to_trade_count_ratio_{w}s": "log((1 + new orders) / (1 + trades)) over the last {w} s",
    "log_mean_new_order_amount_cny_{w}s": "log of the mean new-order amount (CNY) over the last {w} s",
    "log_channel_sequence_advance_per_s_{w}s": "log(1 + channel sequence numbers advanced per second over the last {w} s): the whole channel's message rate",
    # the whole rebuilt order book (full_book.py)
    "full_book_depth_imbalance": "(bid - ask) / (bid + ask) resting volume over the whole rebuilt book",
    "full_book_amount_imbalance": "(bid - ask) / (bid + ask) resting amount (price x volume) over the whole rebuilt book",
    "full_book_depth_imbalance_within_{bp}bp": "(bid - ask) / (bid + ask) resting volume within {bp} bp of the mid",
    "full_book_depth_imbalance_beyond_level10": "(bid - ask) / (bid + ask) resting volume beyond each side's 10 best price levels",
    "bid_depth_share_beyond_level10_full_book": "share of the whole bid side's volume resting beyond its 10 best price levels",
    "ask_depth_share_beyond_level10_full_book": "share of the whole ask side's volume resting beyond its 10 best price levels",
    "bid_depth_share_beyond_200bp_full_book": "share of the whole bid side's volume resting more than 200 bp below the mid",
    "ask_depth_share_beyond_200bp_full_book": "share of the whole ask side's volume resting more than 200 bp above the mid",
    "bid_depth_weighted_distance_full_book_rel": "volume-weighted distance of the whole bid side below the mid, over the mid",
    "ask_depth_weighted_distance_full_book_rel": "volume-weighted distance of the whole ask side above the mid, over the mid",
    "depth_weighted_distance_ask_minus_bid_full_book_rel": "ask minus bid volume-weighted distance from the mid, over the mid",
    "bid_price_levels_count_log": "log of the number of bid price levels holding volume",
    "ask_price_levels_count_log": "log of the number of ask price levels holding volume",
    "price_levels_count_imbalance": "(bid - ask) / (bid + ask) number of price levels holding volume",
    "resting_order_count_imbalance_full_book": "(bid - ask) / (bid + ask) number of resting orders over the whole book",
    "resting_order_count_imbalance_within_100bp": "(bid - ask) / (bid + ask) number of resting orders within 100 bp of the mid",
    "bid_mean_resting_order_size_log": "log of the mean remaining size of the resting bid orders",
    "ask_mean_resting_order_size_log": "log of the mean remaining size of the resting ask orders",
    "mean_resting_order_size_bid_over_ask_log": "log of (mean resting bid order size / mean resting ask order size)",
    "best_level_mean_order_size_bid_over_ask_log": "log of (mean order size at the best bid / mean order size at the best ask)",
    "best_level_order_count_imbalance": "(bid - ask) / (bid + ask) number of orders resting at the best prices",
    "bid_resting_volume_share_placed_last_{w}s": "share of the resting bid volume from orders that arrived in the last {w} s",
    "ask_resting_volume_share_placed_last_{w}s": "share of the resting ask volume from orders that arrived in the last {w} s",
    "resting_volume_placed_last_{w}s_imbalance": "(bid - ask) / (bid + ask) resting volume from orders that arrived in the last {w} s",
    "full_book_depth_over_10_levels_log": "log of (whole-book resting volume / the volume in each side's 10 best levels)",
    "full_book_bid_depth_log": "log of the whole bid side's resting volume",
    "full_book_ask_depth_log": "log of the whole ask side's resting volume",
}

TIERS = (("under_40k_cny", 0.0, 4e4), ("40k_to_200k_cny", 4e4, 2e5), ("200k_to_1m_cny", 2e5, 1e6), ("over_1m_cny", 1e6, np.inf))
TIER_WINDOWS = (120, 300, 600, 1800)  # seconds; plus since 09:30


def describe(name: str) -> str:
    """Plain description of one of our names (``gen.`` / ``flow.`` prefix allowed), '' when unknown."""
    base = name.removeprefix("gen.").removeprefix("flow.")
    slots = {"k": r"\d+", "w": r"\d+", "bp": r"\d+", "win": r"\d+s|since_0930", "tier": "|".join(t for t, _, _ in TIERS)}
    for pat, text in DESCRIBE.items():
        rx = re.escape(pat)
        for k, g in slots.items():
            rx = rx.replace(re.escape("{" + k + "}"), f"(?P<{k}>{g})")
        if m := re.fullmatch(rx, base):
            return text.format(**m.groupdict())
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
        out.update(_book_open_limits(gf, T))
        out.update(_size_tiers(gs, T))
    out["trading_seconds_since_0930"] = np.asarray(T, float).copy()
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in out.items()}


def _first_pos(v: np.ndarray, fn=np.nanmax) -> float:
    """fn over the positive finite values of v, NaN when there is none."""
    v = v[np.isfinite(v) & (v > 0)]
    return float(fn(v)) if v.size else np.nan


def _book_open_limits(gf: pl.DataFrame | None, T: np.ndarray) -> dict[str, np.ndarray]:
    """Deeper book (needs the ``extra`` columns), the open auction and the first half hour, the price-limit distance."""
    out: dict[str, np.ndarray] = {}
    if gf is None or gf.height <= 3:
        return out
    f = lambda c: gf[c].to_numpy().astype(float)  # noqa: E731 -- column as float array
    a = E.arrays(gf, "fast")
    tl, s, mid = a["t_local"], f("s"), a["mid"]
    k = np.isfinite(mid) & (mid > 0)
    t = tl[k]
    put = lambda name, v: out.__setitem__(name, FI.at_marks(t, v[k], T))  # noqa: E731 -- latest snapshot value at T
    lmT = FI.at_marks(t, np.log(mid[k]), T)
    if "bwp5" in gf.columns:
        bv1, av1, bv5, av5 = f("bv1"), f("av1"), f("bv5"), f("av5")
        bw, aw, b5, a5 = f("bwp5"), f("awp5"), f("b5px"), f("a5px")
        put("depth_weighted_price_5lvl_minus_mid_rel", ((bw * bv5 + aw * av5) / (bv5 + av5) - mid) / mid)
        put("bid_depth_weighted_distance_5lvl_rel", (mid - bw) / mid)
        put("ask_depth_weighted_distance_5lvl_rel", (aw - mid) / mid)
        rb, ra = np.where(b5 > 0, (mid - b5) / mid, np.nan), np.where(a5 > 0, (a5 - mid) / mid, np.nan)
        sb, sa = np.log(bv5 / np.where(rb > 0, rb, np.nan)), np.log(av5 / np.where(ra > 0, ra, np.nan))
        put("log_bid_depth_5lvl_per_rel_distance_to_level5", sb)
        put("log_ask_depth_5lvl_per_rel_distance_to_level5", sa)
        put("book_slope_log_ratio_bid_over_ask_5lvl", sb - sa)
        if "bwp10" in gf.columns:  # the same over 10 levels, and the share of depth beyond level 5
            bv10, av10 = f("bv10"), f("av10")
            bw10, aw10, b10, a10 = f("bwp10"), f("awp10"), f("b10px"), f("a10px")
            put("depth_weighted_price_10lvl_minus_mid_rel", ((bw10 * bv10 + aw10 * av10) / (bv10 + av10) - mid) / mid)
            put("bid_depth_weighted_distance_10lvl_rel", (mid - bw10) / mid)
            put("ask_depth_weighted_distance_10lvl_rel", (aw10 - mid) / mid)
            put("depth_imbalance_10lvl", (bv10 - av10) / (bv10 + av10))
            put("depth_imbalance_levels6to10", ((bv10 - bv5) - (av10 - av5)) / (bv10 - bv5 + av10 - av5))
            put("bid_depth_share_beyond_level5", (bv10 - bv5) / bv10)
            put("ask_depth_share_beyond_level5", (av10 - av5) / av10)
            r10b, r10a = np.where(b10 > 0, (mid - b10) / mid, np.nan), np.where(a10 > 0, (a10 - mid) / mid, np.nan)
            s10b = np.log(bv10 / np.where(r10b > 0, r10b, np.nan))
            s10a = np.log(av10 / np.where(r10a > 0, r10a, np.nan))
            put("book_slope_log_ratio_bid_over_ask_10lvl", s10b - s10a)
        if "bq10" in gf.columns:  # every depth: book imbalance over levels 1..k and at level k alone, k = 1..10
            bq = [bv1] + [f(f"bq{i}") for i in range(2, 11)]
            aq = [av1] + [f(f"aq{i}") for i in range(2, 11)]
            cb, ca = np.cumsum(np.nan_to_num(bq), axis=0), np.cumsum(np.nan_to_num(aq), axis=0)
            for lv in range(1, 11):
                put(f"depth_imbalance_levels1to{lv}", (cb[lv - 1] - ca[lv - 1]) / (cb[lv - 1] + ca[lv - 1]))
                if lv > 1:
                    put(f"depth_imbalance_level{lv}", (bq[lv - 1] - aq[lv - 1]) / (bq[lv - 1] + aq[lv - 1]))
        if "ofi2_5" in gf.columns:  # level-1 event here, levels 2-5 from SQL; both on every snapshot, quoted prices only
            b1, a1 = f("b1"), f("a1")
            e1 = np.zeros(tl.size)
            qb, qa = (b1[1:] > 0) & (b1[:-1] > 0), (a1[1:] > 0) & (a1[:-1] > 0)
            e1[1:] = (np.where(qb, bv1[1:] * (b1[1:] >= b1[:-1]) - bv1[:-1] * (b1[1:] <= b1[:-1]), 0.0)
                      - np.where(qa, av1[1:] * (a1[1:] <= a1[:-1]) - av1[:-1] * (a1[1:] >= a1[:-1]), 0.0))
            e25 = np.nan_to_num(f("ofi2_5"))
            cs15, cs25 = _cs(e1 + e25), _cs(e25)
            cd15, cd25, cn = _cs((bv5 + av5) / 2), _cs((bv5 - bv1 + av5 - av1) / 2), _cs(np.ones(tl.size))
            if "ofi6_10" in gf.columns:
                e610 = np.nan_to_num(f("ofi6_10"))
                bv10_, av10_ = f("bv10"), f("av10")
                cs110, cs610 = _cs(e1 + e25 + e610), _cs(e610)
                cd110, cd610 = _cs((bv10_ + av10_) / 2), _cs((bv10_ - bv5 + av10_ - av5) / 2)
            if "ofil10" in gf.columns:
                evl = {lv: np.nan_to_num(f(f"ofil{lv}")) for lv in range(2, 11)}
                ql = {1: (bv1 + av1) / 2} | {lv: (f(f"bq{lv}") + f(f"aq{lv}")) / 2 for lv in range(2, 11)}
                csl, cdl = {lv: _cs(evl[lv]) for lv in evl}, {lv: _cs(ql[lv]) for lv in evl}
                run_e, run_q, csk, cdk = e1.copy(), np.nan_to_num(ql[1]), {}, {}
                for lv in range(2, 11):
                    run_e, run_q = run_e + evl[lv], run_q + np.nan_to_num(ql[lv])
                    csk[lv], cdk[lv] = _cs(run_e), _cs(run_q)
            for w in (10, 30, 60, 300):
                n = _win(cn, tl, T, w)
                out[f"ofi_levels1to5_norm_{w}s"] = _win(cs15, tl, T, w) / (_win(cd15, tl, T, w) / n)
                out[f"ofi_levels2to5_norm_{w}s"] = _win(cs25, tl, T, w) / (_win(cd25, tl, T, w) / n)
                if "ofil10" in gf.columns:  # OFI over levels 1..k and at level k alone, each over its own mean depth
                    for lv in range(2, 11):
                        if lv not in (5, 10):
                            out[f"ofi_levels1to{lv}_norm_{w}s"] = _win(csk[lv], tl, T, w) / (_win(cdk[lv], tl, T, w) / n)
                        out[f"ofi_level{lv}_norm_{w}s"] = _win(csl[lv], tl, T, w) / (_win(cdl[lv], tl, T, w) / n)
                if "ofi6_10" in gf.columns:
                    out[f"ofi_levels1to10_norm_{w}s"] = _win(cs110, tl, T, w) / (_win(cd110, tl, T, w) / n)
                    out[f"ofi_levels6to10_norm_{w}s"] = _win(cs610, tl, T, w) / (_win(cd610, tl, T, w) / n)
    op, pc = _first_pos(f("op"), np.nanmin), _first_pos(f("pc"))
    one = np.ones(np.shape(T))
    out["overnight_gap_log"] = one * np.log(op / pc)
    out["return_open_to_1000_log"] = FI.at_marks(t, np.log(mid[k]), np.minimum(T, 1800.0)) - np.log(op)
    auc, ca = s < 34200, f("ca")
    aamt = _first_pos(ca[auc]) if auc.any() else np.nan
    out["log_open_auction_amount_cny"] = one * np.log1p(aamt)
    out["open_auction_amount_share_of_day_so_far"] = aamt / FI.at_marks(tl, ca, T)
    bv1, av1, b1 = f("bv1"), f("av1"), f("b1")
    j = np.flatnonzero((s >= 33300) & (s < 33900))  # 09:15-09:25: the auction's indicative book
    out["open_auction_imbalance_level1_last_before_0925"] = one * ((bv1[j[-1]] - av1[j[-1]]) / (bv1[j[-1]] + av1[j[-1]]) if j.size else np.nan)
    j = np.flatnonzero((s >= 33300) & (s < 33600) & (b1 > 0))  # 09:15-09:20, cancels still allowed
    out["open_auction_indicative_change_0920_to_open_log"] = one * (np.log(op / b1[j[-1]]) if j.size else np.nan)
    if "ul" in gf.columns:
        ul, ll = _first_pos(f("ul")), _first_pos(f("ll"), np.nanmin)
        out["log_distance_to_up_limit"] = np.log(ul) - lmT
        out["log_distance_to_down_limit"] = lmT - np.log(ll)
        out["position_in_limit_band"] = (np.exp(lmT) - ll) / (ul - ll)
    return out


def _size_tiers(gs: pl.DataFrame | None, T: np.ndarray) -> dict[str, np.ndarray]:
    """Aggressor amount by the size of the aggressive order (all its fills summed: an aggressive order fills on arrival)."""
    out: dict[str, np.ndarray] = {}
    if gs is None or gs.height <= 3 or "bsq" not in gs.columns:
        return out
    b = E.arrays(gs, "slow")
    cont = gs["s"].to_numpy() >= 34200
    side, amt, ts = b["side"], np.nan_to_num(b["amt"]), b["t_local"]
    bsq, ssq = gs["bsq"].fill_null(-1).to_numpy(), gs["ssq"].fill_null(-1).to_numpy()
    oid = np.where(side > 0, bsq, np.where(side < 0, ssq, -1))
    valid = cont & (side != 0) & (oid >= 0)
    if not valid.any():
        return out
    _, inv = np.unique(oid[valid].astype(np.int64) * 2 + (side[valid] > 0), return_inverse=True)
    oamt = np.full(side.size, np.nan)
    oamt[valid] = np.bincount(inv, weights=amt[valid])[inv]
    tot = _cs(np.where(cont, amt, 0.0))
    at_t = np.searchsorted(ts, T, side="right")
    wins = [(f"{w}s", w) for w in TIER_WINDOWS] + [("since_0930", None)]
    W = lambda cs, w: _win(cs, ts, T, w) if w else cs[at_t]  # noqa: E731 -- window sum, or since 09:30
    sel = lambda lo, hi: valid & (oamt >= lo) & (oamt < hi)  # noqa: E731 -- aggressor fills of one size tier
    for name, lo, hi in TIERS:
        cs = _cs(np.where(sel(lo, hi), amt * side, 0.0))
        for tag, w in wins:
            out[f"net_buy_amount_share_orders_{name}_{tag}"] = W(cs, w) / W(tot, w)
    big = sel(2e5, np.inf)
    cs_bn, cs_ba = _cs(np.where(big, amt * side, 0.0)), _cs(np.where(big, amt, 0.0))
    for tag, w in wins:
        out[f"net_buy_amount_share_orders_over_200k_cny_{tag}"] = W(cs_bn, w) / W(tot, w)
        out[f"amount_share_orders_over_200k_cny_{tag}"] = W(cs_ba, w) / W(tot, w)
    return out


def order_stream(go: pl.DataFrame | None, gf: pl.DataFrame | None, gs: pl.DataFrame | None, T: np.ndarray) -> dict[str, np.ndarray]:
    """Order-stream factors at T from ``tick_flow.load_orders`` rows (continuous session), each new order against the
    stock's latest snapshot at or before it (local arrival)."""
    out: dict[str, np.ndarray] = {}
    if go is None or go.height <= 3:
        return out
    g = lambda c: go[c].to_numpy().astype(float)  # noqa: E731 -- column as float array
    t, s, kind, side, amt, px = g("t_local"), g("s"), g("kind"), g("side"), np.nan_to_num(g("amt")), g("px")
    new = (s >= 34200) & (kind == 1) & (side != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cs_new, cs_cnt = _cs(np.where(new, amt, 0.0)), _cs(new.astype(float))
        trades = gs is not None and gs.height > 0
        if trades:
            ttr = gs["t_local"].to_numpy().astype(float)
            cs_tr = _cs((gs["s"].to_numpy() >= 34200).astype(float))
        if gf is not None and gf.height > 2:
            a = E.arrays(gf, "fast")
            tf = a["t_local"]
            b1, a1, mid = FI.at_marks(tf, a["bid1"], t), FI.at_marks(tf, a["ask1"], t), FI.at_marks(tf, a["mid"], t)
            mkt = new & ((px <= 0) | ((side > 0) & (px >= a1)) | ((side < 0) & (px <= b1)))
            pas = new & ~mkt & (((side > 0) & (px >= b1)) | ((side < 0) & (px <= a1)))
            lim = new & (px > 0) & np.isfinite(mid) & (mid > 0)
            agg = np.clip(np.where(lim, side * (px - mid) / mid, 0.0), -0.2, 0.2)
            cs_m, cs_mn = _cs(np.where(mkt, amt, 0.0)), _cs(np.where(mkt, amt * side, 0.0))
            cs_p, cs_pn = _cs(np.where(pas, amt, 0.0)), _cs(np.where(pas, amt * side, 0.0))
            cs_ag, cs_la = _cs(np.where(lim, amt * agg, 0.0)), _cs(np.where(lim, amt, 0.0))
            for w in (60, 300, 900):
                out[f"marketable_order_amount_share_{w}s"] = _win(cs_m, t, T, w) / _win(cs_new, t, T, w)
                out[f"marketable_order_net_amount_share_{w}s"] = _win(cs_mn, t, T, w) / _win(cs_m, t, T, w)
                out[f"passive_order_net_amount_share_at_or_inside_best_{w}s"] = _win(cs_pn, t, T, w) / _win(cs_p, t, T, w)
                out[f"order_price_aggressiveness_mean_{w}s"] = _win(cs_ag, t, T, w) / _win(cs_la, t, T, w)
        for w in (60, 300, 900):
            n = _win(cs_cnt, t, T, w)
            out[f"log_new_order_count_per_s_{w}s"] = np.log1p(n / w)
            out[f"log_mean_new_order_amount_cny_{w}s"] = np.log(_win(cs_new, t, T, w) / n)
            if trades:
                out[f"log_new_order_to_trade_count_ratio_{w}s"] = np.log((1 + n) / (1 + _win(cs_tr, ttr, T, w)))
        seq = g("seq")
        ok = np.isfinite(seq)
        if ok.sum() > 3:  # the channel's sequence number as seen on this stock's own messages, as a running max
            ts_, sq = t[ok], np.fmax.accumulate(seq[ok])
            for w in (60, 300, 900):
                adv = FI.at_marks(ts_, sq, T) - FI.at_marks(ts_, sq, T - w)
                out[f"log_channel_sequence_advance_per_s_{w}s"] = np.log1p(np.clip(adv, 0, None) / w)
    return {k: np.where(np.isfinite(v), v, np.nan) for k, v in out.items()}
