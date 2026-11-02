// The update rules, a port of research/engine/rules.py; the EMAS table itself arrives as rules.tsv.
#pragma once

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "state.hpp"

namespace eng {

inline double plog(double x) { return x > 0 ? std::log(x) : qnan; }

// Python max(a, b) / min(a, b): the first argument stays unless the second compares strictly better.
inline double pymax(double a, double b) { return b > a ? b : a; }
inline double pymin(double a, double b) { return b < a ? b : a; }

enum class what : int { price, mid, logp, hl, hl0, hl_valid, tr_open, logret, one, snap_px, hl_snap, rvwap, hl_trade, tro, price0, price_pc,
                        ret_open, ret_pc, ret_mid, hl_pc, hl_op, hl_px, rvwap_all };
enum class clock_mod : int { none, wall, exch, o930 };

inline double tc(double s) { return s <= 41400 ? s - 34200 : (s < 46800 ? 7200.0 : s - 46800 + 7200); }

// The time an ema sees: t (local trading clock), t without lunch removal, or the exchange stamp's clock.
inline double clock_time(clock_mod m, double t, double s) {
    if (m == clock_mod::wall) return t > 7200 ? t + 5400.0 : t;
    if (m == clock_mod::exch) return tc(s);
    if (m == clock_mod::o930) return s >= 34200 ? t : qnan;  // NaN = this event does not update
    return t;
}

inline what parse_what(const std::string& w) {
    if (w == "price") return what::price;
    if (w == "mid") return what::mid;
    if (w == "logp") return what::logp;
    if (w == "hl") return what::hl;
    if (w == "hl0") return what::hl0;
    if (w == "hl_valid") return what::hl_valid;
    if (w == "tr_open") return what::tr_open;
    if (w == "logret") return what::logret;
    if (w == "one") return what::one;
    if (w == "snap_px") return what::snap_px;
    if (w == "hl_snap") return what::hl_snap;
    if (w == "rvwap") return what::rvwap;
    if (w == "hl_trade") return what::hl_trade;
    if (w == "tro") return what::tro;
    if (w == "price0") return what::price0;
    if (w == "price_pc") return what::price_pc;
    if (w == "ret_open") return what::ret_open;
    if (w == "ret_pc") return what::ret_pc;
    if (w == "ret_mid") return what::ret_mid;
    if (w == "hl_pc") return what::hl_pc;
    if (w == "hl_op") return what::hl_op;
    if (w == "hl_px") return what::hl_px;
    if (w == "rvwap_all") return what::rvwap_all;
    throw std::runtime_error("unknown value kind " + w);
}

inline clock_mod parse_mod(const std::string& m) {
    if (m.empty()) return clock_mod::none;
    if (m == "wall") return clock_mod::wall;
    if (m == "exch") return clock_mod::exch;
    if (m == "o930") return clock_mod::o930;
    throw std::runtime_error("unknown clock modifier " + m);
}

// Value kinds shared by both clocks: ratios to the open / previous close / mid, normalised ranges, the all-day VWAP.
inline double extra_value(const stock& st, what w) {
    switch (w) {
    case what::ret_open: return st.op > 0 ? st.snap_px / st.op : qnan;
    case what::ret_pc: return st.pc > 0 ? st.snap_px / st.pc : qnan;
    case what::ret_mid: return st.mid > 0 ? st.snap_px / st.mid : qnan;
    case what::hl_pc: return st.pc > 0 ? (st.hi_f - st.lo_f) / st.pc : qnan;
    case what::hl_op: return st.op > 0 ? (st.hi_f - st.lo_f) / st.op : qnan;
    case what::hl_px: return st.snap_px > 0 ? (st.hi_f - st.lo_f) / st.snap_px : qnan;
    case what::rvwap_all: return st.vol_all > 0 ? st.amt_all / st.vol_all : qnan;
    default: throw std::runtime_error("unknown value kind");
    }
}

inline double snap_value(const stock& st, what w) {
    switch (w) {
    case what::price: return st.snap_px;
    case what::mid: return st.mid;
    case what::logp: return plog(st.snap_px);
    case what::hl: return st.hi_f - st.lo_f;
    case what::hl0: { double v = st.hi_f - st.lo_f; return std::isnan(v) ? 0.0 : v; }
    case what::hl_valid: return std::isnan(st.hi_f - st.lo_f) ? 0.0 : 1.0;
    case what::tr_open: return pymax(st.hi_f, st.op) - pymin(st.lo_f, st.op);
    case what::logret: return (!std::isnan(st.snap_prev) && st.snap_prev > 0) ? plog(st.snap_px / st.snap_prev) : qnan;
    case what::one: return 1.0;
    case what::hl_trade: return st.high - st.low;
    case what::tro: return pymax(st.high, st.op) - pymin(st.low, st.op);
    case what::snap_px: return st.snap_px;
    case what::price0: return std::isnan(st.snap_px) ? 0.0 : st.snap_px;
    case what::price_pc: return std::isnan(st.snap_px) ? st.pc : st.snap_px;
    case what::rvwap: return st.vol > 0 ? st.amt / st.vol : qnan;
    default: return extra_value(st, w);
    }
}

inline double trade_value(const stock& st, what w) {
    switch (w) {
    case what::price: return st.trade_px;
    case what::hl: case what::hl_trade: return st.high - st.low;
    case what::tro: return pymax(st.high, st.op) - pymin(st.low, st.op);
    case what::mid: return st.mid;
    case what::snap_px: return st.snap_px;
    case what::hl_snap: return st.hi_f - st.lo_f;
    case what::logp: return plog(st.trade_px);
    case what::logret: return (!std::isnan(st.trade_prev) && st.trade_prev > 0) ? plog(st.trade_px / st.trade_prev) : qnan;
    case what::rvwap: return st.vol > 0 ? st.amt / st.vol : qnan;
    default: return extra_value(st, w);
    }
}

// Rules resolved once: clock and value kind per ema, table order.
struct compiled_rules {
    std::vector<int> clock;
    std::vector<clock_mod> mod;
    std::vector<what> kind;
    explicit compiled_rules(const std::vector<rule>& r) {
        for (const auto& x : r) {
            clock.push_back(x.clock);
            mod.push_back(parse_mod(x.mod));
            kind.push_back(parse_what(x.what));
        }
    }
};

inline void on_snap(stock& st, const compiled_rules& cr, double t, double s, double lsod, double price, double b1, double a1,
                    double hi, double lo, double op, double pc, double amount) {
    st.snap_prev = st.snap_px;
    st.snap_px = price > 0 ? price : qnan;
    st.bid = b1 > 0 ? b1 : qnan;
    st.ask = a1 > 0 ? a1 : qnan;
    st.mid = (b1 > 0 && a1 > 0) ? (b1 + a1) / 2 : st.snap_px;
    st.hi_f = hi > 0 ? hi : qnan;
    st.lo_f = lo > 0 ? lo : qnan;
    if (op > 0) st.op = op;
    if (pc > 0) st.pc = pc;
    if (34200 <= s && s < 53820 && price > 0 && !(st.hm >= price)) st.hm = price;
    for (std::size_t i = 0; i < st.emas.size(); ++i)
        if (cr.clock[i] != 1) {
            double tt = clock_time(cr.mod[i], t, s);
            if (!std::isnan(tt)) st.emas[i].update(tt, snap_value(st, cr.kind[i]));
        }
    st.zzfb.feed(t, amount);
    st.jb_px.feed(t, st.snap_px);
    for (int k = 0; k < n_hj; ++k)
        if (t <= hj_sample(k)) st.hj[k] = st.mid;
    if (price > 0) {
        for (int k = 0; k < n_xo; ++k)
            if (lsod <= 53820 - (n_xo - 1 - k) * 10) st.xo[k] = price;
        for (int k = 0; k < n_ui; ++k)
            if (lsod <= 53820 - (n_ui - 1 - k) * 10) st.ui[k] = price;
    }
}

// An order event: the all-event averages (clock 4) move on the state as it stands; the order-flow imbalances count it.
inline void on_order(stock& st, const compiled_rules& cr, double t, double s, double price, double vol, double side) {
    if (34200 <= s && s < 53820 && price > 0) {
        double amt = vol * price, b = side > 0 ? 1.0 : 0.0, sl = side < 0 ? 1.0 : 0.0;
        st.nb_ord[0] += amt * b;
        st.nb_ord[1] += amt * sl;
        st.nb_ordvol[0] += vol * b;
        st.nb_ordvol[1] += vol * sl;
    }
    for (std::size_t i = 0; i < st.emas.size(); ++i)
        if (cr.clock[i] == 4) {
            double tt = clock_time(cr.mod[i], t, s);
            if (!std::isnan(tt)) st.emas[i].update(tt, snap_value(st, cr.kind[i]));
        }
}

// A cancel record of the transaction stream: the bothc (5), tradec (6) and all (4) averages move, on the state as it stands.
inline void on_cancel(stock& st, const compiled_rules& cr, double t, double s) {
    for (std::size_t i = 0; i < st.emas.size(); ++i)
        if (cr.clock[i] == 4 || cr.clock[i] == 5 || cr.clock[i] == 6) {
            double tt = clock_time(cr.mod[i], t, s);
            if (!std::isnan(tt)) st.emas[i].update(tt, cr.clock[i] == 6 ? trade_value(st, cr.kind[i]) : snap_value(st, cr.kind[i]));
        }
}

inline void on_trade(stock& st, const compiled_rules& cr, double t, double s, double price, double vol, double side) {
    st.trade_prev = st.trade_px;
    st.trade_px = price > 0 ? price : qnan;
    if (price > 0) {
        if (!(st.high >= price)) st.high = price;
        if (!(st.low <= price)) st.low = price;
        if (std::isnan(st.open)) st.open = price;
        st.last = price;
    }
    double amt = vol * price;
    double b = side > 0 ? 1.0 : 0.0, sl = side < 0 ? 1.0 : 0.0;
    if (34200 <= s && s < 53820) {  // zzgh.nb variants: trade count, amount at the last snapshot price
        st.nb_cnt[0] += b;
        st.nb_cnt[1] += sl;
        double px = std::isnan(st.snap_px) ? price : st.snap_px;
        st.nb_snappx[0] += vol * px * b;
        st.nb_snappx[1] += vol * px * sl;
    }
    st.vol_all += vol;  // zzqj.v / tpv variants: the whole day, the day without the close auction, the local-clock session
    st.amt_all += amt;
    if (34200 <= s && s < 54000) {
        st.vol_o += vol;
        st.amt_o += amt;
    }
    if (0 <= t && t < session) {
        st.vol_loc += vol;
        st.amt_loc += amt;
    }
    if (s >= 34200) {
        st.vol += vol;
        st.amt += amt;
        if (s < 53820) {
            if (side > 0) st.buy += amt;
            else if (side < 0) st.sell += amt;
            st.nb_vol[0] += vol * b;
            st.nb_vol[1] += vol * sl;
            st.gh_feed(t, side > 0 ? amt : 0.0, side < 0 ? amt : 0.0);
        }
    }
    for (std::size_t i = 0; i < st.emas.size(); ++i)
        if (cr.clock[i] != 0) {
            double tt = clock_time(cr.mod[i], t, s);
            if (!std::isnan(tt))
                st.emas[i].update(tt, (cr.clock[i] == 1 || cr.clock[i] == 3 || cr.clock[i] == 6) ? trade_value(st, cr.kind[i]) : snap_value(st, cr.kind[i]));
        }
}

}  // namespace eng
