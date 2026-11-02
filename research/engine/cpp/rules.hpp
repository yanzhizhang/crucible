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

enum class what : int { price, mid, logp, hl, hl0, hl_valid, tr_open, logret, one, snap_px, hl_snap, rvwap };

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
    throw std::runtime_error("unknown value kind " + w);
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
    default: throw std::runtime_error("value kind not on the snapshot clock");
    }
}

inline double trade_value(const stock& st, what w) {
    switch (w) {
    case what::price: return st.trade_px;
    case what::hl: return st.high - st.low;
    case what::mid: return st.mid;
    case what::snap_px: return st.snap_px;
    case what::hl_snap: return st.hi_f - st.lo_f;
    case what::logp: return plog(st.trade_px);
    case what::logret: return (!std::isnan(st.trade_prev) && st.trade_prev > 0) ? plog(st.trade_px / st.trade_prev) : qnan;
    case what::rvwap: return st.vol > 0 ? st.amt / st.vol : qnan;
    default: throw std::runtime_error("value kind not on the trade clock");
    }
}

// Rules resolved once: clock and value kind per ema, table order.
struct compiled_rules {
    std::vector<int> clock;
    std::vector<what> kind;
    explicit compiled_rules(const std::vector<rule>& r) {
        for (const auto& x : r) {
            clock.push_back(x.clock);
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
        if (cr.clock[i] == 0) st.emas[i].update(t, snap_value(st, cr.kind[i]));
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
    if (s >= 34200) {
        st.vol += vol;
        st.amt += amt;
        if (s < 53820) {
            if (side > 0) st.buy += amt;
            else if (side < 0) st.sell += amt;
            st.gh_feed(t, side > 0 ? amt : 0.0, side < 0 ? amt : 0.0);
        }
    }
    for (std::size_t i = 0; i < st.emas.size(); ++i)
        if (cr.clock[i] == 1) st.emas[i].update(t, trade_value(st, cr.kind[i]));
}

}  // namespace eng
