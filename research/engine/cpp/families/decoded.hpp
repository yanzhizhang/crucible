// decoded: the averages read from the runtime's own update slots (zzhj.p, zztx.r, zzeb.r, zzui.k / zzui.r).
// Their constants (half-lives, slot steps and counts, the adaptive average's window) live in a private key<TAB>value file
// named by $CRUCIBLE_RUNTIME_CONSTS, kept off git; without it the family is inert and dumps nothing.
#pragma once

#include <cmath>
#include <cstdlib>
#include <deque>
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "base.hpp"

namespace eng {

// The private constants, read once; empty when the env var is unset or the file is missing.
inline const std::map<std::string, double>& runtime_consts() {
    static const std::map<std::string, double> m = [] {
        std::map<std::string, double> out;
        const char* p = std::getenv("CRUCIBLE_RUNTIME_CONSTS");
        if (!p) return out;
        std::ifstream f(p);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::string k;
            double v;
            if (ss >> k >> v) out[k] = v;
        }
        return out;
    }();
    return m;
}

inline double rc(const char* key) {
    const auto& m = runtime_consts();
    auto it = m.find(key);
    return it == m.end() ? std::nan("") : it->second;
}

// The runtime's half-life average: time runs through max(), a gap over 20 half-lives restarts it, tiny sums snap to 0.
struct pm_ema {
    double half = std::nan(""), s = 0, w = 0, t = 0;
    bool has_t = false;
    void update(double now, double x) {
        if (std::isnan(x) || std::isnan(now)) return;
        double nl = has_t && t > now ? t : now;
        double g = has_t ? (nl - t) / half : 0.0;
        double d = has_t ? (g > 20.0 ? 0.0 : std::exp(-g * 0.6931471805599453)) : 0.0;
        w = w * d + 1.0;
        s = s * d + x;
        if (std::fabs(s) < 1e-10) s = 0.0;
        t = nl;
        has_t = true;
    }
    void dump(std::string& o, const std::string& key) const {
        if (!has_t) return;
        put(o, key + ".sum", s);
        put(o, key + ".wgt", w);
        put(o, key + ".t", t);
    }
};

// First snapshot of each new slot of a fixed-count grid (the runtime's slot samplers: no update past the last slot).
struct slot_first {
    double step = std::nan(""), count = 0;
    long last = -1;
    bool take(double t) {
        if (!(t > 0) || std::isnan(step)) return false;
        long k = static_cast<long>(std::floor(t / step));
        if (k <= last || k >= count) return false;
        last = k;
        return true;
    }
};

struct decoded : family {
    static constexpr const char* stem = "decoded";
    static constexpr status st = status::guessed;
    bool on = !runtime_consts().empty();
    pm_ema hj, tx, eb1, eb2, ui;
    slot_first hj_slot, ui_slot;
    double hj_prev = 0, hj_last = std::nan(""), amt_now = std::nan(""), kama = std::nan("");
    double hj_cy[16] = {};      // yesterday's zzhj.c.0..15 (CRUCIBLE_HJ_PREV), read before today's overwrite
    double hj_acc1 = 0, hj_acc2 = 0;  // [+0x100]: sum of x * log(mid / c_yesterday[idx]); 1 = idx on the 30 s slot, 2 = quarter-hour index
    std::deque<double> ui_win;
    double buy_q = 0, buy_a = 0, sell_q = 0, sell_a = 0;
    decoded() {
        hj.half = rc("zzhj.p.half");
        hj_slot.step = rc("zzhj.p.step");
        hj_slot.count = rc("zzhj.p.count");
        tx.half = rc("zztx.r.half");
        eb1.half = rc("zzeb.p.half");
        eb2.half = rc("zzeb.r.half");
        ui.half = rc("zzui.r.half");
        ui_slot.step = rc("zzui.step");
        ui_slot.count = rc("zzui.count");
    }
    double am_lo = rc("window.am_open"), am_hi = rc("window.am_close"), pm_lo = rc("window.pm_open"), pm_hi = rc("window.pm_close");
    double t_max = rc("window.t_max");  // the same close on the trading clock, for the trade hook (it has no local time of day)
    bool in_window(double lsod) const { return (am_lo <= lsod && lsod <= am_hi) || (pm_lo <= lsod && lsod <= pm_hi); }
    // The runtime's price: the mid when both sides quote (and do not cross), else the side that quotes, else the snapshot price.
    static double px_of(const tape& t) {
        bool b = !std::isnan(t.bid), a = !std::isnan(t.ask);
        if (b && a && t.ask >= t.bid) return (t.bid + t.ask) / 2;
        if (a) return t.ask;
        if (b) return t.bid;
        return t.snap_px;
    }
    void on_snap(const tape& tp, double t, double, double lsod, double amount) {
        if (!on || !in_window(lsod) || !(t > 0)) return;
        double px = px_of(tp);
        amt_now = amount;
        if (hj_slot.take(t)) {  // zzhj.p: amount since the previous update, at the first snapshot of each slot
            hj_last = amount - hj_prev;
            if (!std::isnan(px) && px > 0 && hj_last > 0) {
                long i1 = static_cast<long>(std::floor(t / hj_slot.step));
                long i2 = static_cast<long>(std::floor((t + 30.0) / 900.0));
                if (i1 >= 0 && i1 < 16 && hj_cy[i1] > 1e-10) hj_acc1 += hj_last * std::log(px / hj_cy[i1]);
                if (i2 >= 0 && i2 < 16 && hj_cy[i2] > 1e-10) hj_acc2 += hj_last * std::log(px / hj_cy[i2]);
            }
            hj.update(t, hj_last);
            hj_prev = amount;
        }
        if (!std::isnan(px)) {  // zzeb.r: -p log2 p, p = the price over the 100 s price average's sum just updated
            eb1.update(t, px);
            double p = eb1.s != 0 ? px / eb1.s : std::nan("");
            if (p > 0) eb2.update(t, -p * std::log2(p));
        }
        if (!std::isnan(px) && ui_slot.take(t)) {  // zzui: Kaufman adaptive average on the slot samples, then its average
            std::size_t n = static_cast<std::size_t>(rc("zzui.window"));
            double f = 2.0 / (rc("zzui.fast") + 1), sl = 2.0 / (rc("zzui.slow") + 1);
            ui_win.push_back(px);
            if (ui_win.size() > n + 1) ui_win.pop_front();
            if (std::isnan(kama) || ui_win.size() <= n) {
                if (std::isnan(kama)) kama = px;
            } else {
                double path = 0;
                for (std::size_t i = 1; i < ui_win.size(); ++i) path += std::fabs(ui_win[i] - ui_win[i - 1]);
                double er = path > 0 ? std::fabs(px - ui_win.front()) / path : 0.0;
                double sc = er * (f - sl) + sl;
                kama += sc * sc * (px - kama);
            }
            ui.update(t, kama);
        }
    }
    void on_trade(const tape& tp, double t, double, double amt, double side) {
        if (!on || !(t > 0) || !(t <= t_max) || !(tp.trade_px > 0)) return;  // zztx.r: the cumulative sell VWAP minus the buy VWAP
        double q = amt / tp.trade_px;
        if (side > 0) {
            buy_q += q;
            buy_a += amt;
        } else if (side < 0) {
            sell_q += q;
            sell_a += amt;
        }
        if (buy_q > 0 && sell_q > 0) tx.update(t, sell_a / sell_q - buy_a / buy_q);
    }
    // ZZHJ term output (slot 7 @ e5150): [+0x100] / (s / w) of the zzhj.p average = its last input over its mean (0 before 09:30).
    double hj_out_last() const { return (hj.has_t && hj.w != 0 && hj.s != 0) ? hj_last / (hj.s / hj.w) : std::nan(""); }
    double hj_out1() const { return (hj.has_t && hj.w != 0 && hj.s != 0) ? hj_acc1 / (hj.s / hj.w) : std::nan(""); }
    double hj_out2() const { return (hj.has_t && hj.w != 0 && hj.s != 0) ? hj_acc2 / (hj.s / hj.w) : std::nan(""); }
    double hj_out_now() const { return (hj.has_t && hj.w != 0 && hj.s != 0) ? (amt_now - hj_prev) / (hj.s / hj.w) : std::nan(""); }
    void dump(std::string& o, const tape&) {
        if (!on) return;
        hj.dump(o, "zzhj.p");
        tx.dump(o, "zztx.r");
        eb2.dump(o, "zzeb.r");
        ui.dump(o, "zzui.r");
        if (!std::isnan(kama)) put(o, "zzui.k", kama);
    }
};

}  // namespace eng
