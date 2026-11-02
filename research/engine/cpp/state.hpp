// Per-stock state of PM's intraday engine, a line-for-line port of research/engine/state.py.
#pragma once

#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace eng {

constexpr double session = 14220.0;  // 09:30-11:30 + 13:00-14:57 on the trading clock
constexpr double jb_drop = 0.0304;   // zzjb: 0 when the slot's price is this far below the previous close
constexpr double gh_edges[4] = {-0.6, -0.2, 0.2, 0.6};
constexpr int gh_empty = 2;
constexpr int n_hj = 16;
constexpr int n_xo = 30, n_ui = 10;  // 10 s price grids ending at 14:57:00 local
constexpr int n_gh = 948;
inline const double qnan = std::nan("");

// zzhj sample k: 30 s before each quarter-hour end, the last at 14:56:30.
inline double hj_sample(int k) { double v = k * 900.0 + 870.0; return v < 14190.0 ? v : 14190.0; }

// One decayed average: sum, weight, time of the last update (has_t false = never updated).
struct ema {
    double k = 0, s = 0, w = 0, t = 0;
    bool has_t = false;
    void update(double now, double x) {
        if (std::isnan(x)) return;
        double d = has_t ? std::exp(-(now - t) * k) : 1.0;
        s = s * d + x;
        w = w * d + 1.0;
        t = now;
        has_t = true;
    }
};

// Grid of n slots: slot i = the last value fed strictly before its end (i + 1) * step.
struct last_before {
    double step, cur = std::nan("");
    std::vector<double> vals;
    std::size_t nxt = 0;
    last_before(int n, double fill) : step(session / n), vals(n, fill) {}
    void feed(double t, double v) {
        while (nxt < vals.size() && (nxt + 1) * step <= t) {
            if (!std::isnan(cur)) vals[nxt] = cur;
            ++nxt;
        }
        cur = v;
    }
    const std::vector<double>& close() {
        while (nxt < vals.size()) {
            if (!std::isnan(cur)) vals[nxt] = cur;
            ++nxt;
        }
        return vals;
    }
};

// One line of rules.tsv: clock 0 snap / 1 trade / 2 both / 3 bothn / 4 all / 5 bothc / 6 tradec, clock modifier, half-life s, value kind,
// seed ("-" none, "c<N>" sum + wgt, "w<N>" wgt only, "wt<N>" wgt only with t = 0; N = decay seconds).
struct rule {
    std::string name;
    int clock;
    std::string mod;
    double half;
    std::string what;
    std::string seed;
};

struct stock {
    std::vector<ema> emas;  // one per rule, table order
    double high = qnan, low = qnan, open = qnan, last = qnan;
    double vol = 0, amt = 0, buy = 0, sell = 0;
    double nb_vol[2] = {0, 0}, nb_cnt[2] = {0, 0}, nb_snappx[2] = {0, 0}, nb_ord[2] = {0, 0}, nb_ordvol[2] = {0, 0};  // zzgh.nb variants
    double vol_all = 0, amt_all = 0, vol_o = 0, amt_o = 0, vol_loc = 0, amt_loc = 0;  // zzqj.v / tpv variants
    double hm = qnan, pc = qnan, op = qnan;
    double snap_px = qnan, snap_prev = qnan, trade_px = qnan, trade_prev = qnan;
    double bid = qnan, ask = qnan, mid = qnan, hi_f = qnan, lo_f = qnan;
    last_before zzfb{711, 0.0}, jb_px{948, qnan};
    std::vector<double> gh_buy, gh_sell;  // SlotSum(948, 2)
    double hj[n_hj];
    double xo[n_xo], ui[n_ui];

    explicit stock(const std::vector<rule>& rules) : emas(rules.size()), gh_buy(n_gh, 0.0), gh_sell(n_gh, 0.0) {
        for (std::size_t i = 0; i < rules.size(); ++i) emas[i].k = std::log(2.0) / rules[i].half;
        for (double& x : hj) x = qnan;
        for (double& x : xo) x = qnan;
        for (double& x : ui) x = qnan;
    }
    // Seed rule i from yesterday's sum / wgt per its seed tag (Python Stock.__init__).
    void seed(std::size_t i, const std::string& tag, double sum, double wgt) {
        if (wgt == 0 || tag == "-") return;
        std::size_t k = tag.find_first_of("0123456789");
        std::string kind = tag.substr(0, k);
        double n = std::stod(tag.substr(k));
        double d = std::exp(-n * emas[i].k);
        emas[i].s = kind == "c" ? sum * d : 0.0;
        emas[i].w = wgt * d;
        if (kind == "wt") {
            emas[i].t = 0.0;
            emas[i].has_t = true;
        }
    }
    void gh_feed(double t, double b, double s) {
        if (0 <= t && t < session) {
            std::size_t k = static_cast<std::size_t>(std::floor(t / (session / n_gh)));
            gh_buy[k] += b;
            gh_sell[k] += s;
        }
    }
};

}  // namespace eng
