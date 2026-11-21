// Per-stock tape of PM's intraday engine (raw running values + the averages), a port of research/engine/state.py.
// The per-family state and keys live in families/*.hpp; stock = tape + one object per family (families/all.hpp).
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>

namespace eng {

constexpr double session = 14220.0;  // 09:30-11:30 + 13:00-14:57 on the trading clock
constexpr double jb_drop = 0.0304;   // zzjb: 0 when the slot's price is this far below the previous close
constexpr double gh_edges[4] = {-0.6, -0.2, 0.2, 0.6};
constexpr int gh_empty = 2;
constexpr int n_hj = 16;
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

// "snap" / "trade" / "both" ... [+ "_wall" / "_exch" / "_o930"] -> clock code and modifier.
inline rule make_rule(const std::string& name, const std::string& clock, double half, const std::string& what, const std::string& seed) {
    rule r;
    r.name = name;
    r.half = half;
    r.what = what;
    r.seed = seed;
    std::string base = clock.substr(0, clock.find('_'));
    r.mod = clock.find('_') == std::string::npos ? "" : clock.substr(clock.find('_') + 1);
    r.clock = base == "trade" ? 1 : base == "both" ? 2 : base == "bothn" ? 3 : base == "all" ? 4 : base == "bothc" ? 5 : base == "tradec" ? 6 : 0;
    return r;
}

// A family's built-in rule: the seed is the name's "|c<N>" / "|w<N>" / "|wt<N>" tag, "-" without one (export.seed_tag).
inline rule family_rule(const std::string& name, const std::string& clock, double half, const std::string& what) {
    std::string seed = "-";
    std::size_t bar = name.find('|');
    if (bar != std::string::npos) {
        std::string tag = name.substr(bar + 1);
        std::size_t k = tag.find_first_of("0123456789");
        std::string kind = k == std::string::npos ? "" : tag.substr(0, k);
        if ((kind == "c" || kind == "w" || kind == "wt") && tag.find_first_not_of("0123456789", k) == std::string::npos) seed = tag;
    }
    return make_rule(name, clock, half, what, seed);
}

// %.17g round-trips every double exactly; NaN becomes JSON null.
inline void put(std::string& o, const char* key, double v) {
    o += '"';
    o += key;
    o += "\":";
    if (std::isnan(v)) o += "null";
    else {
        char b[32];
        std::snprintf(b, sizeof b, "%.17g", v);
        o += b;
    }
    o += ',';
}
inline void put(std::string& o, const std::string& key, double v) { put(o, key.c_str(), v); }

struct tape {
    std::vector<ema> emas;  // one per rule, table order
    double high = qnan, low = qnan, open = qnan, last = qnan;
    double vol = 0, amt = 0, buy = 0, sell = 0;
    double nb_vol[2] = {0, 0}, nb_cnt[2] = {0, 0}, nb_snappx[2] = {0, 0}, nb_ord[2] = {0, 0}, nb_ordvol[2] = {0, 0};  // zzgh.nb variants
    double vol_all = 0, amt_all = 0, vol_o = 0, amt_o = 0, vol_loc = 0, amt_loc = 0;  // zzqj.v / tpv variants
    double hm = qnan, pc = qnan, op = qnan;
    double hm_c = qnan;  // zzmh.hm variant: snapshot high 09:30:00 .. 15:00:00 inclusive
    double lm = qnan, sp_hi = qnan, sp_lo = qnan, c_high = qnan, c_low = qnan;  // narrower ranges: snapshot prices (session / day), session trades
    double ch_h = qnan, ch_l = qnan;  // zzeg.c: the previous 3 trading days' snapshot high / low (chan.bin)
    double snap_px = qnan, snap_prev = qnan, trade_px = qnan, trade_prev = qnan;
    double bid = qnan, ask = qnan, mid = qnan, hi_f = qnan, lo_f = qnan;

    int ix_zzxu_a = -1, ix_zzxu_ac = -1, ix_zzxu_ac1 = -1;  // "zzxu.a" and its carried twins "zzxu.a|c0" / "|c1" (the zzxu band's width)
    double band_seed[3] = {qnan, qnan, qnan};  // yesterday's zzxu.upper / lower / dir (band.bin), for the carried-band variants
    int ix_vx_tr = -1, ix_vx_dema = -1, ix_zzxu_p = -1;  // "zzvx.trEmaD", "zzvx.trDEmaD", "zzxu.p": averages read as inputs elsewhere

    explicit tape(const std::vector<rule>& rules) : emas(rules.size()) {
        for (std::size_t i = 0; i < rules.size(); ++i) {
            emas[i].k = std::log(2.0) / rules[i].half;
            if (rules[i].name == "zzxu.a") ix_zzxu_a = static_cast<int>(i);
            if (rules[i].name == "zzxu.a|c0") ix_zzxu_ac = static_cast<int>(i);
            if (rules[i].name == "zzxu.a|c1") ix_zzxu_ac1 = static_cast<int>(i);
            if (rules[i].name == "zzvx.trEmaD") ix_vx_tr = static_cast<int>(i);
            if (rules[i].name == "zzvx.trDEmaD") ix_vx_dema = static_cast<int>(i);
            if (rules[i].name == "zzxu.p") ix_zzxu_p = static_cast<int>(i);
        }
    }
    double ema_value(int i) const { return (i >= 0 && emas[i].w != 0) ? emas[i].s / emas[i].w : qnan; }
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
};

}  // namespace eng
