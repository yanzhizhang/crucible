// zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).
// SUBJECT TO CHANGE: several band rules run side by side (see zzxu.py) and are scored against PM's zzxu.upper / lower / dir.
#pragma once

#include "base.hpp"

namespace eng {

// One band rule's state. rule 0 simple (the inactive band is recomputed, the active one only tightens), 1 classic (each
// band tightens unless the previous price was beyond it), 2 classic with the current price.
struct band {
    int rule;
    double dir;
    bool use_mid;
    const char* tag;
    double upper = qnan, lower = qnan, prev = qnan;
    void step(double bu, double bl, double px) {
        if (std::isnan(upper)) {
            upper = bu;
            lower = bl;
            prev = px;
            return;
        }
        if (rule == 0) {
            if (dir > 0) {
                upper = bu;
                if (bl > lower) lower = bl;
                if (px < lower) {
                    dir = -1.0;
                    lower = bl;
                }
            } else {
                lower = bl;
                if (bu < upper) upper = bu;
                if (px > upper) {
                    dir = 1.0;
                    upper = bu;
                }
            }
        } else {
            double ref = rule == 2 ? px : prev, pu = upper, pl = lower;
            if (bu < pu || ref > pu) upper = bu;
            if (bl > pl || ref < pl) lower = bl;
            if (dir > 0) {
                if (px < lower) dir = -1.0;
            } else if (px > upper) {
                dir = 1.0;
            }
        }
        prev = px;
    }
};

struct zzxu : family {
    static constexpr const char* stem = "zzxu";
    static constexpr status st = status::pinned;
    static constexpr double band_mult = 3.0;
    band bands[6] = {{0, 1.0, false, "st1"}, {1, 1.0, false, "st2"}, {2, 1.0, false, "st3"}, {1, -1.0, false, "st4"}, {1, 1.0, true, "st5"}, {0, 1.0, true, "st6"}};
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzxu.p", "snap", 300, "price"));
        o.push_back(family_rule("zzxu.a", "snap", 300, "hl"));
    }
    void on_snap(const tape& t, double, double, double, double) {
        if (t.ix_zzxu_a < 0) return;
        const ema& e = t.emas[t.ix_zzxu_a];
        double w = e.w != 0 ? e.s / e.w : qnan;
        if (std::isnan(w) || std::isnan(t.snap_px) || std::isnan(t.hi_f) || std::isnan(t.lo_f)) return;
        double mid = (t.hi_f + t.lo_f) / 2, bu = mid + band_mult * w, bl = mid - band_mult * w;
        for (band& b : bands) b.step(bu, bl, b.use_mid ? t.mid : t.snap_px);
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzxu.high", t.high);
        put(o, "zzxu.low", t.low);
        put(o, "zzxu.last", t.last);
        for (const band& b : bands) {
            put(o, std::string("zzxu.upper|") + b.tag, b.upper);
            put(o, std::string("zzxu.lower|") + b.tag, b.lower);
            put(o, std::string("zzxu.dir|") + b.tag, b.dir);
        }
    }
    static double col_a(double a, double low) { return a / low; }                      // stand-in, wrong scale
    static double col_b(double px, double a, double last) { return (px - a) / last; }  // stand-in, wrong scale
};

}  // namespace eng
