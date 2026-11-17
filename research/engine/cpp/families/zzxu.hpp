// zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).
// SUBJECT TO CHANGE: the band is a first guess (multiplier 3 around (high + low) / 2, width = the range average), see zzxu.py.
#pragma once

#include "base.hpp"

namespace eng {

struct zzxu : family {
    static constexpr const char* stem = "zzxu";
    static constexpr status st = status::pinned;
    static constexpr double band_mult = 3.0;
    double upper = qnan, lower = qnan, dir = 1.0;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzxu.p", "snap", 300, "price"));
        o.push_back(family_rule("zzxu.a", "snap", 300, "hl"));
    }
    void on_snap(const tape& t, double, double, double, double) {
        if (t.ix_zzxu_a < 0) return;
        const ema& e = t.emas[t.ix_zzxu_a];
        double w = e.w != 0 ? e.s / e.w : qnan, px = t.snap_px;
        if (std::isnan(w) || std::isnan(px) || std::isnan(t.hi_f) || std::isnan(t.lo_f)) return;
        double mid = (t.hi_f + t.lo_f) / 2, bu = mid + band_mult * w, bl = mid - band_mult * w;
        if (std::isnan(upper)) {
            upper = bu;
            lower = bl;
        } else if (dir > 0) {  // up trend: the lower band only rises, the upper is recomputed
            upper = bu;
            if (bl > lower) lower = bl;
            if (px < lower) {
                dir = -1.0;
                lower = bl;
            }
        } else {  // down trend: the mirror
            lower = bl;
            if (bu < upper) upper = bu;
            if (px > upper) {
                dir = 1.0;
                upper = bu;
            }
        }
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzxu.high", t.high);
        put(o, "zzxu.low", t.low);
        put(o, "zzxu.last", t.last);
        put(o, "zzxu.upper|st", upper);
        put(o, "zzxu.lower|st", lower);
        put(o, "zzxu.dir|st", dir);
    }
    static double col_a(double a, double low) { return a / low; }                            // stand-in, wrong scale
    static double col_b(double px, double a, double last) { return (px - a) / last; }        // stand-in, wrong scale
};

}  // namespace eng
