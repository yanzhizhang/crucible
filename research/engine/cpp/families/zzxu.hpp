// zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).
// SUBJECT TO CHANGE: the simple band rule from a fresh band, six choices for the open (see zzxu.py): width c0 (carried, fed by
// every snapshot) / c1 (carried, fed from 09:30) x start a (first valid snapshot) / p (09:25) / c (09:30).
#pragma once

#include "base.hpp"

namespace eng {

struct band {
    const char* tag;
    bool width_c1;
    double start;
    double upper = qnan, lower = qnan, dir = 1.0;
    void step(double bu, double bl, double px) {
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
};

struct zzxu : family {
    static constexpr const char* stem = "zzxu";
    static constexpr status st = status::pinned;
    static constexpr double band_mult = 3.0;
    band bands[6] = {{"c0a", false, 0.0}, {"c0p", false, 33900.0}, {"c0c", false, 34200.0}, {"c1a", true, 0.0}, {"c1p", true, 33900.0}, {"c1c", true, 34200.0}};
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzxu.p", "snap", 300, "price"));
        o.push_back(family_rule("zzxu.a", "snap", 300, "hl"));
        o.push_back(family_rule("zzxu.a|c0", "snap", 300, "hl"));
        o.push_back(family_rule("zzxu.a|c1", "snap_o930", 300, "hl"));
    }
    void on_snap(const tape& t, double, double s, double, double) {
        double px = t.snap_px;
        if (std::isnan(px) || std::isnan(t.hi_f) || std::isnan(t.lo_f)) return;
        double mid = (t.hi_f + t.lo_f) / 2;
        for (band& b : bands) {
            if (s < b.start) continue;
            double w = t.ema_value(b.width_c1 ? t.ix_zzxu_ac1 : t.ix_zzxu_ac);
            if (!std::isnan(w)) b.step(mid + band_mult * w, mid - band_mult * w, px);
        }
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
