// zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).
// SUBJECT TO CHANGE: the simple band rule with four choices of where its state comes from (see zzxu.py): b1 fresh width and
// band, b2 width carried from yesterday, b3 width and band carried, b4 band carried.
#pragma once

#include "base.hpp"

namespace eng {

struct band {
    const char* tag;
    bool carried_width, carried_band;
    double upper = qnan, lower = qnan, dir = 1.0;
    void seed(const double* s) {
        if (carried_band && !std::isnan(s[0]) && !std::isnan(s[1]) && !std::isnan(s[2])) {
            upper = s[0];
            lower = s[1];
            dir = s[2];
        }
    }
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
    band bands[4] = {{"b1", false, false}, {"b2", true, false}, {"b3", true, true}, {"b4", false, true}};
    bool seeded = false;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzxu.p", "snap", 300, "price"));
        o.push_back(family_rule("zzxu.a", "snap", 300, "hl"));
        o.push_back(family_rule("zzxu.a|c0", "snap", 300, "hl"));
    }
    void seed_once(const tape& t) {
        if (seeded) return;
        for (band& b : bands) b.seed(t.band_seed);
        seeded = true;
    }
    void on_snap(const tape& t, double, double, double, double) {
        seed_once(t);
        double px = t.snap_px;
        if (std::isnan(px) || std::isnan(t.hi_f) || std::isnan(t.lo_f)) return;
        double mid = (t.hi_f + t.lo_f) / 2;
        for (band& b : bands) {
            double w = t.ema_value(b.carried_width ? t.ix_zzxu_ac : t.ix_zzxu_a);
            if (!std::isnan(w)) b.step(mid + band_mult * w, mid - band_mult * w, px);
        }
    }
    void dump(std::string& o, const tape& t) {
        seed_once(t);
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
