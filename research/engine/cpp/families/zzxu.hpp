// zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).
// SUBJECT TO CHANGE: the simple band rule on the carried width ("|c0a") plus running extremes of centre -+ mult x width as
// candidates for the active band ("|m.<centre>.<mult>.<width>", see zzxu.py).
#pragma once

#include <cstdio>

#include "base.hpp"

namespace eng {

struct zzxu : family {
    static constexpr const char* stem = "zzxu";
    static constexpr status st = status::pinned;
    static constexpr double band_mult = 3.0;
    static constexpr int n_c = 5, n_m = 5, n_w = 3, n_cand = n_c * n_m * n_w;
    static constexpr double mults[n_m] = {1.0, 1.5, 2.0, 2.5, 3.0};
    double upper = qnan, lower = qnan, dir = 1.0;
    double max_lower[n_cand], min_upper[n_cand];
    zzxu() {
        for (int i = 0; i < n_cand; ++i) max_lower[i] = min_upper[i] = qnan;
    }
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzxu.p", "snap", 300, "price"));
        o.push_back(family_rule("zzxu.a", "snap", 300, "hl"));
        o.push_back(family_rule("zzxu.a|c0", "snap", 300, "hl"));
    }
    void on_snap(const tape& t, double, double, double, double) {
        double px = t.snap_px;
        if (std::isnan(px) || std::isnan(t.hi_f) || std::isnan(t.lo_f)) return;
        double mid = (t.hi_f + t.lo_f) / 2;
        const double ws[n_w] = {t.ema_value(t.ix_zzxu_ac), t.ema_value(t.ix_vx_tr), t.ema_value(t.ix_vx_dema)};
        const double cs[n_c] = {mid, px, t.ema_value(t.ix_zzxu_p), t.hi_f, t.lo_f};
        int i = 0;
        for (double c : cs)
            for (double m : mults)
                for (double w : ws) {
                    if (!std::isnan(c) && !std::isnan(w)) {
                        double lo = c - m * w, up = c + m * w;
                        if (!(max_lower[i] >= lo)) max_lower[i] = lo;
                        if (!(min_upper[i] <= up)) min_upper[i] = up;
                    }
                    ++i;
                }
        double w = ws[0];
        if (std::isnan(w)) return;
        double bu = mid + band_mult * w, bl = mid - band_mult * w;
        if (std::isnan(upper)) {
            upper = bu;
            lower = bl;
        } else if (dir > 0) {
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
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzxu.high", t.high);
        put(o, "zzxu.low", t.low);
        put(o, "zzxu.last", t.last);
        put(o, "zzxu.upper|c0a", upper);
        put(o, "zzxu.lower|c0a", lower);
        put(o, "zzxu.dir|c0a", dir);
        static const char* cn[n_c] = {"mid", "px", "p", "hi", "lo"};
        static const char* mn[n_m] = {"1", "1.5", "2", "2.5", "3"};
        static const char* wn[n_w] = {"c0", "tr", "de"};
        int i = 0;
        for (const char* c : cn)
            for (const char* m : mn)
                for (const char* w : wn) {
                    std::string tag = std::string("|m.") + c + "." + m + "." + w;
                    put(o, "zzxu.lower" + tag, max_lower[i]);
                    put(o, "zzxu.upper" + tag, min_upper[i]);
                    ++i;
                }
    }
    static double col_a(double a, double low) { return a / low; }                      // stand-in, wrong scale
    static double col_b(double px, double a, double last) { return (px - a) / last; }  // stand-in, wrong scale
};

}  // namespace eng
