// zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; a trend band (upper / lower / dir).
// SUBJECT TO CHANGE: the simple band rule ("|st1") gets the inactive band right and the ratcheted one wrong (R162); the
// "|m.<centre>.<width>.<start>" keys are running extremes of centre -+ 3 x width, the candidates for the ratchet (see zzxu.py).
#pragma once

#include "base.hpp"

namespace eng {

struct zzxu : family {
    static constexpr const char* stem = "zzxu";
    static constexpr status st = status::pinned;
    static constexpr double band_mult = 3.0;
    static constexpr int n_cand = 18;  // 3 centres (mid, px, p) x 3 widths (a, tr, de) x 2 starts (a, c)
    double upper = qnan, lower = qnan, dir = 1.0;
    double max_lower[n_cand], min_upper[n_cand];
    zzxu() {
        for (int i = 0; i < n_cand; ++i) max_lower[i] = min_upper[i] = qnan;
    }
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzxu.p", "snap", 300, "price"));
        o.push_back(family_rule("zzxu.a", "snap", 300, "hl"));
    }
    void on_snap(const tape& t, double, double s, double, double) {
        const double ws[3] = {t.ema_value(t.ix_zzxu_a), t.ema_value(t.ix_vx_tr), t.ema_value(t.ix_vx_dema)};
        double px = t.snap_px;
        if (std::isnan(px) || std::isnan(t.hi_f) || std::isnan(t.lo_f)) return;
        double mid = (t.hi_f + t.lo_f) / 2;
        const double cs[3] = {mid, px, t.ema_value(t.ix_zzxu_p)};
        int i = 0;
        for (double c : cs)
            for (double w : ws)
                for (int start = 0; start < 2; ++start) {
                    if (!std::isnan(c) && !std::isnan(w) && (start == 0 || s >= 34200)) {
                        double lo = c - band_mult * w, up = c + band_mult * w;
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
        put(o, "zzxu.upper|st1", upper);
        put(o, "zzxu.lower|st1", lower);
        put(o, "zzxu.dir|st1", dir);
        static const char* cn[3] = {"mid", "px", "p"};
        static const char* wn[3] = {"a", "tr", "de"};
        static const char* sn[2] = {"a", "c"};
        int i = 0;
        for (const char* c : cn)
            for (const char* w : wn)
                for (const char* s : sn) {
                    std::string tag = std::string("|m.") + c + "." + w + "." + s;
                    put(o, "zzxu.lower" + tag, max_lower[i]);
                    put(o, "zzxu.upper" + tag, min_upper[i]);
                    ++i;
                }
    }
    static double col_a(double a, double low) { return a / low; }                      // stand-in, wrong scale
    static double col_b(double px, double a, double last) { return (px - a) / last; }  // stand-in, wrong scale
};

}  // namespace eng
