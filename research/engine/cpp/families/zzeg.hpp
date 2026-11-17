// zzeg: the 4-day channel (previous 3 days + today's snapshot high / low) and the 300 s average of its mid, carried overnight.
#pragma once

#include "base.hpp"

namespace eng {

// NaN-skipping max / min (NaN only when both are).
inline double nmax(double a, double b) { return std::isnan(a) ? b : (std::isnan(b) ? a : (b > a ? b : a)); }
inline double nmin(double a, double b) { return std::isnan(a) ? b : (std::isnan(b) ? a : (b < a ? b : a)); }
inline double chan_high(const tape& t) { return nmax(t.ch_h, t.hi_f); }
inline double chan_low(const tape& t) { return nmin(t.ch_l, t.lo_f); }
inline double chan_span(const tape& t) {
    double h = chan_high(t), l = chan_low(t);
    return (std::isnan(h) || std::isnan(l)) ? qnan : (h + l) / 2;
}

struct zzeg : family {
    static constexpr const char* stem = "zzeg";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) { o.push_back(family_rule("zzeg.c.rEma|c0", "snap", 300, "span")); }
    void dump(std::string& o, const tape& t) {
        put(o, "zzeg.c.high", chan_high(t));
        put(o, "zzeg.c.low", chan_low(t));
        put(o, "zzeg.c.span", chan_span(t));
    }
};

}  // namespace eng
