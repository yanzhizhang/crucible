// zzxu: 300 s averages of the snapshot price and of the day range; day high / low / last; columns zzxu.a, zzxu.b exact.
#pragma once

#include "base.hpp"

namespace eng {

struct zzxu : family {
    static constexpr const char* stem = "zzxu";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzxu.p", "snap", 300, "price"));
        o.push_back(family_rule("zzxu.a", "snap", 300, "hl"));
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzxu.high", t.high);
        put(o, "zzxu.low", t.low);
        put(o, "zzxu.last", t.last);
    }
    static double col_a(double a, double low) { return a / low; }
    static double col_b(double px, double a, double last) { return (px - a) / last; }
};

}  // namespace eng
