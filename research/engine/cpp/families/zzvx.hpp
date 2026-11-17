// zzvx: 30 s averages of the day range (four names, one value) and their update counts; day high / low; column zzvx.a exact.
#pragma once

#include "base.hpp"

namespace eng {

struct zzvx : family {
    static constexpr const char* stem = "zzvx";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzvx.trEmaD", "snap", 30, "hl"));
        o.push_back(family_rule("zzvx.trEmaO", "snap", 30, "hl"));
        o.push_back(family_rule("zzvx.trDEmaD", "snap", 30, "hl"));
        o.push_back(family_rule("zzvx.trDEmaO", "snap", 30, "hl"));
        o.push_back(family_rule("zzvx.rEmaD", "snap", 30, "one"));
        o.push_back(family_rule("zzvx.rEmaO", "snap", 30, "one"));
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzvx.high", t.high);
        put(o, "zzvx.low", t.low);
    }
    static double col_a(double tr_d_ema_d, double px) { return tr_d_ema_d / px; }
};

}  // namespace eng
