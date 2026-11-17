// zzvx: the 30 s average of the day range (trEma*), its double average (trDEma* = the 30 s average of trEma's value) and the
// average of their ratio (rEma* = average of trEma / trDEma); day high / low. All equal PM's day-end file (R162).
#pragma once

#include "base.hpp"

namespace eng {

struct zzvx : family {
    static constexpr const char* stem = "zzvx";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzvx.trEmaD", "snap", 30, "hl"));
        o.push_back(family_rule("zzvx.trEmaO", "snap", 30, "hl"));
        o.push_back(family_rule("zzvx.trDEmaD", "snap", 30, "ema_tr"));
        o.push_back(family_rule("zzvx.trDEmaO", "snap", 30, "ema_tr"));
        o.push_back(family_rule("zzvx.rEmaD", "snap", 30, "r_ema_dema"));
        o.push_back(family_rule("zzvx.rEmaO", "snap", 30, "r_ema_dema"));
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzvx.high", t.high);
        put(o, "zzvx.low", t.low);
    }
    static double col_a(double tr_d_ema_d, double px) { return tr_d_ema_d / px; }  // stand-in, wrong scale
};

}  // namespace eng
