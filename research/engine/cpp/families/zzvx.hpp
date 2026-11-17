// zzvx: 30 s averages of the day range (trEma*, exact), a lagging second series (trDEma*: the double average is under test) and a ratio (rEma*).
// SUBJECT TO CHANGE: trDEma* / rEma* are right only where PM's two ranges agree; the "|..." rules are the candidates under test.
#pragma once

#include "base.hpp"

namespace eng {

struct zzvx : family {
    static constexpr const char* stem = "zzvx";
    static constexpr status st = status::guessed;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzvx.trEmaD", "snap", 30, "hl"));
        o.push_back(family_rule("zzvx.trEmaO", "snap", 30, "hl"));
        o.push_back(family_rule("zzvx.trDEmaD", "snap", 30, "hl"));
        o.push_back(family_rule("zzvx.trDEmaO", "snap", 30, "hl"));
        o.push_back(family_rule("zzvx.rEmaD", "snap", 30, "one"));
        o.push_back(family_rule("zzvx.rEmaO", "snap", 30, "one"));
        o.push_back(family_rule("zzvx.trDEmaD|dema", "snap", 30, "ema_tr"));
        o.push_back(family_rule("zzvx.rEmaD|rte", "snap", 30, "r_tr_ema"));
        o.push_back(family_rule("zzvx.rEmaD|rtd", "snap", 30, "r_tr_dema"));
        o.push_back(family_rule("zzvx.rEmaD|red", "snap", 30, "r_ema_dema"));
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzvx.high", t.high);
        put(o, "zzvx.low", t.low);
    }
    static double col_a(double tr_d_ema_d, double px) { return tr_d_ema_d / px; }  // stand-in, wrong scale
};

}  // namespace eng
