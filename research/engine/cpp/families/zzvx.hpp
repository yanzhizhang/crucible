// zzvx: 30 s averages of the day range and of a wider "true" range, and of their ratio; day high / low. SUBJECT TO CHANGE:
// PM's trEma* >= trDEma* and rEma* > 1 on ~13-17% of stock-days; the "|..." rules are the candidates under test.
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
        o.push_back(family_rule("zzvx.trEmaD|pc", "snap", 30, "tr_pc"));
        o.push_back(family_rule("zzvx.trEmaD|trade", "snap", 30, "hl_trade"));
        o.push_back(family_rule("zzvx.trEmaD|tro", "snap", 30, "tro"));
        o.push_back(family_rule("zzvx.rEmaD|tr", "snap", 30, "tr_ratio"));
        o.push_back(family_rule("zzvx.rEmaD|tratio", "snap", 30, "trade_ratio"));
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzvx.high", t.high);
        put(o, "zzvx.low", t.low);
    }
    static double col_a(double tr_d_ema_d, double px) { return tr_d_ema_d / px; }  // stand-in, wrong scale
};

}  // namespace eng
