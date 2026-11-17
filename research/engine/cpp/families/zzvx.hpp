// zzvx: 30 s averages of the day range (trEma*, exact), of a narrower range (trDEma*) and of their ratio (rEma*); day high / low.
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
        o.push_back(family_rule("zzvx.trDEmaD|sp", "snap", 30, "hl_sp"));
        o.push_back(family_rule("zzvx.trDEmaD|spall", "snap", 30, "hl_spall"));
        o.push_back(family_rule("zzvx.trDEmaD|cont", "snap", 30, "hl_cont"));
        o.push_back(family_rule("zzvx.rEmaD|rsp", "snap", 30, "r_sp"));
        o.push_back(family_rule("zzvx.rEmaD|rspall", "snap", 30, "r_spall"));
        o.push_back(family_rule("zzvx.rEmaD|rcont", "snap", 30, "r_cont"));
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzvx.high", t.high);
        put(o, "zzvx.low", t.low);
    }
    static double col_a(double tr_d_ema_d, double px) { return tr_d_ema_d / px; }  // stand-in, wrong scale
};

}  // namespace eng
