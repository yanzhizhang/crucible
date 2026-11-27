// zzmh: the continuous session's highest snapshot price and the 10 s average (cfg hl=10) of (2 x drawdown from it)^2.
#pragma once

#include "base.hpp"

namespace eng {

struct zzmh : family {
    static constexpr const char* stem = "zzmh";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzmh.r", "snap", 10, "mh_dd2"));
    }
    void dump(std::string& o, const tape& t) { put(o, "zzmh.hm", t.hm); }
};

}  // namespace eng
