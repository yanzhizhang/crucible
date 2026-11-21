// zzmh: the continuous session's highest snapshot price and the 1 s average of (2 x drawdown from it)^2.
#pragma once

#include "base.hpp"

namespace eng {

struct zzmh : family {
    static constexpr const char* stem = "zzmh";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzmh.r", "snap", 1, "mh_dd2"));
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzmh.hm", t.hm);
        put(o, "zzmh.hm|le1500", t.hm_c);
        put(o, "zzmh.hm|all", t.sp_hi);
    }
};

}  // namespace eng
