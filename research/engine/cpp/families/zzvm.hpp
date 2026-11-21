// zzvm: volume and amount from 09:30, and zzvm.c.re = the 300 s trade-clock average of the running VWAP from 09:30.
#pragma once

#include "base.hpp"

namespace eng {

struct zzvm : family {
    static constexpr const char* stem = "zzvm";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzvm.c.re", "trade", 300, "rvwap"));
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzvm.c.v", t.vol);
        put(o, "zzvm.c.tpv", t.amt);
    }
};

}  // namespace eng
