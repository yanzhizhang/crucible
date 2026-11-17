// zzqj: 30 s trade-clock average of the running VWAP; volume and amount from 09:30 (|all / |noclose / |loc = rejected variants).
#pragma once

#include "base.hpp"

namespace eng {

struct zzqj : family {
    static constexpr const char* stem = "zzqj";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzqj.r", "trade", 30, "rvwap"));
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzqj.v", t.vol);
        put(o, "zzqj.tpv", t.amt);
        put(o, "zzqj.v|all", t.vol_all);
        put(o, "zzqj.tpv|all", t.amt_all);
        put(o, "zzqj.v|noclose", t.vol_o);
        put(o, "zzqj.tpv|noclose", t.amt_o);
        put(o, "zzqj.v|loc", t.vol_loc);
        put(o, "zzqj.tpv|loc", t.amt_loc);
    }
};

}  // namespace eng
