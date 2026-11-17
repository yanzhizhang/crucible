// zzba: 300 s averages, updated by snapshots and trades, of the snapshot price and of the trades' day range.
#pragma once

#include "base.hpp"

namespace eng {

struct zzba : family {
    static constexpr const char* stem = "zzba";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzba.p", "both", 300, "snap_px"));
        o.push_back(family_rule("zzba.r", "both", 300, "hl_trade"));
    }
};

}  // namespace eng
