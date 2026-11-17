// zzbd: log-price (100 s) and log-return (200 s) averages on the snapshot clock and on the trade clock.
#pragma once

#include "base.hpp"

namespace eng {

struct zzbd : family {
    static constexpr const char* stem = "zzbd";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzbd.pEmaS", "snap", 100, "logp"));
        o.push_back(family_rule("zzbd.rEmaS", "snap", 200, "logret"));
        o.push_back(family_rule("zzbd.pEmaT", "trade", 100, "logp"));
        o.push_back(family_rule("zzbd.rEmaT", "trade", 200, "logret"));
    }
};

}  // namespace eng
