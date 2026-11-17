// zzab: 300 s snapshot-price average (key prefix zzab2).
#pragma once

#include "base.hpp"

namespace eng {

struct zzab : family {
    static constexpr const char* stem = "zzab";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzab2.r", "snap", 300, "price"));
    }
};

}  // namespace eng
