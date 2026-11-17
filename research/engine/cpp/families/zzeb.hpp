// zzeb: 100 s mid average; zzeb.r is open.
#pragma once

#include "base.hpp"

namespace eng {

struct zzeb : family {
    static constexpr const char* stem = "zzeb";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzeb.p", "snap", 100, "mid"));
    }
};

}  // namespace eng
