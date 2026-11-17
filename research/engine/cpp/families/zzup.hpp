// zzup: fast (300 s) over slow (600 s) snapshot-price average; column zzup exact.
#pragma once

#include "base.hpp"

namespace eng {

struct zzup : family {
    static constexpr const char* stem = "zzup";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzup.pf", "snap", 300, "price"));
        o.push_back(family_rule("zzup.ps", "snap", 600, "price"));
    }
    static double col_zzup(double pf, double ps) { return pf / ps; }
};

}  // namespace eng
