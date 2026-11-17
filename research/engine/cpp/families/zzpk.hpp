// zzpk: 300 s snapshot-price average, day high / low / open.
#pragma once

#include "base.hpp"

namespace eng {

struct zzpk : family {
    static constexpr const char* stem = "zzpk";
    static constexpr status st = status::pinned;
    static void rules(std::vector<rule>& o) {
        o.push_back(family_rule("zzpk.p", "snap", 300, "price"));
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzpk.high", t.high);
        put(o, "zzpk.low", t.low);
        put(o, "zzpk.open", t.open);
    }
};

}  // namespace eng
