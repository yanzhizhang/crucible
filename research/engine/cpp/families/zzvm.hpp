// zzvm: volume and amount from 09:30; zzvm.c.re (a carried trade-price average) is a lead only, not built.
#pragma once

#include "base.hpp"

namespace eng {

struct zzvm : family {
    static constexpr const char* stem = "zzvm";
    static constexpr status st = status::pinned;
    void dump(std::string& o, const tape& t) {
        put(o, "zzvm.c.v", t.vol);
        put(o, "zzvm.c.tpv", t.amt);
    }
};

}  // namespace eng
