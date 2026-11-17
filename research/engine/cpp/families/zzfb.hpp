// zzfb: the snapshot's total amount on a 711-slot grid (last value before each slot's end).
#pragma once

#include "base.hpp"

namespace eng {

struct zzfb : family {
    static constexpr const char* stem = "zzfb";
    static constexpr status st = status::guessed;
    last_before c{711, 0.0};
    void on_snap(const tape&, double t, double, double, double amount) { c.feed(t, amount); }
    void dump(std::string& o, const tape&) {
        const auto& v = c.close();
        for (std::size_t k = 0; k < v.size(); ++k) put(o, "zzfb.c." + std::to_string(k), v[k]);
    }
};

}  // namespace eng
