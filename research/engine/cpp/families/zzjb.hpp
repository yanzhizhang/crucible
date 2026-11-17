// zzjb: per 15 s slot, code 0 when the slot's snapshot price is jb_drop or more below the previous close, else 4.
#pragma once

#include "base.hpp"

namespace eng {

struct zzjb : family {
    static constexpr const char* stem = "zzjb";
    static constexpr status st = status::guessed;
    last_before px{948, qnan};
    void on_snap(const tape& t, double tt, double, double, double) { px.feed(tt, t.snap_px); }
    void dump(std::string& o, const tape& t) {
        const auto& v = px.close();
        for (std::size_t k = 0; k < v.size(); ++k) {
            bool drop = !std::isnan(v[k]) && !std::isnan(t.pc) && v[k] <= t.pc * (1 - jb_drop);
            put(o, "zzjb.c." + std::to_string(k), drop ? 0.0 : 4.0);
        }
    }
};

}  // namespace eng
