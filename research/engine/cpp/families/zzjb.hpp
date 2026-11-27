// zzjb / zzjb2: per slot, code 0 when the slot's snapshot price is jb_drop or more below the previous close, else 4.
// The runtime's cfg.json has two ZZJB terms: zzjb (blen 15 -> 948 slots) and zzjb2 (blen 10 -> 1422 slots).
#pragma once

#include "base.hpp"

namespace eng {

template <int N>
struct zzjb_t : family {
    static constexpr status st = status::guessed;
    last_before px{N, qnan};
    void on_snap(const tape& t, double tt, double, double, double) { px.feed(tt, t.snap_px); }
    void dump_as(std::string& o, const tape& t, const char* name) {
        const auto& v = px.close();
        for (std::size_t k = 0; k < v.size(); ++k) {
            bool drop = !std::isnan(v[k]) && !std::isnan(t.pc) && v[k] <= t.pc * (1 - jb_drop);
            put(o, std::string(name) + ".c." + std::to_string(k), drop ? 0.0 : 4.0);
        }
    }
};

struct zzjb : zzjb_t<948> {
    static constexpr const char* stem = "zzjb";
    void dump(std::string& o, const tape& t) { dump_as(o, t, "zzjb"); }
};

struct zzjb2 : zzjb_t<1422> {
    static constexpr const char* stem = "zzjb2";
    void dump(std::string& o, const tape& t) { dump_as(o, t, "zzjb2"); }
};

}  // namespace eng
