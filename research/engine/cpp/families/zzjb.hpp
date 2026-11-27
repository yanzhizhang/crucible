// zzjb / zzjb2: per slot, code 0 when the FIRST snapshot price at or after the slot's start (local trading clock) is below
// 0.97 x the previous close, else 4. cfg.json: zzjb blen 15 (948 slots), zzjb2 blen 10 (1422 slots).
// r239 (1119): 0 mismatches on all 300 x 948 and 300 x 1422 codes (max ratio of a PM 0 = 0.96999, min of a PM 4 = 0.970003).
#pragma once

#include "base.hpp"

namespace eng {

template <int N>
struct zzjb_t : family {
    static constexpr status st = status::pinned;
    std::vector<double> px = std::vector<double>(N, qnan);
    std::size_t nxt = 0;
    static double tclock(double sod) { return sod < 41400.0 ? sod - 34200.0 : sod - 46800.0 + 7200.0; }
    void on_snap(const tape& t, double, double, double lsod, double) {
        double tl = tclock(lsod), step = session / N;
        while (nxt < static_cast<std::size_t>(N) && tl >= nxt * step) px[nxt++] = t.snap_px;
    }
    void dump_as(std::string& o, const tape& t, const char* name) {
        for (std::size_t k = 0; k < px.size(); ++k) {
            bool drop = !std::isnan(px[k]) && !std::isnan(t.pc) && px[k] / t.pc < 0.97;
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
