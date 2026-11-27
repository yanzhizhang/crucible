// zzhj: zzhj.c holds 474 slots in PM's .jz; the first 16 are the mid of the FIRST snapshot at or after each sample time
// (30 s before each quarter-hour end, the last at 14:56:30) on the local trading clock; the remaining slots are 0.
// r237 (1119): 1.0000 exact on all 16 samples (last-before on the local clock: 0.63; exchange clock first-after: 0.9998).
#pragma once

#include "base.hpp"

namespace eng {

struct zzhj : family {
    static constexpr const char* stem = "zzhj";
    static constexpr status st = status::pinned;
    double c[n_hj];
    bool done[n_hj] = {};
    zzhj() { for (double& x : c) x = qnan; }
    static double tclock(double sod) { return sod < 41400.0 ? sod - 34200.0 : sod - 46800.0 + 7200.0; }
    void on_snap(const tape& t, double, double, double lsod, double) {
        double tl = tclock(lsod);
        for (int k = 0; k < n_hj; ++k)
            if (!done[k] && tl >= hj_sample(k)) {
                c[k] = t.mid;
                done[k] = true;
            }
    }
    void dump(std::string& o, const tape&) {
        for (int k = 0; k < 474; ++k) put(o, "zzhj.c." + std::to_string(k), k < n_hj ? c[k] : 0.0);
    }
};

}  // namespace eng
