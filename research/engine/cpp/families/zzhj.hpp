// zzhj: the mid 30 s before each quarter-hour end (16 samples, the last at 14:56:30); zzhj.p is open.
#pragma once

#include "base.hpp"

namespace eng {

struct zzhj : family {
    static constexpr const char* stem = "zzhj";
    static constexpr status st = status::guessed;
    double c[n_hj];
    zzhj() { for (double& x : c) x = qnan; }
    void on_snap(const tape& t, double tt, double, double, double) {
        for (int k = 0; k < n_hj; ++k)
            if (tt <= hj_sample(k)) c[k] = t.mid;
    }
    void dump(std::string& o, const tape&) {
        for (int k = 0; k < n_hj; ++k) put(o, "zzhj.c." + std::to_string(k), c[k]);
    }
};

}  // namespace eng
