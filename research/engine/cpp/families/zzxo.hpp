// zzxo: last trade price (pinned); a 30-slot 10 s snapshot-price grid ending at 14:57:00 local (guess).
#pragma once

#include "base.hpp"

namespace eng {

// N slots: slot k = the last snapshot price received at or before 14:57:00 - (N - 1 - k) x 10 s, local time.
template <int N>
struct price_grid : family {
    double g[N];
    price_grid() { for (double& x : g) x = qnan; }
    void on_snap(const tape& t, double, double, double lsod, double) {
        if (std::isnan(t.snap_px)) return;
        for (int k = 0; k < N; ++k)
            if (lsod <= 53820 - (N - 1 - k) * 10) g[k] = t.snap_px;
    }
    void dump_grid(std::string& o, const char* prefix) {
        for (int k = 0; k < N; ++k) put(o, prefix + std::to_string(k), g[k]);
    }
};

struct zzxo : price_grid<30> {
    static constexpr const char* stem = "zzxo";
    static constexpr status st = status::pinned;
    void dump(std::string& o, const tape& t) {
        put(o, "zzxo.last", t.last);
        dump_grid(o, "zzxo.r.");
    }
};

}  // namespace eng
