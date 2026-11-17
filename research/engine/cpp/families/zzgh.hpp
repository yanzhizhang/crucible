// zzgh: per 15 s slot the buy / sell amount imbalance as a code 0..4; the session's amount imbalance zzgh.nb and its variants.
#pragma once

#include "base.hpp"

namespace eng {

struct zzgh : family {
    static constexpr const char* stem = "zzgh";
    static constexpr status st = status::guessed;
    std::vector<double> buy = std::vector<double>(n_gh, 0.0), sell = std::vector<double>(n_gh, 0.0);  // SlotSum(948, 2)
    static double imb(double b, double s) { return b + s ? (b - s) / (b + s) : qnan; }
    void on_trade(const tape&, double t, double s, double amt, double side) {
        if (34200 <= s && s < 53820 && 0 <= t && t < session) {
            std::size_t k = static_cast<std::size_t>(std::floor(t / (session / n_gh)));
            buy[k] += side > 0 ? amt : 0.0;
            sell[k] += side < 0 ? amt : 0.0;
        }
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzgh.nb", imb(t.buy, t.sell));
        put(o, "zzgh.nb|vol", imb(t.nb_vol[0], t.nb_vol[1]));
        put(o, "zzgh.nb|cnt", imb(t.nb_cnt[0], t.nb_cnt[1]));
        put(o, "zzgh.nb|snappx", imb(t.nb_snappx[0], t.nb_snappx[1]));
        put(o, "zzgh.nb|ord", imb(t.nb_ord[0], t.nb_ord[1]));
        put(o, "zzgh.nb|ordvol", imb(t.nb_ordvol[0], t.nb_ordvol[1]));
        for (int k = 0; k < n_gh; ++k) {
            double b = buy[k], s = sell[k];
            int code = gh_empty;
            if (b + s > 0) {
                double x = (b - s) / (b + s);
                code = 0;
                for (double e : gh_edges) code += e <= x;  // searchsorted right: a value on an edge goes up
            }
            put(o, "zzgh.c." + std::to_string(k), static_cast<double>(code));
        }
    }
};

}  // namespace eng
