// zzgh / zzgh2: per slot the buy / sell trade-amount imbalance as a code 0..4 (edges -0.6 / -0.2 / 0.2 / 0.6, empty slot 2),
// trades placed by their local trading-clock time (r241: best of local / exchange / snapshot-gated; r242: amount weighting).
// cfg.json: zzgh blen 15 (948 slots), zzgh2 blen 10 (1422 slots). zzgh.nb: the session's amount imbalance (+ variants);
// zzgh2.nb / zzgh.nb|slots: the imbalance over slots 0 .. n-2 (the last slot dropped, r225b's best).
#pragma once

#include <cstdlib>

#include "base.hpp"

namespace eng {

template <int N>
struct zzgh_t : family {
    static constexpr status st = status::guessed;
    std::vector<double> buy = std::vector<double>(N, 0.0), sell = std::vector<double>(N, 0.0);
    static double imb(double b, double s) { return b + s ? (b - s) / (b + s) : qnan; }
    void on_trade(const tape&, double t, double s, double amt, double side) {
        if (34200 <= s && s < 53820 && 0 <= t && t < session) {
            std::size_t k = static_cast<std::size_t>(std::floor(t / (session / N)));
            buy[k] += side > 0 ? amt : 0.0;
            sell[k] += side < 0 ? amt : 0.0;
        }
    }
    double nb_slots() const {
        double b = 0, s = 0;
        for (int k = 0; k + 1 < N; ++k) { b += buy[k]; s += sell[k]; }
        return imb(b, s);
    }
    void codes(std::string& o, const char* name) const {
        for (int k = 0; k < N; ++k) {
            double b = buy[k], s = sell[k];
            int code = gh_empty;
            if (b + s > 0) {
                double x = (b - s) / (b + s);
                code = 0;
                for (double e : gh_edges) code += e <= x;  // searchsorted right: a value on an edge goes up
            }
            put(o, std::string(name) + ".c." + std::to_string(k), static_cast<double>(code));
        }
    }
};

struct zzgh : zzgh_t<948> {
    static constexpr const char* stem = "zzgh";
    void dump(std::string& o, const tape& t) {
        put(o, "zzgh.nb", imb(t.buy, t.sell));
        put(o, "zzgh.nb|slots", nb_slots());
        put(o, "zzgh.nb|vol", imb(t.nb_vol[0], t.nb_vol[1]));
        put(o, "zzgh.nb|cnt", imb(t.nb_cnt[0], t.nb_cnt[1]));
        put(o, "zzgh.nb|snappx", imb(t.nb_snappx[0], t.nb_snappx[1]));
        put(o, "zzgh.nb|ord", imb(t.nb_ord[0], t.nb_ord[1]));
        put(o, "zzgh.nb|ordvol", imb(t.nb_ordvol[0], t.nb_ordvol[1]));
        codes(o, "zzgh");
        if (std::getenv("ZZGH_PROBE"))
            for (int k = 0; k < 948; ++k) {
                put(o, "zzghb.c." + std::to_string(k), buy[k]);
                put(o, "zzghs.c." + std::to_string(k), sell[k]);
            }
    }
};

struct zzgh2 : zzgh_t<1422> {
    static constexpr const char* stem = "zzgh2";
    void dump(std::string& o, const tape&) {
        put(o, "zzgh2.nb", nb_slots());
        codes(o, "zzgh2");
    }
};

}  // namespace eng
