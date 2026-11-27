// zzgh: per 15 s slot the buy / sell amount imbalance as a code 0..4; the session's amount imbalance zzgh.nb and its variants.
// r241: slot by the local trade clock is best (0.968; exchange clock 0.949, snapshot-gated 0.73-0.79).
// Probe variants (zzghvN.c.k): 1 = volume-weighted, 2 = trade count, 3 = amount, trades whose exchange slot agrees only.
#pragma once

#include "base.hpp"

namespace eng {

struct zzgh : family {
    static constexpr const char* stem = "zzgh";
    static constexpr status st = status::guessed;
    static constexpr int nv = 4;
    std::vector<double> buy[nv], sell[nv];
        zzgh() { for (int v = 0; v < nv; ++v) { buy[v].assign(n_gh, 0.0); sell[v].assign(n_gh, 0.0); } }
    static double imb(double b, double s) { return b + s ? (b - s) / (b + s) : qnan; }
    static double tclock(double sod) { return sod < 41400.0 ? sod - 34200.0 : sod - 46800.0 + 7200.0; }
    static int slot(double t) { return (0 <= t && t < session) ? static_cast<int>(std::floor(t / (session / n_gh))) : -1; }
    void add(int v, int k, double amt, double side) {
        if (k < 0) return;
        buy[v][k] += side > 0 ? amt : 0.0;
        sell[v][k] += side < 0 ? amt : 0.0;
    }
    void on_trade(const tape& tp, double t, double s, double amt, double side) {
        if (!(34200 <= s && s < 53820)) return;
        int k = slot(t);
        add(0, k, amt, side);
        double vol = tp.trade_px > 0 ? amt / tp.trade_px : 0.0;
        add(1, k, vol, side);   // volume-weighted
        add(2, k, 1.0, side);   // trade count
        add(3, slot(tclock(s)) == k ? k : -1, amt, side);  // amount, only trades whose exchange slot agrees
    }
    static int code_of(double b, double s) {
        int code = gh_empty;
        if (b + s > 0) {
            double x = (b - s) / (b + s);
            code = 0;
            for (double e : gh_edges) code += e <= x;
        }
        return code;
    }
    void dump(std::string& o, const tape& t) {
        put(o, "zzgh.nb", imb(t.buy, t.sell));
        put(o, "zzgh.nb|vol", imb(t.nb_vol[0], t.nb_vol[1]));
        put(o, "zzgh.nb|cnt", imb(t.nb_cnt[0], t.nb_cnt[1]));
        put(o, "zzgh.nb|snappx", imb(t.nb_snappx[0], t.nb_snappx[1]));
        put(o, "zzgh.nb|ord", imb(t.nb_ord[0], t.nb_ord[1]));
        put(o, "zzgh.nb|ordvol", imb(t.nb_ordvol[0], t.nb_ordvol[1]));
        for (int k = 0; k < n_gh; ++k) {
            put(o, "zzgh.c." + std::to_string(k), static_cast<double>(code_of(buy[0][k], sell[0][k])));
            for (int v = 1; v < nv; ++v) put(o, "zzghv" + std::to_string(v) + ".c." + std::to_string(k), static_cast<double>(code_of(buy[v][k], sell[v][k])));
        }
    }
};

}  // namespace eng
