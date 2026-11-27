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
    // Term output (slot 7 @ f5a60): idx = floor(t / blen) on the trading clock; b = 4 if price / prev close - 1 > -0.03 else 0
    // (the -0.01 / 0.01 / 0.03 branches follow the > -0.03 test and never fire); value = 25 idx + 5 b + code[idx]; .a = the return.
    double code_at(std::size_t k, const tape& t) const {
        return (!std::isnan(px[k]) && !std::isnan(t.pc) && px[k] / t.pc < 0.97) ? 0.0 : 4.0;
    }
    double out_a(const tape& t) const { return (t.snap_px > 0 && t.pc > 0) ? t.snap_px / t.pc - 1.0 : qnan; }
    double out_b(double tm, const tape& t) const {
        if (!(tm > 0)) return 0.0;
        double step = session / N;
        long idx = static_cast<long>(std::floor(tm / step));
        if (idx < 0 || idx >= N) return qnan;
        double r = out_a(t);
        double b = (r > -0.03) ? 4.0 : 0.0;
        return 25.0 * idx + 5.0 * b + code_at(static_cast<std::size_t>(idx), t);
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
