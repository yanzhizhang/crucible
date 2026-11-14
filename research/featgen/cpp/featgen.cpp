// featgen: the base quantities behind the baseline's 144 column formulas (research/engine/baseline_r41c.txt), per stock
// at every time of a grid T (trading-clock seconds of local arrival: 09:30 = 0, lunch removed).
//
// A port of the Python definitions, operation for operation, so the two can be compared value by value:
//   ic_score.generic                         snapshot ("fast") and trade ("slow") quantities
//   lit_factors.literature / _book_open_limits / _size_tiers / order_stream
//   tick_flow.flow                           flow_* quantities
//   ic_score.seed_feats                      sd_* quantities (yesterday's state)
// Only what the 144 formulas use is here. The whole-book quantity (best_level_order_count_imbalance) comes from the book
// replay, not from this file.
//
//   featgen <in.bin> <out.bin> [--names names.txt] [--threads N] [--simb-exact]
// in.bin  (featio.py writes it): "CRUFG001", u64 n_stocks, u64 n_T, T[n_T], then per stock
//         i64 sid, u64 nf, u64 ns, u64 no, f64 seed[3], fast[NF][nf], slow[NS][ns], orders[NO][no]   (all f64, column-major)
// out.bin: "CRUFO001", u64 n_stocks, u64 n_T, u64 n_base, then per stock i64 sid, f64 v[n_base][n_T]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using vec = std::vector<double>;
constexpr double nan_ = std::numeric_limits<double>::quiet_NaN();
constexpr double inf_ = std::numeric_limits<double>::infinity();
bool simb_exact = false;  // --simb-exact: sign the snapshot flows against the mid itself, not exp(log(mid))

// column order of the three streams (featio.py FAST / SLOW / ORD)
enum fast_col { F_T, F_S, F_PRICE, F_OP, F_PC, F_B1, F_A1, F_BV1, F_AV1, F_CV, F_CA, F_BV5, F_AV5, F_BWP5, F_AWP5, F_OFI25, F_UL, F_LL,
                F_OFIL2, F_BQ2 = F_OFIL2 + 5, F_AQ2 = F_BQ2 + 5, NF = F_AQ2 + 5 };
enum slow_col { S_T, S_S, S_PRICE, S_VOL, S_AMT, S_SIDE, S_BSQ, S_SSQ, NS };
enum ord_col { O_T, O_S, O_KIND, O_SIDE, O_AMT, O_PX, O_SEQ, NO };
enum seed_col { SD_PK_LOW, SD_EG_HIGH, SD_EG_LOW, NSEED };

const char* const names[] = {
    // snapshots (ic_score.generic)
    "ret90", "ret120", "ret1800", "mret1800", "imb5", "rv60", "imb1", "zr60",
    "lpe5", "lpe25", "lpe50", "lpe100", "lpe200", "rpe5", "rpe10", "rpe20", "rpe25", "rpe75", "rpe5400", "rpe14400",
    "lmie5", "lmie10", "lmie30", "rete30", "rete50", "rete75", "rete150", "lpma60", "rpma60", "lee30_14400",
    "lpes600", "lpes1200", "lpes1800", "lpes2400", "lpes3600", "lpes14400", "lpes28800",
    "lpev60", "lpev3600", "lpev14400", "lpew1800", "lpew14400", "lpek50", "ldl1800",
    "simba10", "simba180", "simbn45", "simbv120",
    // snapshots (lit_factors)
    "microprice_minus_mid_rel", "depth_weighted_price_5lvl_minus_mid_rel", "bid_depth_weighted_distance_5lvl_rel",
    "ofi_levels1to5_norm_10s", "ofi_levels2to5_norm_10s", "ofi_levels1to6_norm_30s",
    "overnight_gap_log", "return_open_to_1000_log", "log_distance_to_up_limit", "position_in_limit_band",
    "trading_seconds_since_0930", "amihud_illiq_900s",
    // trades (ic_score.generic, lit_factors._size_tiers, tick_flow)
    "imbv10", "imbv30", "imbn10", "imbn90", "imbn120", "imba10", "imba30", "imba600", "nam15", "vfrac10", "buy300", "stoch",
    "amount_share_orders_over_200k_cny_since_0930", "net_buy_amount_share_orders_over_200k_cny_120s",
    "net_buy_amount_share_orders_under_40k_cny_120s", "net_buy_amount_share_orders_under_40k_cny_300s",
    "flow_big_imb60",
    // orders (tick_flow, lit_factors.order_stream), snapshots + trades (tick_flow refill)
    "flow_ord_imb60", "flow_ord_imb300", "flow_refill_imb60", "flow_refill_imb300",
    "marketable_order_amount_share_60s", "order_price_aggressiveness_mean_900s", "log_new_order_count_per_s_60s",
    "log_channel_sequence_advance_per_s_60s", "log_channel_sequence_advance_per_s_900s",
    // yesterday's state (ic_score.seed_feats), the price at the mark
    "sd_lp_pklo", "sd_eg_rng", "px",
};
constexpr std::size_t n_base = sizeof(names) / sizeof(names[0]);

struct stock {
    std::int64_t sid = 0;
    std::size_t nf = 0, ns = 0, no = 0;
    double seed[NSEED] = {nan_, nan_, nan_};
    std::vector<vec> f, s, o;  // one vec per column
};

// ---- numpy look-alikes ----
inline double nan_to_num(double x) {
    if (std::isnan(x)) return 0.0;
    if (std::isinf(x)) return x > 0 ? std::numeric_limits<double>::max() : -std::numeric_limits<double>::max();
    return x;
}
inline double sign(double x) { return std::isnan(x) ? nan_ : (x > 0 ? 1.0 : (x < 0 ? -1.0 : 0.0)); }
inline double clip(double x, double lo, double hi) { return std::isnan(x) ? nan_ : std::min(std::max(x, lo), hi); }
// np.searchsorted(t, x, side="right")
inline std::size_t upper(const vec& t, double x) { return static_cast<std::size_t>(std::upper_bound(t.begin(), t.end(), x) - t.begin()); }
// feature_intraday.at_marks: the last value received at or before x
inline double at(const vec& t, const vec& v, double x) {
    const std::size_t k = upper(t, x);
    return k > 0 ? v[k - 1] : nan_;
}
// [0, cumsum(v)]
vec csum(const vec& v) {
    vec c(v.size() + 1);
    c[0] = 0.0;
    double a = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) c[i + 1] = (a += v[i]);
    return c;
}
// lit_factors._cs: the same with NaN counted as 0
vec csum0(const vec& v) {
    vec c(v.size() + 1);
    c[0] = 0.0;
    double a = 0.0;
    for (std::size_t i = 0; i < v.size(); ++i) c[i + 1] = (a += nan_to_num(v[i]));
    return c;
}
// running total over (x - w, x] of a csum over events at times t
inline double win(const vec& c, const vec& t, double x, double w) { return c[upper(t, x)] - c[upper(t, x - w)]; }

// ic_kernel.ema_irregular: weight 1 per update, decayed by 2^(-dt / half) between updates; NaN updates are skipped
vec ema(const vec& t, const vec& x, double half, double s = 0.0, double w = 0.0, bool seeded = false, double t0 = 0.0) {
    vec out(t.size());
    double last = seeded ? t0 : (t.empty() ? 0.0 : t[0]);
    const double k = std::log(2.0) / half;
    for (std::size_t i = 0; i < t.size(); ++i) {
        const double d = t[i] > last ? std::exp(-(t[i] - last) * k) : 1.0;
        s *= d;
        w *= d;
        last = t[i];
        if (std::isfinite(x[i])) {
            s += x[i];
            w += 1.0;
        }
        out[i] = w > 0 ? s / w : nan_;
    }
    return out;
}

struct result {
    std::vector<vec> v;  // [n_base][n_T]
    std::map<std::string, std::size_t> ix;
    result(std::size_t n_t) : v(n_base, vec(n_t, nan_)) {
        for (std::size_t i = 0; i < n_base; ++i) ix[names[i]] = i;
    }
    vec& operator[](const std::string& k) {
        auto it = ix.find(k);
        if (it == ix.end()) throw std::runtime_error("unknown base quantity " + k);
        return v[it->second];
    }
};

void compute(const stock& st, const vec& T, result& out) {
    const std::size_t nt = T.size();
    const auto& F = st.f;
    const auto& S = st.s;
    const auto& O = st.o;
    const std::size_t nf = st.nf, ns = st.ns, no = st.no;
    for (std::size_t j = 0; j < nt; ++j) out["trading_seconds_since_0930"][j] = T[j];

    // ---- per-snapshot arrays (dcp_ema.arrays "fast") ----
    vec pos(nf), mid_u(nf), dvol_u(nf), damt_u(nf), bid1_u(nf), ask1_u(nf);
    for (std::size_t i = 0; i < nf; ++i) {
        const double px = F[F_PRICE][i], b1 = F[F_B1][i], a1 = F[F_A1][i];
        pos[i] = px > 0 ? px : nan_;
        mid_u[i] = (b1 > 0 && a1 > 0) ? (b1 + a1) / 2 : pos[i];
        bid1_u[i] = b1 > 0 ? b1 : nan_;
        ask1_u[i] = a1 > 0 ? a1 : nan_;
        const double dv = i ? F[F_CV][i] - F[F_CV][i - 1] : nan_, da = i ? F[F_CA][i] - F[F_CA][i - 1] : nan_;
        dvol_u[i] = nan_to_num(std::isnan(dv) ? nan_ : std::max(dv, 0.0));
        damt_u[i] = nan_to_num(std::isnan(da) ? nan_ : std::max(da, 0.0));
    }
    // the snapshots with a usable mid
    vec t, lm, midk, b1, a1, bv, av, lpx, dv, da, pck, bw5, aw5, bv5, av5;
    for (std::size_t i = 0; i < nf; ++i) {
        if (!(std::isfinite(mid_u[i]) && mid_u[i] > 0)) continue;
        t.push_back(F[F_T][i]);
        midk.push_back(mid_u[i]);
        lm.push_back(std::log(mid_u[i]));
        b1.push_back(F[F_B1][i]);
        a1.push_back(F[F_A1][i]);
        bv.push_back(F[F_BV1][i]);
        av.push_back(F[F_AV1][i]);
        lpx.push_back(std::log(pos[i]));
        dv.push_back(dvol_u[i]);
        da.push_back(damt_u[i]);
        pck.push_back(F[F_PC][i]);
        bw5.push_back(F[F_BWP5][i]);
        aw5.push_back(F[F_AWP5][i]);
        bv5.push_back(F[F_BV5][i]);
        av5.push_back(F[F_AV5][i]);
    }
    const std::size_t n = t.size();
    vec lm0(nt, nan_);

    if (nf > 2) {  // ic_score.generic, the snapshot half
        vec lmi(n), mid_(n), micro_e(n);
        for (std::size_t i = 0; i < n; ++i) {
            mid_[i] = std::exp(lm[i]);
            const double micro = (bv[i] + av[i] > 0 && b1[i] > 0 && a1[i] > 0) ? (b1[i] * av[i] + a1[i] * bv[i]) / (bv[i] + av[i]) : mid_[i];
            lmi[i] = std::log(micro);
            micro_e[i] = std::exp(lmi[i]);
        }
        vec li0(nt), lp0(nt);
        for (std::size_t j = 0; j < nt; ++j) {
            lm0[j] = at(t, lm, T[j]);
            li0[j] = at(t, lmi, T[j]);
            lp0[j] = at(t, lpx, T[j]);
        }
        auto ret = [&](double h, std::size_t j) { return lm0[j] - at(t, lm, T[j] - h); };
        // realised volatility: c2[i] = sum of squared log-mid changes up to snapshot i
        vec c2(n);
        for (std::size_t i = 0; i < n; ++i) c2[i] = i ? c2[i - 1] + (lm[i] - lm[i - 1]) * (lm[i] - lm[i - 1]) : 0.0;
        auto rv = [&](double h, std::size_t j) {
            const std::size_t i1 = upper(t, T[j]), i0 = upper(t, T[j] - h);
            return std::sqrt((i1 > 0 ? c2[i1 - 1] : nan_) - (i0 > 0 ? c2[i0 - 1] : 0.0));
        };
        vec imb5_u(nf), imb1_u(nf);
        for (std::size_t i = 0; i < nf; ++i) {
            imb5_u[i] = (F[F_BV5][i] - F[F_AV5][i]) / (F[F_BV5][i] + F[F_AV5][i]);
            imb1_u[i] = (F[F_BV1][i] - F[F_AV1][i]) / (F[F_BV1][i] + F[F_AV1][i]);
        }
        // flows signed from snapshots alone
        vec sd(n), w_v(n), w_sv(n), w_a(n), w_sa(n), w_n(n), w_sn(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double pmid = i ? (simb_exact ? midk[i - 1] : mid_[i - 1]) : nan_;
            const double vw = dv[i] > 0 ? da[i] / dv[i] : nan_;
            double s1 = sign(vw - pmid);
            if (!(std::isfinite(s1) && s1 != 0)) s1 = sign(i ? lpx[i] - lpx[i - 1] : nan_);
            sd[i] = dv[i] > 0 ? nan_to_num(s1) : 0.0;
            w_v[i] = dv[i];
            w_sv[i] = dv[i] * sd[i];
            w_a[i] = da[i];
            w_sa[i] = da[i] * sd[i];
            w_n[i] = dv[i] > 0 ? 1.0 : 0.0;
            w_sn[i] = sd[i];
        }
        const vec c_v = csum(w_v), c_sv = csum(w_sv), c_a = csum(w_a), c_sa = csum(w_sa), c_n = csum(w_n), c_sn = csum(w_sn);
        // window low and mean of the log mid (win_reduce / win_mean)
        vec cs_lm(n + 1, 0.0);
        for (std::size_t i = 0; i < n; ++i) cs_lm[i + 1] = cs_lm[i] + lm[i];  // lm is finite on these rows
        auto win_min = [&](double w, std::size_t j) {
            const std::size_t s0 = upper(t, T[j] - w), e0 = upper(t, T[j]);
            if (e0 <= s0) return nan_;
            double m = lm[s0];
            for (std::size_t i = s0 + 1; i < e0; ++i) m = std::fmin(m, lm[i]);
            return m;
        };
        auto win_mean = [&](double w, std::size_t j) {
            const std::size_t s0 = upper(t, T[j] - w), e0 = upper(t, T[j]);
            return e0 > s0 ? (cs_lm[e0] - cs_lm[s0]) / static_cast<double>(e0 - s0) : nan_;
        };
        // EMAs of the mid, the microprice and the log-mid changes, on the trading clock
        std::map<int, vec> em_log, emi_log, em_ret;
        auto log_of = [](vec e) {
            for (double& x : e) x = std::log(x);
            return e;
        };
        for (int h : {5, 10, 20, 25, 30, 50, 60, 75, 100, 200, 3600, 5400, 14400}) em_log[h] = log_of(ema(t, mid_, h));
        for (int h : {5, 10, 30}) emi_log[h] = log_of(ema(t, micro_e, h));
        vec dlm(n);
        for (std::size_t i = 0; i < n; ++i) dlm[i] = i ? lm[i] - lm[i - 1] : nan_;
        for (int h : {30, 50, 75, 150}) em_ret[h] = ema(t, dlm, h);
        // EMAs carried from yesterday: start at the pre-close with a steady-state weight
        std::map<int, vec> ems_log;
        double pc0 = nan_;
        for (double p : pck)
            if (std::isfinite(p) && p > 0) {
                pc0 = p;
                break;
            }
        if (std::isfinite(pc0) && n > 2) {
            double dt = 3.0;
            if (n > 10) {
                vec d(n - 1);
                for (std::size_t i = 1; i < n; ++i) d[i - 1] = t[i] - t[i - 1];
                std::sort(d.begin(), d.end());
                dt = d.size() % 2 ? d[d.size() / 2] : (d[d.size() / 2 - 1] + d[d.size() / 2]) / 2;
            }
            for (int h : {600, 1200, 1800, 2400, 3600, 14400, 28800}) {
                const double w0 = 1.0 / (1.0 - std::pow(2.0, -std::max(dt, 0.1) / h));
                ems_log[h] = log_of(ema(t, mid_, h, pc0 * w0, w0, true, 0.0));
            }
        }
        // the wall clock (lunch back in) and the update counter as EMA clocks
        vec tw(n), ixc(n);
        for (std::size_t i = 0; i < n; ++i) {
            tw[i] = t[i] > 7200 ? t[i] + 5400 : t[i];
            ixc[i] = static_cast<double>(i);
        }
        const vec ew1800 = log_of(ema(tw, mid_, 1800)), ew14400 = log_of(ema(tw, mid_, 14400)), ek50 = log_of(ema(ixc, mid_, 50));

        for (std::size_t j = 0; j < nt; ++j) {
            const double x = T[j];
            out["ret90"][j] = ret(90, j);
            out["ret120"][j] = ret(120, j);
            out["ret1800"][j] = ret(1800, j);
            out["mret1800"][j] = li0[j] - at(t, lmi, x - 1800);
            out["imb5"][j] = at(F[F_T], imb5_u, x);
            out["imb1"][j] = at(F[F_T], imb1_u, x);
            out["rv60"][j] = rv(60, j);
            const double rv900 = rv(900, j);
            out["zr60"][j] = clip(ret(60, j) / (rv900 * std::sqrt(60.0 / 900.0)), -10, 10);
            auto lpe = [&](int h) { return lm0[j] - at(t, em_log[h], x); };
            out["lpe5"][j] = lpe(5);
            out["lpe25"][j] = lpe(25);
            out["lpe50"][j] = lpe(50);
            out["lpe100"][j] = lpe(100);
            out["lpe200"][j] = lpe(200);
            for (int h : {5, 10, 20, 25, 75, 5400, 14400}) out["rpe" + std::to_string(h)][j] = std::exp(lpe(h));
            for (int h : {5, 10, 30}) out["lmie" + std::to_string(h)][j] = li0[j] - at(t, emi_log[h], x);
            for (int h : {30, 50, 75, 150}) out["rete" + std::to_string(h)][j] = at(t, em_ret[h], x);
            const double lpma = lm0[j] - win_mean(60, j);
            out["lpma60"][j] = lpma;
            out["rpma60"][j] = std::exp(lpma);
            out["lee30_14400"][j] = at(t, em_log[30], x) - at(t, em_log[14400], x);
            for (auto& [h, e] : ems_log) out["lpes" + std::to_string(h)][j] = lm0[j] - at(t, e, x);
            for (int h : {60, 3600, 14400}) out["lpev" + std::to_string(h)][j] = clip(lpe(h) / rv900, -10, 10);
            out["lpew1800"][j] = lm0[j] - at(t, ew1800, x);
            out["lpew14400"][j] = lm0[j] - at(t, ew14400, x);
            out["lpek50"][j] = lm0[j] - at(t, ek50, x);
            out["ldl1800"][j] = win_min(1800, j) - lm0[j];
            out["simba10"][j] = win(c_sa, t, x, 10) / win(c_a, t, x, 10);
            out["simba180"][j] = win(c_sa, t, x, 180) / win(c_a, t, x, 180);
            out["simbn45"][j] = win(c_sn, t, x, 45) / win(c_n, t, x, 45);
            out["simbv120"][j] = win(c_sv, t, x, 120) / win(c_v, t, x, 120);
            out["px"][j] = at(F[F_T], pos, x);
        }
    }

    vec m0(nt, nan_);  // lit_factors' log_mid_price
    bool have_m0 = false;
    if (nf > 3) {  // lit_factors.literature (snapshot half) and _book_open_limits
        have_m0 = true;
        vec rel(n), dwp(n), bdd(n);
        for (std::size_t i = 0; i < n; ++i) {
            const double mid = midk[i];
            const double micro = (bv[i] + av[i] > 0 && b1[i] > 0 && a1[i] > 0) ? (b1[i] * av[i] + a1[i] * bv[i]) / (bv[i] + av[i]) : mid;
            rel[i] = (micro - mid) / mid;
            dwp[i] = ((bw5[i] * bv5[i] + aw5[i] * av5[i]) / (bv5[i] + av5[i]) - mid) / mid;
            bdd[i] = (mid - bw5[i]) / mid;
        }
        // multi-level order-flow imbalance, on every snapshot (quoted prices only)
        const vec& tl = F[F_T];
        vec e15(nf), e25(nf), d15(nf), d25(nf), one(nf, 1.0), e16(nf), d16(nf);
        for (std::size_t i = 0; i < nf; ++i) {
            double e1 = 0.0;
            if (i) {
                const double pb = F[F_B1][i - 1], cb = F[F_B1][i], pa = F[F_A1][i - 1], ca = F[F_A1][i];
                const double eb = (cb > 0 && pb > 0) ? F[F_BV1][i] * (cb >= pb) - F[F_BV1][i - 1] * (cb <= pb) : 0.0;
                const double ea = (ca > 0 && pa > 0) ? F[F_AV1][i] * (ca <= pa) - F[F_AV1][i - 1] * (ca >= pa) : 0.0;
                e1 = eb - ea;
            }
            const double x25 = nan_to_num(F[F_OFI25][i]);
            e15[i] = e1 + x25;
            e25[i] = x25;
            d15[i] = (F[F_BV5][i] + F[F_AV5][i]) / 2;
            d25[i] = (F[F_BV5][i] - F[F_BV1][i] + F[F_AV5][i] - F[F_AV1][i]) / 2;
            double re = e1, rq = nan_to_num((F[F_BV1][i] + F[F_AV1][i]) / 2);
            for (int lv = 0; lv < 5; ++lv) {  // levels 2..6
                re = re + nan_to_num(F[F_OFIL2 + lv][i]);
                rq = rq + nan_to_num((F[F_BQ2 + lv][i] + F[F_AQ2 + lv][i]) / 2);
            }
            e16[i] = re;
            d16[i] = rq;
        }
        const vec c15 = csum0(e15), c25 = csum0(e25), cd15 = csum0(d15), cd25 = csum0(d25), cn = csum0(one), c16 = csum0(e16), cd16 = csum0(d16);
        double op = nan_, pc = nan_, ul = nan_, ll = nan_;
        for (std::size_t i = 0; i < nf; ++i) {
            const double o = F[F_OP][i], p = F[F_PC][i], u = F[F_UL][i], l = F[F_LL][i];
            if (std::isfinite(o) && o > 0) op = std::isnan(op) ? o : std::min(op, o);
            if (std::isfinite(p) && p > 0) pc = std::isnan(pc) ? p : std::max(pc, p);
            if (std::isfinite(u) && u > 0) ul = std::isnan(ul) ? u : std::max(ul, u);
            if (std::isfinite(l) && l > 0) ll = std::isnan(ll) ? l : std::min(ll, l);
        }
        for (std::size_t j = 0; j < nt; ++j) {
            const double x = T[j];
            m0[j] = at(t, lm, x);
            out["microprice_minus_mid_rel"][j] = at(t, rel, x);
            out["depth_weighted_price_5lvl_minus_mid_rel"][j] = at(t, dwp, x);
            out["bid_depth_weighted_distance_5lvl_rel"][j] = at(t, bdd, x);
            const double n10 = win(cn, tl, x, 10), n30 = win(cn, tl, x, 30);
            out["ofi_levels1to5_norm_10s"][j] = win(c15, tl, x, 10) / (win(cd15, tl, x, 10) / n10);
            out["ofi_levels2to5_norm_10s"][j] = win(c25, tl, x, 10) / (win(cd25, tl, x, 10) / n10);
            out["ofi_levels1to6_norm_30s"][j] = win(c16, tl, x, 30) / (win(cd16, tl, x, 30) / n30);
            out["overnight_gap_log"][j] = std::log(op / pc);
            out["return_open_to_1000_log"][j] = at(t, lm, std::min(x, 1800.0)) - std::log(op);
            out["log_distance_to_up_limit"][j] = std::log(ul) - m0[j];
            out["position_in_limit_band"][j] = (std::exp(m0[j]) - ll) / (ul - ll);
        }
    }

    // ---- trades ----
    const vec& ts = ns ? S[S_T] : t;
    std::vector<char> cont(ns);
    vec tpos(ns);
    for (std::size_t i = 0; i < ns; ++i) {
        cont[i] = S[S_S][i] >= 34200;
        tpos[i] = S[S_PRICE][i] > 0 ? S[S_PRICE][i] : nan_;
    }
    if (ns > 2) {  // ic_score.generic, the trade half; tick_flow big_imb
        vec w_v(ns), w_nv(ns), w_n(ns), w_nn(ns), w_a(ns), w_na(ns), w_big(ns), w_bign(ns), hi(ns), lo(ns);
        double rh = nan_, rl = nan_;
        for (std::size_t i = 0; i < ns; ++i) {
            const double c = cont[i] ? 1.0 : 0.0, vol = S[S_VOL][i], amt = S[S_AMT][i], side = S[S_SIDE][i];
            w_v[i] = vol * c;
            w_nv[i] = vol * side * c;
            w_n[i] = c;
            w_nn[i] = side * c;
            w_a[i] = amt * c;
            w_na[i] = amt * side * c;
            const bool big = amt >= 2e5 && cont[i];
            w_big[i] = amt * (big ? 1.0 : 0.0);
            w_bign[i] = amt * side * (big ? 1.0 : 0.0);
            const double p = cont[i] ? tpos[i] : nan_;
            rh = std::fmax(rh, p);
            rl = std::fmin(rl, p);
            hi[i] = rh;
            lo[i] = rl;
        }
        const vec c_v = csum(w_v), c_nv = csum(w_nv), c_n = csum(w_n), c_nn = csum(w_nn), c_a = csum(w_a), c_na = csum(w_na);
        const vec c_big = csum(w_big), c_bign = csum(w_bign);
        for (std::size_t j = 0; j < nt; ++j) {
            const double x = T[j];
            out["imbv10"][j] = win(c_nv, ts, x, 10) / win(c_v, ts, x, 10);
            out["imbv30"][j] = win(c_nv, ts, x, 30) / win(c_v, ts, x, 30);
            out["imbn10"][j] = win(c_nn, ts, x, 10) / win(c_n, ts, x, 10);
            out["imbn90"][j] = win(c_nn, ts, x, 90) / win(c_n, ts, x, 90);
            out["imbn120"][j] = win(c_nn, ts, x, 120) / win(c_n, ts, x, 120);
            out["imba10"][j] = win(c_na, ts, x, 10) / win(c_a, ts, x, 10);
            out["imba30"][j] = win(c_na, ts, x, 30) / win(c_a, ts, x, 30);
            out["imba600"][j] = win(c_na, ts, x, 600) / win(c_a, ts, x, 600);
            out["nam15"][j] = win(c_na, ts, x, 15);
            out["vfrac10"][j] = std::log(win(c_v, ts, x, 10) / c_v[upper(ts, x)]);
            out["buy300"][j] = (win(c_a, ts, x, 300) + win(c_na, ts, x, 300)) / 2;
            const double px = at(ts, tpos, x), l = at(ts, lo, x), h = at(ts, hi, x);
            out["stoch"][j] = (px - l) / (h - l);
            out["flow_big_imb60"][j] = win(c_bign, ts, x, 60) / win(c_big, ts, x, 60);
        }
    }
    if (ns > 3) {  // amihud (needs the snapshot half too) and lit_factors._size_tiers
        vec amt_c(ns);
        for (std::size_t i = 0; i < ns; ++i) amt_c[i] = cont[i] ? S[S_AMT][i] : 0.0;
        const vec cs_a = csum0(amt_c);
        if (have_m0)
            for (std::size_t j = 0; j < nt; ++j)
                out["amihud_illiq_900s"][j] = std::fabs(m0[j] - at(t, lm, T[j] - 900)) / (win(cs_a, ts, T[j], 900) / 1e8);
        // the aggressive order's whole amount: its fills summed
        std::vector<char> valid(ns);
        std::vector<std::int64_t> key(ns);
        std::unordered_map<std::int64_t, double> tot_of;
        bool any = false;
        for (std::size_t i = 0; i < ns; ++i) {
            const double side = S[S_SIDE][i];
            const double oid = side > 0 ? S[S_BSQ][i] : (side < 0 ? S[S_SSQ][i] : -1.0);
            valid[i] = cont[i] && side != 0 && oid >= 0;
            if (valid[i]) {
                key[i] = static_cast<std::int64_t>(oid) * 2 + (side > 0 ? 1 : 0);
                tot_of[key[i]] += nan_to_num(S[S_AMT][i]);
                any = true;
            }
        }
        if (any) {
            vec small_n(ns), big_n(ns), big_a(ns);
            for (std::size_t i = 0; i < ns; ++i) {
                const double amt = nan_to_num(S[S_AMT][i]), side = S[S_SIDE][i];
                const double oamt = valid[i] ? tot_of[key[i]] : nan_;
                const bool small = valid[i] && oamt >= 0.0 && oamt < 4e4, big = valid[i] && oamt >= 2e5 && oamt < inf_;
                small_n[i] = small ? amt * side : 0.0;
                big_n[i] = big ? amt * side : 0.0;
                big_a[i] = big ? amt : 0.0;
            }
            const vec c_sn = csum0(small_n), c_bn = csum0(big_n), c_ba = csum0(big_a);
            for (std::size_t j = 0; j < nt; ++j) {
                const double x = T[j];
                const std::size_t k = upper(ts, x);
                out["amount_share_orders_over_200k_cny_since_0930"][j] = c_ba[k] / cs_a[k];
                out["net_buy_amount_share_orders_over_200k_cny_120s"][j] = win(c_bn, ts, x, 120) / win(cs_a, ts, x, 120);
                out["net_buy_amount_share_orders_under_40k_cny_120s"][j] = win(c_sn, ts, x, 120) / win(cs_a, ts, x, 120);
                out["net_buy_amount_share_orders_under_40k_cny_300s"][j] = win(c_sn, ts, x, 300) / win(cs_a, ts, x, 300);
            }
        }
    }
    if (nf > 2 && ns > 2) {  // tick_flow: refills at an unchanged best price
        const vec& tf = F[F_T];
        vec rb(nf), ra(nf);
        for (std::size_t i = 0; i < nf; ++i) {
            const bool sb = i && F[F_B1][i] == F[F_B1][i - 1], sa = i && F[F_A1][i] == F[F_A1][i - 1];
            const double db = F[F_BV1][i] - (i ? F[F_BV1][i - 1] : 0.0), dq = F[F_AV1][i] - (i ? F[F_AV1][i - 1] : 0.0);
            rb[i] = sb ? clip(db, 0, inf_) : 0.0;
            ra[i] = sa ? clip(dq, 0, inf_) : 0.0;
        }
        const vec c_b = csum(rb), c_a = csum(ra);
        for (std::size_t j = 0; j < nt; ++j)
            for (int w : {60, 300}) {
                const double fb = win(c_b, tf, T[j], w), fa = win(c_a, tf, T[j], w);
                out["flow_refill_imb" + std::to_string(w)][j] = (fb - fa) / (fb + fa);
            }
    }

    // ---- orders ----
    if (no > 2) {  // tick_flow: new-order amount imbalance
        const vec& to = O[O_T];
        vec w_new(no), w_net(no);
        for (std::size_t i = 0; i < no; ++i) {
            const double is_new = (O[O_KIND][i] == 1 && O[O_S][i] >= 34200) ? 1.0 : 0.0;
            w_new[i] = O[O_AMT][i] * is_new;
            w_net[i] = O[O_AMT][i] * O[O_SIDE][i] * is_new;
        }
        const vec c_new = csum(w_new), c_net = csum(w_net);
        for (std::size_t j = 0; j < nt; ++j)
            for (int w : {60, 300}) out["flow_ord_imb" + std::to_string(w)][j] = win(c_net, to, T[j], w) / win(c_new, to, T[j], w);
    }
    if (no > 3) {  // lit_factors.order_stream
        const vec& to = O[O_T];
        vec w_new(no), w_cnt(no), w_m(no), w_ag(no), w_la(no);
        for (std::size_t i = 0; i < no; ++i) {
            const double side = O[O_SIDE][i], amt = nan_to_num(O[O_AMT][i]), px = O[O_PX][i];
            const bool is_new = O[O_S][i] >= 34200 && O[O_KIND][i] == 1 && side != 0;
            w_new[i] = is_new ? amt : 0.0;
            w_cnt[i] = is_new ? 1.0 : 0.0;
            if (nf > 2) {
                const double qb = at(F[F_T], bid1_u, to[i]), qa = at(F[F_T], ask1_u, to[i]), mid = at(F[F_T], mid_u, to[i]);
                const bool mkt = is_new && (px <= 0 || (side > 0 && px >= qa) || (side < 0 && px <= qb));
                const bool lim = is_new && px > 0 && std::isfinite(mid) && mid > 0;
                const double agg = clip(lim ? side * (px - mid) / mid : 0.0, -0.2, 0.2);
                w_m[i] = mkt ? amt : 0.0;
                w_ag[i] = lim ? amt * agg : 0.0;
                w_la[i] = lim ? amt : 0.0;
            }
        }
        const vec c_new = csum0(w_new), c_cnt = csum0(w_cnt), c_m = csum0(w_m), c_ag = csum0(w_ag), c_la = csum0(w_la);
        vec tq, sq;  // the channel sequence number seen on this stock's own messages, as a running maximum
        double run = nan_;
        for (std::size_t i = 0; i < no; ++i)
            if (std::isfinite(O[O_SEQ][i])) {
                run = std::fmax(run, O[O_SEQ][i]);
                tq.push_back(to[i]);
                sq.push_back(run);
            }
        for (std::size_t j = 0; j < nt; ++j) {
            const double x = T[j];
            if (nf > 2) {
                out["marketable_order_amount_share_60s"][j] = win(c_m, to, x, 60) / win(c_new, to, x, 60);
                out["order_price_aggressiveness_mean_900s"][j] = win(c_ag, to, x, 900) / win(c_la, to, x, 900);
            }
            out["log_new_order_count_per_s_60s"][j] = std::log1p(win(c_cnt, to, x, 60) / 60);
            if (tq.size() > 3)
                for (int w : {60, 900}) {
                    const double adv = at(tq, sq, x) - at(tq, sq, x - w);
                    out["log_channel_sequence_advance_per_s_" + std::to_string(w) + "s"][j] = std::log1p(clip(adv, 0, inf_) / w);
                }
        }
    }

    // ---- yesterday's state (ic_score.seed_feats) ----
    {
        vec tp, pp;
        for (std::size_t i = 0; i < nf; ++i)
            if (std::isfinite(pos[i]) && pos[i] > 0) {
                tp.push_back(F[F_T][i]);
                pp.push_back(pos[i]);
            }
        if (tp.size() >= 3) {
            const double hi = st.seed[SD_EG_HIGH], lo = st.seed[SD_EG_LOW], pk = st.seed[SD_PK_LOW];
            for (std::size_t j = 0; j < nt; ++j) {
                const double p0 = at(tp, pp, T[j]);
                if (std::isfinite(hi) && std::isfinite(lo)) out["sd_eg_rng"][j] = (hi - lo) / p0;
                if (std::isfinite(pk) && pk > 0) out["sd_lp_pklo"][j] = std::log(p0 / pk);
            }
        }
    }
    for (auto& row : out.v)
        for (double& x : row)
            if (!std::isfinite(x)) x = nan_;
}

template <typename X>
void rd(std::FILE* f, X* p, std::size_t n) {
    if (std::fread(p, sizeof(X), n, f) != n) throw std::runtime_error("short read");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: featgen <in.bin> <out.bin> [--names names.txt] [--threads N]");
        std::string names_path;
        unsigned threads = 8;
        for (int i = 3; i < argc; ++i)
            if (!std::strcmp(argv[i], "--simb-exact")) simb_exact = true;
        for (int i = 3; i + 1 < argc; ++i) {
            if (!std::strcmp(argv[i], "--names")) names_path = argv[i + 1];
            if (!std::strcmp(argv[i], "--threads")) threads = static_cast<unsigned>(std::max(1, std::atoi(argv[i + 1])));
        }
        std::FILE* f = std::fopen(argv[1], "rb");
        if (!f) throw std::runtime_error(std::string("cannot open ") + argv[1]);
        char magic[8];
        std::uint64_t n_stock, n_t;
        rd(f, magic, 8);
        if (std::memcmp(magic, "CRUFG001", 8)) throw std::runtime_error("not a featgen input file");
        rd(f, &n_stock, 1);
        rd(f, &n_t, 1);
        vec T(n_t);
        rd(f, T.data(), n_t);
        std::vector<stock> stocks(n_stock);
        for (auto& st : stocks) {
            std::uint64_t n[3];
            rd(f, &st.sid, 1);
            rd(f, n, 3);
            st.nf = n[0], st.ns = n[1], st.no = n[2];
            rd(f, st.seed, NSEED);
            auto cols = [&](std::vector<vec>& c, std::size_t k, std::size_t rows) {
                c.assign(k, vec(rows));
                for (auto& col : c)
                    if (rows) rd(f, col.data(), rows);
            };
            cols(st.f, NF, st.nf);
            cols(st.s, NS, st.ns);
            cols(st.o, NO, st.no);
        }
        std::fclose(f);
        std::vector<result> res(n_stock, result(n_t));
        std::vector<std::thread> pool;
        std::vector<std::string> err(threads);
        for (unsigned w = 0; w < threads; ++w)
            pool.emplace_back([&, w] {
                try {
                    for (std::size_t i = w; i < n_stock; i += threads) compute(stocks[i], T, res[i]);
                } catch (const std::exception& e) {
                    err[w] = e.what();
                }
            });
        for (auto& th : pool) th.join();
        for (auto& e : err)
            if (!e.empty()) throw std::runtime_error(e);
        std::FILE* g = std::fopen(argv[2], "wb");
        if (!g) throw std::runtime_error(std::string("cannot write ") + argv[2]);
        const std::uint64_t nb = n_base;
        std::fwrite("CRUFO001", 1, 8, g);
        std::fwrite(&n_stock, 8, 1, g);
        std::fwrite(&n_t, 8, 1, g);
        std::fwrite(&nb, 8, 1, g);
        for (std::size_t i = 0; i < n_stock; ++i) {
            std::fwrite(&stocks[i].sid, 8, 1, g);
            for (auto& row : res[i].v) std::fwrite(row.data(), 8, n_t, g);
        }
        std::fclose(g);
        if (!names_path.empty()) {
            std::FILE* h = std::fopen(names_path.c_str(), "w");
            if (!h) throw std::runtime_error("cannot write " + names_path);
            for (auto nm : names) std::fprintf(h, "%s\n", nm);
            std::fclose(h);
        }
        std::printf("featgen: %llu stocks x %llu times x %zu base quantities\n", static_cast<unsigned long long>(n_stock),
                    static_cast<unsigned long long>(n_t), n_base);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "featgen: %s\n", e.what());
        return 1;
    }
}
