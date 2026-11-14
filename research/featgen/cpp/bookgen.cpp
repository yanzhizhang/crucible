// bookgen: the whole-order-book replay behind gen.best_level_order_count_imbalance -- a port of bt.engine's book
// (_apply_sse / _apply_szse and what they call) and of full_book.book_at's best-level reading.
//
// Per stock the order and trade events come in exchange sequence order (bt.events' cache); the book is read after the last
// event that had ARRIVED by each time of the grid (running maximum of the arrival stamps <= the time's stamp).
// The reading: resting orders at the best bid / best ask, (bid - ask) / (bid + ask); NaN while one side is empty.
// The per-level order counts are kept as the events apply (full_book scans every order slot at every mark instead).
//
//   bookgen <in.bin> <out.bin> [--threads N]
// in.bin : "CRUBK001", u64 n_stock, u64 n_T, then per stock
//          i64 sid, i64 venue (0 SSE, 1 SZSE), i64 lo_tick, i64 n_ticks, u64 n_ev, i64 cut[n_T] (arrival ns of each time),
//          i64 kind[n], ats[n], side[n], tick[n], qty[n], id_a[n], id_b[n], otype[n]
//          kind 0 = new order, 1 = cancel, 2 = trade
// out.bin: "CRUBO001", u64 n_stock, u64 n_T, then per stock i64 sid, f64 imb[n_T], f64 bid_cnt[n_T], f64 ask_cnt[n_T],
//          i64 stats[3] (unresolved cancels, unresolved passive fills, out-of-band adds)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace {

using i64 = std::int64_t;
constexpr double nan_ = std::numeric_limits<double>::quiet_NaN();
constexpr i64 mtag_cp_best = 11, mtag_fak = 12, mtag_fok_zero = 15;  // bt.events' SZSE market-order tags

struct stock {
    i64 sid = 0, venue = 0, lo_tick = 0, n_ticks = 0;
    std::vector<i64> cut, kind, ats, side, tick, qty, id_a, id_b, otype;
    std::vector<double> imb, bcnt, acnt;
    i64 stats[3] = {0, 0, 0};
};

struct book {
    i64 n_ticks, best_bid, best_ask, nslot = 0, pkey = -1;
    std::vector<i64> bidv, askv, bidn, askn;  // volume and resting-order count per level
    std::vector<i64> o_side, o_tick, o_rem;
    std::vector<std::int8_t> o_placed;
    std::unordered_map<i64, i64> oid;
    i64* stats;

    book(i64 nt, std::size_t n_ev, i64* st) : n_ticks(nt), best_bid(-1), best_ask(nt), bidv(nt), askv(nt), bidn(nt), askn(nt),
                                             o_side(n_ev + 1), o_tick(n_ev + 1), o_rem(n_ev + 1), o_placed(n_ev + 1), stats(st) {
        oid.reserve(n_ev / 2 + 16);
    }
    i64 slot(i64 key) const {
        auto it = oid.find(key);
        return it == oid.end() ? -1 : it->second;
    }
    void count(i64 s, i64 d) {  // a resting order (placed, size left, level in range) joins or leaves its level's count
        const i64 lv = o_tick[s];
        if (lv >= 0 && lv < n_ticks) (o_side[s] > 0 ? bidn : askn)[lv] += d;
    }
    void level_add(i64 sd, i64 lvl, i64 q) {
        if (sd > 0) {
            bidv[lvl] += q;
            if (lvl > best_bid) best_bid = lvl;
        } else {
            askv[lvl] += q;
            if (lvl < best_ask) best_ask = lvl;
        }
    }
    void take(i64 key, i64 s, i64 q) {
        const i64 lvl = o_tick[s];
        const bool was = o_placed[s] == 1 && o_rem[s] > 0;
        if (o_placed[s] == 1) {
            if (o_side[s] > 0) {
                bidv[lvl] = std::max<i64>(bidv[lvl] - q, 0);
                if (lvl == best_bid && bidv[lvl] == 0)
                    while (best_bid >= 0 && bidv[best_bid] == 0) --best_bid;
            } else {
                askv[lvl] = std::max<i64>(askv[lvl] - q, 0);
                if (lvl == best_ask && askv[lvl] == 0)
                    while (best_ask < n_ticks && askv[best_ask] == 0) ++best_ask;
            }
        }
        o_rem[s] -= q;
        if (was && o_rem[s] <= 0) count(s, -1);
        if (o_rem[s] <= 0) oid.erase(key);
    }
    void new_order(i64 key, i64 sd, i64 lvl, i64 q, std::int8_t placed) {
        oid[key] = nslot;
        o_side[nslot] = sd;
        o_tick[nslot] = lvl;
        o_rem[nslot] = q;
        o_placed[nslot] = placed;
        if (placed == 1 && q > 0) count(nslot, +1);
        ++nslot;
    }
    void cancel_or_trade(i64 k, i64 sd, i64 q0, i64 a, i64 b) {
        for (int j = 0; j < (k == 2 ? 2 : 1); ++j) {
            const i64 key = j == 0 ? a : b;
            const i64 s = slot(key);
            if (s < 0) {
                if (k == 1)
                    ++stats[0];
                else if ((j == 0 && sd < 0) || (j == 1 && sd > 0))
                    ++stats[1];
                continue;
            }
            const i64 q = (q0 > 0 && q0 < o_rem[s]) ? q0 : o_rem[s];
            take(key, s, q);
        }
    }
    void apply_sse(i64 k, i64 sd, i64 tk, i64 q, i64 a, i64 b, i64 lo_tick) {
        if (k == 0) {
            if (sd != 0) {
                const i64 lvl = tk - lo_tick;
                if (lvl < 0 || lvl >= n_ticks) {
                    ++stats[2];
                } else {
                    new_order(a, sd, lvl, q, 1);
                    level_add(sd, lvl, q);
                }
            }
        } else {
            cancel_or_trade(k, sd, q, a, b);
        }
    }
    void apply_szse(i64 k, i64 sd, i64 tk, i64 q, i64 a, i64 b, i64 ot, i64 lo_tick) {
        const bool own = (k == 1 && a == pkey) || (k == 2 && ((sd > 0 && a == pkey) || (sd < 0 && b == pkey)));
        if (pkey >= 0 && !own) {  // the held marketable order's burst is over: its remainder rests now
            const i64 s = slot(pkey);
            if (s >= 0 && o_placed[s] == 0 && o_rem[s] > 0 && o_tick[s] >= 0 && o_tick[s] < n_ticks) {
                o_placed[s] = 1;
                level_add(o_side[s], o_tick[s], o_rem[s]);
                count(s, +1);
            }
            pkey = -1;
        }
        if (k == 0) {
            if (sd != 0) {
                if (ot == 1 || ot >= mtag_cp_best) {  // market order: priced by its fill; fak / fok_zero never rest
                    const bool never = ot == mtag_fak || ot == mtag_fok_zero;
                    new_order(a, sd, -1, q, never ? 2 : 0);
                    pkey = a;
                } else {
                    const i64 lvl = ot == 3 ? (sd > 0 ? best_bid : best_ask) : tk - lo_tick;
                    if (lvl < 0 || lvl >= n_ticks) {
                        ++stats[2];
                    } else {
                        const bool crosses = (sd > 0 && best_ask < n_ticks && lvl >= best_ask) || (sd < 0 && best_bid >= 0 && lvl <= best_bid);
                        new_order(a, sd, lvl, q, crosses ? 0 : 1);
                        if (crosses)
                            pkey = a;
                        else
                            level_add(sd, lvl, q);
                    }
                }
            }
            return;
        }
        if (k == 2)  // a market order learns its price from its fill
            for (int j = 0; j < 2; ++j) {
                const i64 s = slot(j == 0 ? a : b);
                if (s >= 0 && o_placed[s] == 0 && o_tick[s] < 0) o_tick[s] = tk - lo_tick;
            }
        cancel_or_trade(k, sd, q, a, b);
    }
};

void replay(stock& st) {
    const std::size_t n = st.kind.size(), nt = st.cut.size();
    st.imb.assign(nt, nan_);
    st.bcnt.assign(nt, nan_);
    st.acnt.assign(nt, nan_);
    if (n == 0 || st.n_ticks <= 0) return;
    // rec[j] = (number of events whose running-maximum arrival stamp is <= cut[j]) - 1
    std::vector<i64> rec(nt);
    {
        std::vector<i64> cm(n);
        i64 run = std::numeric_limits<i64>::min();
        for (std::size_t i = 0; i < n; ++i) cm[i] = run = std::max(run, st.ats[i]);
        for (std::size_t j = 0; j < nt; ++j) rec[j] = static_cast<i64>(std::upper_bound(cm.begin(), cm.end(), st.cut[j]) - cm.begin()) - 1;
    }
    book b(st.n_ticks, n, st.stats);
    std::size_t r = 0;
    while (r < nt && rec[r] < 0) ++r;
    for (std::size_t i = 0; i < n; ++i) {
        if (st.venue == 1)
            b.apply_szse(st.kind[i], st.side[i], st.tick[i], st.qty[i], st.id_a[i], st.id_b[i], st.otype[i], st.lo_tick);
        else
            b.apply_sse(st.kind[i], st.side[i], st.tick[i], st.qty[i], st.id_a[i], st.id_b[i], st.lo_tick);
        if (r < nt && rec[r] == static_cast<i64>(i)) {
            i64 bb = b.n_ticks - 1, ba = 0;  // the best levels by volume, scanned as full_book does
            while (bb >= 0 && b.bidv[bb] == 0) --bb;
            while (ba < b.n_ticks && b.askv[ba] == 0) ++ba;
            const bool both = bb >= 0 && ba < b.n_ticks;
            const double x = both ? static_cast<double>(b.bidn[bb]) : nan_, y = both ? static_cast<double>(b.askn[ba]) : nan_;
            const double v = (x - y) / (x + y);
            while (r < nt && rec[r] == static_cast<i64>(i)) {
                st.bcnt[r] = x;
                st.acnt[r] = y;
                st.imb[r] = std::isfinite(v) ? v : nan_;
                ++r;
            }
        }
    }
}

template <typename X>
void rd(std::FILE* f, X* p, std::size_t n) {
    if (n && std::fread(p, sizeof(X), n, f) != n) throw std::runtime_error("short read");
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 3) throw std::runtime_error("usage: bookgen <in.bin> <out.bin> [--threads N]");
        unsigned threads = 8;
        for (int i = 3; i + 1 < argc; ++i)
            if (!std::strcmp(argv[i], "--threads")) threads = static_cast<unsigned>(std::max(1, std::atoi(argv[i + 1])));
        std::FILE* f = std::fopen(argv[1], "rb");
        if (!f) throw std::runtime_error(std::string("cannot open ") + argv[1]);
        char magic[8];
        std::uint64_t ns, nt;
        rd(f, magic, 8);
        if (std::memcmp(magic, "CRUBK001", 8)) throw std::runtime_error("not a bookgen input file");
        rd(f, &ns, 1);
        rd(f, &nt, 1);
        std::vector<stock> stocks(ns);
        for (auto& st : stocks) {
            i64 h[4];
            std::uint64_t n;
            rd(f, h, 4);
            rd(f, &n, 1);
            st.sid = h[0], st.venue = h[1], st.lo_tick = h[2], st.n_ticks = h[3];
            st.cut.resize(nt);
            rd(f, st.cut.data(), nt);
            for (auto* c : {&st.kind, &st.ats, &st.side, &st.tick, &st.qty, &st.id_a, &st.id_b, &st.otype}) {
                c->resize(n);
                rd(f, c->data(), n);
            }
        }
        std::fclose(f);
        std::vector<std::thread> pool;
        std::vector<std::string> err(threads);
        for (unsigned w = 0; w < threads; ++w)
            pool.emplace_back([&, w] {
                try {
                    for (std::size_t i = w; i < ns; i += threads) replay(stocks[i]);
                } catch (const std::exception& e) {
                    err[w] = e.what();
                }
            });
        for (auto& th : pool) th.join();
        for (auto& e : err)
            if (!e.empty()) throw std::runtime_error(e);
        std::FILE* g = std::fopen(argv[2], "wb");
        if (!g) throw std::runtime_error(std::string("cannot write ") + argv[2]);
        std::fwrite("CRUBO001", 1, 8, g);
        std::fwrite(&ns, 8, 1, g);
        std::fwrite(&nt, 8, 1, g);
        for (auto& st : stocks) {
            std::fwrite(&st.sid, 8, 1, g);
            std::fwrite(st.imb.data(), 8, nt, g);
            std::fwrite(st.bcnt.data(), 8, nt, g);
            std::fwrite(st.acnt.data(), 8, nt, g);
            std::fwrite(st.stats, 8, 3, g);
        }
        std::fclose(g);
        std::printf("bookgen: %llu stocks x %llu times\n", static_cast<unsigned long long>(ns), static_cast<unsigned long long>(nt));
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bookgen: %s\n", e.what());
        return 1;
    }
}
