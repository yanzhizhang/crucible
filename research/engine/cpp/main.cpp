// The engine binary (C++20): events.bin + rules.tsv (+ prev.bin) -> the day-end state as JSON, one thread per stock chunk.
//   engine <events.bin> <rules.tsv> <out.json> [--prev prev.bin] [--threads N]
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include "event.hpp"
#include "rules.hpp"
#include "state.hpp"

namespace {

using namespace eng;

// prev.bin: 16-byte header "CRUPV001" + count, then (sid, rule index, sum, wgt) records.
struct prev_record {
    std::int32_t sid, rule;
    double sum, wgt;
};
static_assert(sizeof(prev_record) == 24, "prev record layout");

struct mapped {
    const char* p = nullptr;
    std::size_t n = 0;
    explicit mapped(const char* path) {
        int fd = ::open(path, O_RDONLY);
        if (fd < 0) throw std::runtime_error(std::string("cannot open ") + path);
        struct stat st {};
        ::fstat(fd, &st);
        n = static_cast<std::size_t>(st.st_size);
        p = static_cast<const char*>(::mmap(nullptr, n, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0));
        ::close(fd);
        if (p == MAP_FAILED) throw std::runtime_error(std::string("cannot mmap ") + path);
    }
    ~mapped() { if (p) ::munmap(const_cast<char*>(p), n); }
};

std::vector<rule> read_rules(const char* path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("cannot open ") + path);
    std::vector<rule> out;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        rule r;
        std::string clock;
        if (!(std::getline(ss, r.name, '\t') && std::getline(ss, clock, '\t') && (ss >> r.half) && ss.ignore() &&
              std::getline(ss, r.what, '\t') && (ss >> r.carry)))
            throw std::runtime_error("bad rules line: " + line);
        std::string base = clock.substr(0, clock.find('_'));  // "snap" / "trade" / "both" [+ "_wall" / "_exch"]
        r.mod = clock.find('_') == std::string::npos ? "" : clock.substr(clock.find('_') + 1);
        r.clock = base == "trade" ? 1 : base == "both" ? 2 : base == "bothn" ? 3 : 0;  // bothn: both events, each its own value
        out.push_back(r);
    }
    return out;
}

std::unordered_map<std::int32_t, std::vector<prev_record>> read_prev(const char* path) {
    std::unordered_map<std::int32_t, std::vector<prev_record>> out;
    if (!path) return out;
    mapped m(path);
    if (m.n < 16 || std::memcmp(m.p, "CRUPV001", 8) != 0) throw std::runtime_error("bad prev.bin header");
    std::uint64_t n;
    std::memcpy(&n, m.p + 8, 8);
    auto* rec = reinterpret_cast<const prev_record*>(m.p + 16);
    for (std::uint64_t i = 0; i < n; ++i) out[rec[i].sid].push_back(rec[i]);
    return out;
}

// %.17g round-trips every double exactly; NaN becomes JSON null.
void put(std::string& o, const char* key, double v) {
    o += '"';
    o += key;
    o += "\":";
    if (std::isnan(v)) o += "null";
    else {
        char b[32];
        std::snprintf(b, sizeof b, "%.17g", v);
        o += b;
    }
    o += ',';
}

void put(std::string& o, const std::string& key, double v) { put(o, key.c_str(), v); }

// Stock::dump, under PM's key names.
std::string dump(stock& st, const std::vector<rule>& rules) {
    std::string o;
    o.reserve(1 << 16);
    o += '{';
    for (std::size_t i = 0; i < rules.size(); ++i) {
        put(o, rules[i].name + ".sum", st.emas[i].s);
        put(o, rules[i].name + ".wgt", st.emas[i].w);
        if (st.emas[i].has_t) put(o, rules[i].name + ".t", st.emas[i].t);
    }
    for (const char* k : {"zzpk.high", "zzvx.high", "zzxu.high"}) put(o, k, st.high);
    for (const char* k : {"zzpk.low", "zzvx.low", "zzxu.low"}) put(o, k, st.low);
    put(o, "zzpk.open", st.open);
    for (const char* k : {"zzxu.last", "zzxo.last", "cdc.lp"}) put(o, k, st.last);
    put(o, "zzqj.v", st.vol);
    put(o, "zzvm.c.v", st.vol);
    put(o, "zzqj.tpv", st.amt);
    put(o, "zzvm.c.tpv", st.amt);
    put(o, "zzgh.nb", st.buy + st.sell ? (st.buy - st.sell) / (st.buy + st.sell) : qnan);
    const std::pair<const char*, const double*> nbs[] = {{"zzgh.nb|all", st.nb_all}, {"zzgh.nb|volall", st.nb_vall}, {"zzgh.nb|vol", st.nb_vol},
                                                         {"zzgh.nb|to1500", st.nb_1500}, {"zzgh.nb|open", st.nb_open},
                                                         {"zzgh.nb|tick", st.nb_tick}, {"zzgh.nb|mid", st.nb_mid}, {"zzgh.nb|loc", st.nb_loc}};
    for (const auto& [k, v] : nbs) put(o, k, v[0] + v[1] ? (v[0] - v[1]) / (v[0] + v[1]) : qnan);
    put(o, "zzgh.nb|den", st.nb_den ? (st.buy - st.sell) / st.nb_den : qnan);
    put(o, "zzmh.hm", st.hm);
    for (int k = 0; k < n_xo; ++k) put(o, "zzxo.r." + std::to_string(k), st.xo[k]);
    for (int k = 0; k < n_ui; ++k) put(o, "zzui.r." + std::to_string(k), st.ui[k]);
    for (int k = 0; k < n_hj; ++k) put(o, "zzhj.c." + std::to_string(k), st.hj[k]);
    const auto& fb = st.zzfb.close();
    for (std::size_t k = 0; k < fb.size(); ++k) put(o, "zzfb.c." + std::to_string(k), fb[k]);
    const auto& jb = st.jb_px.close();
    for (std::size_t k = 0; k < jb.size(); ++k) {
        double px = jb[k];
        bool drop = !std::isnan(px) && !std::isnan(st.pc) && px <= st.pc * (1 - jb_drop);
        put(o, "zzjb.c." + std::to_string(k), drop ? 0.0 : 4.0);
    }
    for (int k = 0; k < n_gh; ++k) {
        double b = st.gh_buy[k], s = st.gh_sell[k];
        int code = gh_empty;
        if (b + s > 0) {
            double x = (b - s) / (b + s);
            code = 0;
            for (double e : gh_edges) code += e <= x;  // searchsorted right: a value on an edge goes up
        }
        put(o, "zzgh.c." + std::to_string(k), static_cast<double>(code));
    }
    o.back() = '}';
    return o;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: engine <events.bin> <rules.tsv> <out.json> [--prev prev.bin] [--threads N]\n");
        return 2;
    }
    const char* prev_path = nullptr;
    int threads = static_cast<int>(std::thread::hardware_concurrency());
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--prev") && i + 1 < argc) prev_path = argv[++i];
        else if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) threads = std::atoi(argv[++i]);
    }
    auto t0 = std::chrono::steady_clock::now();
    std::vector<rule> rules = read_rules(argv[2]);
    compiled_rules cr(rules);
    auto prev = read_prev(prev_path);
    mapped m(argv[1]);
    if (m.n < sizeof(file_header) || std::memcmp(m.p, magic, 8) != 0) {
        std::fprintf(stderr, "bad events header\n");
        return 1;
    }
    file_header h;
    std::memcpy(&h, m.p, sizeof h);
    if (m.n != sizeof h + h.n_events * sizeof(event)) {
        std::fprintf(stderr, "events file size mismatch\n");
        return 1;
    }
    auto* ev = reinterpret_cast<const event*>(m.p + sizeof h);
    // Stocks are contiguous in the file: one (begin, end) range each.
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (std::uint64_t i = 0; i < h.n_events;) {
        std::uint64_t j = i;
        while (j < h.n_events && ev[j].sid == ev[i].sid) ++j;
        ranges.emplace_back(i, j);
        i = j;
    }
    std::vector<std::string> out(ranges.size());
    std::atomic<std::size_t> next{0};
    auto work = [&] {
        for (std::size_t r; (r = next.fetch_add(1)) < ranges.size();) {
            stock st(rules);
            auto it = prev.find(ev[ranges[r].first].sid);
            if (it != prev.end())
                for (const auto& p : it->second) st.seed(p.rule, rules[p.rule].carry, p.sum, p.wgt);
            for (std::uint64_t i = ranges[r].first; i < ranges[r].second; ++i) {
                const event& e = ev[i];
                if (e.kind == 0) on_snap(st, cr, e.t, e.s, e.lsod, e.price, e.f[0], e.f[1], e.f[2], e.f[3], e.f[4], e.f[5], e.f[6]);
                else on_trade(st, cr, e.t, e.s, e.price, e.f[0], e.f[1]);
            }
            out[r] = dump(st, rules);
        }
    };
    if (threads < 1) threads = 1;
    std::vector<std::thread> pool;
    for (int i = 1; i < threads; ++i) pool.emplace_back(work);
    work();
    for (auto& t : pool) t.join();
    std::ofstream f(argv[3], std::ios::binary);
    f << '{';
    for (std::size_t r = 0; r < ranges.size(); ++r) {
        if (r) f << ',';
        f << '"' << ev[ranges[r].first].sid << "\":" << out[r];
    }
    f << "}\n";
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("engine: %zu stocks, %llu events, %zu rules, %d threads, %.2fs\n", ranges.size(),
                static_cast<unsigned long long>(h.n_events), rules.size(), threads, sec);
    return 0;
}
