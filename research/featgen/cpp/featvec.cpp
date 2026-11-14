// featvec: the model columns from the base quantities -- the formula layer of ic_score (Pool.value, xs, demean,
// neutralize, calibrate, FM.evaluate on state atoms), for every time of the grid.
//
// A formula (research/engine/baseline_r41c.txt) is
//   [cal:][z:|rank:|crank:|srank:|dm:|dmi:|neu:][|][-]gen.<base>[|][*1e4]     a base quantity, optionally transformed
//   an arithmetic expression over state names (zzup.pf / zzup.ps)              the engine's state at the time
// Per time, across the stocks: z = z-score, rank in [0, 1], crank in [-0.5, 0.5], srank in [-1, 1] (average ranks),
// dm = minus the mean of all stocks, dmi = minus the Barra industry's mean (groups under 3 give NaN), neu = residual of a
// least-squares fit on [1, exposures]. cal: maps the column's mid-rank ECDF over ALL times and stocks of the run through
// PM's training-CDF points (thresholds t with left shares p), so it needs the whole run: a second pass.
//
//   featvec <in.bin> <values.txt> <formulas.txt> <cal.txt> <out.bin> [--threads N]
// in.bin : "CRUFV001", u64 n_stock, n_T, n_val, n_expo; i64 sid[n_stock]; i64 industry[n_stock] (-1 = none);
//          f64 expo[n_stock][n_expo] (NaN row = no CED row); f64 val[n_stock][n_val][n_T]
// values.txt : n_val names, one per line (gen.lpe50, zzup.pf, px ...)
// formulas.txt : "<column> <formula>" per line, '#' comments;  cal.txt : "<column> n t1 p1 ... tn pn" per line
// out.bin: "CRUFX001", u64 n_col, n_T, n_stock; f64 x[n_col][n_T][n_stock]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

using vec = std::vector<double>;
constexpr double nan_ = std::numeric_limits<double>::quiet_NaN();

struct panel {
    std::size_t ns = 0, nt = 0, nv = 0, ne = 0;
    std::vector<std::int64_t> sid, ind;
    vec expo;  // [ns][ne]
    vec val;   // [ns][nv][nt]
    std::map<std::string, std::size_t> name;
    double at(std::size_t v, std::size_t s, std::size_t t) const { return val[(s * nv + v) * nt + t]; }
};

// ---- arithmetic over state names: + - * / and parentheses ----
struct expr_eval {
    const panel& p;
    const std::string& src;
    std::size_t pos = 0, s = 0, t = 0;
    void skip() {
        while (pos < src.size() && src[pos] == ' ') ++pos;
    }
    double atom() {
        skip();
        if (pos < src.size() && src[pos] == '(') {
            ++pos;
            const double v = sum();
            skip();
            if (pos >= src.size() || src[pos] != ')') throw std::runtime_error("missing ) in " + src);
            ++pos;
            return v;
        }
        if (pos < src.size() && src[pos] == '-') {
            ++pos;
            return -atom();
        }
        std::size_t b = pos;
        while (pos < src.size() && (std::isalnum(static_cast<unsigned char>(src[pos])) || src[pos] == '.' || src[pos] == '_')) ++pos;
        const std::string tok = src.substr(b, pos - b);
        if (tok.empty()) throw std::runtime_error("cannot read " + src);
        auto it = p.name.find(tok);
        if (it != p.name.end()) return p.at(it->second, s, t);
        char* end = nullptr;
        const double c = std::strtod(tok.c_str(), &end);
        if (end && *end == 0) return c;
        throw std::runtime_error("unknown name " + tok + " in " + src);
    }
    double prod() {
        double v = atom();
        for (;;) {
            skip();
            if (pos < src.size() && src[pos] == '*') {
                ++pos;
                v = v * atom();
            } else if (pos < src.size() && src[pos] == '/') {
                ++pos;
                v = v / atom();
            } else {
                return v;
            }
        }
    }
    double sum() {
        double v = prod();
        for (;;) {
            skip();
            if (pos < src.size() && src[pos] == '+') {
                ++pos;
                v = v + prod();
            } else if (pos < src.size() && src[pos] == '-') {
                ++pos;
                v = v - prod();
            } else {
                return v;
            }
        }
    }
};

struct formula {
    std::string column, text, pre, base;
    bool cal = false, pool = false, neg = false, ab = false;
    double scale = 1.0;
};

bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }

formula parse(const std::string& column, std::string e) {
    formula f;
    f.column = column;
    f.text = e;
    if (starts(e, "cal:")) {
        f.cal = true;
        e = e.substr(4);
    }
    f.pool = e.find("gen.") != std::string::npos || e.find("ced.") != std::string::npos;
    if (!f.pool) {
        f.base = e;
        return f;
    }
    for (const char* p : {"z:", "rank:", "crank:", "srank:", "dm:", "dmi:", "neu:"})
        if (starts(e, p)) {
            f.pre = p;
            e = e.substr(std::strlen(p));
            break;
        }
    if (e.size() > 4 && e.compare(e.size() - 4, 4, "*1e4") == 0) {
        f.scale = 1e4;
        e = e.substr(0, e.size() - 4);
    }
    if (e.size() > 1 && e.front() == '|' && e.back() == '|') {
        f.ab = true;
        e = e.substr(1, e.size() - 2);
    }
    if (!e.empty() && e.front() == '-') {
        f.neg = true;
        e = e.substr(1);
    }
    f.base = e;
    return f;
}

// one time's values across the stocks -> the transform (ic_score.xs / demean / neutralize)
void cross_section(const formula& f, const panel& p, double* v) {
    const std::size_t n = p.ns;
    std::vector<std::size_t> fin;
    for (std::size_t i = 0; i < n; ++i)
        if (std::isfinite(v[i])) fin.push_back(i);
    auto all_nan = [&] { std::fill(v, v + n, nan_); };
    if (f.pre == "z:" || f.pre == "rank:" || f.pre == "crank:" || f.pre == "srank:") {
        if (fin.size() < 2) return all_nan();
        const double m = static_cast<double>(fin.size());
        if (f.pre == "z:") {
            double mean = 0;
            for (auto i : fin) mean += v[i];
            mean /= m;
            double var = 0;
            for (auto i : fin) var += (v[i] - mean) * (v[i] - mean);
            const double sd = std::sqrt(var / m);
            for (std::size_t i = 0; i < n; ++i) v[i] = (std::isfinite(v[i]) && sd > 0) ? (v[i] - mean) / sd : nan_;
            return;
        }
        std::vector<std::size_t> o = fin;
        std::sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return v[a] < v[b]; });
        vec r(n, nan_);
        for (std::size_t a = 0; a < o.size();) {  // average ranks over ties, 1-based
            std::size_t b = a;
            while (b + 1 < o.size() && v[o[b + 1]] == v[o[a]]) ++b;
            const double avg = (static_cast<double>(a) + static_cast<double>(b)) / 2 + 1;
            for (std::size_t k = a; k <= b; ++k) r[o[k]] = (avg - 1) / (m - 1);
            a = b + 1;
        }
        for (std::size_t i = 0; i < n; ++i) v[i] = f.pre == "crank:" ? r[i] - 0.5 : (f.pre == "srank:" ? 2 * r[i] - 1 : r[i]);
        return;
    }
    if (f.pre == "dm:" || f.pre == "dmi:") {
        std::map<std::int64_t, std::pair<double, std::size_t>> g;
        auto group = [&](std::size_t i) { return f.pre == "dmi:" ? p.ind[i] : std::int64_t{0}; };
        for (auto i : fin)
            if (group(i) >= 0) {
                auto& a = g[group(i)];
                a.first += v[i];
                ++a.second;
            }
        for (std::size_t i = 0; i < n; ++i) {
            const bool ok = std::isfinite(v[i]) && group(i) >= 0 && g[group(i)].second >= 3;
            v[i] = ok ? v[i] - g[group(i)].first / static_cast<double>(g[group(i)].second) : nan_;
        }
        return;
    }
    if (f.pre == "neu:") {
        if (p.ne == 0) return all_nan();
        std::vector<std::size_t> rows;
        for (auto i : fin) {
            bool ok = true;
            for (std::size_t k = 0; k < p.ne && ok; ++k) ok = std::isfinite(p.expo[i * p.ne + k]);
            if (ok) rows.push_back(i);
        }
        const std::size_t m = rows.size(), k = p.ne + 1;
        if (m < k + 10) return all_nan();
        // residual of v on span([1, E]): modified Gram-Schmidt, dependent columns (industry dummies sum to 1) dropped
        std::vector<vec> q;
        vec res(m);
        for (std::size_t r = 0; r < m; ++r) res[r] = v[rows[r]];
        for (std::size_t c = 0; c < k; ++c) {
            vec x(m);
            double n0 = 0;
            for (std::size_t r = 0; r < m; ++r) {
                x[r] = c ? p.expo[rows[r] * p.ne + c - 1] : 1.0;
                n0 += x[r] * x[r];
            }
            for (int pass = 0; pass < 2; ++pass)
                for (auto& u : q) {
                    double d = 0;
                    for (std::size_t r = 0; r < m; ++r) d += u[r] * x[r];
                    for (std::size_t r = 0; r < m; ++r) x[r] -= d * u[r];
                }
            double n1 = 0;
            for (std::size_t r = 0; r < m; ++r) n1 += x[r] * x[r];
            if (!(n1 > 1e-20 * std::max(n0, 1e-300))) continue;
            const double inv = 1 / std::sqrt(n1);
            for (auto& e : x) e *= inv;
            q.push_back(std::move(x));
        }
        for (int pass = 0; pass < 2; ++pass)
            for (auto& u : q) {
                double d = 0;
                for (std::size_t r = 0; r < m; ++r) d += u[r] * res[r];
                for (std::size_t r = 0; r < m; ++r) res[r] -= d * u[r];
            }
        all_nan();
        for (std::size_t r = 0; r < m; ++r) v[rows[r]] = res[r];
    }
}

// ic_score.calibrate over the whole column
void calibrate(vec& x, const vec& t_in, const vec& p_in) {
    std::vector<std::size_t> o(t_in.size());
    for (std::size_t i = 0; i < o.size(); ++i) o[i] = i;
    std::stable_sort(o.begin(), o.end(), [&](std::size_t a, std::size_t b) { return t_in[a] < t_in[b]; });
    vec t(o.size()), p(o.size());
    double run = -std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < o.size(); ++i) {
        t[i] = t_in[o[i]];
        run = std::max(run, p_in[o[i]]);
        p[i] = run;
    }
    vec s;
    for (double v : x)
        if (std::isfinite(v)) s.push_back(v);
    if (s.size() < 2 || t.size() < 2) {
        std::fill(x.begin(), x.end(), nan_);
        return;
    }
    std::sort(s.begin(), s.end());
    const double two_n = 2.0 * static_cast<double>(s.size());
    for (double& v : x) {
        if (!std::isfinite(v)) {
            v = nan_;
            continue;
        }
        const double u = static_cast<double>((std::lower_bound(s.begin(), s.end(), v) - s.begin()) + (std::upper_bound(s.begin(), s.end(), v) - s.begin())) / two_n;
        if (u <= p.front()) {  // np.interp: clamped at the ends
            v = t.front();
        } else if (u >= p.back()) {
            v = t.back();
        } else {
            const std::size_t j = static_cast<std::size_t>(std::upper_bound(p.begin(), p.end(), u) - p.begin()) - 1;
            v = (t[j + 1] - t[j]) / (p[j + 1] - p[j]) * (u - p[j]) + t[j];
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
        if (argc < 6) throw std::runtime_error("usage: featvec <in.bin> <values.txt> <formulas.txt> <cal.txt> <out.bin> [--threads N]");
        unsigned threads = 8;
        for (int i = 6; i + 1 < argc; i += 2)
            if (!std::strcmp(argv[i], "--threads")) threads = static_cast<unsigned>(std::max(1, std::atoi(argv[i + 1])));
        panel p;
        std::FILE* f = std::fopen(argv[1], "rb");
        if (!f) throw std::runtime_error(std::string("cannot open ") + argv[1]);
        char magic[8];
        std::uint64_t h[4];
        rd(f, magic, 8);
        if (std::memcmp(magic, "CRUFV001", 8)) throw std::runtime_error("not a featvec input file");
        rd(f, h, 4);
        p.ns = h[0], p.nt = h[1], p.nv = h[2], p.ne = h[3];
        p.sid.resize(p.ns);
        p.ind.resize(p.ns);
        p.expo.resize(p.ns * p.ne);
        p.val.resize(p.ns * p.nv * p.nt);
        rd(f, p.sid.data(), p.ns);
        rd(f, p.ind.data(), p.ns);
        rd(f, p.expo.data(), p.expo.size());
        rd(f, p.val.data(), p.val.size());
        std::fclose(f);
        {
            std::ifstream in(argv[2]);
            std::string s;
            while (in >> s) p.name.emplace(s, p.name.size());
            if (p.name.size() != p.nv) throw std::runtime_error("values.txt does not match the input file");
        }
        std::vector<formula> fs;
        {
            std::ifstream in(argv[3]);
            std::string line;
            while (std::getline(in, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (line.empty() || line[0] == '#') continue;
                const auto sp = line.find(' ');
                if (sp == std::string::npos) throw std::runtime_error("bad formula line: " + line);
                fs.push_back(parse(line.substr(0, sp), line.substr(sp + 1)));
            }
        }
        std::map<std::string, std::pair<vec, vec>> cal;
        {
            std::ifstream in(argv[4]);
            std::string line;
            while (std::getline(in, line)) {
                std::istringstream ss(line);
                std::string col;
                std::size_t n = 0;
                if (!(ss >> col >> n)) continue;
                auto& c = cal[col];
                c.first.resize(n);
                c.second.resize(n);
                for (std::size_t i = 0; i < n; ++i) ss >> c.first[i] >> c.second[i];
            }
        }
        for (auto& fm : fs)
            if (fm.pool && !p.name.count(fm.base)) throw std::runtime_error("column " + fm.column + ": no base " + fm.base);
        const std::size_t nc = fs.size(), cells = p.nt * p.ns;
        std::vector<vec> out(nc, vec(cells, nan_));
        std::vector<std::thread> pool;
        std::vector<std::string> err(threads);
        for (unsigned w = 0; w < threads; ++w)
            pool.emplace_back([&, w] {
                try {
                    for (std::size_t c = w; c < nc; c += threads) {
                        const formula& fm = fs[c];
                        vec& x = out[c];
                        const std::size_t b = fm.pool ? p.name.at(fm.base) : 0;
                        for (std::size_t t = 0; t < p.nt; ++t) {
                            double* row = &x[t * p.ns];
                            for (std::size_t s = 0; s < p.ns; ++s) {
                                if (fm.pool) {
                                    double v = p.at(b, s, t);
                                    if (fm.ab) v = std::fabs(v);
                                    row[s] = (fm.neg ? -v : v) * fm.scale;
                                } else {
                                    expr_eval ev{p, fm.base, 0, s, t};
                                    const double v = ev.sum();
                                    row[s] = std::isfinite(v) ? v : nan_;
                                }
                            }
                            if (fm.pool && !fm.pre.empty()) cross_section(fm, p, row);
                        }
                        if (fm.cal) {
                            auto it = cal.find(fm.column);
                            if (it != cal.end()) calibrate(x, it->second.first, it->second.second);  // no points: left as is
                        }
                    }
                } catch (const std::exception& e) {
                    err[w] = e.what();
                }
            });
        for (auto& th : pool) th.join();
        for (auto& e : err)
            if (!e.empty()) throw std::runtime_error(e);
        std::FILE* g = std::fopen(argv[5], "wb");
        if (!g) throw std::runtime_error(std::string("cannot write ") + argv[5]);
        const std::uint64_t o[3] = {nc, p.nt, p.ns};
        std::fwrite("CRUFX001", 1, 8, g);
        std::fwrite(o, 8, 3, g);
        for (auto& x : out) std::fwrite(x.data(), 8, cells, g);
        std::fclose(g);
        std::printf("featvec: %zu columns x %zu times x %zu stocks\n", nc, p.nt, p.ns);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "featvec: %s\n", e.what());
        return 1;
    }
}
