// lgbpred: PM's LightGBM text models, predicted without lightgbm -- a port of ic_score.TreeModel.predict, tree for tree
// and in the same order of additions, so the scores are the same doubles.
//
// The raw score of a row is the sum of one leaf value per tree (the model's shrinkage and its average start are already
// inside the leaf values); several model files are averaged. A NaN feature follows each split's own missing rule.
//
//   lgbpred <x.bin> <out.bin> <model.txt> [<model.txt> ...] [--threads N]
// x.bin  : "CRULX001", u64 n_rows, u64 n_cols, f64 x[n_rows][n_cols]   (columns in the model's feature order, NaN = missing)
// out.bin: "CRULP001", u64 n_rows, f64 score[n_rows]
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct tree {
    std::vector<int> feat, dt, left, right, cat_b;
    std::vector<double> thr, leaf;
    std::vector<std::int64_t> cat_t;
};

template <typename X>
std::vector<X> nums(const std::string& s) {
    std::vector<X> out;
    const char* p = s.c_str();
    char* end = nullptr;
    for (;;) {
        while (*p == ' ' || *p == '\t') ++p;
        if (!*p) break;
        const double v = std::strtod(p, &end);  // the same decimal -> double rounding as Python's float()
        if (end == p) throw std::runtime_error("cannot read a number in: " + s.substr(0, 60));
        out.push_back(static_cast<X>(v));
        p = end;
    }
    return out;
}

std::vector<std::int64_t> ints64(const std::string& s) {
    std::vector<std::int64_t> out;
    std::istringstream ss(s);
    std::int64_t v;
    while (ss >> v) out.push_back(v);
    return out;
}

// the trees of one model file: the blocks between "Tree=<n>" lines, up to "end of trees"
std::vector<tree> load(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<tree> trees;
    std::string line;
    bool inside = false;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.rfind("end of trees", 0) == 0) break;
        if (line.rfind("Tree=", 0) == 0) {
            trees.emplace_back();
            inside = true;
            continue;
        }
        if (!inside) continue;
        const auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string k = line.substr(0, eq), v = line.substr(eq + 1);
        tree& t = trees.back();
        if (k == "leaf_value") t.leaf = nums<double>(v);
        else if (k == "split_feature") t.feat = nums<int>(v);
        else if (k == "threshold") t.thr = nums<double>(v);
        else if (k == "decision_type") t.dt = nums<int>(v);
        else if (k == "left_child") t.left = nums<int>(v);
        else if (k == "right_child") t.right = nums<int>(v);
        else if (k == "cat_boundaries") t.cat_b = nums<int>(v);
        else if (k == "cat_threshold") t.cat_t = ints64(v);
    }
    if (trees.empty()) throw std::runtime_error("no trees in " + path);
    return trees;
}

double score(const std::vector<tree>& trees, const double* x) {
    double out = 0.0;
    for (const tree& t : trees) {
        if (t.feat.empty()) {
            out += t.leaf[0];
            continue;
        }
        int n = 0;
        for (;;) {
            const double v = x[t.feat[n]];
            const int dt = t.dt[n], default_left = (dt >> 1) & 1, missing = (dt >> 2) & 3;  // missing: 0 none, 1 zero, 2 nan
            const bool isnan = std::isnan(v);
            const double xz = (isnan && missing != 2) ? 0.0 : v;  // LightGBM turns NaN into 0 unless NaN is the missing type
            bool go_left = xz <= t.thr[n];
            if ((missing == 2 && isnan) || (missing == 1 && std::fabs(xz) <= 1e-35)) go_left = default_left == 1;
            if (dt & 1) {  // categorical split: the threshold is an index into the bitset table
                const int ci = static_cast<int>(t.thr[n]);
                const long long xi = isnan ? -1 : static_cast<long long>(v);
                const long long word = t.cat_b[ci] + (xi >= 0 ? xi / 32 : 0);
                const bool ok = xi >= 0 && word < t.cat_b[ci + 1];
                go_left = ok && ((t.cat_t[static_cast<std::size_t>(word)] >> (xi % 32)) & 1) == 1;
            }
            const int nxt = go_left ? t.left[n] : t.right[n];
            if (nxt < 0) {
                out += t.leaf[static_cast<std::size_t>(~nxt)];
                break;
            }
            n = nxt;
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        unsigned threads = 8;
        std::vector<std::string> files;
        for (int i = 3; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
                threads = static_cast<unsigned>(std::max(1, std::atoi(argv[++i])));
            } else {
                files.emplace_back(argv[i]);
            }
        }
        if (argc < 4 || files.empty()) throw std::runtime_error("usage: lgbpred <x.bin> <out.bin> <model.txt> [...] [--threads N]");
        std::vector<std::vector<tree>> models;
        for (auto& f : files) models.push_back(load(f));
        std::FILE* f = std::fopen(argv[1], "rb");
        if (!f) throw std::runtime_error(std::string("cannot open ") + argv[1]);
        char magic[8];
        std::uint64_t n_rows, n_cols;
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "CRULX001", 8)) throw std::runtime_error("not an lgbpred input file");
        if (std::fread(&n_rows, 8, 1, f) != 1 || std::fread(&n_cols, 8, 1, f) != 1) throw std::runtime_error("short read");
        std::vector<double> x(n_rows * n_cols);
        if (!x.empty() && std::fread(x.data(), 8, x.size(), f) != x.size()) throw std::runtime_error("short read");
        std::fclose(f);
        for (auto& m : models)
            for (auto& t : m)
                for (int c : t.feat)
                    if (c < 0 || static_cast<std::uint64_t>(c) >= n_cols) throw std::runtime_error("a tree splits on a column the matrix does not have");
        std::vector<double> out(n_rows);
        std::vector<std::thread> pool;
        for (unsigned w = 0; w < threads; ++w)
            pool.emplace_back([&, w] {
                for (std::uint64_t r = w; r < n_rows; r += threads) {
                    double s = 0.0;  // np.mean over the models: their scores added in order, then divided
                    for (auto& m : models) s += score(m, &x[r * n_cols]);
                    out[r] = s / static_cast<double>(models.size());
                }
            });
        for (auto& th : pool) th.join();
        std::FILE* g = std::fopen(argv[2], "wb");
        if (!g) throw std::runtime_error(std::string("cannot write ") + argv[2]);
        std::fwrite("CRULP001", 1, 8, g);
        std::fwrite(&n_rows, 8, 1, g);
        std::fwrite(out.data(), 8, n_rows, g);
        std::fclose(g);
        std::size_t nt = 0;
        for (auto& m : models) nt += m.size();
        std::printf("lgbpred: %llu rows x %llu columns, %zu model files, %zu trees\n", static_cast<unsigned long long>(n_rows),
                    static_cast<unsigned long long>(n_cols), models.size(), nt);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "lgbpred: %s\n", e.what());
        return 1;
    }
}
