// lgbpred: PM's LightGBM text models, predicted without lightgbm -- a port of ic_score.TreeModel.predict, tree for tree
// and in the same order of additions, so the scores are the same doubles.
//
// The raw score of a row is the sum of one leaf value per tree (the model's shrinkage and its average start are already
// inside the leaf values); several model files are averaged. A NaN feature follows each split's own missing rule.
//
//   lgbpred <x.bin> <out.bin> <model.txt> [<model.txt> ...] [--threads N]
//   lgbpred --parquet <vec.parquet> <pred.parquet> --features <names.csv> <model.txt> [...] [--threads N]
//           (only when built with -DCRUCIBLE_ARROW, by CMake inside the conda env, as lgbpred_pq): reads the feature
//           parquet itself (sid, mark, one column per model feature it fills; the others are NaN) and writes sid, mark, pred
// x.bin  : "CRULX001", u64 n_rows, u64 n_cols, f64 x[n_rows][n_cols]   (columns in the model's feature order, NaN = missing)
// out.bin: "CRULP001", u64 n_rows, f64 score[n_rows]
#include <algorithm>
#include <cctype>
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

#ifdef CRUCIBLE_ARROW
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>
#endif

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

// one score per row: np.mean over the models, their scores added in order, then divided
std::vector<double> predict(const std::vector<std::vector<tree>>& models, const std::vector<double>& x, std::uint64_t n_rows, std::uint64_t n_cols,
                            unsigned threads) {
    for (auto& m : models)
        for (auto& t : m)
            for (int c : t.feat)
                if (c < 0 || static_cast<std::uint64_t>(c) >= n_cols) throw std::runtime_error("a tree splits on a column the matrix does not have");
    std::vector<double> out(n_rows);
    std::vector<std::thread> pool;
    for (unsigned w = 0; w < threads; ++w)
        pool.emplace_back([&, w] {
            for (std::uint64_t r = w; r < n_rows; r += threads) {
                double s = 0.0;
                for (auto& m : models) s += score(m, &x[r * n_cols]);
                out[r] = s / static_cast<double>(models.size());
            }
        });
    for (auto& th : pool) th.join();
    return out;
}

#ifdef CRUCIBLE_ARROW
void check(const arrow::Status& s) {
    if (!s.ok()) throw std::runtime_error(s.ToString());
}

// the model's columns in order: the first field of each line of the csv, a "name" header line dropped
std::vector<std::string> feature_names(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> names;
    std::string line;
    while (std::getline(in, line)) {
        std::string k = line.substr(0, line.find(','));
        const auto b = k.find_first_not_of(" \t\r\n"), e = k.find_last_not_of(" \t\r\n");
        if (b == std::string::npos) continue;
        names.push_back(k.substr(b, e - b + 1));
    }
    if (!names.empty()) {
        std::string h = names[0];
        std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (h == "name") names.erase(names.begin());
    }
    return names;
}

// the parquet way: only the filled model columns are held (plus one all-NaN column every other feature points at)
int run_parquet(const std::string& src, const std::string& dst, const std::string& csv, std::vector<std::vector<tree>>& models, unsigned threads) {
    const std::vector<std::string> names = feature_names(csv);
    auto file = arrow::io::ReadableFile::Open(src).ValueOrDie();
    auto reader = parquet::arrow::OpenFile(file, arrow::default_memory_pool()).ValueOrDie();
    reader->set_use_threads(true);
    std::shared_ptr<arrow::Schema> schema;
    check(reader->GetSchema(&schema));
    const int i_sid = schema->GetFieldIndex("sid"), i_mark = schema->GetFieldIndex("mark");
    if (i_sid < 0 || i_mark < 0) throw std::runtime_error("the feature parquet needs sid and mark columns");
    if (schema->GetFieldIndex("date") >= 0) throw std::runtime_error("the feature parquet has a date column: use the x.bin way (pred_day.py does)");
    std::vector<int> pick{i_sid, i_mark}, remap(names.size());
    for (std::size_t k = 0; k < names.size(); ++k) {
        const int i = schema->GetFieldIndex(names[k]);
        remap[k] = i < 0 ? -1 : static_cast<int>(pick.size()) - 2;
        if (i >= 0) pick.push_back(i);
    }
    const std::uint64_t n_have = pick.size() - 2, n_cols = n_have + 1;
    for (int& r : remap)
        if (r < 0) r = static_cast<int>(n_have);
    for (auto& m : models)
        for (auto& t : m)
            for (int& c : t.feat) {
                if (c < 0 || static_cast<std::size_t>(c) >= names.size()) throw std::runtime_error("a tree splits on a column the feature list does not have");
                c = remap[static_cast<std::size_t>(c)];
            }
    const std::shared_ptr<arrow::Table> table = reader->ReadTable(pick).ValueOrDie();
    const std::uint64_t n_rows = static_cast<std::uint64_t>(table->num_rows());
    std::vector<double> x(n_rows * n_cols, std::nan(""));
    std::vector<std::thread> pool;
    std::vector<std::string> errors(threads);
    for (unsigned w = 0; w < threads; ++w)
        pool.emplace_back([&, w] {
            for (std::uint64_t j = w; j < n_have; j += threads) {
                std::uint64_t r = 0;
                for (const auto& chunk : table->column(static_cast<int>(j) + 2)->chunks()) {
                    const std::int64_t n = chunk->length();
                    if (chunk->type_id() == arrow::Type::DOUBLE) {
                        const auto& d = static_cast<const arrow::DoubleArray&>(*chunk);
                        for (std::int64_t i = 0; i < n; ++i, ++r)
                            if (d.IsValid(i)) x[r * n_cols + j] = d.Value(i);
                    } else if (chunk->type_id() == arrow::Type::FLOAT) {
                        const auto& d = static_cast<const arrow::FloatArray&>(*chunk);
                        for (std::int64_t i = 0; i < n; ++i, ++r)
                            if (d.IsValid(i)) x[r * n_cols + j] = static_cast<double>(d.Value(i));
                    } else {
                        errors[w] = "a feature column is neither double nor float";
                        return;
                    }
                }
            }
        });
    for (auto& th : pool) th.join();
    for (auto& e : errors)
        if (!e.empty()) throw std::runtime_error(e);
    const std::vector<double> out = predict(models, x, n_rows, n_cols, threads);
    arrow::DoubleBuilder b;
    check(b.AppendValues(out));
    std::shared_ptr<arrow::Array> pred;
    check(b.Finish(&pred));
    auto res = arrow::Table::Make(arrow::schema({schema->field(i_sid), schema->field(i_mark), arrow::field("pred", arrow::float64())}),
                                  {table->column(0), table->column(1), std::make_shared<arrow::ChunkedArray>(pred)});
    auto sink = arrow::io::FileOutputStream::Open(dst).ValueOrDie();
    auto props = parquet::WriterProperties::Builder().compression(arrow::Compression::ZSTD)->build();
    check(parquet::arrow::WriteTable(*res, arrow::default_memory_pool(), sink, 1 << 20, props));
    check(sink->Close());
    std::size_t nt = 0;
    for (auto& m : models) nt += m.size();
    std::printf("lgbpred: %llu rows, %llu of %zu model columns filled, %zu model files, %zu trees (parquet in and out)\n",
                static_cast<unsigned long long>(n_rows), static_cast<unsigned long long>(n_have), names.size(), models.size(), nt);
    return 0;
}
#endif

}  // namespace

int main(int argc, char** argv) {
    try {
        unsigned threads = 8;
        bool parquet = false;
        std::string csv;
        std::vector<std::string> pos;
        for (int i = 1; i < argc; ++i) {
            if (!std::strcmp(argv[i], "--threads") && i + 1 < argc) {
                threads = static_cast<unsigned>(std::max(1, std::atoi(argv[++i])));
            } else if (!std::strcmp(argv[i], "--features") && i + 1 < argc) {
                csv = argv[++i];
            } else if (!std::strcmp(argv[i], "--parquet")) {
                parquet = true;
            } else {
                pos.emplace_back(argv[i]);
            }
        }
        if (pos.size() < 3) throw std::runtime_error("usage: lgbpred [--parquet --features <names.csv>] <x> <out> <model.txt> [...] [--threads N]");
        std::vector<std::vector<tree>> models;
        for (std::size_t i = 2; i < pos.size(); ++i) models.push_back(load(pos[i]));
        if (parquet) {
#ifdef CRUCIBLE_ARROW
            if (csv.empty()) throw std::runtime_error("--parquet needs --features <names.csv>");
            return run_parquet(pos[0], pos[1], csv, models, threads);
#else
            throw std::runtime_error("--parquet: this binary was built without Arrow (build with CMake inside the conda env)");
#endif
        }
        std::FILE* f = std::fopen(pos[0].c_str(), "rb");
        if (!f) throw std::runtime_error("cannot open " + pos[0]);
        char magic[8];
        std::uint64_t n_rows, n_cols;
        if (std::fread(magic, 1, 8, f) != 8 || std::memcmp(magic, "CRULX001", 8)) throw std::runtime_error("not an lgbpred input file");
        if (std::fread(&n_rows, 8, 1, f) != 1 || std::fread(&n_cols, 8, 1, f) != 1) throw std::runtime_error("short read");
        std::vector<double> x(n_rows * n_cols);
        if (!x.empty() && std::fread(x.data(), 8, x.size(), f) != x.size()) throw std::runtime_error("short read");
        std::fclose(f);
        const std::vector<double> out = predict(models, x, n_rows, n_cols, threads);
        std::FILE* g = std::fopen(pos[1].c_str(), "wb");
        if (!g) throw std::runtime_error("cannot write " + pos[1]);
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
