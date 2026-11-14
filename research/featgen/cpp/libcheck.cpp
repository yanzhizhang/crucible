// libcheck: proves the conda C++ libraries of environment.yml are usable together in one program, and times them on a
// real-sized table: DuckDB (query), Arrow + Parquet (write and read back, zstd), Eigen (a least-squares residual).
//
//   libcheck <work dir> [rows] [columns]        default 1,422,000 x 144 (one day of 3-second vectors)
// Prints one line per step with its seconds, then "libcheck ok".
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <arrow/api.h>
#include <arrow/io/api.h>
#include <duckdb.h>
#include <parquet/arrow/reader.h>
#include <parquet/arrow/writer.h>

namespace {
double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
void check(const arrow::Status& s) {
    if (!s.ok()) throw std::runtime_error(s.ToString());
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const std::string dir = argc > 1 ? argv[1] : ".";
        const long long rows = argc > 2 ? std::atoll(argv[2]) : 1422000, cols = argc > 3 ? std::atoll(argv[3]) : 144;
        const std::string path = dir + "/libcheck.parquet";
        double t = now();
        // a table of doubles with some NaN, like a day of feature vectors
        std::vector<std::shared_ptr<arrow::Field>> fields;
        std::vector<std::shared_ptr<arrow::Array>> arrays;
        double sum_w = 0;
        for (long long c = 0; c < cols; ++c) {
            arrow::DoubleBuilder b;
            check(b.Reserve(rows));
            for (long long r = 0; r < rows; ++r) {
                const double v = (r * 31 + c * 17) % 97 == 0 ? std::nan("") : std::sin(0.001 * static_cast<double>(r) + static_cast<double>(c));
                b.UnsafeAppend(v);
                if (!std::isnan(v)) sum_w += v;
            }
            std::shared_ptr<arrow::Array> a;
            check(b.Finish(&a));
            arrays.push_back(a);
            fields.push_back(arrow::field("c" + std::to_string(c), arrow::float64()));
        }
        auto table = arrow::Table::Make(arrow::schema(fields), arrays);
        std::printf("build   %lld rows x %lld columns: %.1fs\n", rows, cols, now() - t);
        t = now();
        {
            auto out = arrow::io::FileOutputStream::Open(path).ValueOrDie();
            auto props = parquet::WriterProperties::Builder().compression(arrow::Compression::ZSTD)->build();
            check(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), out, 1 << 20, props));
            check(out->Close());
        }
        std::printf("parquet write (zstd): %.1fs\n", now() - t);
        t = now();
        std::shared_ptr<arrow::Table> back;
        {
            auto in = arrow::io::ReadableFile::Open(path).ValueOrDie();
            auto reader = parquet::arrow::OpenFile(in, arrow::default_memory_pool()).ValueOrDie();
            reader->set_use_threads(true);
            back = reader->ReadTable().ValueOrDie();
        }
        double sum_r = 0;
        for (int c = 0; c < back->num_columns(); ++c)
            for (const auto& chunk : back->column(c)->chunks()) {
                const auto& d = static_cast<const arrow::DoubleArray&>(*chunk);
                for (long long r = 0; r < d.length(); ++r)
                    if (!std::isnan(d.Value(r))) sum_r += d.Value(r);
            }
        std::printf("parquet read: %.1fs; rows %lld, sum written %.6f, read %.6f (%s)\n", now() - t, static_cast<long long>(back->num_rows()), sum_w,
                    sum_r, sum_w == sum_r ? "equal" : "DIFFERENT");
        t = now();
        // DuckDB through its C interface (duckdb.h): the stable one, the C++ classes change between major versions
        duckdb_database db;
        duckdb_connection con;
        duckdb_result res;
        if (duckdb_open(nullptr, &db) != DuckDBSuccess || duckdb_connect(db, &con) != DuckDBSuccess) throw std::runtime_error("duckdb: cannot open");
        const std::string q = "SELECT count(*), count(c0) FROM read_parquet('" + path + "')";
        if (duckdb_query(con, q.c_str(), &res) != DuckDBSuccess) throw std::runtime_error(std::string("duckdb: ") + duckdb_result_error(&res));
        std::printf("duckdb %s query on the parquet: %.1fs; rows %lld, c0 values %lld\n", duckdb_library_version(), now() - t,
                    static_cast<long long>(duckdb_value_int64(&res, 0, 0)), static_cast<long long>(duckdb_value_int64(&res, 1, 0)));
        duckdb_destroy_result(&res);
        duckdb_disconnect(&con);
        duckdb_close(&db);
        t = now();
        Eigen::MatrixXd X = Eigen::MatrixXd::Random(300, 47);
        Eigen::VectorXd y = Eigen::VectorXd::Random(300), r(300);
        for (int k = 0; k < 4740; ++k) r = y - X * X.colPivHouseholderQr().solve(y);
        std::printf("eigen %d.%d.%d: 4740 least squares of 300 x 47: %.1fs, residual norm %.6f\n", EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION,
                    EIGEN_MINOR_VERSION, now() - t, r.norm());
        std::remove(path.c_str());
        std::printf("libcheck ok\n");
        return sum_w == sum_r ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "libcheck: %s\n", e.what());
        return 1;
    }
}
