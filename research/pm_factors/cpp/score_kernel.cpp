// The implied_match scoring kernel: per column of B, n / median ratio y:B / exact-after-scaling share / spearman with y.
// A C mirror of implied_match.col_scores and rank; built as a shared library and called through ctypes.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <vector>

namespace {

// Average ranks over the finite cells of x (NaN stays NaN), as implied_match.rank does.
void rank_into(const double* x, std::int64_t n, double* out) {
    std::vector<std::int64_t> idx;
    idx.reserve(n);
    for (std::int64_t i = 0; i < n; ++i) {
        if (std::isfinite(x[i])) idx.push_back(i);
        else out[i] = std::nan("");
    }
    std::stable_sort(idx.begin(), idx.end(), [&](std::int64_t a, std::int64_t b) { return x[a] < x[b]; });
    std::size_t i = 0;
    while (i < idx.size()) {
        std::size_t j = i;
        while (j + 1 < idx.size() && x[idx[j + 1]] == x[idx[i]]) ++j;
        double r = (static_cast<double>(i) + static_cast<double>(j)) / 2.0;  // mean of 0-based positions i..j
        for (std::size_t k = i; k <= j; ++k) out[idx[k]] = r;
        i = j + 1;
    }
}

// np.median of a scratch vector (consumed).
double median(std::vector<double>& v) {
    std::size_t m = v.size() / 2;
    std::nth_element(v.begin(), v.begin() + m, v.end());
    double hi = v[m];
    if (v.size() % 2) return hi;
    double lo = *std::max_element(v.begin(), v.begin() + m);
    return (lo + hi) / 2.0;
}

}  // namespace

extern "C" {

// Ranks of one array.
void ck_rank(const double* x, std::int64_t n, double* out) { rank_into(x, n, out); }

// Ranks of every column of a COLUMN-MAJOR K x N matrix (Fortran order: column c is the contiguous run B + c * n) into RB.
void ck_rank_cols(const double* B, std::int64_t n, std::int64_t k, double* RB) {
    for (std::int64_t c = 0; c < k; ++c) rank_into(B + c * n, n, RB + c * n);
}

// Scores of y against every column of B (column-major, see above); RB = column ranks (null: computed here); min_n as in Python.
void ck_col_scores(const double* y, const double* B, const double* RB, std::int64_t n, std::int64_t k, double tol, std::int64_t min_n,
                   std::int64_t* out_n, double* ratio, double* exact, double* rho) {
    std::vector<double> ry(n), rb_local;
    rank_into(y, n, ry.data());
    if (!RB) {
        rb_local.resize(static_cast<std::size_t>(n) * k);
        ck_rank_cols(B, n, k, rb_local.data());
        RB = rb_local.data();
    }
    std::vector<double> q;
    q.reserve(n);
    std::vector<std::int64_t> ok;
    ok.reserve(n);
    for (std::int64_t c = 0; c < k; ++c) {
        const double* b = B + c * n;
        const double* rb = RB + c * n;
        q.clear();
        ok.clear();
        double ma = 0, mb = 0;
        for (std::int64_t i = 0; i < n; ++i) {
            if (std::isfinite(y[i]) && std::isfinite(b[i])) {
                ok.push_back(i);
                ma += ry[i];
                mb += rb[i];
                if (b[i] != 0) q.push_back(y[i] / b[i]);
            }
        }
        std::int64_t cnt = static_cast<std::int64_t>(ok.size());
        out_n[c] = cnt;
        ratio[c] = std::nan("");
        exact[c] = 0.0;
        rho[c] = std::nan("");
        if (cnt < min_n) continue;
        double r = std::nan("");
        if (static_cast<std::int64_t>(q.size()) >= min_n) r = median(q);
        ma /= cnt;
        mb /= cnt;
        std::int64_t good = 0;
        double sab = 0, saa = 0, sbb = 0;
        for (std::int64_t i : ok) {
            if (std::fabs(y[i] - r * b[i]) <= tol * std::fabs(y[i]) + 1e-12) ++good;
            double da = ry[i] - ma, db = rb[i] - mb;
            sab += da * db;
            saa += da * da;
            sbb += db * db;
        }
        if (!std::isnan(r)) {
            ratio[c] = r;
            exact[c] = static_cast<double>(good) / static_cast<double>(cnt);
        }
        if (saa > 0 && sbb > 0) rho[c] = (sab / cnt) / (std::sqrt(saa / cnt) * std::sqrt(sbb / cnt));
    }
}

}  // extern "C"
