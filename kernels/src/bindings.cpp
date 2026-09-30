// Python bindings for crucible_kernels. Decoding runs with the GIL released, so a Python
// ThreadPoolExecutor over minute files uses every core.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <string>
#include <utility>
#include <vector>

#include "crucible_kernels/feitu_v3.hpp"

namespace nb = nanobind;
namespace fe = crucible_kernels::feitu;

namespace {

// Hand a vector's buffer to numpy without copying; the capsule owns the heap vector.
template <class T>
nb::object to_numpy(std::vector<T>&& v, std::size_t cols = 1) {
  auto* owned = new std::vector<T>(std::move(v));
  nb::capsule owner(owned, [](void* p) noexcept { delete static_cast<std::vector<T>*>(p); });
  if (cols == 1) {
    std::size_t shape[1] = {owned->size()};
    return nb::cast(nb::ndarray<nb::numpy, T>(owned->data(), 1, shape, owner));
  }
  std::size_t shape[2] = {owned->size() / cols, cols};
  return nb::cast(nb::ndarray<nb::numpy, T>(owned->data(), 2, shape, owner));
}

#define CK_COL(d, c, name) d[#name] = to_numpy(std::move((c).name))
#define CK_LVL(d, c, name) d[#name] = to_numpy(std::move((c).name), fe::levels)

nb::dict decode_order(const std::string& path) {
  fe::order_cols c;
  {
    nb::gil_scoped_release nogil;
    c = fe::decode_file<fe::order_cols, OrderV3>(path);
  }
  nb::dict d;
  CK_COL(d, c, shm_key); CK_COL(d, c, nano_ts); CK_COL(d, c, symbol_id); CK_COL(d, c, market_id);
  CK_COL(d, c, time); CK_COL(d, c, symbol_type); CK_COL(d, c, price); CK_COL(d, c, volume);
  CK_COL(d, c, dir); CK_COL(d, c, order_id); CK_COL(d, c, channel_id); CK_COL(d, c, seq_id);
  CK_COL(d, c, index_id); CK_COL(d, c, order_type); CK_COL(d, c, update_type);
  CK_COL(d, c, spider_ts); CK_COL(d, c, server_ts);
  return d;
}

nb::dict decode_transaction(const std::string& path) {
  fe::transaction_cols c;
  {
    nb::gil_scoped_release nogil;
    c = fe::decode_file<fe::transaction_cols, TransactionV3>(path);
  }
  nb::dict d;
  CK_COL(d, c, shm_key); CK_COL(d, c, nano_ts); CK_COL(d, c, symbol_id); CK_COL(d, c, market_id);
  CK_COL(d, c, time); CK_COL(d, c, symbol_type); CK_COL(d, c, price); CK_COL(d, c, volume);
  CK_COL(d, c, dir); CK_COL(d, c, channel_id); CK_COL(d, c, index_id); CK_COL(d, c, seq_id);
  CK_COL(d, c, buy_seq_id); CK_COL(d, c, sell_seq_id); CK_COL(d, c, trade_type);
  CK_COL(d, c, spider_ts); CK_COL(d, c, server_ts);
  return d;
}

nb::tuple decode_quotation(const std::string& path) {
  fe::quotation_cols c;
  {
    nb::gil_scoped_release nogil;
    c = fe::decode_file<fe::quotation_cols, QuotationV3>(path);
  }
  const int64_t truncated = c.n_truncated_books;
  nb::dict d;
  CK_COL(d, c, shm_key); CK_COL(d, c, nano_ts); CK_COL(d, c, symbol_id); CK_COL(d, c, market_id);
  CK_COL(d, c, time); CK_COL(d, c, symbol_type); CK_COL(d, c, status); CK_COL(d, c, pre_close);
  CK_COL(d, c, open); CK_COL(d, c, high); CK_COL(d, c, low); CK_COL(d, c, close);
  CK_COL(d, c, price);
  CK_LVL(d, c, bid_px); CK_LVL(d, c, bid_vol); CK_LVL(d, c, bid_no);
  CK_LVL(d, c, ask_px); CK_LVL(d, c, ask_vol); CK_LVL(d, c, ask_no);
  CK_COL(d, c, total_no); CK_COL(d, c, total_buy_no); CK_COL(d, c, total_sell_no);
  CK_COL(d, c, total_volume); CK_COL(d, c, total_amount); CK_COL(d, c, total_bid_volume);
  CK_COL(d, c, total_ask_volume); CK_COL(d, c, weighted_avg_bid_price);
  CK_COL(d, c, weighted_avg_ask_price); CK_COL(d, c, high_limited); CK_COL(d, c, low_limited);
  CK_COL(d, c, buy_cancel_no); CK_COL(d, c, buy_cancel_volume); CK_COL(d, c, buy_cancel_amount);
  CK_COL(d, c, sell_cancel_no); CK_COL(d, c, sell_cancel_volume);
  CK_COL(d, c, sell_cancel_amount); CK_COL(d, c, num_buy_levels); CK_COL(d, c, num_sell_levels);
  CK_COL(d, c, buy_level_queue_no01); CK_COL(d, c, sell_level_queue_no01); CK_COL(d, c, iopv);
  CK_COL(d, c, match_last_px); CK_COL(d, c, auction_volume_trade);
  CK_COL(d, c, auction_value_trade); CK_COL(d, c, spider_ts); CK_COL(d, c, server_ts);
  return nb::make_tuple(d, truncated);
}

nb::dict decode_index(const std::string& path) {
  fe::index_cols c;
  {
    nb::gil_scoped_release nogil;
    c = fe::decode_file<fe::index_cols, IndexV3>(path);
  }
  nb::dict d;
  CK_COL(d, c, shm_key); CK_COL(d, c, nano_ts); CK_COL(d, c, symbol_id); CK_COL(d, c, market_id);
  CK_COL(d, c, time); CK_COL(d, c, last_price); CK_COL(d, c, open_price);
  CK_COL(d, c, high_price); CK_COL(d, c, low_price); CK_COL(d, c, close_price);
  CK_COL(d, c, pre_close_price); CK_COL(d, c, total_volume); CK_COL(d, c, total_amount);
  CK_COL(d, c, server_ts); CK_COL(d, c, spider_ts);
  d["symbol_str"] = nb::cast(std::move(c.symbol_str));
  return d;
}

}  // namespace

NB_MODULE(_feitu, m) {
  m.doc() = "feitu v3 capnp dump decoder: one minute file -> dict of numpy columns (raw vendor values)";
  m.attr("levels") = fe::levels;
  m.def("decode_order", &decode_order, nb::arg("path"));
  m.def("decode_transaction", &decode_transaction, nb::arg("path"));
  m.def("decode_quotation", &decode_quotation, nb::arg("path"),
        "Returns (columns, n_truncated_books). Book levels are (n, levels) arrays.");
  m.def("decode_index", &decode_index, nb::arg("path"));
}
