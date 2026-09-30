// crucible_kernels/feitu_v3.hpp
//
// Columnar decoder for feitu v3 capnp dumps (one file = one minute of one stream).
//
// Wire format: zstd( stream of PACKED capnp ExportedData{shmKey, nanoTs, data} ), where each
// `data` is itself a PACKED capnp message of the stream's struct (OrderV3 / TransactionV3 /
// QuotationV3 / IndexV3). Both layers are packed; reading either unpacked yields garbage.
//
// Output is raw vendor values in column vectors -- no unit conversion, no symbol formatting.
// Normalisation (ns units, 6-digit symbols, price rounding) is vectorised on the Python side,
// once, where it is auditable. This header only has to be fast and exact.
#pragma once

#include <fcntl.h>
#include <unistd.h>
#include <zstd.h>

#include <array>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <capnp/serialize-packed.h>
#include <kj/io.h>

#include "export.capnp.h"
#include "index_v3.capnp.h"
#include "order_v3.capnp.h"
#include "quotation_v3.capnp.h"
#include "transaction_v3.capnp.h"

namespace crucible_kernels::feitu {

inline constexpr int levels = 10;

/// Read a whole .zst file and decompress it (multi-frame safe, size-unknown safe).
/// kj::InputStream that reads a .zst file in bounded chunks and decompresses on the fly, so a
/// file never has to exist decompressed in memory. Reads are chunked because 9p mounts (WSL
/// /mnt/c) fail very large single reads with ENOMEM; EINTR/EAGAIN are retried.
class zstd_file_stream final : public kj::InputStream {
 public:
  explicit zstd_file_stream(const std::string& path)
      : path_(path), dctx_(ZSTD_createDCtx(), &ZSTD_freeDCtx), in_(ZSTD_DStreamInSize()) {
    fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
  }
  ~zstd_file_stream() override {
    if (fd_ >= 0) ::close(fd_);
  }
  zstd_file_stream(const zstd_file_stream&) = delete;
  zstd_file_stream& operator=(const zstd_file_stream&) = delete;

  std::size_t tryRead(void* buffer, std::size_t minBytes, std::size_t maxBytes) override {
    ZSTD_outBuffer out{buffer, maxBytes, 0};
    while (out.pos < minBytes) {
      if (ib_.pos == ib_.size) {
        if (eof_) break;
        fill();
        if (ib_.size == 0) break;
      }
      const std::size_t rc = ZSTD_decompressStream(dctx_.get(), &out, &ib_);
      if (ZSTD_isError(rc))
        throw std::runtime_error(std::string("zstd: ") + ZSTD_getErrorName(rc) + " in " + path_);
    }
    return out.pos;
  }

 private:
  void fill() {
    int retries = 0;
    for (;;) {
      const ssize_t n = ::read(fd_, in_.data(), in_.size());
      if (n > 0) {
        ib_ = ZSTD_inBuffer{in_.data(), static_cast<std::size_t>(n), 0};
        return;
      }
      if (n == 0) {
        eof_ = true;
        ib_ = ZSTD_inBuffer{in_.data(), 0, 0};
        return;
      }
      if ((errno == EINTR || errno == EAGAIN || errno == ENOMEM) && ++retries < 100) continue;
      throw std::runtime_error("read " + path_ + ": " + std::strerror(errno));
    }
  }

  std::string path_;
  int fd_ = -1;
  std::unique_ptr<ZSTD_DCtx, std::size_t (*)(ZSTD_DCtx*)> dctx_;
  std::vector<kj::byte> in_;
  ZSTD_inBuffer ib_{nullptr, 0, 0};
  bool eof_ = false;
};

inline capnp::ReaderOptions reader_options() {
  capnp::ReaderOptions o;
  o.traversalLimitInWords = UINT64_MAX;
  o.nestingLimit = 64;
  return o;
}

/// Stream every ExportedData envelope of a .zst dump and hand its decoded payload root to
/// `fn(shm_key, nano_ts, root)`. Memory stays bounded by one envelope plus the zstd window.
template <class Struct, class Fn>
void for_each_record(const std::string& path, Fn&& fn) {
  const auto opts = reader_options();
  // Reused scratch avoids a heap allocation per message; payloads are small.
  thread_local std::vector<capnp::word> env_scratch(1 << 12);
  thread_local std::vector<capnp::word> msg_scratch(1 << 14);
  zstd_file_stream raw(path);
  kj::BufferedInputStreamWrapper outer(raw);
  while (outer.tryGetReadBuffer().size() != 0) {
    capnp::PackedMessageReader env_reader(outer, opts,
                                          kj::arrayPtr(env_scratch.data(), env_scratch.size()));
    auto env = env_reader.getRoot<ExportedData>();
    auto data = env.getData();
    kj::ArrayInputStream inner(kj::arrayPtr(data.begin(), data.size()));
    capnp::PackedMessageReader msg(inner, opts,
                                   kj::arrayPtr(msg_scratch.data(), msg_scratch.size()));
    fn(env.getShmKey(), env.getNanoTs(), msg.getRoot<Struct>());
  }
}

struct order_cols {
  std::vector<int32_t> shm_key, symbol_id, symbol_type, channel_id;
  std::vector<int64_t> nano_ts, time, volume, order_id, seq_id, index_id, spider_ts, server_ts;
  std::vector<int16_t> market_id, dir;
  std::vector<double> price;
  std::vector<int8_t> order_type, update_type;

  void push(int32_t key, int64_t nts, OrderV3::Reader r) {
    shm_key.push_back(key);
    nano_ts.push_back(nts);
    symbol_id.push_back(r.getSymbolId());
    market_id.push_back(r.getMarketId());
    time.push_back(r.getTime());
    symbol_type.push_back(r.getSymbolType());
    price.push_back(r.getPrice());
    volume.push_back(r.getVolume());
    dir.push_back(r.getDir());
    order_id.push_back(r.getOrderId());
    channel_id.push_back(r.getChannelId());
    seq_id.push_back(r.getSeqId());
    index_id.push_back(r.getIndexId());
    order_type.push_back(r.getOrderType());
    update_type.push_back(r.getUpdateType());
    spider_ts.push_back(r.getSpiderTs());
    server_ts.push_back(r.getServerTs());
  }
};

struct transaction_cols {
  std::vector<int32_t> shm_key, symbol_id, symbol_type, channel_id;
  std::vector<int64_t> nano_ts, time, volume, index_id, seq_id, buy_seq_id, sell_seq_id, spider_ts,
      server_ts;
  std::vector<int16_t> market_id, dir;
  std::vector<double> price;
  std::vector<int8_t> trade_type;

  void push(int32_t key, int64_t nts, TransactionV3::Reader r) {
    shm_key.push_back(key);
    nano_ts.push_back(nts);
    symbol_id.push_back(r.getSymbolId());
    market_id.push_back(r.getMarketId());
    time.push_back(r.getTime());
    symbol_type.push_back(r.getSymbolType());
    price.push_back(r.getPrice());
    volume.push_back(r.getVolume());
    dir.push_back(r.getDir());
    channel_id.push_back(r.getChannelId());
    index_id.push_back(r.getIndexId());
    seq_id.push_back(r.getSeqId());
    buy_seq_id.push_back(r.getBuySeqId());
    sell_seq_id.push_back(r.getSellSeqId());
    trade_type.push_back(r.getTradeType());
    spider_ts.push_back(r.getSpiderTs());
    server_ts.push_back(r.getServerTs());
  }
};

struct quotation_cols {
  std::vector<int32_t> shm_key, symbol_id, symbol_type, num_buy_levels, num_sell_levels,
      buy_level_queue_no01, sell_level_queue_no01;
  std::vector<int16_t> market_id, status;
  std::vector<int64_t> nano_ts, time, total_no, total_buy_no, total_sell_no, total_volume,
      total_bid_volume, total_ask_volume, buy_cancel_no, buy_cancel_volume, sell_cancel_no,
      sell_cancel_volume, auction_volume_trade, spider_ts, server_ts;
  std::vector<double> pre_close, open, high, low, close, price, total_amount,
      weighted_avg_bid_price, weighted_avg_ask_price, high_limited, low_limited, buy_cancel_amount,
      sell_cancel_amount, iopv, match_last_px, auction_value_trade;
  // (n, levels) row-major; missing levels are 0.
  std::vector<double> bid_px, ask_px;
  std::vector<int64_t> bid_vol, ask_vol;
  std::vector<int32_t> bid_no, ask_no;
  // Records whose book side carried more than `levels` entries (truncated, counted, never silent).
  int64_t n_truncated_books = 0;

  template <class List>
  void push_side(List side, std::vector<double>& px, std::vector<int64_t>& vol,
                 std::vector<int32_t>& no) {
    const unsigned n = side.size();
    if (n > static_cast<unsigned>(levels)) ++n_truncated_books;
    for (unsigned i = 0; i < static_cast<unsigned>(levels); ++i) {
      if (i < n) {
        auto l = side[i];
        px.push_back(l.getPrice());
        vol.push_back(l.getVolume());
        no.push_back(l.getNo());
      } else {
        px.push_back(0.0);
        vol.push_back(0);
        no.push_back(0);
      }
    }
  }

  void push(int32_t key, int64_t nts, QuotationV3::Reader r) {
    shm_key.push_back(key);
    nano_ts.push_back(nts);
    symbol_id.push_back(r.getSymbolId());
    market_id.push_back(r.getMarketId());
    time.push_back(r.getTime());
    symbol_type.push_back(r.getSymbolType());
    status.push_back(r.getStatus());
    pre_close.push_back(r.getPreClose());
    open.push_back(r.getOpen());
    high.push_back(r.getHigh());
    low.push_back(r.getLow());
    close.push_back(r.getClose());
    price.push_back(r.getPrice());
    push_side(r.getBids(), bid_px, bid_vol, bid_no);
    push_side(r.getAsks(), ask_px, ask_vol, ask_no);
    total_no.push_back(r.getTotalNo());
    total_buy_no.push_back(r.getTotalBuyNo());
    total_sell_no.push_back(r.getTotalSellNo());
    total_volume.push_back(r.getTotalVolume());
    total_amount.push_back(r.getTotalAmount());
    total_bid_volume.push_back(r.getTotalBidVolume());
    total_ask_volume.push_back(r.getTotalAskVolume());
    weighted_avg_bid_price.push_back(r.getWeightedAvgBidPrice());
    weighted_avg_ask_price.push_back(r.getWeightedAvgAskPrice());
    high_limited.push_back(r.getHighLimited());
    low_limited.push_back(r.getLowLimited());
    buy_cancel_no.push_back(r.getBuyCancelNo());
    buy_cancel_volume.push_back(r.getBuyCancelVolume());
    buy_cancel_amount.push_back(r.getBuyCancelAmount());
    sell_cancel_no.push_back(r.getSellCancelNo());
    sell_cancel_volume.push_back(r.getSellCancelVolume());
    sell_cancel_amount.push_back(r.getSellCancelAmount());
    num_buy_levels.push_back(r.getNumBuyLevels());
    num_sell_levels.push_back(r.getNumSellLevels());
    buy_level_queue_no01.push_back(r.getBuyLevelQueueNo01());
    sell_level_queue_no01.push_back(r.getSellLevelQueueNo01());
    iopv.push_back(r.getIopv());
    match_last_px.push_back(r.getMatchLastPx());
    auction_volume_trade.push_back(r.getAuctionVolumeTrade());
    auction_value_trade.push_back(r.getAuctionValueTrade());
    spider_ts.push_back(r.getSpiderTs());
    server_ts.push_back(r.getServerTs());
  }
};

struct index_cols {
  std::vector<int32_t> shm_key, symbol_id;
  std::vector<int16_t> market_id;
  std::vector<int64_t> nano_ts, time, total_volume, server_ts, spider_ts;
  std::vector<double> last_price, open_price, high_price, low_price, close_price, pre_close_price,
      total_amount;
  std::vector<std::string> symbol_str;

  void push(int32_t key, int64_t nts, IndexV3::Reader r) {
    shm_key.push_back(key);
    nano_ts.push_back(nts);
    symbol_id.push_back(r.getSymbolId());
    market_id.push_back(r.getMarketId());
    time.push_back(r.getTime());
    last_price.push_back(r.getLastPrice());
    open_price.push_back(r.getOpenPrice());
    high_price.push_back(r.getHighPrice());
    low_price.push_back(r.getLowPrice());
    close_price.push_back(r.getClosePrice());
    pre_close_price.push_back(r.getPreClosePrice());
    total_volume.push_back(r.getTotalVolume());
    total_amount.push_back(r.getTotalAmount());
    server_ts.push_back(r.getServerTs());
    spider_ts.push_back(r.getSpiderTs());
    symbol_str.emplace_back(r.hasSymbolStr() ? r.getSymbolStr().cStr() : "");
  }
};

template <class Cols, class Struct>
Cols decode_file(const std::string& path) {
  Cols c;
  for_each_record<Struct>(path, [&](int32_t key, int64_t nts, typename Struct::Reader r) {
    c.push(key, nts, r);
  });
  return c;
}

}  // namespace crucible_kernels::feitu
