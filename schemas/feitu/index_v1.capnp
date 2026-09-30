@0xa1b2c3d4e5f60718;

# Index 指数行情切片
# 自 2025-12-31 起, 指数行情独立 dump 到 v1_index 目录
# 与 v2_quotation 中 symbolType=2 的指数记录在重叠时段共存时, 以 v1_index 为准
# 此后 v2_quotation 不再发布指数记录

struct IndexV1 {
    symbolId @0 : Int32;       # 指数代码 上证50=000016 中证500=000905 沪深300=000300
    marketId @1 : Int16;       # 交易市场ID SSE=3553 SZSE=3554
    time @2 : Int64;           # 数据时间 单位毫秒
    lastPrice @3 : Float64;    # 最新价
    openPrice @4 : Float64;    # 开盘价
    highPrice @5 : Float64;    # 最高价
    lowPrice @6 : Float64;     # 最低价
    closePrice @7 : Float64;   # 收盘价
    preClosePrice @8 : Float64;# 昨收盘价
    totalVolume @9 : Int64;    # 总成交量
    totalAmount @10 : Float64; # 总成交额
    spiderTs @11 : Int64;      # spider 接收时间 单位毫秒
    serverTs @12 : Int64;      # 数据服务商时间 单位毫秒
    receiveTs @13 : Int64;     # 本次反序列化时间 单位毫秒
    symbolStr @14 : Text;      # 一般用于海外指数 无 symbolId 时 例如 N225
}
