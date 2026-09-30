@0xa95d524f2fcf8f06;

struct IndexV3{
    symbolId @0 : Int32; # 指数代码 上证50=000016, 中证500=000905, 沪深300=000300
    marketId @1 : Int16; # 市场ID SSE=3553,SZSE=3554
    time @2: Int64; # 数据生成时间(ns)
    lastPrice @3: Float64; # 最新价
    openPrice @4: Float64; # 开盘价
    highPrice @5: Float64; # 最高价
    lowPrice @6: Float64; # 最低价
    closePrice @7: Float64; # 收盘价
    preClosePrice @8: Float64; # 昨收盘价
    totalVolume @9: Int64; # 总成交量,手
    totalAmount @10: Float64; # 总成交额
    serverTs @11 : Int64; # 数据服务商时间(ns)
    spiderTs @12 : Int64; # 接收时间(ns)
    symbolStr @13 : Text; # 一般用于海外指数（无symbolId），如日经N225指数等
}