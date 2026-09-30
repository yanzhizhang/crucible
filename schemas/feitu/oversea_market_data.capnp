@0xfd836a32a210e24d;
using Base = import "base.capnp";

# 直达市场信息
struct OverseaMarketData {
    exchangeCode @0: Int16;  # 交易所代码 SGX
    symbolId @1: Int16;  # 合约代码 CN
    symbolStr @2: Text; # 合约名称: CN2309
    deliveryCode @3: Int32; # 交割代码: 2309
    currPrice @4: Float64;  # 最新价
    currNumber @5: Float64;  # 最近一笔成交量
    high @6: Float64;  # 当天最高价
    low @7: Float64;  # 当天最低价
    open @8: Float64;  # 开盘价
    close @9: Float64; # 收盘价
    intradaySettlePrice @10: Float64; # 盘中结算价(股票：收盘价)，CME交易所在交易盘中会推出当前交易日结算价，亚洲的交易所是在收盘 (T session) 后推出结算价(通过最后几分钟竞价得到)
    time @11: Int64;  # 行情时间，纳秒
    spiderTime @12: Int64; # 收到数据的时间，纳秒
    filledNum @13: Float64;  # 成交量
    holdNum @14: Float64;  # 持仓量(连接股票的行情前置时，代表成交金额。)
    hideBuyPrice @15: Float64; # 隐藏买价
    hideBuyNum @16: Float64; # 隐藏买量
    hideSellPrice @17: Float64; # 隐藏卖价
    hideSellNum @18: Float64; # 隐藏卖量
    limitDownPrice @19: Float64;  # 跌停价
    limitUpPrice @20: Float64;  # 涨停价
    tradeDate @21: Int32; # 交易日
    tradeFlag @22: Int8;  # 港交所股票行情：成交类型
    dataTime @23: Int64;  # 交易所发出数据时间
    dataSourceId @24: Int8;  # 数据来源
    canSellVol @25: Float64;  # 可卖空股数（美股行情用）
    quoteType @26: Int8;  # 行情区分，分两种情况(意思是Y和2中的成交量可以统计到分钟数据里，Z的不可以)
    preSettlementPrice @27: Float64;  # 当前交易日的前结算（股票：昨收盘价）
    aggressorSide @28: Int8; # 主动买卖方向: 0=不涉及，1=主动买，2=主动卖，3=未知
    bids @29: List(Base.PriceAmount2); # 买盘
    asks @30: List(Base.PriceAmount2); # 卖盘
}
