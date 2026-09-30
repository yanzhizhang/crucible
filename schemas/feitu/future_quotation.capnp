@0xf8412e3fd05c59fb;
using Base = import "base.capnp";

struct FutureQuotation {
    exchangeId @0 : Int16; # 交所代码 CCFEX
    symbolId @1 : Int16; # 期货标的代码 IF IC
    deliveryCode @2 : Int16; # 交割代码 2109, 2112
    time @3 : Int64; # 数据生成时间
    price @4 : Float64; # 最新价
    preSettlementPrice @5 : Float64; # 上次结算价
    preClosePrice @6 : Float64; # 昨收盘价
    openPrice @7 : Float64; # 今开盘价
    highestPrice @8 : Float64; # 最高价
    lowestPrice @9 : Float64; # 最低价
    volume @10 : Int64; # 成交量
    amount @11 : Float64; # 成交额
    openInterest @12 : Float64; # 持仓量
    closePrice @13 : Float64; # 今收盘价
    settlementPrice @14 : Float64; # 本次结算价
    highLimitPrice @15 : Float64; # 涨停价
    lowLimitPrice @16 : Float64; # 跌停价
    preDelta @17: Float64; # 昨虚实度
    currentDelta @18: Float64; # 今虚实度
    bids @19 : List(Base.PriceAmount2); # 买盘
    asks @20 : List(Base.PriceAmount2); # 卖盘
    bandingUpperPrice @21 : Float64; # 上带价
    bandingLowerPrice @22 : Float64; # 下带价
    averagePrice @23 : Float64; # 当日均价
    spiderTs @24 : Int64; # 接收时间
    serverTs @25 : Int64; # 数据服务商时间
    preOpenInterest @26 : Float64; # 昨日持仓量
    symbolStr @27 : Text; # 如果symbolId是空,那么解析symbolStr
    lastVolume @28 : Int64; # 最新一笔成交量
    openInterestChange @29 : Int64; # 持仓量变化（大商所，广期所）
    totalBuyVolume @30: Int64;     # 委托买入总量（郑商所）
    avgBuyPrice @31 : Float64;     # 平均委买价（郑商所）
    totalSellVolume @32 : Int64;   # 委托卖出总量（郑商所）
    avgSellPrice @33 : Float64;    # 平均委卖价（郑商所）
    lifeHighPrice @34 : Float64;   # 历史最高成交价格（郑商所，大商所，广期所）
    lifeLowPrice @35 : Float64;    # 历史最低成交价格（郑商所，大商所，广期所）
    deriveBidPrice @36 : Float64;  # 组合买入价（郑商所）
    deriveAskPrice @37 : Float64;  # 组合卖出价（郑商所）
    deriveBidVolume @38 : Int64;   # 组合买入数量（郑商所）
    deriveAskVolume @39 : Int64;   # 组合卖出数量（郑商所）
    bidAmount @40 : Float64;  # 买报单总金额，买方向所有报单的金额总和(上期所)
    askAmount @41 : Float64;  # 卖报单总金额，卖方向所有报单的金额总和(上期所)
}
