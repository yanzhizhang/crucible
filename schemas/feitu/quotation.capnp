@0x83fd98b35a2e1639;
using Base = import "base.capnp";

struct Quotation {
  symbolId @0 : Int32; # 证券代码 000001
  marketId @1 : Int16; # 市场ID SSE=3553,SZSE=3554
  time @2: Int64; # 数据生成时间
  status @3 : Int16; # 产品状态
  preClose @4 : Float64; # 昨收收盘价
  open @5 : Float64; # 开盘价
  high @6 : Float64; # 最高价
  low @7 : Float64; # 最低价
  price @8 : Float64; # 最新价
  bids @9 : List(Base.PriceAmount); # 买盘
  asks @10 : List(Base.PriceAmount); # 卖盘
  totalNo @11 : Int64; # 成交笔数
  totalVolume @12 : Int64; # 成交总量, 股票:股;权证:份;债券:手
  totalAmount @13 : Float64; # 成交总额, (元)
  totalBidVolume @14 : Int64; # 总买入量, 股票:股;权证:份;债券:手
  totalAskVolume @15 : Int64; # 总卖出量, 股票:股;权证:份;债券:手
  weightedAvgBidPrice @16 : Float64; # 加权平均买价, (元)
  weightedAvgAskPrice @17 : Float64; # 加权平均卖价, (元)
  yml @18 : Float64; # 债券到期收益率
  highLimited @19 : Float64; # 涨停价, (元)
  lowLimited @20: Float64; # 跌停价, (元)
  spiderTs @21 : Int64; # 接收时间
  serverTs @22 : Int64; # 数据服务商时间
}