@0xf946084b58c8811a;
using Base = import "base.capnp";

struct QuotationV2 {
  symbolId @0 : Int32; # 证券代码 000001
  marketId @1 : Int16; # 市场ID SSE=3553,SZSE=3554
  time @2: Int64; # 数据生成时间
  symbolType @3: Int32; # 产品类型 1=股票,2=指数,3=基金,4=债券,0=未知
  status @4 : Int16; # 产品状态 1=开盘前,2=盘前集合竞价,3=连续竞价,4=闭市,5=盘后集合竞价,6=熔断(可恢复),7=熔断(不可恢复),8=全天停牌,9=临时停牌,10=波动性中断,11=盘后交易,0=未知
  preClose @5 : Float64; # 昨收收盘价
  open @6 : Float64; # 开盘价
  high @7 : Float64; # 最高价
  low @8 : Float64; # 最低价
  close @9 : Float64; # 收盘价
  price @10 : Float64; # 最新价
  bids @11 : List(Base.PriceAmount); # 买盘
  asks @12 : List(Base.PriceAmount); # 卖盘
  totalNo @13 : Int64; # 成交笔数
  totalBuyNo @14 : Int64; # 买入总笔数,上海有
  totalSellNo @15 : Int64; # 卖出总笔数,上海有
  totalVolume @16 : Int64; # 成交总量, 股票:股;权证:份;债券:手
  totalAmount @17 : Float64; # 成交总额, (元)
  totalBidVolume @18 : Int64; # 当前订单簿总买入量, 股票:股;权证:份;债券:手
  totalAskVolume @19 : Int64; # 当前订单簿总卖出量, 股票:股;权证:份;债券:手
  weightedAvgBidPrice @20 : Float64; # 加权平均买价, (元)
  weightedAvgAskPrice @21 : Float64; # 加权平均卖价, (元)
  yml @22 : Float64; # 债券到期收益率
  highLimited @23 : Float64; # 涨停价, (元)
  lowLimited @24 : Float64; # 跌停价, (元)
  diffPx1 @25 : Float64; # 升跌1,深圳有
  diffPx2 @26 : Float64; # 升跌2,深圳有
  buyCancelNo @27 : Int64; # 买入撤单笔数,上海有
  buyCancelVolume @28 : Int64; # 买入撤单量,上海有
  buyCancelAmount @29 : Float64; # 买入撤单金额,上海有
  sellCancelNo @30 : Int64; # 卖出撤单笔数,上海有
  sellCancelVolume @31 : Int64; # 卖出撤单量,上海有
  sellCancelAmount @32 : Float64; # 卖出撤单金额,上海有
  buyTradeMaxDuration @33 : Int64; # 买入最大等待时间,上海有
  sellTradeMaxDuration @34 : Int64; # 卖出入最大等待时间,上海有
  numBuyLevels @35 : Int32; # 买方委托档位数,上海有
  numSellLevels @36 : Int32; # 卖方委托档位数,上海有
  buyLevelQueueNo01 @37 : Int32; # 买1档委托笔数
  sellLevelQueueNo01 @38 : Int32; # 卖1档委托笔数
  spiderTs @39 : Int64; # 接收时间
  serverTs @40 : Int64; # 券商服务器收到数据的时间
}