@0x88f0fd8e9e582215;

struct LendingEntry {
  rate @0 : Float64; # 利率
  term @1 : Int32; # 期限（天）
  amount @2 : Int64; # 数量
}

struct SecurityLending {
  symbolId @0 : Int32; # 证券代码 000001
  marketId @1 : Int16; # 市场ID SSE=3553,SZSE=3554
  time @2: Int64; # 数据生成时间
  symbolType @3: Int32; # 产品类型 1=股票,2=指数,3=基金,4=债券,0=未知
  status @4 : Int16; # 产品状态 1=开盘前,2=开盘集合竞价,3=开盘集合竞价阶段结束到连续竞价阶段开始之前,4=连续竞价,5=中午休市,6=收盘集合竞价,7=闭市,8=盘后交易,9=临时停牌,10=波动性中断,11=竞价交易收盘至盘后固定价格交易之前,12=盘后固定交易交易, 0=未知

  prevWeightedRate @5 : Float64; # 昨日融券加权平均出借利率
  prevHighRate @6 : Float64; # 昨日最高融券成交利率
  prevLowRate @7 : Float64; # 昨日最低融券成交利率
  prevPlatformVolume @8 : Int64; # 昨日平台融券成交量
  prevMarketVolume @9 : Int64; # 昨日公开市场融券成交量

  weightedRate @10 : Float64; # 最新融券加权平均出借利率
  highRate @11 : Float64; # 今日最高融券成交利率
  lowRate @12 : Float64; # 今日最低融券成交利率
  platformVolume @13 : Int64; # 今日平台融券成交量
  marketVolume @14 : Int64; # 今日公开市场融券成交量

  bestBorrowRate @15 : Float64; # 当前最优（最高）融入利率
  bestLendRate @16 : Float64; # 当前最优（最低）融出利率
  bestReserveBorrowRate @17 : Float64; # 当前最优（最高）预约融入利率
  bestReserveLendRate @18 : Float64; # 当前最优（最低）预约融出利率

  platformBorrowTradeVolume @19 : Int64; # 平台借入成交量
  platformBorrowWeightedRate @20 : Float64; # 平台借入成交利率加权平均
  prevPlatformBorrowTradeVolume @21 : Int64; # 昨日平台借入成交量
  prevPlatformBorrowWeightedRate @22 : Float64; # 昨日平台借入成交利率加权平均

  borrowsAmount @23 : Int64; # 借入数量, 目前对该证券的需求数量 
  borrowsRate @24 : Float64; # 借入利率, 当前该证券的最高借入利率
  lendsRate @25 : Float64; # 出借利率
  validBorrows @26 : List(LendingEntry); # 当前有效的融券借入委托
  validRealtimeLends @27 : List(LendingEntry); # 当前有效实时券出借委托
  validAuctionLends @28 : List(LendingEntry); # 当前有效的竞拍券出借委托

  spiderTs @29 : Int64; # 接收时间
  serverTs @30 : Int64; # 券商服务器收到数据的时间
}