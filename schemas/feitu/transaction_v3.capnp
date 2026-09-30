@0x932be5f0685a1fa3;

struct TransactionV3 {
    symbolId @0 : Int32; # 证券代码 000001
    marketId @1 : Int16; # 市场ID SSE=3553,SZSE=3554
    time @2 : Int64; # 交易所时间(ns)
    symbolType @3: Int32; # 产品类型 1=股票,2=指数,3=基金,4=债券,0=未知
    price @4 : Float64; # 成交价格
    volume @5 : Int64; # 成交量
    dir @6 : Int16; # 方向 BUY=1063,SELL=1550,未知=0
    channelId @7 : Int32; # 频道ID
    indexId @8 : Int64; # 业务编号,仅沪市有
    seqId @9 : Int64; # 逐笔编号,在同一个channel内连续
    buySeqId @10 : Int64; # 买方订单编号
    sellSeqId @11 : Int64; # 卖方订单编号
    tradeType @12 : Int8; # 成交类型 1=成交,2=取消,0=未知
    spiderTs @13 : Int64; # 接收时间(ns)
    serverTs @14 : Int64; # 券商服务器收到数据的时间(ns)
}