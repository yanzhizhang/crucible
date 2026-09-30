@0xc4d604a2edcbbca1;

struct Transaction {
    symbolId @0 : Int32; # 证券代码 000001
    marketId @1 : Int16; # 市场ID SSE=3553,SZSE=3554
    tradeTs @2 : Int64; # 交易时间
    serverTs @3 : Int64; # 数据服务商时间
    price @4 : Float64; # 成交价格
    volume @5 : Int64; # 成交量
    dir @6 : Int16; # 方向 BUY=1063,SELL=1550,未知=0
    buyId @7 : Int64; # 买方订单ID
    sellId @8 : Int64; # 卖方订单ID
    tradeId @9 : Int64; # 序号
    spiderTs @10 : Int64; # 接收时间
}