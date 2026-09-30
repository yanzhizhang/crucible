@0xb76db76fcb08217c;

struct Order {
    symbolId @0 : Int32; # 证券代码 000001
    marketId @1 : Int16; # 市场ID SSE=3553,SZSE=3554
    putTs @2 : Int64; # 委托时间
    serverTs @3 : Int64; # 数据服务商时间
    price @4 : Float64; # 委托价格
    volume @5 : Int64; # 委托量
    dir @6 : Int16; # 方向 BUY=1063,SELL=1550
    orderId @7 : Int64; # 序号
    spiderTs @8 : Int64; # 接收时间
}