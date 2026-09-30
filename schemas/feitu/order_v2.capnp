@0xf4308d8bb51f0853;

struct OrderV2 {
    symbolId @0 : Int32; # 证券代码 000001
    marketId @1 : Int16; # 市场ID SSE=3553,SZSE=3554
    time @2 : Int64;     # 委托产生时间
    symbolType @3: Int32; # 产品类型 1=股票,2=指数,3=基金,4=债券,0=未知
    price @4 : Float64;  # 委托价格
    volume @5 : Int64;   # 委托量
    dir @6 : Int16;      # 委托方向 BUY=1063,SELL=1550
    orderId @7 : Int64;  # 订单编号
    channelId @8 : Int32; # 频道ID
    seqId @9 : Int64;    # 逐笔编号,在同一个channel内连续
    indexId @10 : Int64;  # 委托编号
    orderType @11 : Int8; # 订单类别 1=市价,2=限价,3=本方最优,0=未知
    updateType @12 : Int8; # 更新类型 0=未知,1=新增,2=取消,
                           # 3=START(产品状态:启动),4=OCALL(产品状态:开市集合竞价),5=TRADE(产品状态:连续自动撮合),6=SUSP(产品状态:停牌),
                           # 7=CCALL(产品状态:收盘集合竞价),8=CLOSE(产品状态:闭市，自动计算闭市价格),9=ENDTR(产品状态:交易结束),10=ADD(产品状态:未上市)
                           # 只会出现沪市的取消单和产品状态单(3~10)
    spiderTs @13 : Int64; # 接收时间
    serverTs @14 : Int64; # 券商服务器收到数据的时间
}