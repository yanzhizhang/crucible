@0xfe3850f4069e0bbb;

struct StockData {
  typeId @0 : Int32; # 数据类型, 1:quotation, 2:order, 3:transaction, 4:index, 5: quotation_v2, 6: order_v2, 7: transaction_v2, 8: index_v3, 9: quotation_v3, 10: order_v3, 11: transaction_v3,
  symbolId @1 : Int32; # 证券代码
  dataTs @2 : Int64; # spider打包的时间
  data @3 : Data; # capnp序列化的二进制数据
}