@0xd0247b219c981085;

struct PriceAmount {
  price @0 : Float64; # 报单价
  volume @1 : Int64; # 报单量
  no @2: Int32; # 档位委托单数
}

struct PriceAmount2 {
  price @0 : Float64; # 报单价
  volume @1 : Int64; # 报单量
  no @2: Int32; # 档位委托单数 (郑商所)
  impliedVolume @3: Int64; # 推导量（由套利定单推导出来; 大商所, 广期所）
}