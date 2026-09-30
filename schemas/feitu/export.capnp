@0xbaca43ae6b0d1939;

struct ExportedData {
  shmKey @0 : Int32; # 导出源
  nanoTs @1 : Int64; # 导出时间戳
  data @2 : Data; # capnp序列化的二进制数据
}
