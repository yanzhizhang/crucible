# so_probe stage 1 summary (97, 20261116) -- relay approved by the user in the supervisor session
Libraries: /data/sht-stg-lib/plugins/libcatchRun.so (47,989,616 B, ELF x86-64 .so, with debug_info, not stripped)
and libshtst.a (66,277,378 B). /data/sht-stg-lib also holds chrono_config.json5 (5333 B) and config2_HS300.json (1181 B).
Tools on 97: readelf objdump nm strings addr2line c++filt ar; gdb 17.2 and pahole 1.31 installed 20:3x via dnf
(internal mirror, user-approved). Full lists on 97: /work/crucible_data/so_probe/{stage1,digest,own_sources,own_functions}.txt

## IT IS A CHRONO PLUGIN
nm -D --defined-only libcatchRun.so:
  0000000001cb1bc0 D chrono_plugin
  0000000000538710 W chrono::strat::describe<cts::qlive::runS>(char const*)::{lambda(std::string, shared_ptr<strategy_order_placer>, log_sink*, std::string, std::string)#1}::_FUN
  00000000005382a0 W chrono::strat::describe<cts::qlive::runS>(...)::{lambda(chrono::strat::strategy*)#2}::_FUN
  (also 0000000001cb8580 D H5E_PLUGIN_g -- hdf5, unrelated)
=> strategy class cts::qlive::runS, exported via chrono::strat::describe<> exactly like hs300.

## Build
libcatchRun.so: GNU C++17 11.5.0 20240719 (Red Hat 11.5.0-11) -mtune=generic -march=x86-64-v2 -g -O3 -fPIC (26 CUs);
  C99 -O2 for bundled hdf5-1.10.4 (252 CUs); comp_dirs /work/sht-strategy/3rd/{hdf5-1.10.4/build/src, netcdf-cxx4-4.3.1/build/cxx4, hdf5.../hl/src}
libshtst.a: GNU C++17 11.5.0 -g -std=gnu++17 (28 CUs), comp_dir /work/sht-strategy/build
Bundled 3rd party: hdf5 1.10.4, netcdf-cxx4 4.3.1, thrift (TCompactProtocol), zstd, msgpack v1/v2/v3, flatbuffers,
  double_conversion, nlohmann json, cxxopts.

## libshtst.a members (28; first 13 seen on screen)
bk.cpp.o cm.cpp.o deser.cpp.o snap.cpp.o simDs.cpp.o jsonc.cpp.o time.cpp.o utils.cpp.o dataparquet.cpp.o dataxarray.cpp.o bk.cpp.o pq.cpp.o models.cp(p.o) ...
(line-table units of libshtst.a include: w utils trd time sys sync snap simDs s pq pos pltr oee models ...)

## Own source tree (/work/sht-strategy, from DWARF line tables, both libs)
ctiny/include/arch: alpha.hpp catchRun.hpp cpp11_zone.hpp engine.hpp model.hpp qutils.hpp
ctiny/include/ds/sim: snap.cpp      ctiny/include/ds: aligned_buffer.h cxxopts.hpp oee.hpp ...
ctiny/include/ele/config: jsonc.cpp  ctiny/include/misc/utils: dataxarray.cpp pq.cpp
ctiny/src/arch: bk.cpp cm.cpp       ctiny/src/ds/sim: cpp11_convert_helper.hpp cpp11_define_array.hpp deser.cpp snap.cpp
ctiny/src/ds: simDs.cpp             ctiny/src/ele/config: jsonc.cpp   ctiny/src/ele/time: time.cpp   ctiny/src/ele/utils: utils.cpp
ctiny/src/misc/utils: dataparquet.cpp dataxarray.cpp string_view.hpp
include/data: alpha.cpp bk.cpp dataparquet.hpp     include/es: string_conversions.h sys.cpp
include/pl: pltr.cpp simDs.hpp       include/qlive/tp/strat: optional.hpp portfolio.hpp   include/qlive/tp/core: optional.hpp
include/qlive: json.hpp optional.hpp strategy.hpp trd.cpp       include/sol: w.cpp
include/utils/statics: bk.hpp ctype_inline.h oee.cpp pltr.hpp s.cpp sys.hpp utils.hpp   include/utils/sync: alpha.hpp
include/utils: abs.hpp catchRun.cpp es.cpp pq.hpp sync.hpp w.hpp     include: bk.hpp cm.hpp cpp11_zone.hpp simDs.cpp strategy.hpp
src/data: bk.cpp compare.h pq.cpp string_builder.h
src/es: models.cpp models.hpp oee.cpp oee.hpp string_conversions.h sys.cpp
src/pl: pltr.cpp
src/qlive: alpha.cpp catchRun.cpp engine.cpp es.cpp models.hpp simDs.hpp string_conversions.h sys.hpp trd.cpp
src/rea: pltr.hpp s.cpp
src/sol: alg.cpp pos.cpp w.cpp
src/utils/statics: gcfg.cpp string_conversions.h      src/utils/sync: sync.cpp      src/utils: ct.cpp ct.hpp
(also catchRun.cpp, alg.cpp, cancel.cc static-init units in libcatchRun)

## Own classes and methods (nm -C, both libs; 63 groups)
LogMessage: LogMessage ~LogMessage
Trader: cancel insert_order ~Trader
  Trader::insert_order(std::string const&, std::string const&, chrono::side_type, unsigned long, double); Trader::cancel(unsigned long)
chrono::infra: log_sink        chrono::infra::log::detail: submit
chrono::strat: asset_view cancel_error describe order_view position_view strategy strategy_order_placer
chrono::strat::strategy: bind cancel_order initialize on_asset on_cancel_rsp on_error_cancel on_insert_rsp on_lv2 on_order_changed
  on_order_event on_portfolio_asset on_portfolio_position on_position on_signal on_trade_event params_path place_order strategy ~strategy
cts: LoadMap
cts::GlobalConfig: GlobalConfig Instance Lmb Lmsr Lvr ToWindSymbol loadEx loadSids loadSymbol setProd ~GlobalConfig
cts::ct: EMS EpochToExTimeInt TimeInt2SecSinceMid TimeInt2SecSinceOpen
cts::ct::EMA: Add EMA GM ~EMA       cts::ct::EMS: Calc EMS Update ~EMS       cts::ct::PosClzer: GVpen PosClzer
cts::data: InTrading PQ aAbst aTuple pDiff sBook     cts::data::PQ: PQ   cts::data::aAbst: aAbst ~aAbst   cts::data::aTuple: aTuple
cts::data::pqAlpha: GetAlpha GetName OnSnapUpdate pqAlpha ~pqAlpha
cts::data::sBook: GASize GAsk GBSize GBid GLASize GLAsk GLBSize GLBid GMid GS IsInTrading Update sBook
cts::es: AF CF EE EF EpochId FF NewOrder OS OT Order RF configOrder eSys oAbs round100
cts::es::EE: AplFi EE FillCallback NewOrderCallback OnSnapUpdate Po Sno Sof aAo lisNo lisOf ~EE
cts::es::EE::FillCallback / NewOrderCallback: ctor/dtor     cts::es::EF: SideStr     cts::es::Order: Order
cts::es::eSys: GAsq GPpp GWoq GscNum Gwfn Gwfq OrderId2cfgID Pnof Po SetEE SidList aAsq eSys initPos ~eSys
cts::es::oAbs: oAbs ~oAbs
cts::pl: tPL     cts::pl::PLTr: Final MkDir OnSnapUpdate PLTr Pnof SetEs ~PLTr     cts::pl::tPL: Calc Hef Sid tPL
cts::qlive: AlpInfo AlphaMsg Data ECM EOM Engine Event FCM FOM MPSCPoolQueue PosM TsCompare TsEvent exOrdInfo nowTs qAlp qTrader runS
cts::qlive::AlpInfo: AlpInfo     cts::qlive::AlphaMsg: AlphaMsg operator ~AlphaMsg
cts::qlive::ECM / EOM / FOM / PosM: Clear     cts::qlive::FCM: FCM     cts::qlive::Event: Event ~Event
cts::qlive::Engine: Engine EnqueueAlphaMsg EnqueueMdMsg Manage StartTimer Stop StopTimer TimerLoop WaitUntilEmpty onAlpEvent onInnerMsg toAlg ~Engine
cts::qlive::LiveEs: GAsq GPpp Gwfn Gwfq LiveEs OnECM OnEOM OnFCM OnFOM OnPosM PlaceExOrd Po Start Stop autoCxl cancelExOrd dOrd ~LiveEs
cts::qlive::TsCompare: operator     cts::qlive::exOrdInfo: exOrdInfo
cts::qlive::qAlp: GetAlpha GetName OnSnapUpdate Weight qAlp ~qAlp
cts::qlive::qAlpEngine: Check GetAlphas OnAlphaMsg aDur onInnerMsg qAlpEngine ~qAlpEngine
cts::qlive::qTrader: Final OnAlphaMsg OnMsg Start loadCfgs qTrader setup setupAlg setupEs ~qTrader
cts::qlive::runS: book2msg cancel_order initialize on_cancel_rsp on_error_cancel on_insert_rsp on_lv2 on_order_changed on_order_event
  on_portfolio_position on_position on_signal on_trade_event place_order runS sid_of sid_of_idx signal2alpha ~runS
cts::sTrader: OnNewOrder OnSnapUpdate OnTickdataTeardown Pnof loadCfgs sTrader setup setupAlg setupAlpha setupEs setupPLTr ~sTrader
cts::sol: Alg AlgW GMktVolPool InitSidIdx2MktVolPool MktVolPool roundBy roundPrc
cts::sol::Alg: Alg GAlpha GTp GVpen OVld cFtp cOrd onInnerMsg onWins oqWins
cts::sol::AlgW: AddFa AlgW GscNum OnMsg SetEs Start describeSid loadCfgs loadDaily onInnerMsg setInProd setPosLimit ~AlgW
cts::sol::Estitp: Calc Estitp GBp GSp HandleSpf SyncBs SyncP ~Estitp
cts::sol::MktVolPool: AEeMktVol GEeMktVol MktVolPool Update ~MktVolPool
cts::sol::MktVs: GetGMV GetNMV GetPos HandleSpf MktVs SyncP ~MktVs
cts::sol::sAbs: sAbs ~sAbs     cts::sync::WaitGroup: Add Done Wait WaitGroup ~WaitGroup     cts::testObs: Final OnMsg Start testObs

Note: the chrono session's strategy interface has on_signal/on_lv2/on_trade_event/on_order_changed/on_error_cancel; the
runS build also has on_insert_rsp / on_cancel_rsp / on_order_event / on_portfolio_position / on_portfolio_asset /
place_order -- i.e. built against a chrono strategy.hpp version different from the chrono session's current one.
