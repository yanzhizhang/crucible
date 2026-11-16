# chrono strategy-plugin ABI (from the chrono session, 2026-11-16)

Caveat from chrono: libcatchRun.so / libshtst.a and the fields fa=, hs=, lthr, winsdQty, asf, emsMktVol,
emsOurVol, executp, vpen, asq, liid have zero hits in chrono/. They are likely old-platform (shtcommon) code.
Check on 97 (read-only): `nm -D --defined-only libcatchRun.so | grep chrono_plugin`. If the data symbol is
absent, the .so is not a chrono plugin and the ABI below does not apply.

## Loader -- chrono/cpp/chrono/strat/plugin_loader.hpp
- dlopen(path, RTLD_NOW | RTLD_LOCAL) (:24); dlsym of ONE data symbol "chrono_plugin" (:32); no version symbol.
- strategy.hpp:205 `struct plugin_descriptor { const char* type_names; create_strategy_fn create; void (*destroy)(strategy*); };`
- strategy.hpp:200 create_strategy_fn = strategy*(std::string id, shared_ptr<strategy_order_placer>, log_sink*, std::string params_json, std::string params_path)
- The symbol name is extern "C", but the ABI is C++ (std::string, shared_ptr, vtable): gcc 11.4, C++17, same headers.
- Export example: strategy_plugins/hs300/hs300.cpp:231-234 via describe<T> (strategy.hpp:212).
- Caller: strategy_manager.cpp:548-558 (open -> provides(type) -> create); the handle lives for the process.

## Lifecycle -- class chrono::strat::strategy (strategy.hpp:125)
- Config: chrono_config.json5 strategy_plugins[] {strategy_id, strategy_type, plugin_path, params, params_path},
  parsed in config/bootstrap_config.hpp:410-415.
- initialize() once (strategy_manager.cpp:80); at boot on_position per symbol and on_asset.
- on_lv2(const lv2&) per L2 snapshot (core/book.hpp:22).
- on_signal(signal&&) per prediction (core/model_signal.hpp:15: symbol, px, vol, prediction, confidence, name, ts).
- insert_order(symbol, qty, px, side) -> idx (limit only); cancel_order(idx) -> bool; both async via
  strategy_order_placer (strategy.hpp:113).
- on_trade_event(idx, px, qty); on_order_changed(order_view) (ack, terminal); on_error_cancel(order_view, timeout|not_cancelable).
- Single core thread, noexcept. Shutdown is destroy() via the descriptor (the destructor).
