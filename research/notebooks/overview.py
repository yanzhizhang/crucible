import marimo

__generated_with = "0.25.0"
app = marimo.App(width="full", app_title="crucible — 数据与进度总览")


@app.cell
def _():
    import duckdb
    import marimo as mo
    import plotly.express as px
    import plotly.graph_objects as go
    import polars as pl
    from plotly.subplots import make_subplots

    CATALOG = "/work/crucible_data/catalog.duckdb"
    con = duckdb.connect(CATALOG, read_only=True)
    q = lambda sql: con.sql(sql).pl()  # noqa: E731
    return con, go, make_subplots, mo, pl, px, q


@app.cell
def _(mo):
    mo.md(
        r"""
        # crucible — 数据与进度总览

        这一页读的是 `/work/crucible_data/catalog.duckdb`：所有数据集都是这个库里的 SQL 视图
        （直接读 Parquet，不复制），页底有 SQL 查询框可以自己查。
        重算汇总：`python research/catalog.py`；原始数据、质量检查、bar 都在 WSL 的 `/work/crucible_data/`。
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## 1. 做了什么（计划各阶段状态）

        | 阶段 | 内容 | 状态 |
        |---|---|---|
        | 0 | 环境：外网 uv（Windows + WSL），内网 conda 由 `tools/gen_conda_env.py` 从 pyproject 生成 | ✅ 完成 |
        | 0.5 | 本机三天 L2 解码：C++ 解码核心（`kernels/`，比 pycapnp 快约 9 倍），一天约 6.5 分钟 | ✅ 完成 |
        | 0.6 | WindPy 封装 + 永不重拉的缓存（`research/wind/`） | ⏸ 代码就绪，等万得终端登录 |
        | 0.7 | 行情数据质量标准（`docs/MD_QUALITY.md`），PASS / WARN / FAIL 关卡 | ✅ v1 完成；订单簿重建核对待做 |
        | — | 公开数据（a-stock-data skill v3.10）：通达信盘后、中证指数权重、交易日历、申万行业、ST | ✅ 已缓存，已接入质量标准做第二数据源 |
        | 1 | CED 日频数据在 crucible 重写 | ⬜ 未开始（Wind SQL 要内网） |
        | 2 | 1 分钟 bar（逐笔构建，带 PIT 可见时间） | ✅ 两个合格日；与 Wind / PM 对照待做 |
        | 3 | samplerR 阈值（三个候选定义，沪深300，两个合格日） | 🟡 本地算好；定义要回 97 与 PM 逐格对照才能定 |
        | 4–6 | 15 个因子族、target、LightGBM 月度重训 | ⬜ 未开始 |
        | 7 | 高频回测（延迟、排队、成本、PnL 分解） | 🟡 v1 完成：沪深两所分别重放、两天 × 30 只跑通（见 7c）；信号是占位的 |
        | 8 | C++ 落地 + 可视化 | 🟡 解码核心、本页面 |
        | praxis | 标准 XTP + XTP Pro 双接口；两个 Pro 账户实盘只读探测 6/6 | ✅ 完成（未提交） |
        """
    )
    return


@app.cell
def _(mo, px, q):
    inv = q("SELECT * FROM summary_inventory ORDER BY dataset, date")
    _fig = px.bar(
        inv.to_pandas(), x="date", y="rows", color="dataset", barmode="group", log_y=True,
        title="每个数据集每天的行数（对数刻度）", height=380,
    )
    mo.vstack([mo.md("## 2. 有哪些数据"), mo.ui.table(inv, selection=None), mo.ui.plotly(_fig)])
    return


@app.cell
def _(mo, q):
    views = q(
        """SELECT table_name AS name, table_type AS type
           FROM information_schema.tables ORDER BY table_type, table_name"""
    )
    mo.accordion({"库里所有视图和表（点开）": mo.ui.table(views, selection=None)})
    return


@app.cell
def _(mo, q):
    verdicts = q(
        """SELECT CAST(date AS VARCHAR) AS date,
                  CASE WHEN bool_or(level='FAIL') THEN 'FAIL'
                       WHEN bool_or(level='WARN') THEN 'WARN' ELSE 'PASS' END AS verdict,
                  count(*) FILTER (level='FAIL') AS n_fail,
                  count(*) FILTER (level='WARN') AS n_warn,
                  count(*) FILTER (level='PASS') AS n_pass,
                  count(*) FILTER (level='SKIP') AS n_skip
           FROM quality_checks GROUP BY 1 ORDER BY 1"""
    )
    qday = mo.ui.dropdown(options=verdicts["date"].to_list(), value=verdicts["date"][0], label="日期")
    mo.vstack([mo.md("## 3. 数据质量关卡"), mo.ui.table(verdicts, selection=None), qday])
    return (qday,)


@app.cell
def _(mo, pl, px, q, qday):
    chk = q(
        f"""SELECT "check", scope, level, value, warn_at, fail_at, n, detail
            FROM quality_checks WHERE CAST(date AS VARCHAR) = '{qday.value}'"""
    )
    _heat = q(
        """SELECT "check" || ' · ' || scope AS item, CAST(date AS VARCHAR) AS date,
                  CASE level WHEN 'FAIL' THEN 3 WHEN 'WARN' THEN 2 WHEN 'PASS' THEN 1 ELSE 0 END AS s
           FROM quality_checks"""
    ).pivot(on="date", index="item", values="s").sort("item")
    _fig = px.imshow(
        _heat.drop("item").to_numpy(), x=_heat.columns[1:], y=_heat["item"].to_list(),
        color_continuous_scale=[[0, "#d0d0d0"], [0.34, "#2e9e5b"], [0.67, "#e0a800"], [1, "#d62728"]],
        zmin=0, zmax=3, aspect="auto", height=max(400, 14 * _heat.height),
        title="所有检查 × 日期（灰 SKIP / 绿 PASS / 黄 WARN / 红 FAIL）",
    )
    _fig.update_coloraxes(showscale=False)
    _order = {"FAIL": 0, "WARN": 1, "SKIP": 2, "PASS": 3}
    mo.vstack([
        mo.md(f"**{qday.value} 的明细（坏的在前）**"),
        mo.ui.table(chk.sort(pl.col("level").replace_strict(_order, return_dtype=pl.Int8)),
                    selection=None, page_size=15),
        mo.ui.plotly(_fig),
    ])
    return


@app.cell
def _(mo, q):
    days = q("SELECT DISTINCT date FROM summary_minute ORDER BY 1")["date"].to_list()
    lday = mo.ui.dropdown(options=days, value=days[0], label="日期")
    lkind = mo.ui.dropdown(options=["transaction", "order", "quotation", "index"],
                           value="transaction", label="数据流")
    mo.vstack([
        mo.md(
            "## 4. 延迟与活跃度（逐分钟）\n"
            "延迟 = 本机收到时间 − 交易所时间。开盘那一刻采集会积压几十秒——回测必须按收到时间回放。"
        ),
        mo.hstack([lday, lkind], justify="start"),
    ])
    return lday, lkind


@app.cell
def _(go, lday, lkind, make_subplots, mo, q):
    m = q(
        f"""SELECT *, strptime(date || ' ' || minute, '%Y%m%d %H:%M') AS t FROM summary_minute
            WHERE date='{lday.value}' AND kind='{lkind.value}' AND exchange IN ('XSHG','XSHE')
              AND minute BETWEEN '09:14' AND '15:01'
            ORDER BY exchange, minute"""
    )
    _fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                         subplot_titles=("延迟 p50 / p99（毫秒，对数刻度）", "每分钟记录数"))
    for _ex, _c in (("XSHG", "#1f77b4"), ("XSHE", "#ff7f0e")):
        _d = m.filter(m["exchange"] == _ex)
        _fig.add_trace(go.Scatter(x=_d["t"], y=_d["lat_p50_ms"], name=f"{_ex} p50",
                                  line={"color": _c}), row=1, col=1)
        _fig.add_trace(go.Scatter(x=_d["t"], y=_d["lat_p99_ms"], name=f"{_ex} p99",
                                  line={"color": _c, "dash": "dot"}), row=1, col=1)
        _fig.add_trace(go.Bar(x=_d["t"], y=_d["n"], name=f"{_ex} 记录数",
                              marker_color=_c, opacity=0.6), row=2, col=1)
    _fig.update_yaxes(type="log", row=1, col=1)
    _fig.update_layout(height=620, barmode="stack", legend={"orientation": "h"})
    _fig.update_xaxes(rangebreaks=[{"bounds": [11.52, 12.98], "pattern": "hour"}])
    mo.ui.plotly(_fig)
    return


@app.cell
def _(mo, q):
    xs = q(
        """SELECT CAST(date AS VARCHAR) AS date, "check", scope, level, n, detail
           FROM quality_checks WHERE "check" LIKE 'multisource.%' AND scope LIKE 'tdx%'
           ORDER BY date, "check", scope"""
    )
    mo.vstack([
        mo.md(
            "## 5. 第二数据源核对：我们的逐笔成交 vs 通达信官网盘后包\n"
            "每只股票全天成交量、成交额逐只比；三天全部一致。"
        ),
        mo.ui.table(xs, selection=None, page_size=20),
    ])
    return


@app.cell
def _(mo, q):
    bdays = q("SELECT DISTINCT CAST(date AS VARCHAR) AS d FROM bars_1min ORDER BY 1")["d"].to_list()
    bday = mo.ui.dropdown(options=bdays, value=bdays[0], label="日期")
    bsym = mo.ui.text(value="600000", label="股票代码")
    mo.vstack([
        mo.md(
            "## 6. 1 分钟 bar 浏览\n"
            "由逐笔成交构建，右标签 (t−1分钟, t]。`可见延迟` = 这根 bar 在本机真正可知的时间 − 标签时间。"
        ),
        mo.hstack([bday, bsym], justify="start"),
    ])
    return bday, bsym


@app.cell
def _(bday, bsym, go, make_subplots, mo, q):
    b = q(
        f"""SELECT ts, open, high, low, close, volume, vwap, n_trades,
                   epoch_ms(available_ts) / 1000.0 - epoch_ms(ts) / 1000.0 AS lag_s
            FROM bars_1min WHERE CAST(date AS VARCHAR)='{bday.value}' AND symbol='{bsym.value}'
            ORDER BY ts"""
    )
    if b.height == 0:
        _out = mo.md(f"**{bsym.value}** 在 {bday.value} 没有 bar（不是 A 股代码，或当天停牌）")
    else:
        _fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.6, 0.2, 0.2],
                             vertical_spacing=0.03,
                             subplot_titles=(f"{bsym.value} {bday.value}", "成交量（股）", "可见延迟（秒）"))
        _fig.add_trace(go.Candlestick(x=b["ts"], open=b["open"], high=b["high"], low=b["low"],
                                      close=b["close"], name="K线"), row=1, col=1)
        _fig.add_trace(go.Scatter(x=b["ts"], y=b["vwap"], name="VWAP", line={"width": 1}), row=1, col=1)
        _fig.add_trace(go.Bar(x=b["ts"], y=b["volume"], name="volume"), row=2, col=1)
        _fig.add_trace(go.Bar(x=b["ts"], y=b["lag_s"], name="lag"), row=3, col=1)
        _fig.update_layout(height=720, xaxis_rangeslider_visible=False, showlegend=False)
        _fig.update_xaxes(rangebreaks=[{"bounds": [11.5, 13], "pattern": "hour"}])
        _out = mo.ui.plotly(_fig)
    _out
    return


@app.cell
def _(mo, px, q):
    idx = mo.ui.dropdown(options={"沪深300": "000300", "中证500": "000905", "中证1000": "000852"},
                         value="沪深300", label="指数")
    mo.vstack([mo.md("## 7. 公开参考数据：指数成分与权重（中证官网）"), idx])
    return (idx,)


@app.cell
def _(idx, mo, px, q):
    w = q(
        f"""SELECT date, code, name, exchange, weight_percent FROM pub_index_weights
            WHERE index_code='{idx.value}' ORDER BY weight_percent DESC"""
    )
    _fig = px.bar(w.head(25).to_pandas(), x="name", y="weight_percent",
                  title=f"{idx.selected_key} 权重前 25（{w['date'][0]}，共 {w.height} 只）", height=380)
    mo.vstack([mo.ui.plotly(_fig), mo.ui.table(w, selection=None, page_size=10)])
    return


@app.cell
def _(mo, q):
    srv = q("SELECT DISTINCT variant, CAST(date AS VARCHAR) AS d FROM pm_sampler_r ORDER BY 1, 2")
    sr_day = mo.ui.dropdown(options=sorted(set(srv["d"].to_list())),
                            value=sorted(set(srv["d"].to_list()))[0], label="交易日")
    sr_sid = mo.ui.text(value="1", label="sid（代码数字，000001 → 1）")
    mo.vstack([
        mo.md(
            "## 7b. 阶段 3：samplerR 候选（沪深300 每只股票的订单大小分位阈值）\n"
            "as/ps/ab/pb = 主动卖/被动卖/主动买/被动买，q50/q90/q95/q98，单位股。三个候选定义并排算好，"
            "回 97 和 PM 的 samplerR 逐格比，看哪个对上：`order_all` 按订单汇总全部成交（主假设）、"
            "`order_cont` 只算连续竞价、`trade_all` 按单笔成交（对照组）。"
        ),
        mo.hstack([sr_day, sr_sid], justify="start"),
    ])
    return sr_day, sr_sid


@app.cell
def _(mo, pl, px, q, sr_day, sr_sid):
    _one = q(
        f"""SELECT * EXCLUDE (date) FROM pm_sampler_r
            WHERE CAST(date AS VARCHAR)='{sr_day.value}' AND sid = {float(sr_sid.value or 1)}"""
    )
    if _one.height == 0:
        _out = mo.md(f"sid {sr_sid.value} 不在 {sr_day.value} 的沪深300 里")
    else:
        _long = _one.unpivot(index="variant", variable_name="col", value_name="shares").filter(
            pl.col("col").str.contains(r"\.q")
        ).with_columns(
            pl.col("col").str.split(".").list.get(0).alias("role"),
            pl.col("col").str.split(".").list.get(1).alias("quantile"),
        )
        _fig = px.bar(_long.to_pandas(), x="quantile", y="shares", color="variant", barmode="group",
                      facet_col="role", log_y=True, height=360,
                      title=f"sid {sr_sid.value} · {sr_day.value}：三个候选定义的分位数（对数）")
        _out = mo.vstack([mo.ui.plotly(_fig), mo.ui.table(_one, selection=None)])
    _out
    return


@app.cell
def _(mo, q):
    _d = q("SELECT DISTINCT CAST(date AS VARCHAR) AS d FROM bt_summary ORDER BY 1")["d"].to_list()
    bt_day = mo.ui.dropdown(options=_d, value=_d[-1] if _d else None, label="交易日")
    mo.vstack([
        mo.md(
            "## 7c. 阶段 7：高频回测——钱在哪一步没了\n"
            "同一个信号，从“按 mid 成交、零成本”一步步加上真实约束：吃对手价 → 延迟 + 真实盘口 → "
            "被动排队 → 费用 → T+1 与尾盘回补。每笔成交拆成 信号 + 延迟 + 价差 + 费用（+ 回补），"
            "四项加总等于总 PnL。设计与已知问题见 `docs/BACKTEST.md`。"
        ),
        bt_day,
    ])
    return (bt_day,)


@app.cell
def _(bt_day, go, mo, px, q):
    mo.stop(bt_day.value is None, mo.md("_还没有回测结果：`python research/bt/run.py`_"))
    _s = q(f"SELECT * FROM bt_summary WHERE CAST(date AS VARCHAR)='{bt_day.value}' ORDER BY config")
    _labels = {"1_ideal_mid": "1 按 mid", "2_cross_spread": "2 吃对手价", "3_latency_true_book": "3 延迟+真实盘口",
               "4_passive_queue": "4 被动排队", "5_fees": "5 加费用", "6_t1_restore": "6 T+1 回补"}
    _parts = {"signal": ("信号", "#2e7d32"), "latency": ("延迟", "#f9a825"), "spread": ("价差", "#e53935"),
              "fees": ("费用", "#6a1b9a"), "restore": ("回补", "#607d8b")}
    _x = [_labels.get(c, c) for c in _s["config"].to_list()]
    _fig = go.Figure()
    for _k, (_name, _color) in _parts.items():
        _fig.add_bar(x=_x, y=_s[_k].to_list(), name=_name, marker_color=_color)
    _fig.add_scatter(x=_x, y=_s["total"].to_list(), name="总 PnL", mode="markers+text",
                     marker=dict(symbol="diamond", size=12, color="black"),
                     text=[f"{v:,.0f}" for v in _s["total"].to_list()], textposition="top center")
    _fig.update_layout(barmode="relative", height=460, legend_orientation="h", legend_y=-0.15,
                       title=f"{bt_day.value}：每一步真实约束后 PnL 的构成（元，黑点 = 总计）")
    _w = q(f"SELECT * FROM bt_sweep WHERE CAST(date AS VARCHAR)='{bt_day.value}' ORDER BY lat_send_ms")
    _lat = px.line(_w.with_columns(_w["lat_send_ms"].cast(str)).to_pandas(), x="lat_send_ms",
                   y=["total", "latency", "spread"], markers=True, height=340,
                   labels={"lat_send_ms": "下单链路延迟（毫秒）", "value": "元", "variable": ""},
                   title="延迟敏感性（配置 3，只改下单链路延迟）")
    mo.vstack([mo.ui.plotly(_fig), mo.ui.plotly(_lat), mo.ui.table(_s, selection=None)])
    return


@app.cell
def _(mo, px, q):
    bench = q(
        """SELECT started_at, task, round(wall_s,1) AS wall_s, rows,
                  round(rows_per_s/1e6,2) AS m_rows_per_s, round(cpu_avg_pct) AS cpu_pct,
                  round(rss_peak_mb) AS rss_peak_mb, round(read_mb_s) AS read_mb_s
           FROM bench_history ORDER BY started_at"""
    )
    _fig = px.scatter(bench.to_pandas(), x="started_at", y="rss_peak_mb", color="task",
                      size="wall_s", title="每次重任务的内存峰值（点越大耗时越长）", height=380)
    mo.vstack([mo.md("## 8. 性能记录（每次重任务自动记录）"), mo.ui.plotly(_fig),
               mo.ui.table(bench, selection=None, page_size=10)])
    return


@app.cell
def _(mo):
    sql = mo.ui.code_editor(
        value=(
            "-- 可查任何视图；原始逐笔很大，务必带 date 过滤\n"
            "SELECT symbol_id, count(*) AS trades, sum(volume) AS volume\n"
            "FROM raw_transaction\n"
            "WHERE date = 20260615 AND trade_type = 1 AND symbol_id = 600000\n"
            "GROUP BY symbol_id"
        ),
        language="sql", min_height=120,
    )
    run = mo.ui.run_button(label="运行")
    mo.vstack([mo.md("## 9. SQL 查询框"), sql, run])
    return run, sql


@app.cell
def _(con, mo, run, sql):
    mo.stop(not run.value, mo.md("_点「运行」执行上面的 SQL（结果最多显示 5000 行）_"))
    try:
        _res = con.sql(sql.value).pl().head(5000)
        _out = mo.ui.table(_res, selection=None, page_size=20)
    except Exception as _e:  # noqa: BLE001 - show the SQL error to the user
        _out = mo.md(f"**SQL 出错：** `{_e}`")
    _out
    return


if __name__ == "__main__":
    app.run()
