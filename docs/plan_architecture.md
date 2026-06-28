# quant 项目架构层重构方案（Plan-only）

> 范围：**只针对 `stock_trade_demo/` 子项目**。不动策略数学逻辑，不动 best_profile JSON schema，不动前端模板，不动 `data/live_trades.csv`。

## A. 现状诊断

### 问题 1：两套回测引擎并行未抽象

**月度选股引擎 `stock_trade_demo/backtest.py`**（829 行，10 个顶层符号）
- `apply_take_profit()` @ `backtest.py:364` — 单股逐日止盈+卖出成本，简版撮合。
- `build_period_daily_curve()` @ `backtest.py:402` — 等权组合日线。
- `select_and_backtest()` @ `backtest.py:426` — **签名 9 个位置参数（行 426–432）；函数体 280 行（行 426–705）**，把"排名→选股→regime 取参→止盈→手续费→资金曲线→holdings 明细 JSON"全揉一团。
- `strategy_evaluate()` @ `backtest.py:708` — 评估指标。
- 手续费来源：`c_rate` / `t_rate` 两个标量；卖出成本 `sell_cost = c_rate + t_rate`（行 458），没有"印花税 / 过户费 / 滑点 / 佣金最低值"细分。

**择时 ETF 引擎 `stock_trade_demo/timing/backtest.py`**（1253 行，19 个顶层符号）
- `_attach_etf_prices()` @ `timing/backtest.py:36` — 关键的 t+1 撮合接线，把 signal_date 映射到下一交易日 open/close（CLAUDE.md 第 10 条）。
- `_rebuild_timing_actions()` @ `timing/backtest.py:77` — exposure → action 状态机。
- `_replay_timing_positions()` @ `timing/backtest.py:119`（≈440 行）— 真正的"工业级"撮合：T+1 结算、涨跌停 FIFO 重试、滑点 bps、佣金/印花税/过户费拆分、闲置现金计息、profit_lock 多档止盈（test_timing_realism.py 验证）。
- `run_timing_backtest()` @ `timing/backtest.py:574` — 装配入口。
- `evaluate_timing_result()` @ `timing/backtest.py:716`、`summarize_timing_windows()` @ `timing/backtest.py:874`、`timing_result_to_json()` @ `timing/backtest.py:1005`。
- 反向依赖：`timing/backtest.py:7` 从 `backtest` 反向 `import compute_alpha_beta`，构成 `backtest ↔ timing.backtest` 弱耦合。

**重复 / 不一致点**

| 子能力 | 月度引擎 | 择时引擎 |
|---|---|---|
| 手续费模型 | `c_rate + t_rate` 两标量 | 6 项：commission/stamp/transfer/slippage/min/transfer_rate |
| 撮合时点 | 隐式 close-to-close | t 日 close → t+1 open 成交（`_attach_etf_prices`） |
| 止盈 | `apply_take_profit` 逐日累乘 | `profit_lock_level_1/2/3 + drawdown` 多档 |
| 评估口径 | `strategy_evaluate` 用月度 returns | `evaluate_timing_result` 用日线 + reset_capital 切窗 |
| 日历切片 | `filter_by_date` @ web_app.py:1038 | `filter_timing_result` @ timing/backtest.py:806 |

两套各写一份，但**择时引擎已经实现的滑点/佣金细分/T+1/涨跌停**月度引擎完全没有；任何想给月度策略加"真实滑点"的需求都得复制粘贴 400 行 `_replay_timing_positions`。

---

### 问题 2：`web_app.py` 3897 行上帝文件

```
$ wc -l stock_trade_demo/web_app.py   # 3897
```

**装载 + 路由 + 业务计算 + 持久化全混**。`grep -n "^def \|@app\.route" web_app.py` 给出 ~80 个顶层符号。粗分块：

| 行号区间 | 关注点 |
|---|---|
| 26–58 | 大段 import（含 `get_stock_info` 数据抓取模块） |
| 66–120 | 全局可变状态：`DATA_DF / INDEX_RETURNS / BACKTEST_CACHE / TIMING_PANEL / TIMING_CACHE / US_TIMING_PANEL / US_TIMING_CACHE / CSI1000_SIGNAL_SERIES / _LIVE_*` 全部 module-level |
| 208–267 | `_save_disk_cache / _load_disk_cache / _load_factor_backtest_cache`（IO 层） |
| 292–311 | 三张硬编码 `_STRATEGY_MAP / TIMING_STRATEGY_MAP / US_TIMING_STRATEGY_MAP` |
| 611–725 | `ensure_*_loaded / init_cache / init_timing_cache / init_us_timing_cache`（启动预热） |
| 825–957 | `build_strategy / build_timing_strategy / build_us_timing_strategy + best_profile/risk_signals 装载` |
| 1012 | **`run_backtest_fresh`** — 在请求路径上跑全量月度回测，**直接违反 CLAUDE.md 第 12 条** |
| 1038 | `filter_by_date` — 选股切窗 |
| 1188–1505 | holdings payload / 交易日历 / 单股快报 / benchmark 曲线（应属"展示层"） |
| 1506–1822 | `build_selection_interval_windows / compute_split_metrics / result_to_json` ≈ 1300 行 JSON 序列化 |
| 2066–3617 | ~40 个 `@app.route`，含 `/`、`/timing`、`/us_timing`、`/live`、`/api/info`、`/api/backtest`、`/api/timing/*`、`/api/us_timing/*`、`/api/live/*`、`/api/update_data*`、`/api/update_index_data*`、`/api/sector_heat`、`/api/factor_single_backtest` |
| 2904–3157 | `/api/live/*` 内联文件锁 + CSV 读写（CLAUDE.md 第 15 条保护对象） |
| 3365–3577 | 数据更新（A 股月度 + 指数/ETF）触发 + 状态轮询，含 ETF freshness guard `_check_a_share_index_etf_alignment` |

**请求路径上现存的"strategy.run()"重计算点（违反 CLAUDE.md 第 12 条）**：
- `web_app.py:657` — `init_cache` 启动时 `csi_strategy.run(TIMING_PANEL.copy())`（启动期可接受）
- `web_app.py:680` — `init_cache` 启动时 OriginalStrategy
- `web_app.py:804` — `init_timing_cache` 启动期遍历 TIMING_STRATEGY_MAP 全部 `.run()`
- `web_app.py:954` — `run_timing_backtest_fresh` 在 `/api/timing/backtest` cache miss 时调用（参数偏离默认就走 fresh）
- `web_app.py:1020` — `run_backtest_fresh` 同上，在 `/api/backtest` cache miss 时全量月度重算
- `web_app.py:2260` — `/api/us_timing/backtest` 调用 fresh 路径

`run_backtest_fresh`（行 1012）执行 `strategy.run(DATA_DF.copy())` + `select_and_backtest(...)`，对 GB 级数据全量重算，每次请求都可能命中（参数 slider 一动就 miss cache）。

---

### 问题 3：策略硬编码注册 + best_profile/CHANGELOG meta 散落

- `web_app.py:292–311` 三张 `*_MAP` 静态字典，每加一个策略都得改 web_app。
- `web_app.py:318–361` `TIMING_CHANGELOG_META / US_TIMING_CHANGELOG_META` 内嵌大段中文文案，本应作为策略元数据归属策略类。
- `_BEST_PROFILE_CACHE / _PROFILE_SUMMARY_CACHE / _HOLDOUT_REPORT_CACHE / _RISK_SIGNALS_CACHE` 四个 module-level 缓存字典分散在 70–96 行。
- `BaseStrategy` (`strategies/base.py:19`) 已经有 `strategy_id` / `display_name` 类属性，但**没有自动注册机制**，注册仍要回到 web_app 手动登记。
- `BaseTimingStrategy` 在 `timing/base.py:6`，同样无注册。

---

## B. 目标包结构

```
stock_trade_demo/
├── engine/                           # 新增：与 Flask/前端完全解耦的纯计算层
│   ├── __init__.py
│   ├── types.py                      # Trade / Position / RegimeTag / BacktestResult dataclass
│   ├── costs.py                      # CommissionModel(commission_rate, stamp, transfer, slippage_bps, min)
│   ├── execution.py                  # 撮合规则：T+1 / 涨跌停 / FIFO retry / slippage（搬自 _replay_timing_positions 抽公共部分）
│   ├── take_profit.py                # 月度等权止盈（apply_take_profit + build_period_daily_curve 搬家，不改逻辑）
│   ├── portfolio.py                  # 等权 / 排名加权 / build_position_weights 入口
│   ├── metrics.py                    # alpha/beta/IR/drawdown/calmar，把 backtest.compute_alpha_beta 和 evaluate_timing_result 共享部分汇拢
│   └── window.py                     # 通用"全历史 replay + 视窗切片 + reset_capital"（CLAUDE.md 第 13 条）
│
├── backtest_monthly.py               # 旧 backtest.py 改名，select_and_backtest 拆为薄装配层
├── timing/
│   ├── backtest.py                   # 保留 _attach_etf_prices / _replay_timing_positions 中 ETF 专属部分（t+1、has_real_etf_bar、etf_inception_date），底层撮合改调 engine.execution
│   └── ...
│
├── strategies/
│   ├── base.py                       # BaseStrategy.__init_subclass__: 自动登记 + strategy_id 校验
│   ├── registry.py                   # 唯一 STRATEGY_REGISTRY / TIMING_REGISTRY 出口
│   ├── meta.py                       # ChangelogMeta dataclass + 给类挂 meta 字段（替代 web_app TIMING_CHANGELOG_META 大块）
│   └── ...（既有策略文件，加 strategy_id 类属性）
│
├── services/                         # 新增：Flask 之外可复用的"业务编排"层
│   ├── cache_store.py                # _save_disk_cache / _load_disk_cache / FACTOR_BACKTEST_CACHE 集中
│   ├── data_loader.py                # ensure_*_loaded + DATA_DF / TIMING_PANEL / INDEX_RETURNS 包成 AppState
│   ├── live_trades.py                # 文件锁 + CSV 读写，给 /api/live/* 用（CLAUDE.md 第 15 条隔离层）
│   ├── timing_signal_gate.py         # CSI1000_SIGNAL_SERIES 装载 + 提供给 select 流程
│   ├── best_profile.py               # _load_best_profile / _load_holdout_report / _load_risk_signals
│   └── jobs.py                       # _run_data_update / _run_index_data_update 后台线程
│
└── web/
    ├── app.py                        # ≤ 200 行：create_app() + Blueprint 注册 + 启动钩子（init_cache）
    ├── api_common.py                 # _normalize_benchmark_id / _parse_realism_bool / _collect_realism_params 共用
    ├── serializers.py                # result_to_json / timing_result_to_json wrappers（从 web_app 1505–1822 搬出）
    └── blueprints/
        ├── pages.py                  # /、/timing、/us_timing、/live HTML 渲染
        ├── select_api.py             # /api/info /api/strategy_list /api/backtest /api/factors /api/factor_overview
        ├── timing_api.py             # /api/timing/* 全部 A 股择时
        ├── us_timing_api.py          # /api/us_timing/*
        ├── live_api.py               # /api/live/* （唯一允许写 live_trades.csv 的入口）
        ├── data_admin_api.py         # /api/update_data* /api/update_index_data*
        └── factor_explore_api.py     # /api/sector_heat /api/factor_single_backtest /api/timing/explore_compare
```

---

## C. 迁移步骤（按风险递增）

每步独立 PR-able，按列出顺序合并；前 5 步**纯文件搬家 + 增加薄层**，第 6 步起才动撮合代码。

### Step 1 — 抽出 `web/serializers.py`（纯函数搬家，零风险）

- **改动范围**：把 `web_app.py:1188–1822`（`_safe_float_or_none / _normalize_stock_code / _extract_open_stock_codes / _fetch_open_stock_quotes / _build_stock_payload / _build_holdings_payload / _compute_single_benchmark_curve / _compute_benchmark_curves / _build_period_benchmark_returns / build_selection_interval_windows / compute_split_metrics / result_to_json`）原样搬到 `stock_trade_demo/web/serializers.py`，web_app 改 `from web.serializers import *`。
- **验收**：`pytest stock_trade_demo/tests/`（应全绿，本步不动 timing 引擎）+ 手测 `/api/backtest?strategy=original` 返回 JSON 字节级等价（diff 旧/新响应）。
- **回滚成本**：极低，纯 rename + import；revert 单 commit 即可。
- **Cache rebuild**：不需要。

### Step 2 — 抽出 `services/live_trades.py`（受保护资源隔离）

- **改动范围**：把 `web_app.py:2904–3157` 的 `_ensure_live_trades_file / _read_live_trades / _write_live_trades / _next_record_id` + `_LIVE_TRADES_LOCK / _LIVE_TRADES_FILE / _LIVE_TRADES_COLUMNS / _LIVE_INITIAL_CAPITAL / _LIVE_LOT_SIZE / _LIVE_CURRENCY` 搬到 `services/live_trades.py`。**严格保留追加列向后兼容写法**（`_LIVE_TRADES_COLUMNS` 顺序、`r.get(k, '')` 默认空字符串）。
- 验证：手测 `GET /api/live/records` 与 `POST /api/live/record` + `DELETE /api/live/record/<id>`；用 `git stash` + 文件对比 `data/live_trades.csv` 内容**绝对未变**。
- **风险点**：CLAUDE.md 第 15 条 — 任何对 `data/live_trades.csv` 的写动作必须保持原有 atomic rename (`tmp_path` + `os.replace`) + 文件锁；本步**禁止改写 schema**。
- **回滚成本**：低。
- **Cache rebuild**：无关。

### Step 3 — 抽出 `services/cache_store.py + data_loader.py + best_profile.py`（IO 层归拢）

- **改动范围**：搬 `web_app.py:208–267`（disk cache）/`611–725`（ensure_*_loaded）/`834–895`（_load_best_profile / _load_holdout_report / _load_risk_signals）。把 5 个 module-level 缓存字典封装为 `AppState` 单例 dataclass。`web_app.py` 改为 `from services.data_loader import APP_STATE`，引用 `APP_STATE.DATA_DF / APP_STATE.TIMING_PANEL` 等。
- **验收**：
  - 启动日志保持 `[init] CSI1000 择时信号预加载完成，共 N 条` 一致；
  - `/api/info`、`/api/timing/info`、`/api/us_timing/info` 返回 max_date/min_date 不变；
  - 跑 `pytest tests/test_timing_realism.py`。
- **回滚成本**：低；如果 import 顺序问题，回退即可。
- **Cache rebuild**：**不要**重建 `.cache/web_cache.pkl` / `us_timing/*.pkl` / `best_profile_*.json`；本步只是改读取入口。

### Step 4 — Blueprint 化路由（拆 web_app 到 5 个 blueprint）

- **改动范围**：建 `web/app.py` 工厂 + 6 个 blueprint（pages / select_api / timing_api / us_timing_api / live_api / data_admin_api / factor_explore_api），原 `@app.route` 注册函数原样搬过去。`web_app.py` 缩为 30 行兼容 shim：`from web.app import create_app; app = create_app()` 供旧 `python web_app.py` 启动命令继续工作（CLAUDE.md 第 5 条：不能让 8080 上的旧启动命令失效）。
- **请求路径上的 strategy.run()/run_backtest_fresh 暂不动**，先保持等价。
- **验收**：
  1. `lsof -nP -iTCP:8080`，重启后 PID 切换；
  2. 逐个 hit `/`、`/timing`、`/us_timing`、`/live`、`/api/info`、`/api/strategy_list`、`/api/backtest?strategy=original`、`/api/timing/backtest?strategy=csi1000_timing`、`/api/us_timing/backtest?strategy=macro_v32_timing`、`/api/live/records`、`/api/update_data/status`，diff JSON 字节相等；
  3. **运行 `frontend-screenshot-verify` skill**，截图首页 + /timing + /us_timing + /live 各 1 张，对比改动前后 DOM 文本无回归（CLAUDE.md 第 7 条）。
- **回滚成本**：中（涉及路由注册顺序、Flask context、blueprint url_prefix）；建议在独立分支 + 临时端口 8081 上跑通后再切。
- **Cache rebuild**：无。

### Step 5 — 策略自动注册：`BaseStrategy.__init_subclass__`

- **改动范围**：
  - `strategies/base.py`：在 `BaseStrategy.__init_subclass__(cls, **kwargs)` 里读 `cls.strategy_id`，写入 `strategies/registry.py:STRATEGY_REGISTRY`，重复 id 抛 `ValueError`。
  - 同理为 `timing/base.py:BaseTimingStrategy` + `TIMING_REGISTRY / US_TIMING_REGISTRY`（用 `cls.market_group` 区分）。
  - `web_app.py` 行 292–311 删除三张硬编码 `*_MAP`，改 `from strategies.registry import STRATEGY_REGISTRY as STRATEGY_MAP`。
  - 把 `TIMING_CHANGELOG_META / US_TIMING_CHANGELOG_META`（行 318–361）以 `cls.changelog_meta = {...}` 放到每个择时策略类上；`api/timing/strategy_list` 改为从类属性读。
- **验收**：`/api/strategy_list`、`/api/timing/strategy_list`、`/api/us_timing/strategy_list` 返回的 id 列表 + changelog 文案与旧版完全一致；`pytest`。
- **回滚成本**：中（注册副作用依赖 import 顺序；要在 `services/data_loader.py` 顶部强制 `import strategies; import timing.strategies` 触发注册）。
- **Cache rebuild**：无。

### Step 6 — 把 `request-path` 上的 `run_backtest_fresh` / `run_timing_backtest_fresh` 改成 "load-only"（落实 CLAUDE.md 第 12 条）

- **改动范围**：
  - 写离线脚本 `scripts/build_select_cache.py`（参考已有 `scripts/build_us_timing_cache.py`），把 `_STRATEGY_MAP` 全部 × 默认参数预跑，写盘到 `.cache/select/<strategy>.pkl`。
  - `select_api.api_backtest`：cache miss（参数偏离默认）时 **返回 400 + 文案"该参数组合无离线缓存，请运行 scripts/build_select_cache.py"**，不再现场跑 `run_backtest_fresh`。可选保留 `?allow_fresh=1` 调试开关供本地 dev。
  - 同理处理 `timing_api.api_timing_backtest`（行 3251）和 `us_timing_api.api_us_timing_backtest`（行 2260）。
- **验收**：
  - `/api/backtest?strategy=original` 在缓存命中时 < 1s；
  - 故意传 `?val_pct_cutoff=0.7` 这种非默认值，返回 400 + 错误指向脚本路径；
  - `test_timing_realism.py` 全绿；
  - `frontend-screenshot-verify` 看默认页面无回归。
- **回滚成本**：中高（前端可能依赖某些 slider 实时改参；需要核 `web/templates/index.html` 里的 slider 行为，本步只能限制 select 页 slider 默认在"仅展示"模式）。
- **Cache rebuild**：**首次部署必须先跑 `scripts/build_select_cache.py`**，否则页面会全报 400。

### Step 7 — 抽 `engine/costs.py + engine/take_profit.py + engine/portfolio.py`（月度引擎纯函数下沉）

- **改动范围**：
  - 把 `backtest.py:364–423` (`apply_take_profit`, `build_period_daily_curve`) 整体搬 `engine/take_profit.py`，签名不动。
  - 把 `BaseStrategy.build_position_weights` 移 `engine/portfolio.py`（保留 base.py 的 thin wrapper 兼容旧调用）。
  - 把 `c_rate / t_rate / sell_cost` 包成 `engine/costs.py:CommissionModel(c_rate=..., t_rate=..., slippage_bps=0, commission_min=0)`；月度引擎只用前两项，择时引擎可用全套。
  - `select_and_backtest` 内部改为构造 `CommissionModel` 后调用新模块；**body 数学逻辑不变**。
- **验收**：跑 `python compare_strategies.py` + `python choose_stock.py` 输出与改动前 byte-equal；`/api/backtest?strategy=original` 累积净值小数点后 6 位等同。
- **回滚成本**：中（数学结果必须严格等价；建议先用 `pytest -k "test_apply_take_profit"` 加一组黄金值固化测试再搬）。
- **Cache rebuild**：建议**清掉 `.cache/web_cache.pkl`** 重建一次，确认 byte-equal。

### Step 8 — 抽 `engine/execution.py + engine/window.py`（择时引擎下沉，最高风险）

- **改动范围**：
  - `_replay_timing_positions` (行 119–571) 拆为：
    - `engine/execution.py:replay_positions(panel, costs, settlement, limit_pct, slippage_bps, ...)` — 通用撮合循环（T+1 / 涨跌停 FIFO / 滑点 / 现金计息 / 费用拆分）。
    - `timing/backtest.py` 保留 ETF 专属预处理：`_attach_etf_prices`、`has_real_etf_bar` 过滤（CLAUDE.md 第 11 条）、`signal_date` t+1 接线、`first_real_etf_date` attrs。
  - `filter_timing_result` + `filter_by_date` 共用 `engine/window.py:slice_with_warmup(result, start, end, reset_capital)`，强制保留全历史 replay → 仅切片显示（CLAUDE.md 第 13 条）。
- **验收**：
  1. `pytest stock_trade_demo/tests/test_timing_realism.py` **必须 9 个测试用例全绿**（特别是 `TestTPlusOneSettlement / TestBug4Bug5InceptionLeak / TestLimitUpBlocking / TestSlippage / TestFeeSplit / TestCashInterest`）；
  2. 对每个择时策略 × {`recent_1m, recent_1q, recent_6m, 全历史`} 4 窗口，diff 改动前后 `metrics.cumulative_return / max_drawdown / cash_interest_total / commission_total` 应严格相等（落差 < 1e-9）；
  3. **CLAUDE.md 第 14 条产品目标硬约束**：每个 best_profile 在三个默认窗口下"累计收益 ≥ ETF & 最大回撤 ≤ ETF"必须保持；用 `summarize_timing_windows` 自动对比；
  4. `frontend-screenshot-verify` 看 /timing 页面所有指标卡数值不变。
- **回滚成本**：高 — 这是真改撮合代码，任何 off-by-one 都会把 best_profile 评估带偏。建议在 feature branch 长期 review，**至少 1 周观察**才合并。
- **Cache rebuild**：**强制重建** `.cache/timing/*.pkl` 与 `.cache/us_timing/*.pkl`；首次重建后 byte-diff 校验。

---

## D. 不做什么（显式 out-of-scope）

1. **不改任何策略的数学逻辑**：`compute_factors / rank_stocks / build_position_weights` 实现不动；权重公式、止盈阈值、staged 状态机阈值不动。
2. **不改 `strategy/best_profile_*.json` 的 schema**：键名、`all_params` 嵌套结构、`metrics` 字段保持兼容；只允许在 `_load_best_profile` 入口扩展默认值合并。
3. **不改前端模板**：`web/templates/index.html`、`/timing`、`/us_timing`、`/live` 页面 DOM 结构与 ECharts 选项不动；JSON payload 字段 / 顺序保持 byte-equal（用 diff 校验）。
4. **不改 `data/live_trades.csv` 的列结构、文件锁、atomic write 方式**（CLAUDE.md 第 15 条）。
5. **不改 `.gitignore` 中 `data/live_trades.csv` 的忽略规则**。
6. **不改数据更新链路的副作用**：`POST /api/update_data` / `POST /api/update_index_data` 行为、ETF freshness guard（`_check_a_share_index_etf_alignment` 行 3365）、`get_index_daily(force_refetch=True)` 的回退规则不动；只是把代码挪到 `data_admin_api.py`。
7. **不动 timing 与 monthly 的 cache 文件路径**（`.cache/web_cache.pkl`、`.cache/us_timing/*.pkl`）。
8. **不引入新依赖**（pydantic / fastapi / dependency-injector 等）；保持 Flask + dataclass。
9. **不改 `web_app.py` 启动命令**：保留 30 行 shim 让 `/Users/fatcat/opt/anaconda3/bin/python web_app.py` 仍能正常起 8080（CLAUDE.md 第 5 条）。
10. **不删 legacy 脚本**：`choose_stock.py / compare_strategies.py / run_weekly_experiment.py` 通过 import path 旧→新别名继续工作。
11. **不增加 cold-start fit**：CLAUDE.md 第 13 条 — 视窗切片绝不允许在窗口内冷启动。

---

## E. 风险清单（按 CLAUDE.md 规则编号）

| # | CLAUDE.md 规则 | 风险点 | 防御手段 |
|---|---|---|---|
| 1 | **第 5 条** 8080 端口热加载 | Step 4 blueprint 化后旧 `web_app.py` 启动命令失效；旧进程未重启导致用户看到 stale 代码 | 保留 `web_app.py` 30 行 shim；merge 时强制 `lsof -nP -iTCP:8080 -sTCP:LISTEN` 验证 PID 与 Anaconda 解释器；记录精确启动命令 |
| 2 | **第 10 条** t+1 撮合 | Step 8 把 `_replay_timing_positions` 抽到 engine 时，可能把 signal_date / etf_open / etf_close 三列的对齐挪错位置；fixed-point bug 影响所有择时策略历史净值 | `_attach_etf_prices` 保留在 `timing/backtest.py` 不下沉；`engine/execution.py` 入参必须是 *已经标好 signal_date → exec_open/exec_close 的 panel*，签名层面拒绝接受未对齐数据；`test_timing_realism.py::TestTPlusOneSettlement` 必须先红再绿 |
| 3 | **第 11 条** ETF 不可伪造 | 抽出 `engine/window.py` 时如果误用 `forward fill` 处理切片缺失会触发 | `slice_with_warmup` 严禁 fillna/interp；`first_real_etf_date < start_date` 时直接返回 `non_tradable=True` |
| 4 | **第 12 条** 请求路径无重算 | Step 6 前 web_app 仍允许 fresh 路径；若 Step 6 推迟，Step 8 的撮合改动可能在请求路径上放大复杂度 | Step 6 必须在 Step 8 前合并 |
| 5 | **第 13 条** 全历史 replay → 视窗切片 | `engine/window.py` 抽象化后，新调用方可能把"视窗内"panel 当作"训练集"喂给 strategy.run | `slice_with_warmup` 签名强制 `(full_result_df, view_start, view_end)` 而不是 `(view_panel, ...)`；docstring 列入第 13 条 |
| 6 | **第 14 条** 择时最低产品目标 | Step 8 撮合下沉若引入 1e-6 数值漂移，可能把临界 best_profile（如 csi1000 近 1 月恰好 ≥ ETF）拉到 < ETF，触发"默认不准发布"约束 | Step 8 验收时跑全套 `summarize_timing_windows` × 3 窗口 × {csi1000, chinext, star50, sp500, macro_v32}，结果与改动前 diff < 1e-9 才能合并 |
| 7 | **第 15 条** `data/live_trades.csv` 受保护 | Step 2 抽出 `services/live_trades.py` 时，新增的"批量 migrate / seed demo" 写法绝对禁止 | `live_trades.py` 顶部 docstring 写明禁令；`/api/live/record` 的 POST 入参依然只支持单条 append；不暴露 truncate / replace endpoint；`_LIVE_TRADES_FILE` 路径写为 readonly constant，引用方只能通过函数访问 |
| 8 | ETF freshness guard | Step 3 把 `_run_index_data_update` / `_check_a_share_index_etf_alignment`（web_app.py:3365, 3391）搬到 `services/jobs.py` 时，若漏掉 A 股指数 vs ETF 最新日期比对，会绕过 freshness guard | 把 guard 改为独立纯函数 `services/jobs.py:check_alignment(...)`，并在 `data_admin_api.api_update_index_data_status` 单测 + 启动期烟雾测试 |
| 9 | 注册副作用 | Step 5 `__init_subclass__` 自动注册依赖 import 顺序；Flask CLI / pytest fixture 可能没 import 所有策略 → `STRATEGY_REGISTRY` 缺项 | `strategies/__init__.py` / `timing/__init__.py` 显式 re-export 所有策略；`services/data_loader.py` 顶部 `import strategies, timing.strategies` 强制触发 |
| 10 | best_profile 兼容 | Step 5 后类自带 `changelog_meta`，但旧 `strategy/best_profile_*.json` 仍权威；前端混合显示时若 dataclass field 与 JSON key 不一致，UI 会闪退 | meta dataclass 仅承载 changelog 文本，不承载 `all_params`；后者继续从 JSON 读 |
| 11 | 缓存 pickle 版本 | Step 7/8 改 result_df 列名或 attrs 会导致旧 `.cache/web_cache.pkl` 反序列化后字段缺失 | 保持列名与 attrs 严格不变；如要变，bump `_CACHE_VERSION` 并强制重建 |
| 12 | 实盘 reconcile 链路 | `/api/live/reconcile`（行 3038）直接读 `TIMING_CACHE / US_TIMING_CACHE`，blueprint 化后若 cache 装载顺序错，会返回空 | `services/data_loader.py:get_timing_cache(strategy)` 内部 lazy init + 抛明确错误，不返回 None |
| 13 | 长尾路由 | `/api/sector_heat`（行 3755）、`/api/factor_single_backtest`（行 3814）依赖 `_run_single_factor_backtest`（行 3678），是 web_app 内最后一个"请求路径上跑回测"的隐蔽点 | Step 6 一并处理：把 `_run_single_factor_backtest` 改为读 `FACTOR_BACKTEST_CACHE` 离线产物，缺失返回 400 |

---

## 附：迁移先后顺序决策依据

- 1→2→3 是**纯文件搬家**，无任何语义改动，最先做以建立目录骨架。
- 4 是**Flask 层重组**，独立分支验证后切换，触发 8080 重启。
- 5 是**注册机制**，依赖 1–4 已稳定。
- 6 是**最重要的合规修正**（CLAUDE.md 第 12 条），必须在引擎抽象前完成，避免请求路径上引擎调用复杂化。
- 7 月度引擎下沉风险低（无 t+1 / 涨跌停）。
- 8 择时引擎下沉风险最高，单独长周期 review。
