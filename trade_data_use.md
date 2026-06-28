# 量化策略数据使用清单

> 本文件梳理 `stock_trade_demo` 当前所有**选股策略**与**择时策略**实际依赖的数据，包含字段、文件路径以及最上游来源。生成日期：2026-05-28。

---

## 0. 共享主数据

所有 A 股选股策略都继承 `BaseStrategy.prepare_data()`，共用同一份月度面板。

| 项目 | 内容 |
|---|---|
| 主数据文件 | `stock_trade_demo/stock_data.csv` (GBK, 月度面板, 键 `(交易日期, 股票代码)`) |
| 镜像 | `stock_trade_demo/stock_data.parquet`（`convert_data.py` 生成） |
| 抓取入口 | `get_stock_info.py --mode supplement` |
| 日 K 上游 | Sina Finance — `https://quotes.sina.cn/cn/api/jsonp_v2.php/.../CN_MarketData.getKLineData?scale=240` |
| 实时 PE/PB/换手 上游 | Tencent Finance — `https://qt.gtimg.cn/q=...` |
| 中间缓存 | `.cache/daily_YYYY-MM.pkl`, `.cache/rtquotes_YYYY-MM.pkl` |
| `BaseStrategy` 必须列 | `上市至今交易天数`, `股票代码`, `总市值`, `bias_20`, `成交额std_10`, `市盈率倒数`, `市净率倒数`, `最高价`, `最低价`, `收盘价`, `MACD`, `DIF`, `DEA`, `涨跌幅_20`, `涨跌幅std_20`, `成交额` |

---

## 1. 选股策略

### 1.1 `original` — 原版小市值策略
- 文件：`stock_trade_demo/strategies/original.py`
- 读取列：`总市值`, `市盈率倒数`, `市净率倒数`, `新版申万二级行业名称`, `bias_20`, `成交额std_10`, `归母净利润_ttm`, `净资产`, `交易日期`
- 数据来源：`stock_data.csv`
- 额外缓存：无

### 1.2 `original_ensemble` — 原版多窗口投票增强
- 文件：`stock_trade_demo/strategies/original_ensemble.py`
- 读取列：与 `original` 相同 + `股票代码`（用于推断 688/689/300/301 板块）+ `下周期每天涨跌幅`
- 额外指数面板：`csi1000_close`, `chinext_close`, `star50_close`（通过 `index_data.build_index_panel` 加载）
- 指数缓存：`.cache/csi1000_daily.csv`, `.cache/chinext_daily.csv`, `.cache/star50_daily.csv`
- 可选 timing 子模块：`ChiNextTimingStrategy` 与 `Star50TimingStrategy`（仅当 `growth_timing_mode='both_signals'`，默认关闭）

### 1.3 `chan_enhanced` — 小市值 + 缠论 tilt
- 文件：`stock_trade_demo/strategies/chan_enhanced.py`
- 读取列：`original` 全部列 + `chan_factors.compute_chan_factors(df)` 在内存中产出的：`chan_above_zs`, `chan_bearish_div`, `chan_top_fractal`, `chan_signal_score`
- 额外缓存：无（缠论因子由月 K 在线计算）

### 1.4 `chan_only` — 纯缠论 + 市值 rank blend
- 文件：`stock_trade_demo/strategies/chan_only.py`
- 读取列：`总市值`, `交易日期` + 在线缠论代理列：`chan_bearish_div`, `chan_top_fractal`, `chan_zs_valid`, `chan_above_zs`, `chan_signal_score`

### 1.5 `method_a` — 日线缠论聚合到月度
- 文件：`stock_trade_demo/strategies/method_a.py`
- 读取列：`original` 列 + 外部缠论因子文件
- 外部因子缓存：`stock_trade_demo/.cache/chan_factors_v2/chan_factors_500.csv` (GBK)
  - 列（运行时 `chan_*` → `ma_*` 重命名）：`chan_top_fractal`, `chan_bottom_fractal`, `chan_fractal_ratio`, `chan_stroke_dir/count/strength`, `chan_zhongshu_count/position/width`, `chan_top_div`, `chan_bottom_div`, `chan_div_signal`, `chan_buy_signals`, `chan_sell_signals`, `chan_segment_count`
- 生产脚本：`chan_monthly_factor_builder.py`（日线缠论 pipeline 聚合到月度）
- 兜底：缓存缺失时退回 `compute_chan_factors` 内存代理

### 1.6 `quality_value` — 质量 + 价值多因子线性合成
- 文件：`stock_trade_demo/strategies/quality_value.py`
- 读取列：`总市值`, `净资产`, `归母净利润_ttm`, `成交额`, `成交额std_10`, `bias_20`, `交易日期`
- 额外缓存：无

### 1.7 `sector_heat` — 行业热度叠加小市值
- 文件：`stock_trade_demo/strategies/sector_heat.py`
- 读取列：`original` 列 + `新版申万一级行业名称`
- 额外输入：`/Users/fatcat/Desktop/quant/strategy/sector_weekly_heat.csv` (UTF-8-SIG)
  - 列：`year_month`, `week_in_month`, `week_label`, `industry`, `weekly_ret_pct`, `n_stocks`, `is_partial`, `n_days_in_week`
  - 策略层只用 `is_partial=False` 的完整周；`is_partial=True` 的当周（不足 5 个交易日）仅供首页热力图展示
- 生产脚本：`scripts/compute_sector_weekly_heat.py`（已接入「更新数据」按钮的 factor 阶段）

---

## 2. 择时策略

所有择时策略继承 `stock_trade_demo/timing/base.py:BaseTimingStrategy`，定义在 `stock_trade_demo/timing/strategies.py`。
最优参数：`/Users/fatcat/Desktop/quant/strategy/best_profile_<sid>.json`（`tuned_params` / `all_params`，离线 walk-forward 产出）。

**信号-成交语义统一**：信号由 close(t) 生成 → 次交易日 (t+1) ETF 开盘价成交 → 次交易日收盘价做估值。

### 2.1 `csi1000_timing` — 中证 1000 择时
| 项目 | 内容 |
|---|---|
| 指数面板列 | `csi1000_close`, `csi1000_high`, `csi1000_low` |
| 指数日线缓存 | `.cache/csi1000_daily.csv`（Sina sh000852 → East Money 兜底） |
| ETF 价格（成交/估值） | `.cache/timing_etf/csi1000_etf_daily_qfq.csv` |
| ETF 上游 | AkShare `fund_etf_hist_em(symbol='510980', adjust='qfq')` |
| 结果缓存 | `.cache/web_cache.pkl` (TIMING `csi1000_timing`) |
| 最优参数 | `strategy/best_profile_csi1000_timing.json` |

### 2.2 `star50_timing` — 科创 50 择时
- 指数列：`star50_close/high/low`（来自 `.cache/star50_daily.csv`，Sina sh000688）
- ETF：`.cache/timing_etf/star50_etf_daily_qfq.csv` ← AkShare `fund_etf_hist_em` 代码 `589850`
- 缓存：`.cache/web_cache.pkl`；最优参数：`strategy/best_profile_star50_timing.json`

### 2.3 `chinext_timing` — 创业板择时
- 指数列：`chinext_close`（来自 `.cache/chinext_daily.csv`，Sina sz399006）
- ETF：`.cache/timing_etf/chinext_etf_daily_qfq.csv` ← AkShare `fund_etf_hist_em` 代码 `159205`
- 缓存：`.cache/web_cache.pkl`；最优参数：`strategy/best_profile_chinext_timing.json`

### 2.4 `sp500_timing` — 标普 500 择时
- 指数列：`sp500_close`（来自 `.cache/sp500_daily.csv`，Sina sh513500 → East Money 兜底）
- ETF：`.cache/timing_etf/sp500_etf_daily_qfq.csv` ← AkShare `fund_etf_hist_em` 代码 `513500`
- 缓存：`.cache/us_timing/sp500_timing.pkl`；最优参数：`strategy/best_profile_sp500_timing.json`

### 2.5 `macro_v32_timing` — 纳指 + 宏观因子择时（v3.2）
- 指数列：`nasdaq_close`（来自 `.cache/nasdaq_daily.csv`，Sina sz159941 → East Money 兜底）
- ETF：`.cache/timing_etf/nasdaq_etf_daily_qfq.csv` ← AkShare `fund_etf_hist_em` 代码 `159941`
- 宏观因子（运行时 `_load_fred_series(name)` 从 `data/fred_<name>.csv` 加载）：
  - `fred_FedFundsRate.csv` (FRED `FEDFUNDS`)
  - `fred_YieldCurve_10Y2Y.csv` (FRED `T10Y2Y`)
  - `fred_CPI_core.csv` (FRED `CPILFESL`)
  - `fred_Unemployment.csv` (FRED `UNRATE`)
  - `fred_VIX.csv` (FRED `VIXCLS`)
  - `fred_HighYieldSpread.csv` (FRED `BAMLH0A0HYM2`)
  - `fred_Treasury10Y.csv` (FRED `DGS10`)
- 缓存：`.cache/us_timing/macro_v32_timing.pkl`；最优参数：`strategy/best_profile_macro_v32_timing.json`

### 2.6 `nasdaq_timing`（未注册，遗留对照）
- 价格源与 `macro_v32_timing` 相同；`registry = None`，仅作历史对比，不进 web。

---

## 3. A 股估值/情绪风险面板（仅展示，不入策略仓位）

| 文件 | 列 | AkShare 上游 |
|---|---|---|
| `data/a_share_macro/pe_ttm.csv` | `date`, `pe_median`, `pe_mean`, `pe_median_q10y` | `stock_a_ttm_lyr`（全 A 个股 PE-TTM 中位数 + 10y 分位） |
| `data/a_share_macro/cn10y.csv` | `date`, `cn10y_pct` | `bond_china_yield`（中债国债收益率曲线，10 年） |
| `data/a_share_macro/sse_daily.csv` | `date`, `sse_float_mcap_yi`, `sse_amount_yi`, `sse_turnover_float_pct`, `sse_margin_buy_yi`, `sse_margin_balance_yi` | `stock_sse_deal_daily` + `stock_margin_sse` |

- 抓取入口：`scripts/fetch_a_share_macro.py`
- 风险信号合成：`scripts/build_risk_signals.py` → `strategy/risk_signals.json`，挂到 `csi1000_timing` / `chinext_timing` / `star50_timing` 看多风险列表
- `build_risk_signals.py` 还读取：`data/fred_VIX.csv`, `fred_Treasury10Y.csv`, `fred_YieldCurve_10Y2Y.csv`, `fred_HighYieldSpread.csv`；以及 `.cache/{csi1000,star50,chinext,nasdaq,sp500}_daily.csv`

---

## 4. 宏观 FRED 数据全集

由 `scripts/download_macro_data.py` 通过 `pandas_datareader.data.DataReader(<code>, 'fred')` 抓取。

| 文件名 | FRED 代码 | 用途 |
|---|---|---|
| `fred_CPI_headline.csv` | `CPIAUCSL` | 通胀（备用） |
| `fred_CPI_core.csv` | `CPILFESL` | **macro_v32 核心通胀** |
| `fred_PCE_headline.csv` | `PCEPI` | 备用 |
| `fred_PCE_core.csv` | `PCEPILFE` | 备用 |
| `fred_Unemployment.csv` | `UNRATE` | **macro_v32 失业率** |
| `fred_NonFarmPayrolls.csv` | `PAYEMS` | 备用 |
| `fred_FedFundsRate.csv` | `FEDFUNDS` | **macro_v32 联邦基金利率** |
| `fred_Treasury10Y.csv` | `DGS10` | **macro_v32 10Y + 风险面板** |
| `fred_Treasury2Y.csv` | `DGS2` | 备用 |
| `fred_YieldCurve_10Y2Y.csv` | `T10Y2Y` | **macro_v32 期限利差 + 风险面板** |
| `fred_DollarIndex.csv` | `DTWEXBGS` | 备用 |
| `fred_VIX.csv` | `VIXCLS` | **macro_v32 VIX + 风险面板** |
| `fred_WTI_Oil.csv` | `DCOILWTICO` | 备用 |
| `fred_GDP_Real.csv` | `GDPC1` | 备用 |
| `fred_IndustrialProduction.csv` | `INDPRO` | 备用 |
| `fred_ConsumerSentiment.csv` | `UMCSENT` | 备用 |
| `fred_M2_MoneySupply.csv` | `M2SL` | 备用 |
| `fred_HighYieldSpread.csv` | `BAMLH0A0HYM2` | **macro_v32 高收益债利差 + 风险面板** |
| `fred_SP500_FRED.csv` | `SP500` | macro_v32 离线研究脚本输入 |
| `fred_NASDAQ100_FRED.csv` | `NASDAQ100` | macro_v32 离线研究脚本输入 |
| `fred_NasdaqComposite_FRED.csv` | `NASDAQCOM` | 备用 |
| `fred_DowJones_FRED.csv` | `DJIA` | 备用 |

同脚本另抓 Yahoo Finance（`yfinance.download`）：QQQ, SPY, TQQQ, DIA, IWM, TLT, GLD, `^VIX` → `data/yf_*.csv`（仅研究用，不入生产策略）。

---

## 5. 离线缓存 vs Web 加载

Web 是只读视图，所有重计算都在离线脚本里：

| 缓存文件 | 生产脚本 | Web 用途 |
|---|---|---|
| `.cache/web_cache.pkl` | `scripts/build_select_cache.py` + `scripts/build_timing_cache.py` | 选股 + A 股择时主缓存 |
| `.cache/us_timing/sp500_timing.pkl` | `scripts/build_us_timing_cache.py` | 美股 SP500 择时 |
| `.cache/us_timing/macro_v32_timing.pkl` | `scripts/build_us_timing_cache.py` | 美股 macro_v32 择时 |
| `strategy/sector_weekly_heat.csv` | `scripts/compute_sector_weekly_heat.py` | `sector_heat` 策略因子 + 首页行业热度热力图（含 partial 当周） |
| `strategy/best_profile_*.json` | `scripts/walk_forward_train.py` | 各择时策略最优参数 |
| `strategy/factor_signals_v32.csv`, `backtest_v32_*.csv`, `v32_performance.csv`, `v32_annual.csv` | `scripts/macro_timing_strategy_v3_2_final.py` | macro_v32 离线研究产物 |
| `strategy/walk_forward_log_*.csv`, `holdout_report_*.md` | `scripts/walk_forward_train.py`, `scripts/build_holdout_reports.py` | 训练/留出报告 |

### 「更新数据」按钮覆盖范围（首页 `triggerDataUpdate`）

`DataUpdateFlow.runChained({ types: ['index', 'aux', 'stock', 'factor'] })` 四段串跑：

| 阶段 | 后端 | 实际产物 |
|---|---|---|
| ① `index` | `POST /api/update_index_data` → `_run_index_data_update` | 5 个指数日/月线、5 只择时 ETF qfq、重建 A 股 + 美股择时缓存 |
| ② `aux` | `POST /api/update_aux_data` → 串跑 `download_macro_data.py` / `fetch_a_share_macro.py` / `build_risk_signals.py` | `data/fred_*.csv` + `data/a_share_macro/*.csv` + `strategy/risk_signals.json` |
| ③ `stock` | `POST /api/update_data` → `_run_data_update` | `stock_data.csv` + parquet 增量；末尾再 rebuild 一次选股 + 择时缓存 |
| ④ `factor` | `POST /api/update_factor_data` → `_run_factor_update` | 跑 `compute_sector_weekly_heat.py`，重算 `strategy/sector_weekly_heat.csv`（含 `is_partial` 当周） |

**仍需手动维护**（不挂在按钮上）：
- `stock_trade_demo/.cache/chan_factors_v2/chan_factors_500.csv`（`method_a` 用，日线缠论 pipeline 重，不在每日刷数据时跑）
- `strategy/best_profile_*.json`（择时策略参数，设计上"刷数据"≠"重训参数"，避免参数漂移；只在 walk-forward 研究时手动跑）

---

## 6. 抓取脚本上游一览

| 脚本 | 抓取目标 | 上游 |
|---|---|---|
| `get_stock_info.py` | A 股月度面板 (`stock_data.csv`) | Sina 日 K + Tencent 实时 |
| `stock_trade_demo/index_data.py` | A 股 / 美股指数日/周/月线 | Sina (`money.finance.sina.com.cn/.../getKLineData`) → East Money (`push2his.eastmoney.com/.../kline/get`) 兜底 |
| `stock_trade_demo/index_data.py` | A 股交易日历 `.cache/a_share_calendar_daily.csv` | Sina sh000001 / sz399001 |
| `stock_trade_demo/index_data.py` | 择时 ETF 日线 (`.cache/timing_etf/*_qfq.csv`) | AkShare `fund_etf_hist_em(adjust='qfq')`；Sina/East Money 不复权兜底 |
| `scripts/download_macro_data.py` | `data/fred_*.csv` | `pandas_datareader` → FRED |
| `scripts/download_macro_data.py` | `data/yf_*.csv` | `yfinance` → Yahoo Finance |
| `scripts/download_index_data.py` | `data/idx_*.csv` + 部分 `fred_*.csv` | Stooq + FRED |
| `scripts/fetch_a_share_macro.py` | `data/a_share_macro/*.csv` | AkShare (`stock_a_ttm_lyr`, `bond_china_yield`, `stock_sse_deal_daily`, `stock_margin_sse`) |
| `scripts/compute_sector_weekly_heat.py` | `strategy/sector_weekly_heat.csv` | 离线由 `stock_data.csv` 行业列聚合 |

---

## 7. ETF 代码 / 上游汇总（择时成交价唯一来源）

| 标的 | ETF 代码 | 主 CSV (qfq) | AkShare 调用 |
|---|---|---|---|
| 中证 1000 | `510980` | `.cache/timing_etf/csi1000_etf_daily_qfq.csv` | `fund_etf_hist_em(symbol='510980', period='daily', adjust='qfq')` |
| 创业板 | `159205` | `.cache/timing_etf/chinext_etf_daily_qfq.csv` | `fund_etf_hist_em(symbol='159205', ...)` |
| 科创 50 | `589850` | `.cache/timing_etf/star50_etf_daily_qfq.csv` | `fund_etf_hist_em(symbol='589850', ...)` |
| 纳指 | `159941` | `.cache/timing_etf/nasdaq_etf_daily_qfq.csv` | `fund_etf_hist_em(symbol='159941', ...)` |
| 标普 500 | `513500` | `.cache/timing_etf/sp500_etf_daily_qfq.csv` | `fund_etf_hist_em(symbol='513500', ...)` |

非 `_qfq` 同名 CSV 为 Sina / East Money 不复权兜底，仅在 AkShare 失败时使用。

---

## 8. 实盘记录（受保护，不入策略输入）

- 文件：`data/live_trades.csv`（不可再生数据，已在 `.gitignore`）
- 服务：`stock_trade_demo/services/live_trades.py`（`/api/live/*` 写入加文件锁）
- 起点：空仓 `actual_position=0`, `NAV=1.0`，禁止任何 seed/demo

---

**备注**：本表只列出当前**实际被生产策略或 Web 端读取**的数据；`archive/` 目录下的历史快照、`xingbuxing_stock_data.csv` 等离线实验文件不计入。
