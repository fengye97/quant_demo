# ChangeLog

> 量化选股策略项目变更日志。按日期倒序。

---

## 2026-05-18

### 前端重构：训练/测试集拆分可视化

- **web_app.py**: 新增 `compute_split_metrics()` 函数，按 `2026-02-28` 拆分训练/测试集
  - `result_to_json()` 新增 `train_equity_curve`、`test_equity_curve`（各自从 1 开始）
  - split 节点包含 train/test 各自的 metrics、win_rate、monthly_returns、holdings
  - 测试集 holdings 包含每期股票代码、名称、等权仓位

- **web/templates/index.html**: 完全重写
  - 训练集独立图表（蓝色、对数坐标）
  - 测试集独立图表（橙色、线性坐标、月度收益标注）
  - 测试集持股明细弹窗（每期代码+名称+仓位+止盈说明）
  - 训练/测试指标分开展示

### Bug 修复

- **Flask 模板缓存**: `debug=False` 时模板在首次请求时缓存，修改后需重启服务器才能生效
- **端口冲突**: macOS AirPlay 占用 5000 端口，切换至 8080

### 项目文档

- **README.md**: 新增项目说明文档（架构、策略库、使用方法、数据说明）
- **ChangeLog.md**: 新增变更日志

---

## 2026-05-16

### 策略库重构

- 将 `choose_stock.py` 中的策略逻辑拆分为独立的 class 文件：
  - `strategies/base.py` — BaseStrategy 抽象基类
  - `strategies/original.py` — OriginalStrategy（原版策略，5223x）
  - `strategies/chan_enhanced.py` — ChanEnhancedStrategy（缠论增强 v1.1）
  - `strategies/chan_only.py` — ChanOnlyStrategy（纯缠论 v1.2）
  - `strategies/method_a.py` — MethodAStrategy（Method A v2.0）
- `choose_stock.py` 精简为 ~50 行，仅保留 OriginalStrategy

### Method A 日线流水线策略

- `chan_monthly_factor_builder.py`: 对 500 只股票跑完整日线缠论流水线（7 模块），月度聚合 16 个因子
- `method_a_strategy()`: 在小市值桶内用 Method A 因子排序（5% 倾斜）
- 结果：与原版高度相关（r=1.000），因子覆盖率仅 3.7%，需扩大覆盖面

### 缠论因子完整实现

- `chan_factors.py`（~1,400 行）: 实现完整缠论 7 模块流水线
  - InclusionProcessor → FractalDetector → StrokeBuilder → SegmentBuilder
  - → ZhongshuDetector → DivergenceDetector → TradeSignalGenerator
- 识别 10+ 个可量化因子：分型密度、笔斜率、中枢位置、背驰强度、买卖点状态等
- quant_factor.md 新增 Section 17 缠论因子规范

### 数据补充

- `get_stock_info.py` 从 267 行扩展至 1,370 行
  - 批量股票日线获取、55 列技术指标计算（bias/振幅/std/KDJ/MACD）
  - 月度聚合、CSV 补充模式、缓存机制、多线程并发（20 workers）

### Web 可视化初版

- `web_app.py`: Flask 后端，预缓存 4 个策略回测结果
  - `/api/backtest` — 回测数据接口
  - `/api/factors` — 因子参数配置接口
  - `/api/strategy_list` — 策略列表
- `web/templates/index.html`: ECharts 前端
  - 资金曲线（对数坐标）、回撤、年度收益
  - 因子参数滑块、策略切换、日期范围选择

### RL+LLM 选股调研与实现

- `rl_llm_stock_selection_research.md`: 系统调研 RL 选股技术（DQN/PPO/A2C/SAC/Decision Transformer）
- `rl_stock_selector/`: 方案 A 完整实现
  - environment.py（24 维状态空间）、models.py（PPO Actor-Critic）
  - train.py（Clipped Surrogate + GAE）、backtest.py、main.py

---

## 2026-05-15

### 量化理论基础

- 阅读并整理 4 本量化投资经典著作的核心方法论：
  - 《因子投资：方法与实践》 → `ref_books/factor_invest_key.md`（54050 字）
  - 《主动投资组合管理》 → `ref_books/active_portfolio_management.md`
  - Quantitative Equity Portfolio Management（Ludwig Chincarini）→ `ref_books/qepm_en.md`
  - 《量化股票组合管理》 → `ref_books/qepm_cn.md`
  - 《预期收益》 → `ref_books/expected_returns.md`（32114 字）

### 因子规范

- `quant_factor.md` 初版：将 4 本书中可落地的因子整理为统一实现规范
  - 定义全局研究设定（股票池、频率、成本、黑名单）
  - 13+ 因子详细规范（输入字段、计算公式、排序方向、分组规则）

---

## 2026-05-14

### 初始版本

- 从"邢不行"选股课程代码出发，建立项目框架
- `choose_stock.py`: 原版策略（行业估值 + bias反转 + 小市值），累积净值 5223x
- `backtest.py`: 回测引擎（加载数据、选股、止盈、评估）
- `visualization.py`: Matplotlib 可视化（策略对比图、简洁图）

### 缠论学习

- `ref_books.md`: 建立参考书目清单
- 开始精读《缠中说禅》原文 PDF（392MB），提取可量化因子

---

## 策略版本演进

| 日期 | 版本 | 累积净值 | 年化收益 | 最大回撤 | 关键变更 |
|------|------|---------|---------|---------|---------|
| 05-14 | v1.0 原版 | 5,223x | 55.51% | -71.50% | 行业估值+bias反转+小市值 |
| 05-16 | v1.1 缠论增强 | 5,131x | 55.37% | -71.57% | 原版过滤+缠论代理因子(3%倾斜) |
| 05-16 | v1.2 纯缠论 | 16x | 15.39% | -70.91% | 仅缠论因子(70%缠论+30%市值) |
| 05-16 | v2.0 Method A | 5,223x | 55.51% | -71.50% | 日线流水线+5%倾斜，覆盖3.7% |

详细迭代日志见 [STRATEGY_CHANGELOG.md](../STRATEGY_CHANGELOG.md)。
