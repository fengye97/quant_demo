# 量化选股策略 · 回测与可视化

基于 A 股全市场月度数据的多因子量化选股回测系统，支持策略库管理、CLI 对比分析、Web 可视化，以及训练/测试集拆分评估。

## 项目结构

```
stock_trade_demo/
├── strategies/              # 策略库（类化管理）
│   ├── base.py              # BaseStrategy 抽象基类（流水线模板）
│   ├── original.py          # OriginalStrategy — 原版策略（★ 5223x 累积净值）
│   ├── chan_enhanced.py     # ChanEnhancedStrategy — 缠论增强 v1.1
│   ├── chan_only.py         # ChanOnlyStrategy — 纯缠论 v1.2
│   └── method_a.py          # MethodAStrategy — Method A 日线流水线 v2.0
├── backtest.py              # 回测引擎（数据加载、选股、止盈、评估）
├── chan_factors.py          # 缠论因子计算库（分型/背驰/中枢/笔/买卖点）
├── visualization.py         # Matplotlib 可视化（对比图、简洁图）
├── web_app.py               # Flask Web 后端（API + 缓存）
├── web/templates/
│   └── index.html           # ECharts 前端（训练/测试集拆分、持股明细）
├── choose_stock.py          # CLI 入口 — 单策略回测
├── compare_strategies.py    # CLI 入口 — 多策略对比
├── convert_data.py          # 数据格式转换工具（CSV → Parquet）
└── stock_data.csv           # A 股全市场月度数据（~823MB，需自行准备）
```

## 策略库

所有策略继承自 `BaseStrategy`，遵循统一流水线：

```
prepare_data → compute_factors → apply_filters → rank_stocks → 产出"因子"列
```

| 策略 | 累积净值 | 年化收益 | 最大回撤 | 说明 |
|------|---------|---------|---------|------|
| **原版策略** | 5,223x | 55.51% | -71.50% | 行业估值 + bias反转 + 小市值 |
| 缠论增强 v1.1 | 5,131x | 55.37% | -71.57% | 原版过滤 + 缠论代理因子 |
| 纯缠论 v1.2 | 16x | 15.39% | -70.91% | 仅缠论因子 |
| Method A v2.0 | 5,223x | 55.51% | -71.50% | 日线缠论流水线聚合 |

**核心发现**: A 股小市值效应是过去 20 年压倒性的 alpha 来源，任何偏离纯市值排名的增强都会稀释这一效应。

## 快速开始

### 环境要求

- Python 3.8+
- pandas, numpy, flask, matplotlib

### CLI 回测

```bash
# 单策略回测（原版策略）
python3 choose_stock.py

# 多策略对比
python3 compare_strategies.py

# 可视化风格
python3 choose_stock.py --plot compare   # 详细对比图（默认）
python3 choose_stock.py --plot raw       # 简洁风格
python3 choose_stock.py --plot both      # 两种都输出
```

### Web 可视化

```bash
python3 web_app.py
# 访问 http://localhost:8080
```

功能：
- 策略切换（4 种策略）
- 因子参数实时调节（滑块）
- 训练集 / 测试集拆分图表（训练 ≤ 2026-02-28，测试 > 2026-02-28）
- 测试集持股明细（每期股票代码、名称、仓位）
- 手动日期范围选择

## 回测引擎

### 核心机制

- **调仓频率**: 月末选股，持有一个月
- **选股数量**: 每期 6 只（因子排名最小前 6）
- **市场状态**: 全市场等权累积收益 vs MA12 → 牛市/熊市
- **止盈规则**: 牛市 30% 止盈 / 持仓 6 只；熊市 22% 止盈 / 持仓 4 只
- **交易成本**: 买入万 1.0 佣金 + 卖出千 1 印花税

### 评估指标

- 累积净值、年化收益、最大回撤（含起止日期）、Calmar 比率
- 训练/测试集拆分：各自独立计算累积净值、胜率、月度收益
- 月度胜率、年度收益分布

## 数据

`stock_data.csv` 包含 A 股全市场月度数据（2006-12 ~ 2026-05），字段包括：

股票代码、股票名称、交易日期、总市值、涨跌幅、bias_20、成交额std_10、
市盈率倒数、市净率倒数、MACD/DIF/DEA、下周期每天涨跌幅、市场状态 等 54 列

数据文件约 823MB，已加入 `.gitignore`，需自行准备。可使用 `convert_data.py` 将 CSV 转换为 Parquet 格式以加速加载。

## 训练/测试集拆分

- **训练集**: 交易日期 ≤ 2026-02-28（231 个月，2006-12 ~ 2026-02）
- **测试集**: 交易日期 > 2026-02-28（3 个月，2026-03 ~ 2026-05）
- 拆分在回测完成后进行：先在全量数据上运行策略，再按日期切分结果
- Expanding rank percentile 需要全量历史数据计算，因此不能先切分再回测

## 相关文档

| 文档 | 说明 |
|------|------|
| [quant_factor.md](../quant_factor.md) | 量化因子实现规范（13+ 因子详细定义） |
| [STRATEGY_CHANGELOG.md](../STRATEGY_CHANGELOG.md) | 策略版本迭代日志（v1.0 ~ v2.0） |
| [chan_theory_analysis.md](../ref_books/chan_theory_analysis.md) | 缠论体系分析与量化实现 |
| [todo_list.md](../todo_list.md) | 项目 TODO 追踪 |
