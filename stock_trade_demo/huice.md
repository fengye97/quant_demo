# 区间回测重构记录

## 用户需求

当前交易系统里的固定训练集 / 固定测试集语义需要重构为统一的近端窗口语义，具体要求如下：

1. 去掉旧的固定 train/test 切分方法，不再使用 `2026-03-31 / 2026-04-01` 这类硬编码边界。
2. 用三个快速周期作为默认验证窗口：
   - 近一月
   - 近一季
   - 近半年
3. 半年前的全部历史数据只作为策略可见先验和状态 warm-up 背景，不再对外展示为固定“训练集”。
4. 选股策略与择时策略在不同回测区间中必须使用完全相同的一组参数，不能按窗口单独调参。
5. 任意回测区间都必须使用该区间起点之前已经可见的全部历史信息，不能只在可见区间内冷启动拟合。
6. 用测试窗口做验证时，必须重置初始资金，不能继承窗口之前的累计收益；测试窗口指标只反映窗口内收益、年化、回撤等 period-local 结果。
7. 这些规则需要同步更新到 `CLAUDE.md`，删除冲突的旧逻辑。
8. A 股 `/timing`、美股 `/us_timing`，以及仍显示旧 split 文案的选股页 `/` 都要按新的语义更新。

## 当前已确认的规则与约束

1. Timing 策略必须保持全历史 replay 后再切片展示，不能把可见窗口当作冷启动拟合窗口。
2. Timing 交易价格语义必须保持一致：
   - 信号由 close(t) 生成
   - 成交使用 ETF next open(t+1)
   - 浮盈 / 持仓估值使用 ETF next close(t+1)
3. 缺失 ETF 历史时不能伪造价格，不能 forward fill / backward fill / 插值。
4. Web 前端应只读离线产物，不应在 `web_app.py` 启动或首个请求里在线预热重算。
5. 前端修改验收必须看真实页面和截图，不能只看代码或 API。

## 本轮已完成的改动

### 1. 计划与任务整理

- 已重写并确认区间回测重构方案。
- 已建立并推进以下任务：
  - `#83 Refactor timing interval window semantics`
  - `#84 Update frontend interval labels`
  - `#85 Revise CLAUDE interval rules`
  - `#86 Record interval-refactor notes in huice.md`

### 2. Timing 回测核心常量已开始替换

文件：`stock_trade_demo/timing/backtest.py`

已去掉旧固定 split 常量，改为新的相对窗口常量：

```python
INTERVAL_WINDOW_MONTHS = {
    'recent_1m': 1,
    'recent_1q': 3,
    'recent_6m': 6,
}

INTERVAL_WINDOW_LABELS = {
    'pre_6m_history': '半年前历史',
    'recent_6m': '近半年',
    'recent_1q': '近一季',
    'recent_1m': '近一月',
}
```

这一步说明后端已经开始从旧固定 train/test 定义切到新的近端窗口体系，但窗口摘要和指标口径还没完全替换。

## 当前代码状态检查结论

### `timing/backtest.py`

- `run_timing_backtest(...)` 仍是正确主链路：
  - 先全历史 replay
  - 再挂 ETF next-open / next-close 价格
  - 再生成交易和净值结果
- `filter_timing_result(...)` 已具备“先全历史、后切片”的正确方向。
- `summarize_timing_windows(...)` 仍然还是旧逻辑，当前还在输出：
  - `train`
  - `validation_april`
  - `validation_may`
  - `validation_all`
- `evaluate_timing_result(...)` 当前默认按已有累计资金曲线计算指标，还没有单独支持“窗口内重置资金”的 period-local 统计口径。

### `web_app.py`

以下旧逻辑仍在：

- 文件头部说明仍写着旧 split 语义。
- 仍保留 `SPLIT_DATE = pd.to_datetime('2026-03-31')`。
- Timing API 仍向前端下发：
  - `payload['windows'] = summarize_timing_windows(...)`
  - `payload['train_end']`
  - `payload['test_start']`
- 选股页后端仍保留：
  - `compute_split_metrics(...)`
  - `result_to_json(..., split_date=SPLIT_DATE, ...)`
  - 训练/测试拆分曲线与字段

### `web/templates/timing.html`

以下旧依赖仍在：

- 卡片中仍显示训练集 / 测试集收益与回撤。
- JS 仍依赖：
  - `windows.train`
  - `windows.validation_all`
  - `train_end`
  - `test_start`
- 图表仍保留：
  - `← 训练 | 测试 →` 分隔线
  - `ETF测试段` 图例

### `web/templates/us_timing.html`

以下旧依赖仍在：

- 顶部仍有固定 split 横幅：
  - 训练集 `2015-01-01 ~ 2026-03-31`
  - 测试集 `2026-04-01 起`
- 卡片中仍显示训练集 / 测试集收益与回撤。
- 图表仍保留旧 split 分隔线与 `ETF测试段` 图例。

### `web/templates/index.html`

以下旧 split 文案仍在：

- 标题中的 `v2 · 训练/测试集`
- 副标题中的 `训练集: ≤2026-03-31 | 测试集: >2026-03-31`
- 页面主体中的“训练集 / 测试集”区块与持股说明

### `CLAUDE.md`

当前已包含以下正确原则：

- 选定区间时必须知道区间起点前历史
- Timing 交易价格语义必须保持 next-open / next-close
- 不允许伪造 ETF 历史
- 前端只读离线产物，不在请求路径在线重算

但仍残留旧 train/test 措辞，需要统一改写为：

- 默认验证窗口为近一月 / 近一季 / 近半年
- 半年前历史作为可见先验背景
- 参数跨窗口一致
- 窗口验证指标按窗口首日重置资金

## 接下来要继续推进的工作

### 第一阶段：完成 timing 后端窗口语义替换

目标文件：`stock_trade_demo/timing/backtest.py`

1. 把 `summarize_timing_windows(...)` 改成输出：
   - `pre_6m_history`
   - `recent_6m`
   - `recent_1q`
   - `recent_1m`
2. 增加“窗口内独立资金曲线”的评估逻辑：
   - 保留全历史 replay 的信号与仓位状态连续性
   - 但窗口绩效统计重新以窗口首日资金为基准计算
3. 输出每个窗口的：
   - 起止日期
   - 行数 / 交易日数
   - 是否可交易
   - 收益率 / 年化 / 回撤 / 调仓次数等摘要

### 第二阶段：同步替换 API payload

目标文件：`stock_trade_demo/web_app.py`

1. Timing / US Timing API 不再返回：
   - `train_end`
   - `test_start`
   - 旧 `windows.train` / `windows.validation_all`
2. 改为统一返回新的窗口摘要结构，例如：
   - `interval_windows`
3. 清理头部旧 split 注释。

### 第三阶段：清理前端旧 split 文案

目标文件：
- `stock_trade_demo/web/templates/timing.html`
- `stock_trade_demo/web/templates/us_timing.html`
- `stock_trade_demo/web/templates/index.html`

需要删除：

- 训练集 / 测试集收益回撤卡片
- 固定 split 横幅
- `← 训练 | 测试 →` 标记线
- `ETF测试段` 图例
- 首页所有 `≤2026-03-31 / >2026-03-31` 文案

改为强调：

- 近一月 / 近一季 / 近半年是默认验证窗口
- 半年前历史只作为先验背景
- 可见区间指标按窗口首日重置资金

### 第四阶段：更新 `CLAUDE.md`

写成新的唯一规则，删除冲突旧逻辑。

### 第五阶段：真实页面验证

需要对以下页面做最终验证和截图：

- `/timing`
- `/us_timing`
- `/`

验收重点：

1. 不再出现旧 train/test 固定日期文案。
2. Timing 页面正确显示新的窗口语义。
3. 区间回测首笔可见交易若存在，必须是 buy，不应出现切片后的冷启动卖出。
4. 窗口指标是 period-local，而不是继承更早累计收益。

## 备注

这份文档用于记录本轮“区间回测重构”的需求、约束、已完成改动与后续推进路径，后续每完成一个关键阶段应继续增量更新。
