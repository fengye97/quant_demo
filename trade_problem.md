# 择时回测系统：问题清单与修复计划

> 本文档用于用户 review。Review 通过后再分阶段动手改代码。
> 生成日期：2026-05-23

## Context

对 `stock_trade_demo` 择时策略做了严格 review，目标是：
1. 找出过拟合风险点
2. 校验前端是否真的渲染了最新策略
3. 找出潜在 bug
4. 并据此构建一套"符合真实市场交易规则"的回测系统

经过 3-angle code-review（line scan / removed-behavior / cross-file trace）+ verifier 复核 + 直接 grep/read 二次确认，已经定位 **5 个代码 bug + 1 个设计 bug + 6 类缺失的真实交易规则 + 2 个过拟合结构性风险**。

---

## 一、已确认的代码 Bug（P0：必须修）

### Bug 1 — Star50 staged 模式拿不到 ETF 价格，profit-lock 永不触发
- **位置**：`stock_trade_demo/timing/strategies.py:330` `Star50TimingStrategy.generate_signals`
- **现象**：调用 `_apply_exposure_columns(df, pd.Series(pos, ...), staged_strength=df['strength_score'], ready_mask=ready_mask)`，**漏传 `price_series=df['close']`**。
- **链路**：`base.py:161-162` 收到 `price_series=None` 时把价格全部填 NaN → `_build_staged_target_exposure` 中 `pd.notna(price)` 永远 False → `entry_price` / `peak_price` 一直是 None → `base.py:225` 的 `profit_lock_enabled` 分支不可能命中。
- **影响**：
  - **功能完全失效**：Star50 在 staged 模式下，无论用户把"盈利锁定"打开还是关闭、把锁盈三档阈值调到多少，仓位轨迹完全一致——锁盈分支永远不会进入。
  - **静默错误**：前端不会报错、API 不会报错、回测能正常出图，但实际逻辑与用户在 UI 上看到的不符。这是最危险的一类 bug——你以为参数生效了，其实没有。
  - **实盘风险**：科创 50 ETF 长期波动率高于沪深 300（年化 35% vs 22%），强趋势后的回吐幅度也大。原本设计 profit_lock 就是为了在 +28% 浮盈后从峰值回撤 4% 时锁仓到 25%，避免大幅回吐。失效后实盘账户在科创 50 一波强势行情后，最终可能多回吐 8~15 个点。
  - **历史结论作废**：所有"Star50 staged + profit_lock=on 不如 off"或类似的对比实验都不能作为决策依据。
  - **影响范围**：仅 Star50；CSI1000、ChinExt、Nasdaq、SP500、MacroV32 均正确传入 price_series，不受影响。
- **修复**：补传 `price_series=df['close']`，与 CSI1000 / ChinExt / Nasdaq / SP500 / MacroV32 保持一致。

### Bug 2 — `profit_lock_*` 参数在 API 层被丢弃
- **位置**：`stock_trade_demo/web_app.py:2091` `api_timing_backtest` 的参数提取循环
- **现象**：循环只取 `_TIMING_CACHE_DEFAULTS` 中的键，而 `_TIMING_CACHE_DEFAULTS`（web_app.py:1812-1831）**没有列出** `profit_lock_enabled / profit_lock_drawdown / profit_lock_level_1/2/3`。
- **影响**：
  - **5 个滑块全部失效**：`profit_lock_enabled`、`profit_lock_drawdown`、`profit_lock_level_1/2/3` 这 5 个前端参数全部进不了策略。无论你在 UI 上怎么拖，回测都按 `base.py:11-17` 里 `__init__` 的默认值跑（drawdown=0.04，三档=0.10/0.18/0.28，enabled=False）。
  - **影响所有 5 个 timing 策略**：与 Bug 1 只影响 Star50 不同，这条 bug 是 API 层的，对 CSI1000 / Star50 / ChinExt / Nasdaq / SP500 全部生效——只要走 `/api/timing/backtest` 这条路径就会被吞参数。
  - **静默归因错误**：你做调参实验时会看到"调大 drawdown 反而收益更好"这种反直觉结果，并可能因此得出错误的策略改进方向，浪费数轮调参时间。
  - **回测数据集污染**：所有以"调过 profit_lock_* 参数"为前提的 grid search 结果都作废，需要在修复后重跑。
  - **实盘风险**：实盘按代码默认值跑，不会被这条 bug 影响成交；但生产参数选型如果基于已污染的实验结论，则会把次优配置上线。
- **修复**：把 5 个 `profit_lock_*` 键加入 `_TIMING_CACHE_DEFAULTS` 和 `_US_TIMING_CACHE_DEFAULTS`；并核对前端是否真的把它们写入了 query string（必要时也补 form / json）。

### Bug 3 — `etf_open == 0.0` 触发 ValueError 崩溃
- **位置**：`stock_trade_demo/timing/backtest.py:156-159` `_replay_timing_positions`
- **现象**：进入循环前用 `notna()` 过滤，没过滤 `0.0`；后面在用 `etf_open` 做除数时若拿到 0，会直接 ValueError 中断整段回测。Sina 接口偶发返回 0 价位（临时停牌 / 跨夜停盘 / 数据修补）。
- **影响**：
  - **整段区间崩溃**：脏数据出现在第 N 根 bar，前面 N-1 根的撮合结果一起被丢弃，而不是只跳过当根。前端表现为 500 错误，用户看到"该策略不可用"的提示，但其实策略本身没问题。
  - **触发频率不稳定**：与 Sina 接口当时状态强相关。临时停牌、跨年/跨节首日、数据修补窗口都可能返回 0；正常时段几乎不出现。一旦命中，整个时间区间作废，用户需要换区间或等数据更新。
  - **复现困难**：调用方看到的是 ValueError，但堆栈不会指向脏数据本身，定位需要翻 panel。如果发生在 demo 现场，体验非常糟糕。
  - **数据风险**：当前没有任何 fallback。一旦 Sina API 升级或某只 ETF 长期返回异常价位，整个 timing 系统的某个 ETF 维度会持续不可用。
  - **不影响实盘**：实盘不走这条回测路径。
- **修复**：把判定改成 `pd.notna(etf_open) and etf_open > 0`，并把异常路径降级为 skip 当根 + 记录 warning，而不是 raise。

### Bug 4 — `filter_timing_result` 后 `etf_inception_date` 仍是全量 attr
- **位置**：`stock_trade_demo/timing/backtest.py:408` `filter_timing_result`
- **现象**：`result.attrs.update(result_df.attrs)` 把全量回测的 `etf_inception_date` 原样塞回筛选后的结果。区间切到 ETF 上市之前时，attr 仍指向全量 inception_date，前端"该 ETF 在区间内不可交易"的判定可能错位。
- **影响**：
  - **错位提示**：前端依赖 `etf_inception_date` 判定"该区间是否可交易"。区间起点早于 ETF 上市日时，attrs 仍指向全量起点（远早于 ETF 上市），前端误以为整段区间都可交易。
  - **虚假交易记录**：用户在前端选了一个跨越 ETF 上市日的区间（例如 "2018-01-01 到 2024-12-31" 看创业板 ETF 159205，而 159205 实际 2022 年才上市），可能在 2018-2022 段看到不存在的"持仓"或"指标曲线"。
  - **影响范围最大的 ETF**：
    - 159205（创业板 ETF）：2022 年上市，"全历史"区间会跨越 4 年虚假窗口。
    - 589850（科创 50 ETF）：2020 年才上市，"全历史"和 5 年区间都受影响。
    - 510980（中证 1000 ETF）、513500（标普 500 ETF）、159941（纳指 ETF）：上市时间较早，影响较小但仍存在。
  - **判断失真**：所有"看历史区间表现"的结论在这些 ETF 上都需重新审视。例如"科创 50 timing 在 2018 年熊市表现稳定"——其实 2020 年前根本没有 ETF。
  - **不影响实盘**：实盘只关注当前持仓，不受影响。
- **修复**：在 `filter_timing_result` 内重新计算区间内真实存在 ETF bar 的最早日期，再写回 attrs；区间内若全无真实 ETF bar，应返回明确的不可交易状态而不是套用全量 inception。

### Bug 5 — `first_real_etf_date` 初始化泄漏
- **位置**：`stock_trade_demo/timing/backtest.py:323-326`
- **现象**：`first_real_etf_date = df['交易日期'].min() if len(df) else None`，之后才用 `has_real_etf_bar` 过滤。如果 `has_real_etf_bar` 全为 False（ETF 尚未上市的早期数据），`first_real_etf_date` 仍保留为 panel 全量最早日期。
- **影响**：
  - **比 Bug 4 更严重**：Bug 4 只是 attrs 错位，前端看到的曲线是 0；Bug 5 会让 `_replay_timing_positions` 真的从 panel 全量最早日期开始撮合，用 **指数价格**（或 forward fill 的脏数据）替代 ETF 价格成交，等于在 ETF 上市前凭空跑出 1~5 年的"假交易"。
  - **硬性约束违反**：直接违反 CLAUDE.md 中"不得伪造 ETF 历史""不得使用指数价格/前后填充替代 ETF 价格"的明确要求。
  - **回测虚增收益**：用指数价格在 ETF 上市前撮合时，缺少 ETF 跟踪误差、申赎损耗、流动性溢价，会系统性高估回测收益。修复后，全历史回测的累计收益会下降。
  - **历史结论作废**：基于 159205/589850 全历史回测的所有判断（包括"是否上线生产""参数调优""策略 ranking"）都需要在修复后重新跑。
  - **触发条件**：当 `has_real_etf_bar` 全为 False（panel 数据全部早于 ETF 上市），或者 panel 起点远早于 ETF 上市日时触发。当前默认全历史 panel 从 2018 年起，所以 2020 年才上市的 589850 和 2022 年才上市的 159205 都会命中。
  - **不影响实盘**：实盘只用最新数据，不受影响。
- **修复**：基于 `has_real_etf_bar==True` 子集再取 `min()`；若该子集为空，返回 `None` 并把整段回测降级为 non-tradable。

---

## 二、Staged 模式的设计 Bug（P0）

### 现象
`stock_trade_demo/timing/base.py:244-275` `_apply_exposure_columns`：

```python
if self.exposure_mode == 'staged' and staged_strength is not None:
    target_exposure = self._build_staged_target_exposure(...)
else:
    target_exposure = binary_position.astype(float)
```

当 `exposure_mode == 'staged'` 时，**`binary_position` 被完全忽略**。`binary_position` 是各策略基于 breakout、MACD、TrendMA、Momentum、Macro 等 buy_cond/sell_cond 状态机算出来的执行信号；staged 模式只看 `strength_score` 是否跨过 `enter_threshold` 之类的桶阈值。

### 影响
- breakout 触发条件、跌破清仓、确认天数、试探建仓等核心状态机在 staged 模式下都形同虚设。
- 用户在 binary 模式下回测出来的"逻辑"和 staged 模式下实际跑出来的"逻辑"完全是两套东西。
- 任何用 staged 模式做的过拟合判断都建立在错误前提之上。

### 用户已选定的修复方案：binary 作为门控叠加
- `binary_position == 0` 时强制 `target_exposure = 0`（清仓优先）。
- `binary_position == 1` 时按 `_build_staged_target_exposure` 的 strength bucket 决定档位。
- 把这条 gating 注入 `_build_staged_target_exposure` 的 `ready_mask` 参数（与 `binary_position` 取 AND），或在 `_apply_exposure_columns` 算完 staged 后再 `target_exposure = target_exposure.where(binary_position > 0, 0.0)`。
- 仍需通过 `confirm_days` 等连续性判定，避免与现有状态机重复打架。

---

## 三、缺失的真实市场交易规则（P0，用户要求"全部都加"）

当前回测引擎在 `stock_trade_demo/timing/backtest.py` 中假定 t 收盘出信号、t+1 开盘以未做任何调整的价格成交，且现金可立刻回收复用。这是一套理想化模型，与真实 ETF 交易存在 6 类系统性偏差：

### 3.1 T+1 结算（A 股 ETF）
- **现状**：`_replay_timing_positions` 中卖出后现金当日可用（line 213-228），下一根 bar 立刻可买。
- **真实规则**：A 股 ETF（含 510980 / 159205 / 589850）当日卖出资金 T+1 可用、当日买入份额 T+1 才可卖。跨境 ETF（513500 / 159941）是 T+0，需要区分。
- **修复**：
  - 在 `index_data.TIMING_ETF_CONFIGS` 中给每个 ETF 标注 `settlement: 't+0' | 't+1'`。
  - `_replay_timing_positions` 维护 `pending_cash` 队列与 `pending_shares` 队列，按 settlement 模式延迟释放。
  - 当日卖出释放的现金不计入当日可用资金；当日买入的份额不计入当日可卖份额。

### 3.2 涨跌停板（A 股 ETF 10%、创业板 ETF 20%、科创 ETF 20%、跨境 ETF 无）
- **现状**：成交价 = 直接采用次日开盘价，没有任何撮合限制。
- **真实规则**：
  - 510980（中证 1000 ETF）：±10%
  - 159205（创业板 ETF）：±20%
  - 589850（科创 50 ETF）：±20%
  - 513500 / 159941（跨境）：无涨跌停
- **修复**：
  - `TIMING_ETF_CONFIGS` 标注 `limit_pct`（None 表示无限制）。
  - 取前一交易日真实收盘价（adjusted），算 `upper_limit = prev_close * (1+limit_pct)`、`lower_limit = prev_close * (1-limit_pct)`。
  - 撮合时：若开盘价 >= upper_limit ⇒ 买单视为无法成交（涨停封板），sell 仍可挂单；若 <= lower_limit ⇒ 卖单视为无法成交，buy 仍可挂单。
  - 无法成交时延迟一日，仍不成交则继续延，并在交易明细里标记 `blocked_by_limit`。

### 3.3 ETF 复权（关键准确性问题）
- **现状**：`index_data._fetch_daily_kline` 从 Sina API 拉取 **未复权** 价格。ETF 分红除息日（如 510980 每年 12 月）会出现 1~3% 的"假跌"，被策略误判为下跌信号；持仓收益曲线也会出现"凭空缩水"。
- **修复路径**（按可行性排序）：
  1. **首选**：改用前复权数据源（Tushare ETF 复权接口或 akshare `fund_etf_hist_em`，二者都提供 `qfq` 接口）。
  2. **次选**：从交易所 ETF 公告抓分红记录，按除息日回溯调整 `open / high / low / close`（保持成交量不变）。
  3. **下限**：在 `TIMING_ETF_CONFIGS` 给每个 ETF 列出已知 ex-dividend 日期 + 每份分红金额，在 `_fetch_daily_kline` 后做"前复权调整"层。
- 任何路径都必须保留：CLAUDE.md 中"不得伪造 ETF 历史"的硬性约束依然成立——只复权真实存在的 bar，不补造。

### 3.4 开盘价滑点
- **现状**：成交价 == ETF 开盘价，零滑点。
- **真实规则**：用市价单 / 开盘集合竞价的实际成交价通常会偏离开盘价数 bp。
- **修复**：
  - 新增策略参数 `slippage_bps`，默认 5bp。
  - 买入：`fill_price = open_price * (1 + slippage_bps/1e4)`。
  - 卖出：`fill_price = open_price * (1 - slippage_bps/1e4)`。
  - 暴露在前端 `get_shared_parameter_definitions` 与 API 参数中。

### 3.5 现金计息
- **现状**：空仓 / 半仓的闲置现金按 0 利率持有，等同于零成本现金垫。
- **真实规则**：闲置现金可放货币基金 / 国债逆回购，年化 1.5~2.5%。
- **修复**：
  - 新增 `cash_interest_rate` 参数，默认 1.5% 年化。
  - 每个交易日对 `cash` 余额计 `cash * cash_interest_rate / 252`。
  - 须与回测基准一致暴露（默认基准也加同等现金计息时再做对比）。

### 3.6 费用拆分（佣金 / 印花税 / 过户费）
- **现状**：`buy_cost / sell_cost` 单一 0.1% 系数，不区分项目。
- **真实规则**：
  - 佣金：双边收取，万分之一，最低 5 元 / 笔。
  - 印花税：**ETF 免征**，股票卖出 0.05%（2026 年现行）。
  - 过户费：沪市 0.001%（2025 年起调整），深市无。
- **修复**：
  - 参数拆为 `commission_rate`、`commission_min`、`stamp_tax_rate`（ETF 设 0）、`transfer_fee_rate`。
  - `TIMING_ETF_CONFIGS` 标注 ETF 是否沪市/深市以决定 `transfer_fee_rate`。
  - 在 trade_details 里输出 `commission / stamp / transfer / slippage` 四个分项，便于审计。

---

## 四、过拟合结构性风险（P1，需做实验验证）

### 4.1 MacroV32 的 "OOS 2019-2026" 实际上是 in-sample
- **现象**：`MacroV32TimingStrategy` 的 8 个 FRED 因子、阈值、权重、sigmoid 中点 0.5 都是基于 2010-2026 全样本数据 z-score 化得到的，"OOS"段也用了同一套统计量做归一化。
- **风险**：宣称的 2019-2026 跑分含 look-ahead 偏差。
- **修复方向**：
  - 用 expanding window 重新计算 z-score（每个时点只用当时可得的历史均值/方差）。
  - 把权重 / 阈值的拟合期与评估期严格切分（如 fit 2010-2018、test 2019-2026），不再共享因子归一化统计量。
  - 报告时把 "OOS" 改名为 "in-sample 2010-2026"，重新跑严格 OOS 后再决定是否上线。

### 4.2 CSI1000 / Star50 窗口参数有 in-sample tuning 嫌疑
- **现象**：CSI1000 的 breakout 窗口、MACD 参数、TrendMA 长度都是手工调出来的数值（如 17/35、11/26 这种非常规组合）。
- **风险**：在不同窗口长度上做敏感性测试时收益跌得很快，符合 over-tuned 特征。
- **修复方向**：
  - 用 walk-forward：每年滚动 refit 一次窗口长度，看 OOS 是否仍稳定优于默认参数。
  - 给关键窗口（n1, n2, ma_len, mom_window）做 ±20% 的网格扰动，记录收益方差；方差大说明依赖窄窗口。

---

## 五、P0 Bug 综合影响（决策依据）

下面按"是否影响历史结论 / 是否影响实盘 / 影响幅度"三个维度，把每个 P0 项的实际后果写清楚，作为是否本轮一次性修完的决策依据。

### 5.0.1 Bug 1（Star50 profit-lock 失效）
- **回测可信度**：现有 Star50 staged + profit_lock 的所有回测结论**全部不可信**。日志里看到 "锁盈触发 0 次" 不是策略保守，而是代码根本没进入锁盈分支。
- **实盘偏差**：若按当前代码部署，遇到 18%/28% 阶梯应该锁仓的场景实际不会减仓，强趋势回吐时账户净值会比回测预期多吐 5~15 个点。
- **历史结论是否要重跑**：是。Star50 上一切"profit_lock=on 不如 off"的对比都要在修复后重跑。
- **修复成本**：1 行。

### 5.0.2 Bug 2（profit_lock_* 参数 API 层被吞）
- **回测可信度**：所有 5 个 timing 策略上"调 profit_lock 参数看效果"的实验**全部等同于没调**。前端滑块拖动有视觉反馈，但请求到后端就被丢弃，回测结果永远是策略默认值。
- **实盘偏差**：不直接影响实盘（实盘按代码默认值跑），但会让你在调参时做出错误归因（"调大 drawdown 反而收益更好" → 其实根本没生效）。
- **历史结论是否要重跑**：所有"调 profit_lock_* 参数得到的对比"实验都要作废重跑。
- **修复成本**：加 5 行字典 key。

### 5.0.3 Bug 3（etf_open == 0 崩溃）
- **回测可信度**：偶发，与数据源新鲜度强相关。每次 Sina 接口抽风都可能在某个区间触发，导致该区间整段回测失败而不是局部跳过。
- **实盘偏差**：实盘不调这条路径，但如果生产监控基于回测复跑，会被噪声中断。
- **历史结论是否要重跑**：否，但要保证后续回测不再被单根脏数据拖垮。
- **修复成本**：改 1 个条件 + 1 次 warning。

### 5.0.4 Bug 4（filter_timing_result 残留全量 inception）
- **回测可信度**：用户切换"近 1 月 / 近 1 季 / 近 6 月"区间时，区间起点早于 ETF 上市日的边界情况下，前端可能看到不该存在的"在 ETF 上市前已经持仓"的伪迹。
- **实盘偏差**：不影响实盘，但会让"用某段时间验证策略"的判断失真，可能误以为策略在 ETF 上市前就有 alpha。
- **历史结论是否要重跑**：仅在区间跨越 ETF 上市日时存在；159205（创业板）2022 年上市、589850（科创 50）2020 年上市，这两个 ETF 的"全历史"实验最容易撞到。
- **修复成本**：filter_timing_result 内补 5~10 行重算。

### 5.0.5 Bug 5（first_real_etf_date 初始化泄漏）
- **回测可信度**：与 Bug 4 同源，但更严重——`_replay_timing_positions` 会从 panel 全量最早日期开始撮合，相当于在 ETF 上市前用指数价格"模拟"了一段虚假成交。这直接违反 CLAUDE.md 中"不得伪造 ETF 历史"的硬约束。
- **实盘偏差**：不影响实盘，但任何引用全历史回测的对比基准都偏高（多了几年免费 alpha）。
- **历史结论是否要重跑**：是。159205 / 589850 的全历史回测、原仓位 ensemble 的 timing gate 全历史曲线都要重跑确认。
- **修复成本**：改 1 行 + 1 个分支。

### 5.0.6 Staged 设计 Bug（binary_position 被完全忽略）
- **回测可信度**：**这是 6 项里最严重的一个**。所有 timing 策略只要切到 staged，breakout / MACD / TrendMA / Momentum / Macro 的状态机就被旁路，仓位完全由 strength_score 的桶阈值决定。`STRATEGY_CHANGELOG.md` 里所有 "staged vs binary" 对比都不是同一套逻辑在比，而是 "状态机 vs 单 sigmoid 阈值" 在比。
- **实盘偏差**：当前 `original_ensemble` 默认走 staged 路径；这意味着生产策略的实际持仓决策与你以为的逻辑**完全不同**，靠的是 strength_score 而非各因子触发。如果近期 strength_score 因数据漂移而偏高，可能一直 100% 持仓而忽略 breakout 已经失效。
- **历史结论是否要重跑**：必须。所有"staged 模式更稳"的结论都需要在 binary-gating 修复后重新验证；很可能修复后 staged 收益曲线会向 binary 收敛。
- **修复成本**：5~10 行。

### 5.0.7 6 项真实交易规则缺失（集合影响）
- **回测可信度**：当前回测系统性高估 ETF 策略真实表现。粗估各项影响幅度：
  - 未复权价格：CSI1000/Star50 全历史年化少算 1~2%（ETF 分红被当成下跌）。
  - 零滑点：每次开盘价成交相比真实成交多赚 ~5bp / 笔，年换手 12 次时约 0.6%/年。
  - 零费用拆分：当前 0.1% 双边 = 0.2% 单次往返，真实 ETF 万分之一双边 = 万分之二，回测**高估了成本**，修完反而是利好（年化 +0.1~0.3%）。
  - 无 T+1：当日卖出当日买回的"小动作"在 A 股 ETF 上做不到，但当前代码允许；这种快速反手在回撤段往往是收益来源，修完后部分策略收益会下降。
  - 无涨跌停：单边连续涨停的极端行情下，当前代码假装能成交；真实账户会卡在板上，错过整段行情或下跌段被套住。
  - 无现金计息：空仓期每年少算 1.5% 收益；MacroV32 这类长期空仓策略影响最大（修完反而是利好）。
- **实盘偏差**：上述偏差**有正有负**，净效应不可凭直觉判断，必须全部接入后才能给出真实的策略 ranking。
- **历史结论是否要重跑**：是。`STRATEGY_CHANGELOG.md` 中所有 timing 策略之间的相对排名都需要在真实规则下重跑确认。
- **修复成本**：这是最重的一块，预计 200~400 行新代码 + 数据源切换 + 元数据补全。

### 5.0.8 综合判断
- **必须本轮一次性修完的**：Bug 1、Bug 2、Staged 设计 bug、6 项真实规则——因为它们共同决定了"我们能不能相信现在和未来的所有回测结论"。
- **可以本轮修也可以下轮修的**：Bug 3、Bug 4、Bug 5——影响范围小，但成本极低（合计 < 20 行），建议顺手一起改。
- **修复后必做**：对 5 个 ETF × 3 个区间做修复前/后对照基准跑，把"哪些策略实际上是被失效的代码"和"哪些策略在真实规则下仍然稳"分开列表，再决定哪些策略可以留在生产。

---

## 六、修复计划与优先级

### P0（本轮一起修，用户已批准 "全部一起修"）
1. 改 `Star50TimingStrategy.generate_signals`，补 `price_series=df['close']`。
2. 在 `_TIMING_CACHE_DEFAULTS / _US_TIMING_CACHE_DEFAULTS` 加入 5 个 `profit_lock_*` key；核对前端是否真的提交。
3. `_replay_timing_positions` 的 `etf_open` 守护从 `notna` 改为 `notna and > 0`；脏数据 skip 而非 raise。
4. `filter_timing_result` 内重算 `etf_inception_date`；无真实 bar 时给出 non-tradable。
5. `first_real_etf_date` 改成基于 `has_real_etf_bar==True` 取 min。
6. Staged 模式接入 binary gating：`_apply_exposure_columns` 中先按 binary_position mask 强制清仓，再叠加 strength bucket。
7. T+1 结算队列、涨跌停板撮合、复权价格、滑点、现金计息、费用拆分 6 项全部接入；新增的参数从前端 → API → 策略实例化全链路打通。
8. `TIMING_ETF_CONFIGS` 补 `settlement / limit_pct / market / dividend_adjust_method` 元数据。

### P1（修完 P0 后立即跟进）
- MacroV32 expanding-window z-score 与严格 OOS 切分。
- CSI1000 / Star50 参数敏感性 + walk-forward 报告。

### P2（结构性，等 P0/P1 验证完再讨论）
- 是否引入独立 regime 切换层（避免 timing gate 只在小市值月度选股之外做"开关"）。
- 是否把择时引擎与 `original_ensemble` 的 `growth_timing_mode` 解耦。

---

## 七、验证方案

### 7.1 回归测试（必须新增）
新建 `stock_trade_demo/tests/test_timing_realism.py`（或扩展现有脚本），覆盖：
- Star50 staged + profit_lock 开启时，至少在历史某段命中过 lock 触发；修复前命中数应为 0、修复后 > 0。
- 构造 mock panel：当日 ETF 开盘 == 0 → 不再 raise，且当日 trade 被跳过。
- 区间起点早于 ETF 上市日：返回 non-tradable，`etf_inception_date` 不再泄漏全量值。
- Staged 模式：构造 `binary_position` 长期为 0 但 `strength_score` 高 的样本，目标仓位必须保持 0。
- T+1：构造连续 buy → 当日 sell → 次日 buy 的场景，验证可用资金/可卖份额延迟。
- 涨跌停：构造开盘价 = 前收 × 1.1 的样本，验证买单不成交。

### 7.2 端到端验证
按 CLAUDE.md 的"先确认占用 8080 的是哪个进程"流程：
1. `lsof -nP -iTCP:8080 -sTCP:LISTEN` 查活进程。
2. 用 `/Users/fatcat/opt/anaconda3/bin/python web_app.py` 重启。
3. 命中 `/api/timing/backtest`，确认返回里出现新字段：`settlement_mode / limit_blocks / cash_interest / commission / stamp / transfer / slippage`。
4. 前端 `timing.html` 各区间切换不报 500；trade_details 里能看到费用拆分。
5. 用 `frontend-screenshot-verify` 技能截图确认。

### 7.3 对照基准
- 修复前后各跑一次 5 个 ETF 的全历史 + 三个验证区间。
- 报告：累计收益、年化、最大回撤、交易笔数、平均费用比、被涨跌停拦下的次数。
- 任何一项指标若出现 > 5% 的反向偏差，必须能用上述新规则解释清楚。

---

## 八、Review 待确认事项

请逐项确认/修改后再让我动代码：
1. **P0 修复范围**是否全部认可？是否要剔除/延后某一项？（参考 §5.0.8 综合判断）
   1. 回复：全部认可
2. **复权数据源**优先级：是否同意首选 akshare `fund_etf_hist_em`（无需 token），还是要先评估 Tushare？
   1. 回复：同意优先使用akshare
3. **默认参数**：`slippage_bps=5`、`cash_interest_rate=1.5%`、`commission_rate=0.0001`（万分之一）、`commission_min=5` 这套默认值是否接受？
   1. 回复：接受
4. **是否要保留 binary 模式不变**（即 binary 模式下不引入 T+1 之外的真实规则），还是 binary/staged 共用同一套真实规则层？建议共用。
   1. 回复：公用一套真实规则层
5. **过拟合 P1 项**：是否本轮一起做实验，还是 P0 修复落地后再开新工单？
   1. P0修复落地后再开新工单
